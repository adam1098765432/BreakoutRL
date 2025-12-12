"""
Terms
- Policy: The probability of taking an action given a state
- Value: The predicted winner or final score
- Immediate Reward: The reward for taking an action
- Trajectories: A sequence of states and actions

MuZero
- Learns a latent model of the environment
- Uses Monte Carlo Tree Search (MCTS)
- Requires past sequences of observations (frames/states) and actions to be stored
- Actions are encoded as constant bias planes (additional channels to the latent representation)
- MCTS predicts the next state using the previous state and action
  - Only masks valid actions at the root of the search tree
- MCTS can proceed past a terminal node, and is expected to return the same terminal state

Functions
- h: Representation function (predicts latent representation for a given state)
- g: Dynamics function (predicts next state and immediate reward given previous state and action)
  - r^k: Reward network (predicts immediate reward for a given state and action)
  - s^k: State network (predicts next state for a given state and action)
- f: Prediction function
  - p^k: Policy network (predicts the immediate action probabilities for a given state)
  - v^k: Value network (predicts the final reward for a given state)

Loss
- Loss = MSE(reward) + MSE(value) + CELoss(policy) + L2_Regularization
- L2_Regularization (weight decay) = theta_f + theta_h + theta_g

Replay Buffer
- Stores trajectories: K * (prev_state, next_state, action, reward, is_done)

Targets
- Value target: the discounted cumulative rewards over multiple timesteps
- Policy target: the final action probabilities from the MCTS search tree
  (normalized visit counts from MCTS at the root)
- Reward target: the observed immediate reward at each step

Categorical reward and value
- Rewards and values are encoded as probability distributions for scores -300 to 300
- Targets are mapped to this distribution (phi)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq

STATE_SIZE = 10
ACTION_SIZE = 3
SUPPORT_SIZE = 101 # Categorical reward and value [-50, 50] (see Appendix F of MuZero paper)
K_STEPS = 5

def upper_confidence_bound(mcts_value, policy_score, parent_visits, child_visits, c1=1.25, c2=19652):
  """
  The mcts_value should be normalized across all actions [0, 1].
  
  :param mcts_value: The propogated score from MCTS.
  :param total_mcts_value: The propogated scores for all actions from the parent node.
  :param policy_score: The score of the action predicted by the policy.
  :param parent_visits: The number of times the parent node has been visited.
  :param child_visits: The number of times the child node has been visited.
  :param c1: The exploration weight.
  :param c2: The exploration decay.
  """
  return mcts_value + policy_score * np.sqrt(parent_visits) / (1 + child_visits) * (c1 + np.log((parent_visits + c2 + 1) / c2))

def scale_targets(x, eps=1e-3):
  """
  MuZero Appendix F: Network Architecture says for a value and reward prediction
  we scale the targets before we obtain the categorical representations.

  :param x: The target (value or reward)
  :param eps: Epsilon
  """
  return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + eps * x)

def one_hot_score(x):
  """
  One-hot encoding for the reward and value
  
  :param x: The target (value or reward)
  """
  if x < -SUPPORT_SIZE // 2 or x > SUPPORT_SIZE // 2:
    raise ValueError(f"x must be between -{SUPPORT_SIZE // 2} and {SUPPORT_SIZE // 2}")
  arr = torch.zeros(SUPPORT_SIZE)
  arr[x + 50] = 1
  return arr

def one_hot_action(x):
  """
  One-hot encoding for the action

  :param x: The action index
  """
  arr = torch.zeros(ACTION_SIZE)
  arr[x] = 1
  return arr

class ReplayBuffer:
  """
  Stores the trajectories: (prev_state, next_state, action, reward, is_done)
  """
  def __init__(self, capacity):
    self.buffer = []
    self.capacity = capacity

  def add_trajectory(self, trajectory):
    """
    :param trajectory: Comes from the MCTS search
    """
    self.buffer.append(trajectory)
    if len(self.buffer) > self.capacity:
      self.buffer.pop(0)

  def get_sampling_priority(mcts_value, target_value):
    """
    Based on the MuZero Appendix G<br>
    This function is for choosing the replay sample from the replay buffer to train with.
    The higher the difference in mcts_value and target_value, the higher the
    priority (this corresponds to uncertainty).
    
    ### Important!
    To get a probability, normalize all priorities across all replay samples
    so that you can correct for sampling bias in the future. This is done by scaling
    the loss by w_i = (1 / N) * (1 / P(i)), where N is the number of replay samples
    and P(i) is the priority of the ith replay sample.

    :param mcts_value: The search value for the replay sample
    :param target_value: The target value from the replay sample
    """
    return np.abs(mcts_value - target_value) # Don't forget to normalize to get the probability!

class PredictionModel(nn.Module):
  """
  Two-headed model for predicting policy and value from a state.
  """
  def __init__(self, latent_size=16):
    super().__init__()
    self.fc1 = nn.Linear(STATE_SIZE, latent_size)
    self.fc2 = nn.Linear(latent_size, latent_size)
    self.policy = nn.Linear(latent_size, ACTION_SIZE)
    self.value = nn.Linear(latent_size, SUPPORT_SIZE)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return self.policy(x), self.value(x)

class ResBlock(nn.Module):
  """
  This residual block is based on https://arxiv.org/pdf/1603.05027.pdf<br>
  It is uses the constant scaling method since it is not a CNN.

  :param channels: The number of channels in the input
  :param alpha: The scaling factor
  """
  def __init__(self, channels, alpha=0.2):
    super().__init__()
    self.alpha = alpha
    self.fc1 = nn.Linear(channels, channels)
    self.fc2 = nn.Linear(channels, channels)

  def forward(self, x):
    identity = x
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return identity + x * self.alpha

class DynamicsModel(nn.Module):
  """
  Model Architecture Based On: https://arxiv.org/pdf/1603.05027<br>
  Two-headed model for predicting next state and reward from a state and action.
  Usually, you pass in multiple previous states and actions, but for now we will
  only pass in a single previous state and action. This may make it harder to train.

  ### Note
  MuZero Appendix G says to scale the gradient of the dynamics function by 0.5.
  To do this, the input state is multiplied by 0.5.

  ### Note
  MuZero Appendix G says to scale the hidden state after running the dynamics
  function to [0, 1] (once per unroll step).

  :param latent_size: The number of channels in the latent representation.
  :param n_blocks: The number of residual blocks.
  """
  def __init__(self, latent_size=16, n_blocks=3):
    super().__init__()
    self.first = nn.Linear(STATE_SIZE + ACTION_SIZE, latent_size)
    self.model = nn.Sequential(*[ResBlock(latent_size) for _ in range(n_blocks)])
    self.state = nn.Linear(latent_size, STATE_SIZE)
    self.reward = nn.Linear(latent_size, SUPPORT_SIZE)

  def forward(self, state, action):
    state = state * 0.5 # Gradient Scaling
    x = torch.cat([state, action], dim=1)
    x = F.relu(self.first(x))
    x = self.model(x)
    state = F.relu(self.state(x))
    reward = self.reward(x)
    state_mins = state.min(dim=1, keepdim=True)[0]
    state_maxs = state.max(dim=1, keepdim=True)[0]
    state = (state - state_mins) / (state_maxs - state_mins + 1e-6)
    return state, reward

class Node:
  """
  Node in the MCTS search tree.
  """
  largest_Q = []

  def __init__(self, state, action, parent=None):
    self.state = state
    self.parent = parent
    self.visits = 0
    self.G_k = 0
    self.children = []
    self.action = action
    self.reward = 0
    self.mcts_value = 0
    self.policy_score = 0

  def backprop(self, child):
    """
    Based on the MuZero Appendix B
      G_k = sum from rho=0 to l-1-k of gamma^rho * r_{k+1+rho} + gamma^(l-k) * v^l
      For k=l...0, we form an l-k step estimate of the cumulative discounted reward
    
    Convert to recurrence relation:
      G_k = r_{k+1} + gamma * G_{k+1}
    
    mcts_value recurrence relation:
      Q_k = (N_k * Q_k + G_k) / (N_k + 1)
    
    visits recurrence relation:
      N_k = N_k + 1
    
    Legend:
      gamma = discount factor
      G = cumulative discounted reward
      r = reward
      N = number of visits
      Q = mcts_value
      l = leaf node
      k = parent node (this node)
      k+1 = child node (argument)
    """
    gamma = 0.95 # Discount factor
    r_kp1 = child.reward # Immediate reward
    G_kp1 = child.G_k # Cumulative discounted reward

    # Update formulas denoted in MuZero Appendix B
    self.G_k = r_kp1 + gamma * G_kp1
    self.mcts_value = (self.visits * self.mcts_value + self.G_k) / (self.visits + 1)
    self.visits += 1

class MCTS:
  """
  The MCTS search tree for the MuZero model.
  ### Note:
  There is no rollout since a value estimate is used instead.
  """
  def __init__(self, dynamics_model, prediction_model):
    self.dynamics_model = dynamics_model
    self.prediction_model = prediction_model

  def search(self, root):
    """
    ### Selection
    Select the action with the highest upper confidence bound.
    Repeat until a leaf node (s^l, a^l) is reached.
    """
    node = root
    action_idx = None

    while node.children != []:
      best_ucb = -float('inf')
      best_child = None
      
      for idx, child in enumerate(node.children):
        mcts_value = child.mcts_value
        policy_score = child.policy_score
        parent_visits = node.visits
        child_visits = child.visits
        child_ucb = upper_confidence_bound(mcts_value, policy_score, parent_visits, child_visits)
        
        if child_ucb > best_ucb:
          best_ucb = child_ucb
          best_child = child
          action_idx = idx

      node = best_child

    self.expand(node, one_hot_action(action_idx))

  def expand(self, child, action):
    """
    ### Expansion
    Reward and state are computed by the dynamics function and stored in tables.
    In this case, they are stored in the child node.
    Policy and value are computed by the prediction function.
    A new node (s^l, a^l) is added to the search tree.
    Each edge is initialized to N=0, Q=0, P=p^l.

    :param child: The node to expand
    :param action: The action that led to the child node
    """
    # Expand child node
    for _ in range(ACTION_SIZE):
      child_child = Node(None, None, child)
      child.children.append(child_child)

    # Compute statistics for child
    state, reward = self.dynamics_model(child.parent.state, action)
    policy, value = self.prediction_model(state)
    child.state = state
    child.action = action
    child.policy_score = policy
    child.reward = reward
    child.mcts_value = value
    child.G_k = value

    # Backup
    node = child
    while node.parent is not None:
      node.parent.backprop(node)
      node = node.parent

def train():
  """
  Training loop.
  
  ### Note
  Remember to scale the loss by 1 / K_STEPS to ensure the gradient has a similar magnitude
  regardless of the number of unroll steps.
  """
  pass
