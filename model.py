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
import random

"""
TODO:
- Fix Node and MCTS class to reflect the pseudocode << Emmett
- Finish the replay buffer class << Justin
- Add self play (data generation)
- Complete training loop (how are the gradients stored?)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


MAX_FLOAT = float('inf')
STATE_SIZE = 10
ACTION_SIZE = 3
SUPPORT_SIZE = 101 # Categorical reward and value [-50, 50] (see Appendix F of MuZero paper)
K_STEPS = 5
DISCOUNT_FACTOR = 0.997

class MinMaxStats:
  def __init__(self, min_val=None, max_val=None):
    self.max = max_val if max_val is not None else -MAX_FLOAT
    self.min = min_val if min_val is not None else MAX_FLOAT

  def update(self, val):
    self.max = max(self.max, val)
    self.min = min(self.min, val)

  def normalize(self, val):
    if self.max > self.min:
      return (val - self.min) / (self.max - self.min)
    return val

class Action(object):

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class Player(object):
  pass

class Node:
  """
  Node in the MCTS search tree.
  """
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: List[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> List[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()

class Environment(object):
  """The environment MuZero is interacting with."""

  def step(self, action):
    pass

class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(self, action_space_size: int, discount: float):
    self.environment = Environment()  # Game specific environment.
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    pass

  def legal_actions(self) -> List[Action]:
    # Game specific calculation of legal actions.
    return []

  def apply(self, action: Action):
    reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in action_space
    ])
    self.root_values.append(root.value())

  def make_image(self, state_index: int):
    # Game specific feature planes.
    return []

  def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                  to_play: Player):
    # The value target is the discounted root value of the search tree N steps
    # into the future, plus the discounted sum of all rewards until then.
    targets = []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
      bootstrap_index = current_index + td_steps
      if bootstrap_index < len(self.root_values):
        value = self.root_values[bootstrap_index] * self.discount**td_steps
      else:
        value = 0

      for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
        value += reward * self.discount**i  # pytype: disable=unsupported-operands

      # For simplicity the network always predicts the most recently received
      # reward, even for the initial representation network where we already
      # know this reward.
      if current_index > 0 and current_index <= len(self.rewards):
        last_reward = self.rewards[current_index - 1]
      else:
        last_reward = 0

      if current_index < len(self.root_values):
        targets.append((value, last_reward, self.child_visits[current_index]))
      else:
        # States past the end of games are treated as absorbing states.
        targets.append((0, last_reward, []))
    return targets

  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)

class ReplayBuffer:
  """
  Stores the trajectories: (prev_state, next_state, action, reward, is_done)
  """
  def __init__(self, capacity, batch_size):
    self.buffer = []
    self.batch_size = batch_size
    self.capacity = capacity

  def save_games(self, game):
    self.buffer.append(game)
    if len(self.buffer) > self.capacity:
      self.buffer.pop(0)

  def sample_batch(self, num_unroll_steps: int, td_steps: int):
      games = [self.sample_game() for _ in range(self.batch_size)]
      game_pos = [(g, self.sample_position(g)) for g in games]
      return [(g.make_image(i), g.history[i:i + num_unroll_steps],
               g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
              for (g, i) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return random.choice(self.buffer)

  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    T = len(game.history)
    return random.randint(0, T - 1)

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


class MCTS:
  """
  The MCTS search tree for the MuZero model.
  ### Note:
  There is no rollout since a value estimate is used instead.<br>
  The Dynamics and Prediction models are only used **once** per simulation.
  """
  def __init__(self, dynamics_model: DynamicsModel, prediction_model: PredictionModel):
    self.dynamics_model = dynamics_model
    self.prediction_model = prediction_model
    self.n_simulations = 50

  def select_child(self, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max((
      ucb_score(node, child, min_max_stats),
      action,
      child
    ) for action, child in node.children.items())

    return action, child

  def search(self, root: Node, action_history: list[int]):
    min_max_stats = MinMaxStats()

    for _ in range(self.n_simulations):
      history = action_history.copy()
      node = root
      search_path = [node]

      while node.expanded():
        action, node = self.select_child(node, min_max_stats)
        history.append(action)
        search_path.append(node)

      parent = search_path[-2]
      hidden_state, reward = self.dynamics_model(parent.hidden_state, action)
      policy_logits, value = self.prediction_model(hidden_state)

      self.expand_node(node, hidden_state, reward, policy_logits)
      self.backprop(search_path, value, DISCOUNT_FACTOR, min_max_stats)
  
  def expand_node(self, node: Node, hidden_state: torch.Tensor, reward: torch.Tensor, policy_logits: torch.Tensor):
    node.hidden_state = hidden_state
    node.reward = reward
    policy = F.softmax(policy_logits)

    for i in range(ACTION_SIZE):
      child = Node(policy[i])
      node.children[i] = child

  def backprop(self, search_path: list[Node], value: torch.Tensor, discount: float, min_max_stats: MinMaxStats):
    for node in reversed(search_path):
      node.value_sum += value # Add negative if it's the opponent's turn
      node.visit_count += 1
      min_max_stats.update(node.value())
      value = node.reward + discount * value

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

def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats, c1=1.25, c2=19652):
  """
  :param parent: The parent node
  :param child: The child node
  :param c1: The exploration weight.
  :param c2: The exploration decay.
  """
  discount = DISCOUNT_FACTOR
  prior_weight = np.sqrt(parent.visit_count) / (1 + child.visit_count)
  prior_weight *= (c1 + np.log((parent.visit_count + c2 + 1) / c2))
  prior_score = child.prior * prior_weight
  if child.visit_count > 0:
    value_score = child.reward + discount * min_max_stats.normalize(child.value())
  else:
    value_score = 0
  return prior_score + value_score

def train():
  """
  Training loop.
  
  ### Note
  Remember to scale the loss by 1 / K_STEPS to ensure the gradient has a similar magnitude
  regardless of the number of unroll steps.
  """
  pass
