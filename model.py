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
- Actions are encoded as constant bias planes
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

STATE_SIZE = 10
ACTION_SIZE = 3
SUPPORT_SIZE = 101 # [-50, 50]

def upper_confidence_bound(mcts_score, policy_score, parent_visits, child_visits, c1=1.25, c2=19652):
  """
  Note: mcts_score should be normalized across all actions [0, 1].
  
  :param mcts_score: The propogated score from MCTS.
  :param policy_score: The score of the action predicted by the policy.
  :param parent_visits: The number of times the parent node has been visited.
  :param child_visits: The number of times the child node has been visited.
  :param c1: The exploration weight.
  :param c2: The exploration decay.
  """
  return mcts_score + policy_score * np.sqrt(parent_visits) / (1 + child_visits) * (c1 + np.log((parent_visits + c2 + 1) / c2))

def scale_targets(x, eps=1e-3):
  """
  Appendix F: Network Architecture says for a value and reward prediction
  we scale the targets before we obtain the categorical representations.

  :param x: The target (value or reward)
  :param eps: Epsilon
  """
  return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + eps * x)

class PredictionModel(nn.Module):
  """Two-headed model for predicting policy and value from a state."""
  def __init__(self, latent_size=16):
    super().__init__()
    self.fc1 = nn.Linear(STATE_SIZE, latent_size)
    self.fc2 = nn.Linear(latent_size, latent_size)
    self.policy = nn.Linear(latent_size, ACTION_SIZE)
    self.value = nn.Linear(latent_size, SUPPORT_SIZE)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.policy(x), self.value(x)

class DynamicsModel(nn.Module):
  """
  Model Architecture Based On: https://arxiv.org/pdf/1603.05027<br>
  This model predicts the next state and reward given a state and an action.
  """
  def __init__(self, latent_size=16):
    super().__init__()
    self.fc1 = nn.Linear(STATE_SIZE, latent_size)
