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

"""
TODO:
- Fix Node and MCTS class to reflect the pseudocode << Emmett
- Finish the replay buffer class << Justin
- Add self play (data generation)
- Complete training loop (how are the gradients stored?)
- Sample a position from the replay buffer using get_sampling_priority
- Vectorize targets in get_targets to match output of the network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


MAX_FLOAT = float('inf')
STATE_SIZE = 10
ACTION_SIZE = 3
SUPPORT_SIZE = 101 # Categorical reward and value [-50, 50] (see Appendix F of MuZero paper)
K_STEPS = 5
DISCOUNT_FACTOR = 0.997
MAX_MOVES = 50 * 100
LR_INIT = 0.05
LR_DECAY_RATE = 0.95
LR_DECAY_STEPS = 10
TRAINING_STEPS = 50
SAVE_EVERY = 10
UNROLL_STEPS = 5
TD_STEPS = 5
WEIGHT_DECAY = 0.0001

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

class NetworkOutput:
  """
  :param hidden_state: The latent representation of the state
  :param reward: The reward for the state
  :param policy_logits: The policy for the state
  :param value: The value for the state
  """
  hidden_state: torch.Tensor
  reward: float
  policy_logits: torch.Tensor
  value: float

  def __init__(self, hidden_state, reward, policy_logits, value):
    self.hidden_state = hidden_state
    self.reward = reward
    self.policy_logits = policy_logits
    self.value = value

class Network:
  def __init__(self, dynamics_model: DynamicsModel, prediction_model: PredictionModel):
    self.dynamics_model = dynamics_model
    self.prediction_model = prediction_model
    self.training_steps = 0

  def initial_forward(self, state: torch.Tensor):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(1))
    value = 0
    policy = torch.ones(1, ACTION_SIZE)
    return NetworkOutput(hidden_state, reward, policy, value)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)

    value = one_hot_score_to_scaler(value[0]).item()
    reward = one_hot_score_to_scaler(reward[0]).item()

    return NetworkOutput(hidden_state, reward, policy_logits, value)
  
  def forward_grad(self, state: torch.Tensor, action: int):
    action = one_hot_action(action)

    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)

    return NetworkOutput(hidden_state, reward, policy_logits, value)
  
  def parameters(self):
    return self.dynamics_model.parameters() + self.prediction_model.parameters()

class UniformNetwork(Network):
  def __init__(self):
    super().__init__(None, None)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    policy_logits = torch.ones(1, ACTION_SIZE)
    return NetworkOutput(state, 0, policy_logits, 0)

class Node:
  """
  Node in the MCTS search tree.
  """
  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children: dict[int, Node] = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

class MCTS:
  """
  The MCTS search tree for the MuZero model.
  ### Note:
  There is no rollout since a value estimate is used instead.<br>
  The Dynamics and Prediction models are only used **once** per simulation.
  """
  def __init__(self, network: Network):
    self.network = network
    self.n_simulations = 50

  def select_action(self, node: Node, num_moves: int) -> int:
    actions = node.children.keys()
    visit_counts = [child.visit_count for child in node.children.values()]
    temp = get_temperature(num_moves, self.network.training_steps)
    action_idx = torch.multinomial(torch.tensor(visit_counts) ** (1 / temp), num_samples=1).item()
    return list(actions)[action_idx]

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
      network_output = self.network.recurrent_forward(parent.hidden_state, action)

      self.expand_node(node, network_output)
      self.backprop(search_path, network_output, min_max_stats)
  
  def expand_node(self, node: Node, network_output: NetworkOutput):
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    priors = torch.softmax(network_output.policy_logits, dim=1)[0].tolist()

    for i in range(ACTION_SIZE):
      child = Node(priors[i])
      node.children[i] = child

  def backprop(self, search_path: list[Node], network_output: NetworkOutput, min_max_stats: MinMaxStats):
    value = network_output.value

    for node in reversed(search_path):
      node.value_sum += value # Add negative if it's the opponent's turn (this is a single player game)
      node.visit_count += 1
      min_max_stats.update(node.value())
      value = node.reward + DISCOUNT_FACTOR * value

def get_temperature(num_moves, training_steps):
  return 1

def scale_targets(x, eps=1e-3):
  """
  MuZero Appendix F: Network Architecture says for a value and reward prediction
  we scale the targets before we obtain the categorical representations.

  :param x: The target (value or reward)
  :param eps: Epsilon
  """
  return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + eps * x)

def one_hot_score(x: float):
  """
  One-hot encoding for the reward and value
  
  :param x: The target (value or reward)
  """
  if x < -(SUPPORT_SIZE // 2) or x > SUPPORT_SIZE // 2:
    raise ValueError(f"x must be between -{SUPPORT_SIZE // 2} and {SUPPORT_SIZE // 2}")
  arr = torch.zeros(SUPPORT_SIZE)
  arr[x + 50] = 1
  return arr

def one_hot_action(x: int):
  """
  One-hot encoding for the action

  :param x: The action index
  """
  arr = torch.zeros(size=(1, ACTION_SIZE))
  arr[0,x] = 1
  return arr

def one_hot_score_to_scaler(x: torch.Tensor):
  return torch.dot(x, torch.arange(-(SUPPORT_SIZE // 2), SUPPORT_SIZE // 2 + 1, dtype=torch.float32))

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

""" Self-play """

class NetworkBuffer:
  def __init__(self):
    self.networks: dict[int, Network] = {}

  def latest_network(self):
    if self.networks:
      return self.networks[max(self.networks.keys())] # Return the latest network
    else:
      return UniformNetwork() # Default
    
  def save_network(self, step: int, network: Network):
    self.networks[step] = network

class Environment:
  def step(self, action: int):
    """
    Take a step in the environment
    
    :param action: The action
    :return: The state and reward
    """
    state = np.zeros(STATE_SIZE)
    reward = 0
    return state, reward

  def terminal(self):
    return False

class Game:
  """
  A single episode of interaction with the environment.
  """

  def __init__(self, action_space_size: int, discount_factor: float):
    self.environment = Environment()
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.states: list[torch.Tensor] = []
    self.action_space_size = action_space_size
    self.discount_factor = discount_factor

  def terminal(self):
    return self.environment.terminal()
  
  def legal_actions(self):
    return [i for i in range(self.action_space_size)]
  
  def apply(self, action: int):
    state, reward = self.environment.step(action)
    self.states.append(state)
    self.rewards.append(reward)
    self.history.append(action)

  def store_search_stats(self, root: Node):
    """
    This function is called after a full MCTS search.
    
    ### Store:
    1. Proportion of visits to each child (policy target)
    2. Mean value of the root (part of the value target)
    
    :param root: The root node
    """
    sum_visits = sum(child.visit_count for child in root.children.values())
    self.child_visits.append([
      root.children[i].visit_count / sum_visits if i in root.children else 0
      for i in range(self.action_space_size)
    ])
    self.root_values.append(root.value())

  def get_targets(self, start_state_idx, unroll_steps, td_steps):
    """
    The objective of this function is to compute the targets
    (value, reward, policy) for the replay buffer.
    
    TODO: Vectorize targets to match output of the network.

    :param state_idx: The index of the initial state for this trajectory
    :param unroll_steps: The length of the trajectory
    :param td_steps: The number of steps to look ahead
    """
    targets = []

    for current_state_idx in range(start_state_idx, start_state_idx + unroll_steps + 1):
      # First we grab the boostrap value if it exists
      boostrap_state_idx = current_state_idx + td_steps

      if boostrap_state_idx < len(self.root_values):
        value = self.root_values[boostrap_state_idx] * self.discount_factor ** td_steps
      else:
        value = 0

      # Now we add the sum of all the individual rewards up to the bootstrap state to the value
      for i, reward in enumerate(self.rewards[current_state_idx:boostrap_state_idx]):
        value += reward * self.discount_factor ** i

      # The reward predicted will be the reward before the current state
      if current_state_idx > 0 and current_state_idx < len(self.rewards):
        last_reward = self.rewards[current_state_idx - 1]
      else:
        last_reward = 0
      
      # The policy is taken from the proportion of visits to each action
      policy = self.child_visits[current_state_idx]

      # Finally, we append the targets
      if current_state_idx < len(self.root_values):
        targets.append((value, last_reward, policy))
      else:
        # States past the end of games are absorbing states
        uniform_policy = [1 / self.action_space_size] * self.action_space_size
        targets.append((0, last_reward, uniform_policy))

    return targets

  def get_initial_state(self):
    return torch.zeros(size=(1, STATE_SIZE))

class ReplayBuffer:
  """
  Stores the trajectories: (prev_state, next_state, action, reward, is_done)
  """
  def __init__(self, capacity, batch_size):
    self.buffer = []
    self.batch_size = batch_size
    self.capacity = capacity

  def add_game(self, game: Game):
    self.buffer.append(game)
    if len(self.buffer) > self.capacity:
      self.buffer.pop(0)

  def sample_batch(self, unroll_steps: int, td_steps: int):
    """
    Samples a batch of trajectories from the replay buffer.
    (current_state, actions, targets)
    """
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]

    return [(
      game.states[state_idx],
      game.history[state_idx:state_idx + unroll_steps],
      game.get_targets(state_idx, unroll_steps, td_steps)
    ) for (game, state_idx) in game_pos]

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return random.choice(self.buffer)

  def sample_position(self, game) -> int:
    # TODO: sample from get_sampling_priority.
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

def get_root_node(mcts: MCTS, game: Game):
  root = Node(0)
  current_state = game.get_initial_state()
  network_output = mcts.network.initial_forward(current_state)
  mcts.expand_node(root, network_output)
  return root

def run_selfplay(replay_buffer: ReplayBuffer, network_buffer: NetworkBuffer, iterations: int):
  for _ in range(iterations):
    game = play_game(MCTS(network_buffer.latest_network()))
    replay_buffer.add_game(game)

def play_game(mcts: MCTS):
  game = Game(action_space_size=ACTION_SIZE, discount_factor=DISCOUNT_FACTOR)

  while not game.terminal() and len(game.history) < MAX_MOVES:
    root = get_root_node(mcts, game)
    mcts.search(root, game.history.copy())
    action = mcts.select_action(root)
    game.apply(action)
    game.store_search_stats(root)
  
  return game

""" Training """

def scale_gradient(tensor, scale):
  """
  Scales the gradient by a factor.
  """
  return tensor * scale + tensor.detach() * (1 - scale)

def train(replay_buffer: ReplayBuffer, network_buffer: NetworkBuffer):
  """
  Training loop.
  
  ### Note
  Remember to scale the loss by 1 / K_STEPS to ensure the gradient has a similar magnitude
  regardless of the number of unroll steps.
  """
  dynamics_model = DynamicsModel()
  prediction_model = PredictionModel()
  network = Network(dynamics_model, prediction_model)
  optimizer = torch.optim.AdamW(network.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)

  for i in range(TRAINING_STEPS):
    if i % SAVE_EVERY == 0:
      network_buffer.save_network(network)
    batch = replay_buffer.sample_batch(UNROLL_STEPS, TD_STEPS)
    update_weights(optimizer, network, batch)
  
  network_buffer.save_network(network)

def update_weights(optimizer: torch.optim, network: Network, batch: list[tuple]):
  learning_rate = LR_INIT * LR_DECAY_RATE ** (network.training_steps / LR_DECAY_STEPS)
  optimizer.param_groups[0]['lr'] = learning_rate
  loss = 0

  for game_state, actions, targets in batch:
    optimizer.zero_grad()
    
    # Initial step
    network_output = network.initial_forward(game_state)
    predictions = [
      1.0,
      network_output.value,
      network_output.reward,
      network_output.policy_logits
    ]

    # Recurrent steps
    for action in actions:
      network_output = network.recurrent_forward(network_output, action)
      predictions += [
        1 / len(actions),
        network_output.value,
        network_output.reward,
        network_output.policy_logits
      ]
      hidden_state = scale_gradient(hidden_state, 0.5)

    # Compute losses
    for prediction, target in zip(predictions, targets):
      gradient_scale, value, reward, policy_logits = prediction
      target_value, target_reward, target_policy = target

      raw_loss = (
        F.mse_loss(value, target_value) +
        F.mse_loss(reward, target_reward) +
        F.cross_entropy(policy_logits, target_policy)
      )

      loss += scale_gradient(raw_loss, gradient_scale)

  # Backpropagate
  loss.backward()
  optimizer.step()
  network.training_steps += 1


    