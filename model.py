from multiprocessing import Process, Queue
from tqdm import tqdm
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Training parameters
NETWORK_PATH = "Models/network.pt"
MAX_FLOAT = float('inf')
STATE_SIZE = 76
ACTION_SIZE = 3
SUPPORT_SIZE = 601 # Categorical reward and value [-300, 300] (see Appendix F of MuZero paper)
HIDDEN_SIZE = 128
K_STEPS = 5
DISCOUNT_FACTOR = 0.997
N_SIMULATIONS = 50
MAX_MOVES = 27000 # Taken from psudo code
LR_INIT = 0.05 # Taken from psudo code
LR_DECAY_RATE = 0.1 # Taken from psudo code
LR_DECAY_STEPS = 350e3 # Taken from psudo code
TRAINING_STEPS = 1000e3 # Taken from psudo code
SAVE_EVERY = 5
UNROLL_STEPS = 5 # Unroll for K=5 steps (see MuZero Appendix G)
TD_STEPS = 10 # Bootstrap 10 steps into the future (see MuZero Appendix G)
WEIGHT_DECAY = 0.0001 # Taken from psudo code
BATCH_SIZE = 32
NUM_ACTORS = 1
DIRICHLET_ALPHA = 0.25
DIRICHLET_FRAC = 0.25

""" Threading """

class Bridge:
  def __init__(self, max_games=1000, num_actors=NUM_ACTORS):
    self.num_actors = num_actors
    self.game_queue = Queue(maxsize=max_games)
    self.weight_queue = [Queue(maxsize=1) for _ in range(num_actors)]

  def send_game(self, game):
    self.game_queue.put(game)

  def receive_game(self):
    return self.game_queue.get()

  def has_game(self):
    return not self.game_queue.empty()

  def has_network(self, actor_id):
    return not self.weight_queue[actor_id].empty()

  def broadcast_network(self, network):
    state_dict = network.state_dict()
    # Send new weights to all actors
    for actor_id in range(self.num_actors):
      # Drop old weights, keep only newest
      while not self.weight_queue[actor_id].empty():
        self.weight_queue[actor_id].get()
      # Send new weights
      self.weight_queue[actor_id].put(state_dict)

  def receive_network(self, actor_id):
    if not self.weight_queue[actor_id].empty():
      network = Network()
      network.load_state_dict(self.weight_queue[actor_id].get())
      return network
    return UniformNetwork()

""" Network """

class RepresentationModel(nn.Module):
  """
  Take a state as input and output a latent representation.
  """
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
    self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return x

class PredictionModel(nn.Module):
  """
  Two-headed model for predicting policy and value from a state.
  """
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    self.policy = nn.Linear(HIDDEN_SIZE, ACTION_SIZE)
    self.value = nn.Linear(HIDDEN_SIZE, SUPPORT_SIZE)

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
    self.norm = nn.LayerNorm(channels)

  def forward(self, x):
    identity = x
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = identity + x * self.alpha
    x = self.norm(x)
    return x

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
  def __init__(self, n_blocks=2):
    super().__init__()
    self.first = nn.Linear(HIDDEN_SIZE + ACTION_SIZE, HIDDEN_SIZE)
    self.model = nn.Sequential(*[ResBlock(HIDDEN_SIZE) for _ in range(n_blocks)])
    self.state = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    self.reward = nn.Linear(HIDDEN_SIZE, SUPPORT_SIZE)

  def forward(self, state, action):
    x = torch.cat([state, action], dim=1)
    x = F.relu(self.first(x))
    x = self.model(x)
    state = F.relu(self.state(x))
    reward = self.reward(x)
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

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.latent_model = RepresentationModel()
    self.dynamics_model = DynamicsModel()
    self.prediction_model = PredictionModel()
    self.training_steps = 0

  def initial_forward(self, state: torch.Tensor):
    hidden_state = self.latent_model(state)
    policy_logits, value = self.prediction_model(hidden_state)

    value = inverse_value_transform(support_to_scalar(value))

    return NetworkOutput(hidden_state, 0.0, policy_logits, value)
  
  def initial_forward_grad(self, state: torch.Tensor):
    hidden_state = self.latent_model(state)
    policy_logits, value = self.prediction_model(hidden_state)

    reward = scalar_to_support(value_transform(0))

    return NetworkOutput(hidden_state, reward, policy_logits, value)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)

    value = inverse_value_transform(support_to_scalar(value))
    reward = inverse_value_transform(support_to_scalar(reward))

    return NetworkOutput(hidden_state, reward, policy_logits, value)
  
  def recurrent_forward_grad(self, state: torch.Tensor, action: int):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)

    return NetworkOutput(hidden_state, reward, policy_logits, value)

  @staticmethod
  def save(network: nn.Module, path: str):
    try:
      torch.save({
        'model': network.state_dict(),
        'steps': network.training_steps
      }, path + '.tmp')
      os.replace(path + '.tmp', path)
    except Exception as e:
      print(f"Failed to save model: {e}")
    
  @staticmethod
  def load(path: str):
    try:
      checkpoint = torch.load(path)
      network = Network()
      network.load_state_dict(checkpoint['model'])
      network.training_steps = checkpoint['steps']
      print(f"Loaded model from {path}")
      return network
    except Exception as e:
      print(f"Failed to load model: {e}")
      print('Making a new model...')
      return Network()

class UniformNetwork(Network):
  def __init__(self):
    super().__init__()

  def initial_forward(self, state):
    policy_logits = torch.ones(1, ACTION_SIZE)
    return NetworkOutput(state, 0, policy_logits, 0)
  
  def initial_forward_grad(self, state):
    policy_logits = torch.ones(1, ACTION_SIZE)
    reward = scalar_to_support(value_transform(0))
    value = scalar_to_support(value_transform(0))
    return NetworkOutput(state, reward, policy_logits, value)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    policy_logits = torch.ones(1, ACTION_SIZE)
    return NetworkOutput(state, 0, policy_logits, 0)
  
  def recurrent_forward_grad(self, state: torch.Tensor, action: int):
    policy_logits = torch.ones(1, ACTION_SIZE)
    reward = scalar_to_support(value_transform(0))
    value = scalar_to_support(value_transform(0))
    return NetworkOutput(state, reward, policy_logits, value)

""" MCTS """

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
    self.n_simulations = N_SIMULATIONS

  def select_action(self, node: Node, num_moves: int) -> int:
    """
    Docstring for select_action
    
    :param node: The node to select an action from
    :param num_moves: The number of moves made so far
    :return: The action to take
    """
    actions = node.children.keys()
    visit_counts = [child.visit_count for child in node.children.values()]
    temp = get_temperature(num_moves, self.network.training_steps)
    action_idx = torch.multinomial(torch.tensor(visit_counts) ** (1 / temp), num_samples=1).item()
    return list(actions)[action_idx]

  def select_child(self, node: Node, min_max_stats: MinMaxStats):
    scores = [(
      ucb_score(node, child, min_max_stats),
      action,
      child
    ) for action, child in node.children.items()]
    
    random.shuffle(scores) # Randomly break ties (stops from constantly picking the last action)
    _, action, child = max(scores)

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

      # print(network_output.policy_logits)

      self.expand_node(node, network_output)
      self.backprop(search_path, network_output, min_max_stats)
  
  def expand_node(self, node: Node, network_output: NetworkOutput):
    if node.expanded():
      raise Exception("Tried to expand node twice")

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

  def add_exploration_noise(self, node: Node, alpha=DIRICHLET_ALPHA, frac=DIRICHLET_FRAC):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
      node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

""" Self-play """

class NetworkBuffer:
  def __init__(self):
    self.network = UniformNetwork()
    # self.network = Network()

  def latest_network(self):
    return self.network
    
  def save_network(self, network: Network):
    self.network = network

class Environment:
  def __init__(self):
    self.state = torch.zeros(size=(1, STATE_SIZE))

  def step(self, action: int):
    """
    Take a step in the environment
    
    :param action: The action
    :return: The state and reward
    """
    state = torch.zeros(size=(1, STATE_SIZE))
    reward = 0.0
    return state, reward

  def get_state(self):
    return self.state

  def terminal(self):
    return False

class Game:
  """
  A single episode of interaction with the environment.
  """

  def __init__(self, action_space_size=ACTION_SIZE, discount_factor=DISCOUNT_FACTOR, Env=Environment):
    self.Env = Env
    self.action_space_size = action_space_size
    self.discount_factor = discount_factor
    self.environment = Env()
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.states: list[torch.Tensor] = []
    self.priorities = []

  def reset(self):
    self.environment = self.Env()
    self.history = []
    self.rewards = []
    self.child_visits = []
    self.root_values = []
    self.states = []
    self.priorities = []

  def terminal(self):
    return self.environment.terminal()
  
  def legal_actions(self):
    return [i for i in range(self.action_space_size)]
  
  def apply(self, action: int):
    state, reward = self.environment.step(action)
    self.states.append(state)
    self.rewards.append(reward)
    self.history.append(action)

  def get_current_state(self):
    return self.environment.get_state()

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
        value = 0.0

      # Now we add the sum of all the individual rewards up to the bootstrap state to the value
      for i, reward in enumerate(self.rewards[current_state_idx:boostrap_state_idx]):
        value += reward * self.discount_factor ** i

      # The reward predicted will be the reward before the current state
      if current_state_idx > 0 and current_state_idx < len(self.rewards):
        last_reward = self.rewards[current_state_idx - 1]
      else:
        last_reward = 0.0
      
      # Finally, we append the targets
      if current_state_idx < len(self.root_values):
        # The policy is taken from the proportion of visits to each action
        policy = self.child_visits[current_state_idx]
        targets.append((
          scalar_to_support(value_transform(value)),
          scalar_to_support(value_transform(last_reward)),
          torch.tensor(policy).unsqueeze(0)
        ))
      else:
        # States past the end of games are absorbing states
        targets.append((
          scalar_to_support(value_transform(0.0)),
          scalar_to_support(value_transform(last_reward)),
          torch.ones(size=(1, self.action_space_size)) / self.action_space_size
        ))

    return targets

  def compute_priorities(self, td_steps: int):
    """
    Compute sampling priorities based on difference between
    MCTS value and bootstrapped target value.
    """
    self.priorities = []
    
    for state_idx in range(len(self.root_values)):
      # Get the bootstrapped target value (same logic as in get_targets)
      bootstrap_idx = state_idx + td_steps
      
      if bootstrap_idx < len(self.root_values):
        target_value = self.root_values[bootstrap_idx] * (self.discount_factor ** td_steps)
      else:
        target_value = 0.0
      
      # Add discounted rewards
      for i, reward in enumerate(self.rewards[state_idx:bootstrap_idx]):
        target_value += reward * (self.discount_factor ** i)
      
      # Compare MCTS value with target value
      mcts_value = self.root_values[state_idx]
      priority = self.get_sampling_priority(mcts_value, target_value)
      self.priorities.append(priority)
  
  def get_sampling_priority(self, mcts_value, target_value):
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
    return np.abs(mcts_value - target_value) + 1e-8 # Don't forget to normalize to get the probability!

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
    Weight is to account for sampling bias
    """
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    batch = []

    for (game, state_idx) in game_pos:
      hidden_state = game.states[state_idx]
      actions = game.history[state_idx:state_idx + unroll_steps]
      targets = game.get_targets(state_idx, unroll_steps, td_steps)
      weight = sum(game.priorities) / len(game.priorities) / game.priorities[state_idx]
      batch.append((hidden_state, actions, targets, weight))

    return batch

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return random.choice(self.buffer)

  def sample_position(self, game: Game) -> int:
    """
    Sample position from game using priorities
    """
    if len(game.priorities) == 0:
      raise Exception("Game has no priorities")
      # Fallback to uniform sampling
      # return random.randint(0, len(game.history) - 1)
    
    # Normalize priorities to get probabilities
    priorities = np.array(game.priorities)
    probabilities = priorities / priorities.sum()
    
    # Sample according to priorities
    position = np.random.choice(len(game.priorities), p=probabilities)

    return position

def get_root_node(mcts: MCTS, game: Game):
  """
  Get the root node as the current state of the game.
  """
  root = Node(0)
  current_state = game.get_current_state()
  network_output = mcts.network.initial_forward(current_state)
  mcts.expand_node(root, network_output)
  return root

def run_selfplay(actor_id: int, bridge: Bridge, iterations: int, Env: Environment):
  iterations = int(iterations)
  network_buffer = NetworkBuffer()
  print(f"Playing {iterations} games...")
  for _ in range(iterations):
    fetch_network(actor_id, network_buffer, bridge)
    game = play_game(MCTS(network_buffer.latest_network()), Env)
    game.compute_priorities(TD_STEPS)
    bridge.send_game(game)
    # print(f"Game completed in {len(game.history)} moves")
    
def play_game(mcts: MCTS, Env: Environment):
  game = Game(Env=Env)
  game.states.append(game.get_current_state())

  action_hist = np.array([0 for _ in range(ACTION_SIZE)])

  with torch.no_grad():
    while not game.terminal() and len(game.history) < MAX_MOVES:
      root = get_root_node(mcts, game)
      mcts.add_exploration_noise(root)
      mcts.search(root, game.history.copy())
      action = mcts.select_action(root, len(game.history))
      action_hist[action] += 1
      game.apply(action)
      game.store_search_stats(root)

  total_actions = np.sum(action_hist)
  print(f"Action distribution: {(action_hist / total_actions * 100).astype(int)}")

  return game

def fetch_network(actor_id: int, network_buffer: NetworkBuffer, bridge: Bridge):
  if bridge.has_network(actor_id):
    network = bridge.receive_network(actor_id)
    network_buffer.save_network(network)
    print(f"Actor {actor_id}: Received latest network")

""" Training """

def muzero(replay_buffer: ReplayBuffer, Env: Environment):
  bridge = Bridge()

  for actor_id in range(NUM_ACTORS):
    launch_job(run_selfplay, actor_id, bridge, TRAINING_STEPS // NUM_ACTORS, Env)

  train(replay_buffer, bridge)

def train(replay_buffer: ReplayBuffer, bridge: Bridge):
  """
  Training loop.
  
  ### Note
  Remember to scale the loss by 1 / K_STEPS to ensure the gradient has a similar magnitude
  regardless of the number of unroll steps.
  """
  network = Network.load(NETWORK_PATH)
  # freeze_value_and_reward(network)
  optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, network.parameters()),
    lr=LR_INIT,
    weight_decay=WEIGHT_DECAY
  )

  # Wait for a game to complete
  attempts = 0
  while len(replay_buffer.buffer) < 1:
    attempts += 1
    fetch_games(replay_buffer, bridge)
    print("Waiting for a game to complete, attempt", attempts)
    time.sleep(5)

  print("Training...")
  for i in range(int(TRAINING_STEPS)):
    fetch_games(replay_buffer, bridge)
    if (i + 1) % SAVE_EVERY == 0:
      bridge.broadcast_network(network)
      Network.save(network, NETWORK_PATH)
    batch = replay_buffer.sample_batch(UNROLL_STEPS, TD_STEPS)
    update_weights(optimizer, network, batch)

def update_weights(optimizer: torch.optim, network: Network, batch: list[tuple]):
  learning_rate = LR_INIT * LR_DECAY_RATE ** (network.training_steps / LR_DECAY_STEPS)
  optimizer.param_groups[0]['lr'] = learning_rate
  optimizer.zero_grad()
  loss = 0
  n_losses = 0
  prog_bar = tqdm(batch, desc=f"Training step {network.training_steps}")

  for game_state, actions, targets, weight in prog_bar:
    
    # Initial step
    network_output = network.initial_forward_grad(game_state)
    predictions = [(
      1.0,
      network_output.value,
      network_output.reward,
      network_output.policy_logits
    )]

    # Recurrent steps
    for action in actions:
      network_output = network.recurrent_forward_grad(network_output.hidden_state, action)
      predictions += [(
        1 / len(actions),
        network_output.value,
        network_output.reward,
        network_output.policy_logits
      )]
      network_output.hidden_state = scale_gradient(network_output.hidden_state, 0.5)

    # Accumulate predictions and targets
    for prediction, target in zip(predictions, targets):
      gradient_scale, value, reward, policy_logits = prediction
      target_value, target_reward, target_policy = target

      # Cross entropy loss with target probabilities
      raw_loss = -(
        (F.log_softmax(value, dim=1) * target_value).sum(dim=1) +
        (F.log_softmax(reward, dim=1) * target_reward).sum(dim=1) +
        (F.log_softmax(policy_logits, dim=1) * target_policy).sum(dim=1)
      ).mean()

      # Weight is to account for sampling bias
      loss += weight * scale_gradient(raw_loss, gradient_scale)
      n_losses += 1

  # Backpropagate
  n_losses = len(batch) * (1 + UNROLL_STEPS)
  loss = loss / n_losses
  print(f"Loss: {loss}")
  loss.backward()
  torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
  optimizer.step()
  network.training_steps += 1

def fetch_games(replay_buffer: ReplayBuffer, bridge: Bridge):
  games_received = 0
  while bridge.has_game():
    game = bridge.receive_game()
    replay_buffer.add_game(game)
    games_received += 1
  print(f"Received {games_received} games")

def freeze_value_and_reward(network: Network):
  # Freeze value head
  for p in network.prediction_model.value.parameters():
    p.requires_grad = False

  # Freeze reward head
  for p in network.dynamics_model.reward.parameters():
    p.requires_grad = False

""" Utility Functions """

def launch_job(func, *args):
  p = Process(target=func, args=args)
  p.start()
  return p

def scale_gradient(tensor, scale):
  """
  Scales the gradient by a factor.
  """
  return tensor * scale + tensor.detach() * (1 - scale)

def get_temperature(num_moves, training_steps):
  if training_steps < 500e3:
    return 1.0
  elif training_steps < 750e3:
    return 0.5
  else:
    return 0.25

def scale_targets(x, eps=1e-3):
  """
  MuZero Appendix F: Network Architecture says for a value and reward prediction
  we scale the targets before we obtain the categorical representations.

  :param x: The target (value or reward)
  :param eps: Epsilon
  """
  return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + eps * x)

def one_hot_action(x: int):
  """
  One-hot encoding for the action

  :param x: The action index
  """
  arr = torch.zeros(size=(1, ACTION_SIZE))
  arr[0,x] = 1
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

def value_transform(x: float, eps=1e-3):
  return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x

def inverse_value_transform(x: float, eps=1e-3):
  return np.sign(x) * (
    ((np.sqrt(1 + 4 * eps * (np.abs(x) + 1 + eps)) - 1) / (2 * eps))**2 - 1
  )

def scalar_to_support(x: float, support_size=SUPPORT_SIZE):
  # x is already transformed
  x = np.clip(x, -support_size//2, support_size//2)

  floor = np.floor(x)
  ceil = np.ceil(x)

  prob_upper = x - floor
  prob_lower = 1.0 - prob_upper

  support = torch.zeros(size=(1, support_size), dtype=torch.float32)

  idx_lower = int(floor + support_size // 2)
  idx_upper = int(ceil + support_size // 2)

  support[0, idx_lower] += prob_lower
  support[0, idx_upper] += prob_upper

  return support

def support_to_scalar(probs: torch.Tensor, support_size=SUPPORT_SIZE):
  support = torch.arange(
    -(support_size // 2),
    support_size // 2 + 1,
    device=probs.device,
    dtype=torch.float32
  )
  return torch.dot(probs[0], support).item()

