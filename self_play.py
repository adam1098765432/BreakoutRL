import numpy as np
from bridge import Bridge, NetworkBuffer
from config import *
from mcts import MCTS, Node

class Environment:
  def __init__(self, device):
    self.device = device
    self.state = torch.zeros(size=(1, STATE_SIZE), device=device)

  def step(self, action: int):
    """
    Take a step in the environment
    
    :param action: The action
    :return: The state and reward
    """
    state = torch.zeros(size=(1, STATE_SIZE), device=self.device)
    reward = 0.0
    return state, reward

  def get_state(self):
    return self.state

  def terminal(self):
    return False

class Observation:
  """
  An observation from the environment
  (based on the result of a MCTS search for a given node).

  The motivation behind this class is to prevent misalignment
  between each of these statistics during the batching process.
  """

  def __init__(self, state, action, reward, child_visit_priors, mcts_value):
    self.state: torch.Tensor = state # The state of this observation
    self.action: int = action # The action taken from this observation
    self.reward: float = reward # The reward for taking the action
    self.child_visit_priors: list[float] = child_visit_priors # The visit proportions for each child 
    self.mcts_value: float = mcts_value # The propagated MCTS value for this observation

class Game:
  """
  A single episode of interaction with the environment.
  """

  def __init__(self, action_space_size=ACTION_SIZE, discount_factor=DISCOUNT_FACTOR, Env=Environment):
    self.Env = Env
    self.action_space_size = action_space_size
    self.discount_factor = discount_factor
    self.environment = Env(device)
    self.priority = None
    self.priorities = []
    self.observations: list[Observation] = []

  def reset(self):
    self.environment = self.Env(device)
    self.priority = None
    self.priorities = []
    self.observations = []

  def terminal(self):
    return self.environment.terminal()
  
  def legal_actions(self):
    return [i for i in range(self.action_space_size)]
  
  def apply(self, action: int, root: Node):
    state, reward = self.environment.step(action)
    mcts_value = root.value()
    sum_visits = sum(child.visit_count for child in root.children.values())
    child_visit_priors = [
      root.children[i].visit_count / sum_visits if i in root.children else 0
      for i in range(self.action_space_size)
    ]

    observation = Observation(
      state,
      action,
      reward,
      child_visit_priors,
      mcts_value
    )

    self.observations.append(observation)

  def get_current_state(self):
    return self.environment.get_state()

  ### Observation Helpers ###

  def get_target_value(self, observation_idx: int, td_steps: int):
    bootstrap_idx = observation_idx + td_steps
    value = 0

    if bootstrap_idx < len(self.observations):
      # If the bootstrap index has been observed, use it as the mcts value
      value = self.observations[bootstrap_idx].mcts_value * DISCOUNT_FACTOR ** td_steps

    # Add the rewards from the observation index to the bootstrap index
    for i, observation in enumerate(self.observations[observation_idx:bootstrap_idx]):
      value += observation.reward * DISCOUNT_FACTOR ** i

    return value

  def get_target_reward(self, observation_idx: int):
    if observation_idx >= len(self.observations):
      return 0
    
    return self.observations[observation_idx].reward

  def get_target_policy(self, observation_idx: int):
    if observation_idx >= len(self.observations):
      return [1 / self.action_space_size] * self.action_space_size
    
    return self.observations[observation_idx].child_visit_priors

  def compute_priorities(self, td_steps: int):
    """
    Compute sampling priorities based on difference between
    MCTS value and bootstrapped target value.
    """
    self.priorities = []
    
    if len(self.observations) == 0:
      raise Exception("Game has no observations")

    # Compute priority for each observation
    for observation_idx in range(len(self.observations)):
      target_value = self.get_target_value(observation_idx, td_steps)
      mcts_value = self.observations[observation_idx].mcts_value
      priority = self.get_sampling_priority(mcts_value, target_value)
      self.priorities.append(priority)

    self.priority = np.max(self.priorities)
  
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
    # Take this to the power of a constant (alpha) for stronger bias
    # Paper suggests alpha = 1.0
    return np.abs(mcts_value - target_value) + 1e-8 # Don't forget to normalize to get the probability!

### Self Play Helpers ###

def get_root_node(mcts: MCTS, game: Game):
  """
  Get the root node as the current state of the game.
  """
  root = Node(1.0 / ACTION_SIZE)
  current_state = game.get_current_state()
  network_output = mcts.network.initial_forward(current_state)
  mcts.expand_node(root, network_output)
  return root

def run_selfplay(actor_id: int, bridge: Bridge, iterations: int, Env: Environment):
  global device
  device = get_device() if ACTORS_USE_CUDA else torch.device("cpu")
  print(f"Actor {actor_id} using device: {device}")
  
  if device.type == "cuda": torch.set_num_threads(1)

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
  
  # Are we supposed to add an initial observation?
  # game.states.append(game.get_current_state())

  action_histogram = np.array([0 for _ in range(ACTION_SIZE)])

  with torch.no_grad():
    while not game.terminal() and len(game.observations) < MAX_MOVES:
      root = get_root_node(mcts, game)
      mcts.add_exploration_noise(root)
      mcts.search(root)
      action = mcts.select_action(root, len(game.observations))
      action_histogram[action] += 1
      game.apply(action, root)

  total_actions = np.sum(action_histogram)
  # print(f"Action distribution: {(action_histogram / total_actions * 100).astype(int)}")

  return game

def fetch_network(actor_id: int, network_buffer: NetworkBuffer, bridge: Bridge):
  if bridge.has_network(actor_id):
    network = bridge.receive_network(actor_id)
    network_buffer.save_network(network)
    # print(f"Actor {actor_id}: Received latest network")
