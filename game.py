import numpy as np
from config import *
from mcts import Node

class Environment:
  def __init__(self, device):
    self.frame_skip = 1
    self.device = device
    self.reward = 0

  def _step(self, action: int):
    """
    Take a step in the environment
    
    :param action: The action
    :return: The state and reward
    """
    pass

  def step(self, action: int):
    total_reward = 0
    for _ in range(self.frame_skip):
      state, reward = self._step(action)
      total_reward += reward
    self.state = state
    self.reward = total_reward
    return state[:, :STATE_SIZE].clone(), total_reward

  def get_state(self):
    return self.state[:, :STATE_SIZE].clone()

  def get_reward(self):
    return self.reward

  def terminal(self):
    return False

class Game:
  """
  A single episode of interaction with the environment.
  """

  def __init__(self, Env=Environment):
    self.Env = Env
    self.environment = Env(device)
    self.priority = None
    self.priorities = []
    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []
    self.priors = []

  def reset(self):
    old_frame_skip = self.environment.frame_skip
    self.environment = self.Env(device)
    self.environment.frame_skip = old_frame_skip
    self.priority = None
    self.priorities = []
    self.observations = []

  def terminal(self):
    return self.environment.terminal()
  
  def legal_actions(self):
    return [i for i in range(ACTION_SIZE)]
  
  def apply_action(self, action: int):
    state, reward = self.environment.step(action)
    return state, reward

  def apply(self, action: int, root: Node):
    state, reward = self.environment.step(action)
    mcts_value = root.value()
    sum_visits = sum(child.visit_count for child in root.children.values())
    child_visit_priors = [
      root.children[i].visit_count / sum_visits if i in root.children else 0
      for i in range(ACTION_SIZE)
    ]

    self.states += [state] # State of the next observation
    self.rewards += [reward] # Reward of the next observation

    self.actions += [action] # Action for the current state
    self.values += [mcts_value] # MCTS value for the current state
    self.priors += [child_visit_priors] # Child visit proportions for the current state

  def get_current_state(self):
    return self.environment.get_state()

  ### Observation Helpers ###

  def get_target_value(self, state_idx: int, td_steps: int):
    bootstrap_idx = state_idx + td_steps
    value = 0

    # If the bootstrap index has been observed, use it as the mcts value
    if bootstrap_idx < len(self.values):
      value = self.values[bootstrap_idx] * DISCOUNT_FACTOR ** td_steps

    # Add the rewards from the observation index to the bootstrap index
    for i, reward in enumerate(self.rewards[state_idx + 1:bootstrap_idx + 1]):
      value += reward * DISCOUNT_FACTOR ** i

    return value

  def get_target_reward(self, state_idx: int):
    if state_idx >= len(self.rewards):
      return 0
    
    return self.rewards[state_idx]

  def get_target_policy(self, state_idx: int):
    if state_idx >= len(self.priors):
      return [1 / ACTION_SIZE] * ACTION_SIZE

    return self.priors[state_idx]

  def compute_priorities(self, td_steps: int):
    """
    Compute sampling priorities based on difference between
    MCTS value and bootstrapped target value.
    """
    self.priorities = []
    
    if len(self.actions) == 0:
      raise Exception("Game has no actions")

    # Compute priority for each observation
    for state_idx, root_value in enumerate(self.values):
      target_value = self.get_target_value(state_idx, td_steps)
      priority = self.get_sampling_priority(root_value, target_value)
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

  @staticmethod
  def serialize(game):
    game_dict = {
      "priority": game.priority,
      "priorities": game.priorities,
      "states": [state.cpu().detach().tolist() for state in game.states],
      "actions": game.actions,
      "rewards": game.rewards,
      "values": game.values,
      "priors": game.priors,
    }
    return game_dict
  
  @staticmethod
  def deserialize(game_dict, Env):
    game = Game(Env=Env)
    game.priority = game_dict["priority"]
    game.priorities = game_dict["priorities"]
    game.states = [torch.tensor(state, device=device) for state in game_dict["states"]]
    game.actions = game_dict["actions"]
    game.rewards = game_dict["rewards"]
    game.values = game_dict["values"]
    game.priors = game_dict["priors"]
    return game
