import numpy as np
from config import *
from mcts import Node

class Environment:
  def __init__(self, device):
    self.device = device
    self.state = torch.zeros(size=(1, STATE_SIZE), device=device)
    self.reward = 0

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

  def get_reward(self):
    return self.reward

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

  @staticmethod
  def serealize(observation):
    return {
      "state": observation.state.tolist(),
      "action": observation.action,
      "reward": observation.reward,
      "child_visit_priors": observation.child_visit_priors,
      "mcts_value": observation.mcts_value
    }
  
  @staticmethod
  def deserialize(observation_dict):
    return Observation(
      torch.tensor(observation_dict["state"], device=device),
      observation_dict["action"],
      observation_dict["reward"],
      observation_dict["child_visit_priors"],
      observation_dict["mcts_value"]
    )

class Game:
  """
  A single episode of interaction with the environment.
  """

  def __init__(self, Env=Environment):
    self.Env = Env
    self.environment = Env(device)
    self.priority = None
    self.priorities = []
    # self.observations: list[Observation] = []
    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []
    self.priors = []

  def reset(self):
    self.environment = self.Env(device)
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

    if bootstrap_idx >= len(self.values):
      return value

    # If the bootstrap index has been observed, use it as the mcts value
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
    for state_idx in range(len(self.actions)):
      target_value = self.get_target_value(state_idx, td_steps)
      mcts_value = self.values[state_idx]
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

  @staticmethod
  def serealize(game):
    game_dict = {
      "priority": game.priority,
      "priorities": game.priorities,
      "states": [state.cpu().detach().tolist() for state in game.states],
      "actions": game.actions,
      "rewards": game.rewards,
      "values": game.values,
      "priors": game.priors,
      # "observations": [Observation.serealize(o) for o in game.observations]
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
    # game.observations = [Observation.deserialize(o) for o in game_dict["observations"]]
    return game
