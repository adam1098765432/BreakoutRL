import numpy as np
from config import *
from utility import *
from game import Game


class ReplayBuffer:
  """
  Stores the trajectories: (prev_state, next_state, action, reward, is_done)
  """
  def __init__(self, capacity, batch_size):
    self.buffer: list[Game] = []
    self.batch_size = batch_size
    self.capacity = capacity

    if batch_size > capacity:
      raise Exception("Batch size cannot be greater than capacity")

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
    batch = []
    game_idxs, game_probs = self.sample_games(self.batch_size)
    state_idxs, state_probs = self.sample_states(game_idxs)
    weights = np.empty(self.batch_size)

    # Compute weights first
    for i, (game_prob, state_prob) in enumerate(zip(game_probs, state_probs)):
      weights[i] = 1 / (self.batch_size * game_prob * state_prob)

    weights /= weights.max()

    for i, (game_idx, state_idx) in enumerate(zip(game_idxs, state_idxs)):
      # Get the initial environment state and played actions
      game = self.buffer[game_idx]
      states = game.states[state_idx:state_idx + unroll_steps + 1]
      actions = game.actions[state_idx:state_idx + unroll_steps]
      targets = []

      for j in range(unroll_steps + 1):
        # Compute targets
        target_value = game.get_target_value(state_idx + j, td_steps)
        target_reward = game.get_target_reward(state_idx + j)
        target_policy = game.get_target_policy(state_idx + j)

        # Convert to probability distributions
        target_value = scalar_to_support(target_value)
        target_reward = scalar_to_support(target_reward)
        target_policy = torch.tensor(target_policy, device=device)

        # Add to batch
        targets += [(target_value, target_reward, target_policy)]

      # Add to batch
      batch.append((states, actions, targets, weights[i]))

    return batch, game_idxs, state_idxs

  def sample_games(self, num_samples: int) -> tuple[list[int], list[float]]:
    # Sample game from buffer either uniformly or according to some priority.
    priorities = np.array([g.priority for g in self.buffer])

    if len(priorities) == 0:
      raise Exception("Replay buffer is empty")

    probabilities = priorities / priorities.sum()
    idxs = np.random.choice(len(self.buffer), p=probabilities, size=num_samples)
    
    return idxs, probabilities[idxs].tolist()

  def sample_states(self, game_idxs: list[int]) -> tuple[list[int], list[float]]:
    samples = [self.sample_state(game_idx) for game_idx in game_idxs]
    states = [state for state, _ in samples]
    probs = [prob for _, prob in samples]
    return states, probs

  def sample_state(self, game_idx: int) -> tuple[int, float]:
    """
    Sample an observation from game using priorities
    Take from len(game.actions) because of the initial state
    """
    game = self.buffer[game_idx]

    if len(game.priorities) == 0:
      raise Exception("Game has no priorities")
    
    priorities = np.array(game.priorities)
    probabilities = priorities / priorities.sum()
    idx = np.random.choice(len(game.values), p=probabilities)

    return idx, probabilities[idx]
  
  def update_priorities(self, game_idxs: list[int], state_idxs: list[int], priorities: np.ndarray):
    # If the games are being updated asynchronously, the game_idxs
    # may not correspond to the same game anymore.
    for i in range(len(priorities)):
      game_idx, state_idx = game_idxs[i], state_idxs[i]
      priority = priorities[i, :]
      end_idx = min(state_idx + len(priority), len(self.buffer[game_idx].priorities))
      self.buffer[game_idx].priorities[state_idx:end_idx] = priority[:end_idx - state_idx]
      self.buffer[game_idx].priority = np.max(self.buffer[game_idx].priorities)
