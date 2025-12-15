import numpy as np
from config import *
from utility import *
from game import Game, Observation


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
    games = self.sample_games(self.batch_size)
    state_idxs = [self.sample_state(game) for game in games]
    weights = np.empty(len(state_idxs))

    # Compute weights first
    for i, (game, state_idx) in enumerate(zip(games, state_idxs)):
      weights[i] = 1 / (len(state_idxs) * game.priority * game.priorities[state_idx])

    weights /= weights.max()

    for i, (game, state_idx) in enumerate(zip(games, state_idxs)):
      # Get the initial environment state and played actions
      state = game.states[state_idx]
      actions = game.actions[state_idx:state_idx + unroll_steps]
      targets = []

      for j in range(unroll_steps):
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
      batch.append((state, actions, targets, weights[i]))

    return batch

  def sample_games(self, num_samples: int) -> list[Game]:
    # Sample game from buffer either uniformly or according to some priority.
    priorities = np.array([g.priority for g in self.buffer])

    if len(priorities) == 0:
      raise Exception("Replay buffer is empty")

    probabilities = priorities / priorities.sum()
    games = np.random.choice(self.buffer, p=probabilities, size=num_samples)

    return games.tolist()

  def sample_state(self, game: Game) -> int:
    """
    Sample an observation from game using priorities
    Take from len(game.actions) because of the initial state
    """
    if len(game.priorities) == 0:
      raise Exception("Game has no priorities")
    
    priorities = np.array(game.priorities)
    probabilities = priorities / priorities.sum()
    idx = np.random.choice(len(game.actions), p=probabilities)

    return idx
