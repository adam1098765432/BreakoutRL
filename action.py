import torch
import enum

class Action(enum.Enum):
  MOVE_LEFT = 0
  MOVE_RIGHT = 1
  DONT_MOVE = 2

  @staticmethod
  def to_tensor(action: 'Action'):
    return torch.tensor([
      action == Action.MOVE_LEFT,
      action == Action.MOVE_RIGHT,
      action == Action.DONT_MOVE
    ])
  
  @staticmethod
  def from_tensor(tensor: torch.Tensor):
    return Action(tensor[0], tensor[1], tensor[2])