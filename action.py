import torch

class Action:
  def __init__(self, move_left, move_right, dont_move):
    self.move_left = move_left
    self.move_right = move_right
    self.dont_move = dont_move

  @staticmethod
  def to_tensor(self, action):
    return torch.tensor([action.move_left, action.move_right, action.dont_move])
  
  @staticmethod
  def from_tensor(self, tensor):
    return Action(tensor[0], tensor[1], tensor[2])