import torch

class State:
  def __init__(self, n_bricks):
    self.paddle_x = 0
    self.paddle_y = 0
    self.ball_x = 0
    self.ball_y = 0
    self.ball_vx = 0
    self.ball_vy = 0
    self.bricks = [1] * n_bricks

  @staticmethod
  def to_tensor(self, state):
    return torch.tensor([
      state.paddle_x,
      state.paddle_y,
      state.ball_x,
      state.ball_y,
      state.ball_vx,
      state.ball_vy,
      *state.bricks
    ])
  
  @staticmethod
  def from_tensor(self, tensor):
    # TODO: Normalize values between -1 and 1
    state = State()
    state.paddle_x = tensor[0]
    state.paddle_y = tensor[1]
    state.ball_x = tensor[2]
    state.ball_y = tensor[3]
    state.ball_vx = tensor[4]
    state.ball_vy = tensor[5]
    state.bricks = tensor[6:]
    return state