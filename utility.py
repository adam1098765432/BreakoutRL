from multiprocessing import Process
import numpy as np
import torch
from config import *

def launch_job(func, *args):
  p = Process(target=func, args=args)
  p.daemon = True # Close the process if the main process is closed
  p.start()
  return p

def scale_gradient(tensor: torch.Tensor, scale: float):
  """
  Scales the gradient by a factor.
  """
  # return tensor * scale + tensor.detach() * (1 - scale)
  tensor.register_hook(lambda grad: grad * scale)
  return tensor

def get_temperature(num_moves, training_steps):
  if training_steps < 500_000:
    return 1.0
  elif training_steps < 750_000:
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
  arr = torch.zeros(size=(1, ACTION_SIZE), device=device)
  arr[0,x] = 1
  return arr

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

  support = torch.zeros(size=(1, support_size), dtype=torch.float32, device=device)

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
