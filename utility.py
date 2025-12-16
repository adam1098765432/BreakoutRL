from multiprocessing import Process
import numpy as np
import torch
from config import *
import torch.nn.functional as F

def launch_job(func, *args):
  p = Process(target=func, args=args)
  p.daemon = True # Close the process if the main process is closed
  p.start()
  return p

def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
  """
  Scales the gradient by a factor.
  """
  # return tensor * scale + tensor.detach() * (1 - scale)
  tensor.register_hook(lambda grad: grad * scale)
  return tensor

def get_temperature(training_steps):
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

def one_hot_action_batched(actions: torch.Tensor, action_size: int = ACTION_SIZE):
  """
  actions: (B,) long tensor
  returns: (B, action_size)
  """
  return F.one_hot(actions, num_classes=action_size).float()

def support_to_scalar(x: torch.Tensor, support_size=SUPPORT_SIZE, is_prob=False) -> float:
  """
  Inputs support logits or probabilities (if is_prob is True)
  Outputs scalar float
  Taken from https://github.com/werner-duvaud/muzero-general
  Transform a categorical representation to a scalar
  See paper appendix Network Architecture
  """
  support_size = int(support_size)
  # Decode to a scalar
  probabilities = x if is_prob else torch.softmax(x, dim=1)
  support = (
    torch.tensor([x for x in range(-support_size, support_size + 1)])
    .expand(probabilities.shape)
    .float()
    .to(device=probabilities.device)
  )

  x = torch.sum(support * probabilities, dim=1, keepdim=True)

  # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
  x = torch.sign(x) * (
    ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
    ** 2
    - 1
  )

  return x.item() if x.shape[0] == 1 else x

def scalar_to_support(x: float, support_size=SUPPORT_SIZE) -> torch.Tensor:
  """
  Inputs scalar float
  Outputs support probabilities
  Taken from https://github.com/werner-duvaud/muzero-general
  Transform a scalar to a categorical representation with (2 * support_size + 1) categories
  See paper appendix Network Architecture
  """
  support_size = int(support_size)
  x = torch.tensor([x], device=device, dtype=torch.float32).unsqueeze(0)

  # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
  x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

  # Encode on a vector
  x = torch.clamp(x, -support_size, support_size)
  floor = x.floor()
  prob = x - floor
  logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
  logits.scatter_(
    2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
  )
  indexes = floor + support_size + 1
  prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
  indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
  logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
  logits = logits[0]
  return logits
