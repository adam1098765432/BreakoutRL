from utility import *
import torch.nn as nn
import torch.nn.functional as F
import os

class RepresentationModel(nn.Module):
  """
  Take a state as input and output a latent representation.
  """
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)
    self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return x

class PredictionModel(nn.Module):
  """
  Two-headed model for predicting policy and value from a state.
  """
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    self.reduce_policy = nn.Linear(HIDDEN_SIZE, 4)
    self.reduce_value = nn.Linear(HIDDEN_SIZE, 4)
    self.policy = nn.Linear(4, ACTION_SIZE) # The move to play
    self.value = nn.Linear(4, SUPPORT_SIZE) # Expected final reward

    # Zero initialize value head
    nn.init.zeros_(self.value.weight)
    nn.init.zeros_(self.value.bias)

    # Initialize policy head with scaled normal
    nn.init.normal_(self.policy.weight, mean=0, std=0.01)
    nn.init.normal_(self.policy.bias, mean=0, std=0.01)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    p = self.reduce_policy(x)
    v = self.reduce_value(x)
    return self.policy(p), self.value(v)

class ResBlock(nn.Module):
  """
  This residual block is based on https://arxiv.org/pdf/1603.05027.pdf<br>
  It is uses the constant scaling method since it is not a CNN.

  :param channels: The number of channels in the input
  :param alpha: The scaling factor
  """
  def __init__(self, channels, alpha=0.2):
    super().__init__()
    self.alpha = alpha
    self.fc1 = nn.Linear(channels, channels)
    self.n1 = nn.LayerNorm(channels)
    self.fc2 = nn.Linear(channels, channels)
    self.n2 = nn.LayerNorm(channels)

  def forward(self, x):
    identity = x
    x = F.relu(self.fc1(x))
    x = self.n1(x)
    x = F.relu(self.fc2(x))
    x = self.n2(x)
    x = identity + x * self.alpha
    return x

class DynamicsModel(nn.Module):
  """
  Model Architecture Based On: https://arxiv.org/pdf/1603.05027<br>
  Two-headed model for predicting next state and reward from a state and action.
  Usually, you pass in multiple previous states and actions, but for now we will
  only pass in a single previous state and action. This may make it harder to train.

  ### Note
  MuZero Appendix G says to scale the gradient of the dynamics function by 0.5.
  To do this, the input state is multiplied by 0.5.

  ### Note
  MuZero Appendix G says to scale the hidden state after running the dynamics
  function to [0, 1] (once per unroll step).

  :param latent_size: The number of channels in the latent representation.
  :param n_blocks: The number of residual blocks.
  """
  def __init__(self, n_blocks=2):
    super().__init__()
    self.model = nn.Sequential(*[ResBlock(HIDDEN_SIZE + ACTION_SIZE) for _ in range(n_blocks)])
    self.state = nn.Linear(HIDDEN_SIZE + ACTION_SIZE, HIDDEN_SIZE)
    self.reward = nn.Linear(HIDDEN_SIZE + ACTION_SIZE, SUPPORT_SIZE)

    # Initialize reward head with zeros
    nn.init.zeros_(self.reward.weight)
    nn.init.zeros_(self.reward.bias)

    # Initialize state head with scaled normal
    nn.init.normal_(self.state.weight, mean=0, std=0.01)
    nn.init.normal_(self.state.bias, mean=0, std=0.01)

  def forward(self, state, action):
    x = torch.cat([state, action], dim=1)
    x = self.model(x)
    state = F.relu(self.state(x))
    reward = self.reward(x)
    return state, reward

class NetworkOutput:
  """
  :param hidden_state: The latent representation of the state
  :param reward: The reward for the state
  :param policy_logits: The policy for the state
  :param value: The value for the state
  """
  hidden_state: torch.Tensor
  reward: float
  policy_logits: torch.Tensor
  value: float

  def __init__(self, hidden_state, reward, policy_logits, value):
    self.hidden_state = hidden_state
    self.reward = reward
    self.policy_logits = policy_logits
    self.value = value

  def __str__(self):
    return f"NetworkOutput(hidden_state={self.hidden_state}, reward={self.reward}, policy_logits={self.policy_logits}, value={self.value})"

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.latent_model = RepresentationModel()
    self.dynamics_model = DynamicsModel()
    self.prediction_model = PredictionModel()
    self.training_steps = 0

  def initial_forward(self, state: torch.Tensor):
    hidden_state = self.latent_model(state)
    policy_logits, value = self.prediction_model(hidden_state)

    value = inverse_value_transform(support_to_scalar(value))

    return NetworkOutput(hidden_state, 0.0, policy_logits, value)
  
  def initial_forward_grad(self, state: torch.Tensor):
    hidden_state = self.latent_model(state)
    policy_logits, value = self.prediction_model(hidden_state)

    reward = scalar_to_support(value_transform(0))

    return NetworkOutput(hidden_state, reward, policy_logits, value)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)

    value = inverse_value_transform(support_to_scalar(value))
    reward = inverse_value_transform(support_to_scalar(reward))

    return NetworkOutput(hidden_state, reward, policy_logits, value)
  
  def recurrent_forward_grad(self, state: torch.Tensor, action: int):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)

    return NetworkOutput(hidden_state, reward, policy_logits, value)

  @staticmethod
  def save(network: nn.Module, path: str):
    try:
      torch.save({
        'model': network.state_dict(),
        'steps': network.training_steps
      }, path + '.tmp')
      os.replace(path + '.tmp', path)
    except Exception as e:
      print(f"Failed to save model: {e}")
    
  @staticmethod
  def load(path: str):
    try:
      checkpoint = torch.load(path)
      network = Network().to(device)
      network.load_state_dict(checkpoint['model'])
      network.training_steps = checkpoint['steps']
      print(f"Loaded model from {path}")
      return network
    except Exception as e:
      print(f"Failed to load model: {e}")
      print('Making a new model...')
      return Network()

class UniformNetwork(Network):
  def __init__(self):
    super().__init__()

  def initial_forward(self, state):
    hidden_state = torch.ones(1, HIDDEN_SIZE, device=device)
    policy_logits = torch.ones(1, ACTION_SIZE, device=device)
    return NetworkOutput(hidden_state, 0, policy_logits, 0)
  
  def initial_forward_grad(self, state):
    hidden_state = torch.ones(1, HIDDEN_SIZE, device=device)
    policy_logits = torch.ones(1, ACTION_SIZE, device=device)
    reward = scalar_to_support(value_transform(0))
    value = scalar_to_support(value_transform(0))
    return NetworkOutput(hidden_state, reward, policy_logits, value)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    hidden_state = torch.ones(1, HIDDEN_SIZE, device=device)
    policy_logits = torch.ones(1, ACTION_SIZE, device=device)
    return NetworkOutput(hidden_state, 0, policy_logits, 0)
  
  def recurrent_forward_grad(self, state: torch.Tensor, action: int):
    hidden_state = torch.ones(1, HIDDEN_SIZE, device=device)
    policy_logits = torch.ones(1, ACTION_SIZE, device=device)
    reward = scalar_to_support(value_transform(0))
    value = scalar_to_support(value_transform(0))
    return NetworkOutput(hidden_state, reward, policy_logits, value)
