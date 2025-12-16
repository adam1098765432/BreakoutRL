from utility import *
import torch.nn as nn
import torch.nn.functional as F
import os

"""
Might need to make these models bigger
(especially before the heads)
"""

class RepresentationModel(nn.Module):
  """
  Take a state as input and output a latent representation.
  """
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(STATE_SIZE, HIDDEN_SIZE)

  def forward(self, state):
    hidden_state = self.fc1(state)

    # Scale hidden state between [0, 1]
    min_enc = hidden_state.min(1, keepdim=True)[0]
    max_enc = hidden_state.max(1, keepdim=True)[0]
    scl_enc = max_enc - min_enc
    scl_enc[scl_enc < 1e-5] += 1e-5
    hidden_state_norm = (hidden_state - min_enc) / scl_enc

    return hidden_state_norm

class PredictionModel(nn.Module):
  """
  Two-headed model for predicting policy and value from a state.
  """
  def __init__(self):
    super().__init__()
    self.policy_fc1 = nn.Linear(HIDDEN_SIZE, 16)
    self.policy_fc2 = nn.Linear(16, ACTION_SIZE)
    self.value_fc1 = nn.Linear(HIDDEN_SIZE, 16)
    self.value_fc2 = nn.Linear(16, 2 * SUPPORT_SIZE + 1)
    
    # Zero initialize value head
    # nn.init.zeros_(self.value_fc2.weight)
    # nn.init.zeros_(self.value_fc2.bias)

    # Initialize policy head with scaled normal
    # nn.init.normal_(self.policy_fc2.weight, mean=0, std=0.01)
    # nn.init.normal_(self.policy_fc2.bias, mean=0, std=0.01)

  def forward(self, state):
    p = self.policy_fc1(state)
    p = F.elu(p)
    p = self.policy_fc2(p)
    v = self.value_fc1(state)
    v = F.elu(v)
    v = self.value_fc2(v)
    return p, v

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
    self.state_fc1 = nn.Linear(HIDDEN_SIZE + ACTION_SIZE, 16)
    self.state_fc2 = nn.Linear(16, HIDDEN_SIZE)
    self.reward_fc1 = nn.Linear(HIDDEN_SIZE, 16)
    self.reward_fc2 = nn.Linear(16, 2 * SUPPORT_SIZE + 1)

    # Initialize reward head with zeros
    # nn.init.zeros_(self.reward_fc2.weight)
    # nn.init.zeros_(self.reward_fc2.bias)

    # Initialize state head with scaled normal
    # nn.init.normal_(self.state_fc2.weight, mean=0, std=0.01)
    # nn.init.normal_(self.state_fc2.bias, mean=0, std=0.01)

  def forward(self, state, action):
    # print(state.shape, action.shape)
    out = torch.cat([state, action], dim=1)
    out = self.state_fc1(out)
    out = F.elu(out)
    hidden_state = self.state_fc2(out)
    out = self.reward_fc1(hidden_state)
    out = F.elu(out)
    reward = self.reward_fc2(out)

    # Scale hidden state between [0, 1]
    min_enc = hidden_state.min(1, keepdim=True)[0]
    max_enc = hidden_state.max(1, keepdim=True)[0]
    scl_enc = max_enc - min_enc
    scl_enc[scl_enc < 1e-5] += 1e-5
    hidden_state_norm = (hidden_state - min_enc) / scl_enc

    return hidden_state_norm, reward

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
    self.representation_model = RepresentationModel()
    self.dynamics_model = DynamicsModel()
    self.prediction_model = PredictionModel()
    self.training_steps = 0

  def initial_forward(self, state: torch.Tensor):
    hidden_state = self.representation_model(state)
    policy_logits, value = self.prediction_model(hidden_state)
    value = support_to_scalar(value)
    return NetworkOutput(hidden_state, 0.0, policy_logits, value)
  
  def initial_forward_grad(self, state: torch.Tensor):
    hidden_state = self.representation_model(state)
    policy_logits, value = self.prediction_model(hidden_state)
    reward = scalar_to_support(0).repeat(state.shape[0], 1)
    return hidden_state, reward, policy_logits, value

  def recurrent_forward(self, state: torch.Tensor, action: int):
    hidden_state, reward = self.dynamics_model(state, one_hot_action(action))
    policy_logits, value = self.prediction_model(hidden_state)
    value = support_to_scalar(value)
    reward = support_to_scalar(reward)
    return NetworkOutput(hidden_state, reward, policy_logits, value)
  
  def recurrent_forward_grad(self, state: torch.Tensor, action: torch.Tensor):
    hidden_state, reward = self.dynamics_model(state, one_hot_action_batched(action))
    policy_logits, value = self.prediction_model(hidden_state)
    return hidden_state, reward, policy_logits, value

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
    reward = scalar_to_support(0)
    value = scalar_to_support(0)
    return hidden_state, reward, policy_logits, value
    # return NetworkOutput(hidden_state, reward, policy_logits, value)

  def recurrent_forward(self, state: torch.Tensor, action: int):
    hidden_state = torch.ones(1, HIDDEN_SIZE, device=device)
    policy_logits = torch.ones(1, ACTION_SIZE, device=device)
    return NetworkOutput(hidden_state, 0, policy_logits, 0)
  
  def recurrent_forward_grad(self, state: torch.Tensor, action: int):
    hidden_state = torch.ones(1, HIDDEN_SIZE, device=device)
    policy_logits = torch.ones(1, ACTION_SIZE, device=device)
    reward = scalar_to_support(0)
    value = scalar_to_support(0)
    return hidden_state, reward, policy_logits, value
    # return NetworkOutput(hidden_state, reward, policy_logits, value)
