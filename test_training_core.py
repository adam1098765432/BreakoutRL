import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from bridge import Bridge
from config import *
from networks import Network
from training import update_weights_parallel, update_weights
import random

def test_update_weights_equivalence():
  device = torch.device("cpu")

  # --------------------------------------------------
  # Create identical networks
  # --------------------------------------------------
  net_ref = Network().to(device)
  net_par = copy.deepcopy(net_ref)

  net_ref.training_steps = 0
  net_par.training_steps = 0

  # --------------------------------------------------
  # Create identical optimizers
  # --------------------------------------------------
  opt_ref = torch.optim.Adam(net_ref.parameters(), lr=LR_INIT)
  opt_par = clone_optimizer(opt_ref, net_par.parameters())

  # --------------------------------------------------
  # Create a deterministic batch
  # --------------------------------------------------
  B = BATCH_SIZE
  K = UNROLL_STEPS
  z = STATE_SIZE
  s = 2 * SUPPORT_SIZE + 1
  a = ACTION_SIZE

  batch = []

  for _ in range(B):
    game_states = [torch.randn(1, z) for _ in range(K + 1)]
    actions = []
    targets = []
    
    for _ in range(K + 1):
      target_value = torch.softmax(torch.randn(1, s), dim=1)
      target_reward = torch.softmax(torch.randn(1, s), dim=1)
      target_policy = torch.softmax(torch.randn(1, a), dim=1)
      actions.append(random.randint(0, a - 1))
      targets.append((target_value, target_reward, target_policy))

    weight = np.random.rand()

    batch.append((game_states, actions, targets, weight))

  # --------------------------------------------------
  # Run both implementations
  # --------------------------------------------------
  losses_par, priorities_par = update_weights_parallel(
    opt_par, net_par, batch
  )

  priorities_ref = update_weights(
    opt_ref, net_ref, batch, bridge=Bridge()
  )

  # --------------------------------------------------
  # Compare network parameters
  # --------------------------------------------------
  assert_networks_equal(net_ref, net_par)

  # --------------------------------------------------
  # Compare priorities
  # --------------------------------------------------
  np.testing.assert_allclose(
    priorities_ref,
    priorities_par,
    atol=1e-5,
    rtol=1e-5
  )

  # --------------------------------------------------
  # Compare losses
  # --------------------------------------------------
  ref_losses = np.array([
    net_ref.last_value_loss,
    net_ref.last_reward_loss,
    net_ref.last_policy_loss
  ])

  par_losses = np.array(losses_par)

  np.testing.assert_allclose(
    ref_losses,
    par_losses,
    atol=1e-5,
    rtol=1e-5
  )

def clone_optimizer(opt, params):
  new_opt = type(opt)(params, **opt.defaults)
  new_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
  return new_opt

def assert_networks_equal(net1, net2):
  for (n1, p1), (n2, p2) in zip(net1.named_parameters(), net2.named_parameters()):
    assert n1 == n2
    assert_tensors_close(p1, p2)

def assert_tensors_close(a, b, atol=1e-6, rtol=1e-6):
  a = a.detach()
  b = b.detach()
  diff = (a - b).detach().abs()
  assert a.shape == b.shape
  assert torch.allclose(a, b, atol=atol, rtol=rtol), f"max diff: {diff.max().item()}, mean diff: {diff.mean().item()}"
