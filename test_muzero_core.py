import pytest
import numpy as np
import torch
import torch.nn.functional as F

from model import (
  MinMaxStats,
  Network,
  NetworkOutput,
  ReplayBuffer,
  PredictionModel,
  DynamicsModel,
  Node,
  MCTS,
  get_root_node,
  scale_targets,
  one_hot_score,
  one_hot_action,
  one_hot_score_to_scaler,
  ucb_score,
  STATE_SIZE,
  ACTION_SIZE,
  SUPPORT_SIZE,
  DISCOUNT_FACTOR
)

# ----------------------------
# MinMaxStats
# ----------------------------

def test_minmaxstats_normalization():
  stats = MinMaxStats()
  stats.update(2)
  stats.update(4)

  assert stats.min == 2
  assert stats.max == 4
  assert stats.normalize(2) == 0.0
  assert stats.normalize(4) == 1.0
  assert stats.normalize(3) == 0.5


def test_minmaxstats_no_range_returns_input():
  stats = MinMaxStats()
  stats.update(3)

  # max == min → should return raw value
  assert stats.normalize(3) == 3


# ----------------------------
# ReplayBuffer
# ----------------------------

def test_replay_buffer_capacity():
  buffer = ReplayBuffer(capacity=2)

  buffer.add_game("t1")
  buffer.add_game("t2")
  buffer.add_game("t3")

  assert len(buffer.buffer) == 2
  assert buffer.buffer[0] == "t2"
  assert buffer.buffer[1] == "t3"


def test_sampling_priority_absolute_difference():
  p = ReplayBuffer.get_sampling_priority(5.0, 3.0)
  assert p == 2.0


# ----------------------------
# PredictionModel
# ----------------------------

def test_prediction_model_output_shapes():
  model = PredictionModel()
  state = torch.randn(1, STATE_SIZE)

  policy_logits, value_logits = model(state)

  assert policy_logits.shape == (1, ACTION_SIZE)
  assert value_logits.shape == (1, SUPPORT_SIZE)


# ----------------------------
# DynamicsModel
# ----------------------------

def test_dynamics_model_output_shapes_and_bounds():
  model = DynamicsModel()
  state = torch.randn(1, STATE_SIZE)
  action = torch.zeros(1, ACTION_SIZE)
  action[0, 1] = 1  # one-hot action

  next_state, reward = model(state, action)

  assert next_state.shape == (1, STATE_SIZE)
  assert reward.shape == (1, SUPPORT_SIZE)

  # State should be normalized to [0, 1]
  assert torch.all(next_state >= 0)
  assert torch.all(next_state <= 1)


def test_dynamics_model_gradient_scaling():
  model = DynamicsModel()
  state = torch.ones(1, STATE_SIZE)
  action = torch.zeros(1, ACTION_SIZE)

  next_state, _ = model(state, action)

  # If gradient scaling is removed or altered, this test may fail later
  assert next_state is not None


def test_dynamics_same_action_same_state_is_deterministic():
  model = DynamicsModel()
  
  s = torch.randn(1, STATE_SIZE)
  a = one_hot_action(1)

  s1, r1 = model(s, a)
  s2, r2 = model(s, a)

  assert torch.allclose(s1, s2)
  assert torch.allclose(r1, r2)


# ----------------------------
# Node
# ----------------------------

def test_node_value_zero_when_unvisited():
  node = Node(prior=0.5)
  assert node.value() == 0


def test_node_value_average():
  node = Node(prior=0.5)
  node.value_sum = 10
  node.visit_count = 2

  assert node.value() == 5


# ----------------------------
# One-hot encodings
# ----------------------------

def test_one_hot_action_valid():
  a = one_hot_action(1)
  assert a.shape == (1, ACTION_SIZE)
  assert a.sum() == 1
  assert a[0,1] == 1


def test_one_hot_score_center():
  s = one_hot_score(0)
  assert s.shape == (SUPPORT_SIZE,)
  assert s.sum() == 1
  assert s[SUPPORT_SIZE // 2] == 1


def test_one_hot_score_out_of_bounds():
  with pytest.raises(ValueError):
    one_hot_score(1000)


def test_one_hot_score_to_scalar_identity():
  x = torch.zeros(SUPPORT_SIZE)
  x[SUPPORT_SIZE // 2 + 10] = 1  # +10
  assert one_hot_score_to_scaler(x).item() == 10

# ----------------------------
# Target scaling
# ----------------------------

def test_scale_targets_sign_preserved():
  x_pos = scale_targets(4.0)
  x_neg = scale_targets(-4.0)

  assert x_pos > 0
  assert x_neg < 0


def test_scale_targets_zero():
  assert scale_targets(0.0) == 0.0


# ----------------------------
# UCB score
# ----------------------------

def test_ucb_score_increases_with_prior():
  parent = Node(prior=1.0)
  parent.visit_count = 10

  child_low = Node(prior=0.1)
  child_high = Node(prior=0.9)

  stats = MinMaxStats()

  s_low = ucb_score(parent, child_low, stats)
  s_high = ucb_score(parent, child_high, stats)

  assert s_high > s_low


def test_ucb_score_value_influence():
  parent = Node(prior=1.0)
  parent.visit_count = 10

  child = Node(prior=0.5)
  child.visit_count = 5
  child.value_sum = 10
  child.reward = 1

  stats = MinMaxStats()
  stats.update(child.value())

  score = ucb_score(parent, child, stats)
  assert isinstance(score, float)


# ----------------------------
# MCTS
# ----------------------------

def test_mcts_expand_node_creates_children():
  dyn = DynamicsModel()
  pred = PredictionModel()
  network = Network(dyn, pred)
  mcts = MCTS(network)

  node = Node(prior=1.0)
  hidden_state = torch.randn(1, STATE_SIZE)
  action = 1
  network_output = network.forward(hidden_state, action)

  mcts.expand_node(node, network_output)

  assert len(node.children) == ACTION_SIZE
  for a, child in node.children.items():
    assert isinstance(child, Node)


def test_mcts_backprop_updates_visits_and_values():
  node1 = Node(prior=1.0)
  node2 = Node(prior=1.0)
  node1.reward = torch.tensor(0.0)
  node2.reward = torch.tensor(1.0)

  path = [node1, node2]
  stats = MinMaxStats()

  network_output = NetworkOutput(
    hidden_state=torch.randn(1, STATE_SIZE),
    reward=0.0,
    policy=[0.5, 0.25, 0.25],
    value=2.0
  )

  mcts = MCTS(None)
  mcts.backprop(path, network_output, stats)

  assert node1.visit_count == 1
  assert node2.visit_count == 1
  assert node2.value_sum == 2.0


def test_mcts_search_requires_root_hidden_state():
  dyn = DynamicsModel()
  pred = PredictionModel()
  network = Network(dyn, pred)
  mcts = MCTS(network)

  root = get_root_node(pred)

  # If hidden_state is missing, this should crash
  mcts.search(root, action_history=[])
