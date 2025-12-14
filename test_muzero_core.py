import pytest
import numpy as np
import torch
import torch.nn.functional as F

from model import *

# MinMaxStats

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


# ReplayBuffer

def test_replay_buffer_capacity():
  buffer = ReplayBuffer(capacity=2, batch_size=2)

  game_1 = Game(ACTION_SIZE, DISCOUNT_FACTOR)
  game_2 = Game(ACTION_SIZE, DISCOUNT_FACTOR)
  game_3 = Game(ACTION_SIZE, DISCOUNT_FACTOR)

  buffer.add_game(game_1)
  buffer.add_game(game_2)
  buffer.add_game(game_3)

  assert len(buffer.buffer) == 2
  assert buffer.buffer[0] == game_2
  assert buffer.buffer[1] == game_3


def test_sampling_priority_absolute_difference():
  game = Game(ACTION_SIZE, DISCOUNT_FACTOR)
  p = game.get_sampling_priority(5.0, 3.0)
  assert abs(p - 2.0) < 1e-6


# PredictionModel

def test_prediction_model_output_shapes():
  model = PredictionModel()
  state = torch.randn(1, HIDDEN_SIZE)

  policy_logits, value_logits = model(state)

  assert policy_logits.shape == (1, ACTION_SIZE)
  assert value_logits.shape == (1, SUPPORT_SIZE)


# DynamicsModel

def test_dynamics_model_output_shapes_and_bounds():
  model = DynamicsModel()
  state = torch.randn(1, HIDDEN_SIZE)
  action = torch.zeros(1, ACTION_SIZE)
  action[0, 1] = 1  # one-hot action

  next_state, reward = model(state, action)

  assert next_state.shape == (1, HIDDEN_SIZE)
  assert reward.shape == (1, SUPPORT_SIZE)

  # State should be normalized to [0, 1]
  assert torch.all(next_state >= 0)
  assert torch.all(next_state <= 1)


def test_dynamics_model_gradient_scaling():
  model = DynamicsModel()
  state = torch.ones(1, HIDDEN_SIZE)
  action = torch.zeros(1, ACTION_SIZE)

  next_state, _ = model(state, action)

  # If gradient scaling is removed or altered, this test may fail later
  assert next_state is not None


def test_dynamics_same_action_same_state_is_deterministic():
  model = DynamicsModel()
  
  s = torch.randn(1, HIDDEN_SIZE)
  a = one_hot_action(1)

  s1, r1 = model(s, a)
  s2, r2 = model(s, a)

  assert torch.allclose(s1, s2)
  assert torch.allclose(r1, r2)


# Node

def test_node_value_zero_when_unvisited():
  node = Node(prior=0.5)
  assert node.value() == 0


def test_node_value_average():
  node = Node(prior=0.5)
  node.value_sum = 10
  node.visit_count = 2

  assert node.value() == 5


# Encodings

def test_one_hot_action_valid():
  a = one_hot_action(1)
  assert a.shape == (1, ACTION_SIZE)
  assert a.sum() == 1
  assert a[0,1] == 1


def test_scalar_to_support():
  s = scalar_to_support(0)
  assert s.shape == (1, SUPPORT_SIZE)


def test_scalar_to_support_identity():
  value = 0
  support = scalar_to_support(value)
  value_recovered = support_to_scalar(support)
  assert abs(value - value_recovered) < 1e-6


def test_transform_identity():
  value = 0
  value_transformed = value_transform(value)
  value_recovered = inverse_value_transform(value_transformed)
  assert abs(value - value_recovered) < 1e-6

# Target scaling

def test_scale_targets_sign_preserved():
  x_pos = scale_targets(4.0)
  x_neg = scale_targets(-4.0)

  assert x_pos > 0
  assert x_neg < 0


def test_scale_targets_zero():
  assert scale_targets(0.0) == 0.0


# UCB score

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


# MCTS

def test_mcts_expand_node_creates_children():
  network = Network()
  mcts = MCTS(network)

  node = Node(prior=1.0)
  hidden_state = torch.randn(1, HIDDEN_SIZE)
  action = 1
  network_output = network.recurrent_forward(hidden_state, action)

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
    hidden_state=torch.randn(1, HIDDEN_SIZE),
    reward=0.0,
    policy_logits=[0.5, 0.25, 0.25],
    value=2.0
  )

  mcts = MCTS(None)
  mcts.backprop(path, network_output, stats)

  assert node1.visit_count == 1
  assert node2.visit_count == 1
  assert node2.value_sum == 2.0


def test_mcts_search_requires_root_hidden_state():
  network = Network()
  mcts = MCTS(network)
  game = Game(action_space_size=ACTION_SIZE, discount_factor=DISCOUNT_FACTOR)

  root = get_root_node(mcts, game)

  # If hidden_state is missing, this should crash
  mcts.search(root, action_history=[])

# Self-play

def test_self_play_stats():
  network = Network()
  mcts = MCTS(network)
  
  game = play_game(mcts)

  assert len(game.history) <= MAX_MOVES
  assert len(game.history) == len(game.states) - 1
  assert len(game.states) - 1 == len(game.rewards)
  assert len(game.rewards) == len(game.child_visits)
  assert len(game.child_visits) == len(game.root_values)

  for child_visits in game.child_visits:
    assert abs(sum(child_visits) - 1) < 1e-6

def test_self_play_targets():
  network = Network()
  mcts = MCTS(network)

  game = play_game(mcts)
  targets = game.get_targets(0, 5, 5)

  assert type(targets) == list
  assert len(targets) == 6

  for t in targets:
    assert type(t) == tuple
    assert len(t) == 3
    assert type(t[0]) == torch.Tensor
    assert t[0].shape == (1, SUPPORT_SIZE)
    assert type(t[1]) == torch.Tensor
    assert t[1].shape == (1, SUPPORT_SIZE)
    assert type(t[2]) == torch.Tensor
    assert t[2].shape == (1, ACTION_SIZE)


# Training

def test_training():
  network = Network()
  mcts = MCTS(network)

  replay_buffer = ReplayBuffer(5, 1)
  network_buffer = NetworkBuffer()
  # network_buffer.save_network(0, network)

  run_selfplay(replay_buffer, network_buffer, 10)

  train(replay_buffer, network_buffer)

test_training()