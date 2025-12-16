from bridge import Bridge
from game import Environment
from networks import Network
from replay_buffer import ReplayBuffer
import time
from utility import *
import torch.nn.functional as F

def train(replay_buffer: ReplayBuffer, bridge: Bridge, Env=Environment):
  """
  Training loop.
  
  ### Note
  Remember to scale the loss by 1 / K_STEPS to ensure the gradient has a similar magnitude
  regardless of the number of unroll steps.
  """
  network = Network.load(NETWORK_PATH)
  network = network.to(device)
  # freeze_value_and_reward(network)
  optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, network.parameters()),
    lr=LR_INIT,
    weight_decay=WEIGHT_DECAY # L2 regularization (keeps weights small)
  )

  # Wait for a game to complete
  attempts = 0
  while len(replay_buffer.buffer) < 1:
    attempts += 1
    fetch_games(replay_buffer, bridge, Env)
    time.sleep(5)

  print("Training...")
  for i in range(int(TRAINING_STEPS)):
    if (i + 1) % SAVE_EVERY == 0:
      print("Saving network...")
      bridge.broadcast_network(network)
      Network.save(network, NETWORK_PATH)
    if (i + 1) % FETCH_EVERY == 0:
      fetch_games(replay_buffer, bridge)
    batch = replay_buffer.sample_batch(UNROLL_STEPS, TD_STEPS)
    update_weights(optimizer, network, batch, bridge)

def update_weights(optimizer: torch.optim, network: Network, batch: list[tuple], bridge: Bridge):
  learning_rate = LR_INIT * LR_DECAY_RATE ** (network.training_steps / LR_DECAY_STEPS)
  optimizer.param_groups[0]['lr'] = learning_rate
  optimizer.zero_grad()
  value_loss = 0
  reward_loss = 0
  policy_loss = 0
  state_loss = 0
  loss = 0
  ZERO_TENSOR = torch.tensor(0.0).to(device)

  for game_states, actions, targets, weight in batch:

    # Initial step
    # This is the hidden state for the first unroll step
    # The value is the expected value of the first unroll step
    # The reward is not used
    # The policy is the policy for the first unroll step
    hidden_state, reward, policy_logits, value = network.initial_forward_grad(game_states[0])
    predictions = [(1.0, value, reward, policy_logits, hidden_state)]

    # Recurrent steps
    for action in actions:
      # The first time this runs, it is ran on the initial hidden state
      hidden_state, reward, policy_logits, value = network.recurrent_forward_grad(hidden_state, action)
      hidden_state = scale_gradient(hidden_state, 0.5)
      predictions += [(1 / len(actions), value, reward, policy_logits, hidden_state)]

    # Accumulate predictions and targets
    for i, (prediction, target) in enumerate(zip(predictions, targets)):
      # Logits
      gradient_scale, pred_value_logits, pred_reward_logits, pred_policy_logits, pred_hidden_state = prediction
      
      # Probabilities
      target_value, target_reward, target_policy = target
      target_hidden_state = None if i == 0 else network.initial_forward_grad(game_states[i])[0].detach()

      # Convert predicted logits to log probabilities
      pred_value_log_probs = F.log_softmax(pred_value_logits, dim=1)
      pred_reward_log_probs = F.log_softmax(pred_reward_logits, dim=1)
      pred_policy_log_probs = F.log_softmax(pred_policy_logits, dim=1)

      # Cross entropy loss with target probabilities
      # Ignore reward loss for the first step
      raw_value_loss = -((pred_value_log_probs * target_value).sum(dim=1)).mean()
      raw_reward_loss = ZERO_TENSOR if i == 0 else -((pred_reward_log_probs * target_reward).sum(dim=1)).mean()
      raw_policy_loss = -((pred_policy_log_probs * target_policy).sum(dim=1)).mean()
      raw_state_loss = ZERO_TENSOR if i == 0 else F.mse_loss(normalize(pred_hidden_state), normalize(target_hidden_state))
      raw_loss = raw_value_loss * VALUE_LOSS_WEIGHT + raw_reward_loss + raw_policy_loss + raw_state_loss * CONSISTENCY_LOSS_WEIGHT

      # Weight is to account for sampling bias
      loss += weight * scale_gradient(raw_loss, gradient_scale)

      # Collect losses
      value_loss += raw_value_loss.item()
      reward_loss += raw_reward_loss.item()
      policy_loss += raw_policy_loss.item()
      state_loss += raw_state_loss.item()

  # Backpropagate (already scaling gradient by 1 / unroll steps so just mean over batch)
  n_losses = len(batch)
  loss = loss / n_losses
  loss.backward()
  torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
  optimizer.step()
  network.training_steps += 1

  # Logs
  if network.training_steps % LOG_EVERY == 0 and bridge.has_log():
    logs = bridge.receive_log()
    sections = [
      f"Step: {network.training_steps:>6d}",
      f"Value Loss: {(value_loss / n_losses):.4f}",
      f"Reward Loss: {(reward_loss / n_losses):.8f}",
      f"Policy Loss: {(policy_loss / n_losses):.4f}",
      f"State Loss: {(state_loss / n_losses):.4f}",
    ]

    for k, v in logs.items():
      sections.append(f"{k}: {v}")

    print(" | ".join(sections))

def fetch_games(replay_buffer: ReplayBuffer, bridge: Bridge, Env=Environment):
  games_received = 0
  while bridge.has_game():
    game = bridge.receive_game(Env)
    replay_buffer.add_game(game)
    games_received += 1
  # print(f"Received {games_received} games")

def freeze_value_and_reward(network: Network):
  # Freeze value head
  for p in network.prediction_model.value.parameters():
    p.requires_grad = False

  # Freeze reward head
  for p in network.dynamics_model.reward.parameters():
    p.requires_grad = False

def normalize(x):
  return x / (x.norm(dim=1, keepdim=True) + 1e-8)
