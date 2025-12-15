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
  loss = 0

  for game_state, actions, targets, weight in batch:

    # Initial step
    network_output = network.initial_forward_grad(game_state)
    predictions = [(
      1.0,
      network_output.value,
      network_output.reward,
      network_output.policy_logits
    )]

    # Recurrent steps
    for action in actions:
      network_output = network.recurrent_forward_grad(network_output.hidden_state, action)
      network_output.hidden_state = scale_gradient(network_output.hidden_state, 0.5)
      predictions += [(
        1 / len(actions), # Scale gradient by number of unroll steps (actions)
        network_output.value,
        network_output.reward,
        network_output.policy_logits
      )]

    # Accumulate predictions and targets
    for i, (prediction, target) in enumerate(zip(predictions, targets)):
      # Logits
      gradient_scale, pred_value_logits, pred_reward_logits, pred_policy_logits = prediction
      
      # Probabilities
      target_value, target_reward, target_policy = target

      # Convert predicted logits to log probabilities
      pred_value_log_probs = F.log_softmax(pred_value_logits, dim=1)
      pred_reward_log_probs = F.log_softmax(pred_reward_logits, dim=1)
      pred_policy_log_probs = F.log_softmax(pred_policy_logits, dim=1)

      # Cross entropy loss with target probabilities
      # Ignore reward loss for the first step
      raw_value_loss = -((pred_value_log_probs * target_value).sum(dim=1)).mean()
      raw_reward_loss = torch.tensor(0.0).to(device) if i == 0 else -((pred_reward_log_probs * target_reward).sum(dim=1)).mean()
      raw_policy_loss = -((pred_policy_log_probs * target_policy).sum(dim=1)).mean()
      
      # Weight is to account for sampling bias
      loss += weight * scale_gradient(
        raw_value_loss * VALUE_LOSS_WEIGHT + \
        raw_reward_loss + \
        raw_policy_loss,
        gradient_scale
      )

      # Collect losses
      value_loss += raw_value_loss.item()
      reward_loss += raw_reward_loss.item()
      policy_loss += raw_policy_loss.item()

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
      f"Reward Loss: {(reward_loss / n_losses):.4f}",
      f"Policy Loss: {(policy_loss / n_losses):.4f}"
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
