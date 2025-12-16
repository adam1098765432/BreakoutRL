from bridge import Bridge
from game import Environment
from networks import Network
from replay_buffer import ReplayBuffer
import time
from utility import *
import torch.nn.functional as F
from rich import inspect

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
  optimizer = torch.optim.Adam(
    network.parameters(),
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
    # Save network
    if (i + 1) % SAVE_EVERY == 0:
      print("Saving network...")
      bridge.broadcast_network(network)
      Network.save(network, NETWORK_PATH)
    
    # Fetch games
    if (i + 1) % FETCH_EVERY == 0:
      fetch_games(replay_buffer, bridge)
    
    # Save games
    if (i + 1) % SAVE_GAMES_EVERY == 0:
      ReplayBuffer.save(replay_buffer, REPLAY_BUFFER_PATH)

    # Train network
    batch, game_idx, state_idxs = replay_buffer.sample_batch(UNROLL_STEPS, TD_STEPS)
    # new_priorities = update_weights(optimizer, network, batch, bridge)
    losses, new_priorities = update_weights_parallel(optimizer, network, batch)

    # Update priorities
    replay_buffer.update_priorities(game_idx, state_idxs, new_priorities)

    # Logging
    if bridge.has_log():
      logs = bridge.receive_log()
      sections = [
        f"Step: {network.training_steps:>6d}",
        f"Value Loss: {losses[0]:.4f}",
        f"Reward Loss: {losses[1]:.8f}",
        f"Policy Loss: {losses[2]:.4f}",
      ]

      for k, v in logs.items():
        sections.append(f"{k}: {v}")

      print(" | ".join(sections))

def update_weights_parallel(optimizer: torch.optim, network: Network, batch: list[tuple]):
  """
  Batch:
  game_states      (B, K+1, z)
  actions          (B, K)
  target_value     (B, K+1, s)
  target_reward    (B, K+1, s)
  target_policy    (B, K+1, a)
  weights          (B,)
  """
  
  optimizer.zero_grad()

  FULL_SUPPORT_SIZE = 2 * SUPPORT_SIZE + 1

  # game_states: list[B] of list[K+1] of (1, z)
  game_states = torch.cat(
    [torch.cat(gs, dim=0).unsqueeze(0) for gs, _, _, _ in batch],
    dim=0
  )  # (B, K+1, z)

  actions = torch.tensor(
    [a for _, a, _, _ in batch],
    device=device,
    dtype=torch.long
  )  # (B, K)

  weights = torch.tensor(
    [w for _, _, _, w in batch],
    device=device
  )  # (B,)

  target_value = torch.cat([
    torch.cat([target[0] for target in targets], dim=0).unsqueeze(0)
    for _, _, targets, _ in batch
  ], dim=0)  # (B, K+1, s)

  target_reward = torch.cat([
    torch.cat([target[1] for target in targets], dim=0).unsqueeze(0)
    for _, _, targets, _ in batch
  ], dim=0)  # (B, K+1, s)

  target_policy = torch.cat([
    torch.cat([target[2] for target in targets], dim=0).unsqueeze(0)
    for _, _, targets, _ in batch
  ], dim=0)  # (B, K+1, a)

  # print(game_states.shape)
  # print(actions.shape)
  # print(weights.shape)
  # print(target_value.shape)
  # print(target_reward.shape)
  # print(target_policy.shape)

  # Initial states
  initial_states = game_states[:, 0]  # (B, z)

  hidden, reward_logits, policy_logits, value_logits = network.initial_forward_grad(initial_states)

  # hidden         (B, h)
  # reward_logits  (B, s)
  # policy_logits  (B, a)
  # value_logits   (B, s)

  pred_value_logits = []
  pred_reward_logits = []
  pred_policy_logits = []

  pred_value_logits.append(value_logits)
  pred_reward_logits.append(reward_logits)
  pred_policy_logits.append(policy_logits)

  for k in range(UNROLL_STEPS):
    hidden, reward_logits, policy_logits, value_logits = network.recurrent_forward_grad(hidden, actions[:, k])

    hidden = scale_gradient(hidden, 0.5)

    pred_value_logits.append(value_logits)
    pred_reward_logits.append(reward_logits)
    pred_policy_logits.append(policy_logits)

  pred_value_logits  = torch.stack(pred_value_logits, dim=1)   # (B, K+1, s)
  pred_reward_logits = torch.stack(pred_reward_logits, dim=1)  # (B, K+1, s)
  pred_policy_logits = torch.stack(pred_policy_logits, dim=1)  # (B, K+1, a)

  value_log_probs  = F.log_softmax(pred_value_logits, dim=-1)
  reward_log_probs = F.log_softmax(pred_reward_logits, dim=-1)
  policy_log_probs = F.log_softmax(pred_policy_logits, dim=-1)

  value_loss = -(value_log_probs * target_value).sum(dim=-1)     # (B, K+1)
  reward_loss = -(reward_log_probs * target_reward).sum(dim=-1)  # (B, K+1)
  policy_loss = -(policy_log_probs * target_policy).sum(dim=-1)  # (B, K+1)

  # Ignore reward loss at k=0
  reward_loss[:, 0] = 0.0

  # total_loss: (B, K+1)
  total_loss = (
    VALUE_LOSS_WEIGHT * value_loss +
    reward_loss +
    policy_loss
  ) # (B, K+1)

  # grad_scale: (1, K+1)
  grad_scale = torch.ones(
    (1, UNROLL_STEPS + 1),
    device=total_loss.device
  )
  grad_scale[:, 1:] = 1.0 / UNROLL_STEPS

  scaled_loss = total_loss * grad_scale
  scaled_loss = scaled_loss.sum(dim=1)
  scaled_loss = weights * scaled_loss
  loss = scaled_loss.mean()

  loss.backward()
  torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
  optimizer.step()
  network.training_steps += 1

  pred_scalar = support_to_scalar(
    pred_value_logits.reshape(-1, FULL_SUPPORT_SIZE),
    is_prob=False
  ).reshape(BATCH_SIZE, UNROLL_STEPS + 1)

  target_scalar = support_to_scalar(
    target_value.reshape(-1, FULL_SUPPORT_SIZE),
    is_prob=True
  ).reshape(BATCH_SIZE, UNROLL_STEPS + 1)

  new_priorities = torch.abs(pred_scalar - target_scalar)  # (B, K+1)

  losses = (
    value_loss.mean(dim=1).mean().item(),
    reward_loss.mean(dim=1).mean().item(),
    policy_loss.mean(dim=1).mean().item()    
  )

  return losses, new_priorities.detach().cpu().numpy()

def update_weights(optimizer: torch.optim, network: Network, batch: list[tuple], bridge: Bridge):
  learning_rate = LR_INIT * LR_DECAY_RATE ** (network.training_steps / LR_DECAY_STEPS)
  optimizer.param_groups[0]['lr'] = learning_rate
  optimizer.zero_grad()
  value_loss = 0
  reward_loss = 0
  policy_loss = 0
  # state_loss = 0
  loss = 0
  ZERO_TENSOR = torch.tensor(0.0).to(device)
  new_priorities = np.empty(shape=(len(batch), UNROLL_STEPS + 1))

  for i, (game_states, actions, targets, weight) in enumerate(batch):

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
      action = torch.tensor(action).to(device).unsqueeze(0)
      hidden_state, reward, policy_logits, value = network.recurrent_forward_grad(hidden_state, action)
      hidden_state = scale_gradient(hidden_state, 0.5)
      predictions += [(1 / len(actions), value, reward, policy_logits, hidden_state)]

    # Accumulate predictions and targets
    for j, (prediction, target) in enumerate(zip(predictions, targets)):
      # Logits
      gradient_scale, pred_value_logits, pred_reward_logits, pred_policy_logits, pred_hidden_state = prediction
      
      # Probabilities
      target_value, target_reward, target_policy = target
      # target_hidden_state = None if i == 0 else network.initial_forward_grad(game_states[i])[0].detach()

      # Convert predicted logits to log probabilities
      pred_value_log_probs = F.log_softmax(pred_value_logits, dim=1)
      pred_reward_log_probs = F.log_softmax(pred_reward_logits, dim=1)
      pred_policy_log_probs = F.log_softmax(pred_policy_logits, dim=1)

      # Cross entropy loss with target probabilities
      # Ignore reward loss for the first step
      raw_value_loss = -((pred_value_log_probs * target_value).sum(dim=1)).mean()
      raw_reward_loss = ZERO_TENSOR if j == 0 else -((pred_reward_log_probs * target_reward).sum(dim=1)).mean()
      raw_policy_loss = -((pred_policy_log_probs * target_policy).sum(dim=1)).mean()
      raw_loss = raw_value_loss * VALUE_LOSS_WEIGHT + raw_reward_loss + raw_policy_loss
      # raw_state_loss = ZERO_TENSOR if i == 0 else F.mse_loss(normalize(pred_hidden_state), normalize(target_hidden_state))
      # raw_loss = raw_value_loss * VALUE_LOSS_WEIGHT + raw_reward_loss + raw_policy_loss + raw_state_loss * CONSISTENCY_LOSS_WEIGHT

      # Weight is to account for sampling bias
      loss += weight * scale_gradient(raw_loss, gradient_scale)

      # Save new priorities
      pred_value_scalar = support_to_scalar(pred_value_logits)
      target_value_scalar = support_to_scalar(target_value, is_prob=True)
      new_priorities[i][j] = np.abs(pred_value_scalar - target_value_scalar)

      # Collect losses
      value_loss += raw_value_loss.item()
      reward_loss += raw_reward_loss.item()
      policy_loss += raw_policy_loss.item()
      # state_loss += raw_state_loss.item()

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
      # f"State Loss: {(state_loss / n_losses):.4f}",
    ]

    for k, v in logs.items():
      sections.append(f"{k}: {v}")

    print(" | ".join(sections))

  return new_priorities

def fetch_games(replay_buffer: ReplayBuffer, bridge: Bridge, Env=Environment):
  games_received = 0
  while bridge.has_game():
    game = bridge.receive_game(Env)
    replay_buffer.add_game(game)
    games_received += 1
  # print(f"Received {games_received} games")

def normalize(x):
  return x / (x.norm(dim=1, keepdim=True) + 1e-8)
