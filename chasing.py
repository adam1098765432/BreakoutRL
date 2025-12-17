from multiprocessing import freeze_support, set_start_method
import os
import random
from typing import Callable
from mcts import MCTS
from muzero import muzero
from networks import Network
from replay_buffer import ReplayBuffer
from self_play import Environment, Game, get_root_node
from config import *
import numpy as np
import pygame
import torch
import sys

# Game parameters
BALL_RADIUS = 0.015

class IDX:
  BALL_X = 0
  BALL_Y = 1
  TARGET_X = 2
  TARGET_Y = 3
  TIME = 4

class Chasing(Environment):
  def __init__(self, device):
    super().__init__(device)
    self.frame_skip = 6

    # Initialize state
    self.state[0, IDX.BALL_X] = random.random() * 0.8 + 0.1
    self.state[0, IDX.BALL_Y] = random.random() * 0.8 + 0.1
    self.state[0, IDX.TARGET_X] = random.random() * 0.8 + 0.1
    self.state[0, IDX.TARGET_Y] = random.random() * 0.8 + 0.1
    self.state[0, IDX.TIME] = 0

    self.is_done = False

  def terminal(self):
    return self.is_done

  def step(self, action: int):
    state = self.state
    reward = 0.0

    # Move ball
    if action == 0: # Left
      state[0, IDX.BALL_X] -= 0.01
    elif action == 1: # Right
      state[0, IDX.BALL_X] += 0.01
      # reward = 1
    elif action == 2: # Up
      state[0, IDX.BALL_Y] -= 0.01
    elif action == 3: # Down
      state[0, IDX.BALL_Y] += 0.01
    elif action == 4: # Up-Left
      state[0, IDX.BALL_X] -= 0.01 / np.sqrt(2)
      state[0, IDX.BALL_Y] -= 0.01 / np.sqrt(2)
    elif action == 5: # Up-Right
      state[0, IDX.BALL_X] += 0.01 / np.sqrt(2)
      state[0, IDX.BALL_Y] -= 0.01 / np.sqrt(2)
    elif action == 6: # Down-Left
      state[0, IDX.BALL_X] -= 0.01 / np.sqrt(2)
      state[0, IDX.BALL_Y] += 0.01 / np.sqrt(2)
    elif action == 7: # Down-Right
      state[0, IDX.BALL_X] += 0.01 / np.sqrt(2)
      state[0, IDX.BALL_Y] += 0.01 / np.sqrt(2)

    # Constrain ball vertically
    if state[0, IDX.BALL_Y] < BALL_RADIUS:
      state[0, IDX.BALL_Y] = BALL_RADIUS
    elif state[0, IDX.BALL_Y] > 1 - BALL_RADIUS:
      state[0, IDX.BALL_Y] = 1 - BALL_RADIUS

    # Constrain ball horizontally
    if state[0, IDX.BALL_X] < BALL_RADIUS:
      state[0, IDX.BALL_X] = BALL_RADIUS
    elif state[0, IDX.BALL_X] > 1 - BALL_RADIUS:
      state[0, IDX.BALL_X] = 1 - BALL_RADIUS

    ball_x = state[0, IDX.BALL_X].item()
    ball_y = state[0, IDX.BALL_Y].item()
    target_x = state[0, IDX.TARGET_X].item()
    target_y = state[0, IDX.TARGET_Y].item()

    # Time
    state[0, IDX.TIME] += 0.01
    if state[0, IDX.TIME] > 1:
      self.is_done = True

    # Reward for getting close to the target
    dist = np.sqrt((ball_x - target_x)**2 + (ball_y - target_y)**2)
    reward += (2 - dist ** 2) * 0.1

    # Reward for hitting the target
    if dist < 0.05:
      reward += 1

    reward = np.clip(reward, -1, 1)
    self.reward = reward

    return state.clone(), reward

def render(state: torch.Tensor, screen: pygame.Surface, screen_width, screen_height):
  TARGET_COLOR = (107, 115, 117)
  BALL_COLOR = (250, 255, 255)

  screen.fill((12, 23, 31))

  # Target
  target_x = state[0, IDX.TARGET_X].item() * screen_width
  target_y = state[0, IDX.TARGET_Y].item() * screen_height
  target_r = BALL_RADIUS * screen_height
  pygame.draw.circle(screen, TARGET_COLOR, (target_x, target_y), target_r)

  # Ball
  ball_x = state[0, IDX.BALL_X].item() * screen_width
  ball_y = state[0, IDX.BALL_Y].item() * screen_height
  ball_r = BALL_RADIUS * screen_height
  pygame.draw.circle(screen, BALL_COLOR, (ball_x, ball_y), ball_r)

def show(state: torch.Tensor):
  screen_width, screen_height = 600, 700
  pygame.init()
  screen = pygame.display.set_mode((screen_width, screen_height))
  clock = pygame.time.Clock()
  running = True

  while running:
    clock.tick(60)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    render(state, screen, screen_width, screen_height)
    pygame.display.flip()

  pygame.quit()

def live(game: Game, get_action: Callable):
  screen_width, screen_height = 600, 700
  pygame.init()
  screen = pygame.display.set_mode((screen_width, screen_height))
  clock = pygame.time.Clock()
  running = True
  total_reward = 0

  while running:
    clock.tick(60)

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    # Get action
    action = get_action()

    # Update state
    if not game.terminal():
      _, reward = game.apply_action(action)
      total_reward += reward
    else:
      game.reset()
      total_reward = 0

    render(game.get_current_state(), screen, screen_width, screen_height)
    
    # Total reward text
    font = pygame.font.Font(None, 36)
    text = font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

  pygame.quit()

def play():
  game = Game(Env=Chasing)

  def get_action():
    dx = 0
    dy = 0
    if pygame.key.get_pressed()[pygame.K_UP]:
      dy -= 1
    if pygame.key.get_pressed()[pygame.K_DOWN]:
      dy += 1 
    if pygame.key.get_pressed()[pygame.K_LEFT]:
      dx -= 1
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
      dx += 1
    action_map = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0)]
    return action_map.index((dx, dy))

  live(game, get_action)

def play_test_game():
  # network = UniformNetwork()
  network = Network.load(NETWORK_PATH)
  # network = Network()
  mcts = MCTS(network)
  game = Game(Env=Chasing)

  def get_action():
    with torch.no_grad():
      root = get_root_node(mcts, game)
      mcts.search(root)
      action = mcts.select_action(root)
      return action

  live(game, get_action)

def train_chasing():
  replay_buffer = ReplayBuffer(1000, BATCH_SIZE)
  muzero(replay_buffer, Chasing)

if __name__ == "__main__":
  freeze_support()

  try:
    set_start_method('spawn', force=True)
  except RuntimeError:
    pass

  if '--test' in sys.argv:
    play_test_game()
  elif '--play' in sys.argv:
    play()
  else:
    train_chasing()
