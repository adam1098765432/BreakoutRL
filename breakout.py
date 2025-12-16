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
BRICK_ROWS = 7
BRICK_COLUMNS = 10
STATE_SIZE = 6 + BRICK_ROWS * BRICK_COLUMNS
PADDLE_SPEED = 0.01
BALL_SPEED = 0.01
HALF_PADDLE_WIDTH = 0.075
HALF_PADDLE_HEIGHT = 0.01
PADDLE_Y = 0.95
BALL_RADIUS = 0.015
BRICK_TOP = 0.1
BRICK_BOTTOM = 0.3

class IDX:
  PADDLE_X = 0
  BALL_X = 1
  BALL_Y = 2
  BALL_VX = 3
  BALL_VY = 4
  BRICK_BEGIN = 5
  BRICK_END = 5 + BRICK_ROWS * BRICK_COLUMNS

class Breakout(Environment):
  def __init__(self, device):
    super().__init__(device)

    self.frame_skip = 4

    # Initialize state
    self.state[0, IDX.PADDLE_X] = 0.5
    self.state[0, IDX.BALL_X] = random.random() * 0.8 + 0.1
    self.state[0, IDX.BALL_Y] = 0.5
    self.state[0, IDX.BALL_VX] = BALL_SPEED * random.choice([-1, 1])
    self.state[0, IDX.BALL_VY] = BALL_SPEED
    self.state[0, IDX.BRICK_BEGIN:IDX.BRICK_END] = 1

    self.is_done = False

  def terminal(self):
    return self.is_done

  def pos_to_brick_idx(self, x, y):
    """Map a position in the world to the index of the corresponding brick."""
    if y < BRICK_TOP or y > BRICK_BOTTOM or x < 0 or x > 1:
      return None
    
    row = np.floor((y - BRICK_TOP) / (BRICK_BOTTOM - BRICK_TOP) * BRICK_ROWS)
    col = np.floor(x * BRICK_COLUMNS)

    return int(row * BRICK_COLUMNS + col)

  def _step(self, action: int):
    state = self.state
    reward = 0.001

    # Move paddle
    if action == 0:
      state[0, IDX.PADDLE_X] -= PADDLE_SPEED
    elif action == 1:
      state[0, IDX.PADDLE_X] += PADDLE_SPEED

    # Constrain paddle
    if state[0, IDX.PADDLE_X] < HALF_PADDLE_WIDTH:
      state[0, IDX.PADDLE_X] = HALF_PADDLE_WIDTH
    if state[0, IDX.PADDLE_X] > 1 - HALF_PADDLE_WIDTH:
      state[0, IDX.PADDLE_X] = 1 - HALF_PADDLE_WIDTH

    # Move ball
    state[0, IDX.BALL_X] += state[0, IDX.BALL_VX]
    state[0, IDX.BALL_Y] += state[0, IDX.BALL_VY]

    # Constrain ball vertically
    if state[0, IDX.BALL_Y] < BALL_RADIUS:
      state[0, IDX.BALL_Y] = BALL_RADIUS
      state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]
    elif state[0, IDX.BALL_Y] > 1 - BALL_RADIUS:
      state[0, IDX.BALL_Y] = 1 - BALL_RADIUS
      state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]
      ball_to_paddle = abs(state[0, IDX.BALL_X].item() - state[0, IDX.PADDLE_X].item())
      reward -= 1 * ball_to_paddle
      self.is_done = True

    # Constrain ball horizontally
    if state[0, IDX.BALL_X] < BALL_RADIUS:
      state[0, IDX.BALL_X] = BALL_RADIUS
      state[0, IDX.BALL_VX] = -state[0, IDX.BALL_VX]
    elif state[0, IDX.BALL_X] > 1 - BALL_RADIUS:
      state[0, IDX.BALL_X] = 1 - BALL_RADIUS
      state[0, IDX.BALL_VX] = -state[0, IDX.BALL_VX]

    ball_top = state[0, IDX.BALL_Y].item() - BALL_RADIUS
    ball_bottom = state[0, IDX.BALL_Y].item() + BALL_RADIUS
    ball_left = state[0, IDX.BALL_X].item() - BALL_RADIUS
    ball_right = state[0, IDX.BALL_X].item() + BALL_RADIUS

    # Ball colliding with paddle
    if state[0, IDX.BALL_VY] > 0:
      paddle_left = state[0, IDX.PADDLE_X].item() - HALF_PADDLE_WIDTH
      paddle_right = state[0, IDX.PADDLE_X].item() + HALF_PADDLE_WIDTH
      paddle_top = PADDLE_Y - HALF_PADDLE_HEIGHT
      paddle_bottom = PADDLE_Y + HALF_PADDLE_HEIGHT

      if (ball_right >= paddle_left and
        ball_left <= paddle_right and
        ball_bottom >= paddle_top and
        ball_top <= paddle_bottom):
        state[0, IDX.BALL_Y] = PADDLE_Y - BALL_RADIUS
        state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]
        reward += 1

    # Ball colliding with bricks
    brick_tl = self.pos_to_brick_idx(ball_left, ball_top)
    brick_tr = self.pos_to_brick_idx(ball_right, ball_top)
    brick_bl = self.pos_to_brick_idx(ball_left, ball_bottom)
    brick_br = self.pos_to_brick_idx(ball_right, ball_bottom)

    if brick_tl is not None and state[0, IDX.BRICK_BEGIN + brick_tl] == 1:
      reward += 1.0
      state[0, IDX.BRICK_BEGIN + brick_tl] = 0
      state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]

    if brick_tr is not None and state[0, IDX.BRICK_BEGIN + brick_tr] == 1:
      reward += 1.0
      state[0, IDX.BRICK_BEGIN + brick_tr] = 0
      state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]

    if brick_bl is not None and state[0, IDX.BRICK_BEGIN + brick_bl] == 1:
      reward += 1.0
      state[0, IDX.BRICK_BEGIN + brick_bl] = 0
      state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]

    if brick_br is not None and state[0, IDX.BRICK_BEGIN + brick_br] == 1:
      reward += 1.0
      state[0, IDX.BRICK_BEGIN + brick_br] = 0
      state[0, IDX.BALL_VY] = -state[0, IDX.BALL_VY]

    # reward = 0

    # if action == 0:
    #   reward = 1
    #   self.is_done = True
    # else:
    #   reward = 0
    #   self.is_done = True

    # if state[0, IDX.BALL_X].item() < state[0, IDX.PADDLE_X] and action != 0:
    #   reward -= 0.1

    # if state[0, IDX.BALL_X].item() > state[0, IDX.PADDLE_X] and action != 1:
    #   reward -= 0.1

    reward = np.clip(reward, -1, 1)

    return state, reward

def render(state: torch.Tensor, screen: pygame.Surface, screen_width, screen_height):
  brick_width = screen_width // BRICK_COLUMNS
  brick_height = screen_height * (BRICK_BOTTOM - BRICK_TOP) // BRICK_ROWS

  PADDLE_COLOR = (107, 115, 117)
  BALL_COLOR = (250, 255, 255)
  RAINBOW = [
    (255, 85, 85),      # red
    (255, 120, 70),    # orange
    (255, 190, 20),    # yellow
    (90, 230, 90),      # green
    (70, 80, 255),    # blue
    (130, 80, 255),    # purple
    (0, 240, 240)   #  light blue
  ]

  screen.fill((12, 23, 31))

  # Bricks
  for i in range(0, IDX.BRICK_END - IDX.BRICK_BEGIN):
    if state[0, i + IDX.BRICK_BEGIN].item() == 0: continue
    col = i % BRICK_COLUMNS
    row = i // BRICK_COLUMNS
    left = col * brick_width + 1
    top = BRICK_TOP * screen_height + row * brick_height + 1
    width = brick_width - 2
    height = brick_height - 2
    rect = pygame.Rect(left, top, width, height)
    pygame.draw.rect(screen, RAINBOW[row], rect)

  # Paddle
  paddle_width = HALF_PADDLE_WIDTH * screen_width * 2
  paddle_height = HALF_PADDLE_HEIGHT * screen_height * 2
  paddle_left = state[0, IDX.PADDLE_X].item() * screen_width - paddle_width / 2
  paddle_top = PADDLE_Y * screen_height - paddle_height / 2
  paddle = pygame.Rect(paddle_left, paddle_top, paddle_width, paddle_height)
  pygame.draw.rect(screen, PADDLE_COLOR, paddle)

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
    text = font.render(f"Total reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

  pygame.quit()

def play(game: Game):
  game.environment.frame_skip = 1
  
  def get_action():
    action = 2
    if pygame.key.get_pressed()[pygame.K_LEFT]:
      action = 0
    if pygame.key.get_pressed()[pygame.K_RIGHT]:
      if action == 0:
        action = 2
      else:
        action = 1
    return action

  live(game, get_action)

def play_test_game():
  # network = UniformNetwork()
  network = Network.load(NETWORK_PATH)
  # network = Network()
  mcts = MCTS(network)
  game = Game(Env=Breakout)
  game.environment.frame_skip = 1

  def get_action():
    with torch.no_grad():
      root = get_root_node(mcts, game)
      mcts.search(root)
      action = mcts.select_action(root)
      return action

  live(game, get_action)

def train_breakout():
  # replay_buffer = ReplayBuffer(10000, BATCH_SIZE)
  replay_buffer = ReplayBuffer.load(REPLAY_BUFFER_PATH, Breakout)
  muzero(replay_buffer, Breakout)

if __name__ == "__main__":
  freeze_support()

  try:
    set_start_method('spawn', force=True)
  except RuntimeError:
    pass

  if '--test' in sys.argv:
    play_test_game()
  elif '--play' in sys.argv:
    game = Game(Env=Breakout)
    play(game)
  else:
    train_breakout()
