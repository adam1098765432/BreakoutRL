import numpy as np
from action import Action
from state import State
from config import *

def update(state: State, action: int, delta_time: float = 1.0):
  reward = 0.0
  done = False

  # Paddle movement
  state.paddle_y = PADDLE_Y

  if action == Action.MOVE_LEFT:
    state.paddle_x -= PADDLE_SPEED * delta_time
  elif action == Action.MOVE_RIGHT:
    state.paddle_x += PADDLE_SPEED * delta_time
  
  # Constrain paddle movement
  if state.paddle_x < 0:
    state.paddle_x = 0
  if state.paddle_x + PADDLE_WIDTH > WIDTH:
    state.paddle_x = WIDTH - PADDLE_WIDTH

  # Ball movement
  state.ball_x += state.ball_vx * delta_time
  state.ball_y += state.ball_vy * delta_time

  # Ball wall collisions
  if state.ball_x < BALL_RADIUS:
    state.ball_x = BALL_RADIUS
    state.ball_vx =  -state.ball_vx
  elif state.ball_x > WIDTH - BALL_RADIUS:
    state.ball_x = WIDTH - BALL_RADIUS
    state.ball_vx = -state.ball_vx
  if state.ball_y < BALL_RADIUS:
    state.ball_y = BALL_RADIUS
    state.ball_vy = -state.ball_vy
  if state.ball_y > HEIGHT - BALL_RADIUS:
    done = True
    return state, reward, done

  # Ball paddle collisions
  if state.ball_vy > 0:
    ball_left = state.ball_x - BALL_RADIUS
    ball_right = state.ball_x + BALL_RADIUS
    ball_top = state.ball_y - BALL_RADIUS
    ball_bottom = state.ball_y + BALL_RADIUS

    paddle_left = state.paddle_x
    paddle_right = state.paddle_x + PADDLE_WIDTH
    paddle_top = state.paddle_y
    paddle_bottom = state.paddle_y + PADDLE_HEIGHT

    if (ball_right >= paddle_left and
      ball_left <= paddle_right and
      ball_bottom >= paddle_top and
      ball_top <= paddle_bottom):
      state.ball_y = state.paddle_y - BALL_RADIUS
      state.ball_vy = -state.ball_vy

  # Ball brick Collisions
  brick_x = int(np.floor(state.ball_x / BRICK_WIDTH))
  brick_y = int(np.floor(state.ball_y / BRICK_HEIGHT))
  brick_i = brick_y * BRICK_COLUMNS + brick_x
  
  if brick_i > -1 and brick_i < len(state.bricks) and state.bricks[brick_i] == 1:
    state.bricks[brick_i] = 0
    state.ball_vy = -state.ball_vy
    reward += 1

  return state, reward, done
