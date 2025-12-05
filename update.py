from state import State
from config import *

def update(state:State, action: int, delta_time: float = 1.0):

    s = state

    reward = 0.0
    done = False

    # paddle movement
    if action == 0: # move left
        s.paddle_x -= PADDLE_SPEED * delta_time
    elif action == 1: # move right
        s.paddle_x += PADDLE_SPEED * delta_time
    # action == 2 - don't move

    if s.paddle_x < 0:
        s.paddle_x = 0
    if s.paddle_x + PADDLE_WIDTH > WIDTH:
        s.paddle_x = WIDTH - PADDLE_WIDTH

    s.paddle_y = PADDLE_Y

    # ball movement
    s.ball_x += s.ball_vx * delta_time
    s.ball_y += s.ball_vy * delta_time

    # Wall Collision
    if s.ball_x - BALL_RADIUS <= 0:
        s.ball_x = BALL_RADIUS
        s.ball_vx =  -s.ball_vx
    elif s.ball_x + BALL_RADIUS >= WIDTH:
        s.ball_x = WIDTH - BALL_RADIUS
        s.ball_vx = -s.ball_vx


    if s.ball_y - BALL_RADIUS <= 0:
        s.ball_y = BALL_RADIUS
        s.ball_vy = -s.ball_vy

    # Bottom = miss = done
    if s.ball_y - BALL_RADIUS > HEIGHT:
        done = True
        return s, reward, done

    # paddle collisions
    if s.ball_vy > 0:
        ball_left = s.ball_x - BALL_RADIUS
        ball_right = s.ball_x + BALL_RADIUS
        ball_top = s.ball_y - BALL_RADIUS
        ball_bottom = s.ball_y + BALL_RADIUS

        paddle_left = s.paddle_x
        paddle_right = s.paddle_x + PADDLE_WIDTH
        paddle_top = s.paddle_y
        paddle_bottom = s.paddle_y + PADDLE_HEIGHT

        if (ball_right >= paddle_left and
            ball_left <= paddle_right and
            ball_bottom >= paddle_top and
            ball_top <= paddle_bottom):

            s.ball_y = s.paddle_y - BALL_RADIUS

            s.ball_vy = -s.ball_vy

    # Brick Collisions
    bricks_destroyed = 0

    for r in range(BRICK_ROWS):
        for c in range(BRICK_COLUMNS):
            idx = r * BRICK_COLUMNS + c

            if s.bricks[idx] == 0:
                continue

            x0 = c * BRICK_WIDTH
            y0 = BRICK_TOP + r * BRICK_HEIGHT
            x1 = x0 + BRICK_WIDTH
            y1 = y0 + BRICK_HEIGHT

            if (x0 <= s.ball_x <= x1 and
                    y0 <= s.ball_y <= y1):

                # destroy brick
                s.bricks[idx] = 0
                bricks_destroyed += 1

                # vertical bounce
                s.ball_vy = -s.ball_vy

                break
        else:
            continue
        break

    # reward for breaking bricks
    if bricks_destroyed > 0:
         reward +=  1.0

    return s, reward, done