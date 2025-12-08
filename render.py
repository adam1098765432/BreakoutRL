import pygame
from config import *
from state import State


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
PADDLE_COLOR = (50, 200, 50)
BALL_COLOR = (50, 150, 255)
RAINBOW = [
    (255, 0, 0),      # red
    (255, 165, 0),    # orange
    (255, 255, 0),    # yellow
    (0, 255, 0),      # green
    (0, 150, 255),    # blue
    (128, 0, 255),    # purple
    (150, 200, 255)   # light blue
]


# render a single frame for the given State
def render_state(state: State, fps: int = 60):

    clock.tick(fps)
    screen.fill(BLACK)

    # bricks
    idx = 0
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLUMNS):
            if state.bricks[idx] == 1:
                rect = pygame.Rect(col * BRICK_WIDTH, BRICK_TOP + row * BRICK_HEIGHT, BRICK_WIDTH - 2, BRICK_HEIGHT - 2)
                pygame.draw.rect(screen, RAINBOW[row], rect)
            idx += 1

    # paddle
    paddle = pygame.Rect(state.paddle_x, state.paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, PADDLE_COLOR, paddle)

    # ball
    pygame.draw.circle(screen, BALL_COLOR,(int(state.ball_x), int(state.ball_y)), BALL_RADIUS)

    pygame.display.flip()
    return True


# for testing
if __name__ == "__main__":
    # simple dummy state
    s = State(N_BRICKS)
    s.paddle_x = (WIDTH - PADDLE_WIDTH) / 2
    s.paddle_y = PADDLE_Y
    s.ball_x = WIDTH // 2
    s.ball_y = HEIGHT // 2
    s.ball_vx = BALL_SPEED_X
    s.ball_vy = BALL_SPEED_Y

    running = True
    while running:
        running = render_state(s)
