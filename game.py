import pygame
from config import *

# screen setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

# colors
BLACK = (0, 0, 0)
BRICK_COLOR = (200, 50, 50)
PADDLE_COLOR = (50, 200, 50)
BALL_COLOR = (50, 150, 255)

# - game settings -
# paddle
paddle = pygame.Rect((WIDTH - PADDLE_WIDTH) / 2, HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)

# ball
ball_x = WIDTH // 2
ball_y = HEIGHT // 2

# bricks
RAINBOW = [
    (255, 0, 0),      # red
    (255, 165, 0),    # orange
    (255, 255, 0),    # yellow
    (0, 255, 0),      # green
    (0, 150, 255),    # blue
    (128, 0, 255),    # purple
    (150, 200, 255)   # light blue
]
bricks = []
for row in range(BRICK_ROWS):
    for col in range(BRICK_COLUMNS):
        brick = pygame.Rect(col * BRICK_WIDTH, BRICK_TOP + row * BRICK_HEIGHT, BRICK_WIDTH - 2, BRICK_HEIGHT - 2)
        color = RAINBOW[row % len(RAINBOW)]
        bricks.append((brick, color))


# - main method / loop -
running = True
while running:
    clock.tick(60)  # 60 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # paddle controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        paddle.x -= PADDLE_SPEED
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        paddle.x += PADDLE_SPEED

    if paddle.left < 0:
        paddle.left = 0
    if paddle.right > WIDTH:
        paddle.right = WIDTH

    # ball movement
    ball_x += BALL_SPEED_X
    ball_y += BALL_SPEED_Y

    # wall collisions
    if ball_x - BALL_RADIUS <= 0 or ball_x + BALL_RADIUS >= WIDTH:
        BALL_SPEED_X = -BALL_SPEED_X
    if ball_y - BALL_RADIUS <= 0:
        BALL_SPEED_Y = -BALL_SPEED_Y

    # cheat mode
    if ball_y - BALL_RADIUS > HEIGHT:
        ball_x = WIDTH // 2
        ball_y = HEIGHT // 2
        BALL_SPEED_X = 5
        BALL_SPEED_Y = -5

        # pygame.quit() # lowkey just ends the game right now lmao

    # paddle collisions
    ball_rect = pygame.Rect(ball_x - BALL_RADIUS, ball_y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
    if ball_rect.colliderect(paddle) and BALL_SPEED_Y > 0:
        BALL_SPEED_Y = -BALL_SPEED_Y

    # brick collisions
    hit_index = ball_rect.collidelist([b[0] for b in bricks])
    if hit_index != -1:
        hit_brick = bricks.pop(hit_index)
        BALL_SPEED_Y = -BALL_SPEED_Y


    # building game
    screen.fill(BLACK)
    for brick, color in bricks:
        pygame.draw.rect(screen, color, brick)

    # draw paddle
    pygame.draw.rect(screen, PADDLE_COLOR, paddle)

    # draw ball
    pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), int(ball_y)), BALL_RADIUS)


    pygame.display.flip()


pygame.quit()
