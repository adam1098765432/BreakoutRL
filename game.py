import pygame

# screen setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()

# colors
BLACK = (0, 0, 0)
BRICK_COLOR = (200, 50, 50)
PADDLE_COLOR = (50, 200, 50)
BALL_COLOR = (50, 150, 255)

# - game settings -
# paddle
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 15
paddle = pygame.Rect((WIDTH - PADDLE_WIDTH) // 2, HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)

paddle_speed = 8

# ball
BALL_RADIUS = 8
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_dx = 5   # horizontal speed
ball_dy = -5  # vertical speed

# bricks
BRICK_ROWS = 7
BRICK_COLUMNS = 10
BRICK_WIDTH = WIDTH // BRICK_COLUMNS
BRICK_HEIGHT = 40
bricks = []
for row in range(BRICK_ROWS):
    for col in range(BRICK_COLUMNS):
        brick = pygame.Rect(col * BRICK_WIDTH, 50 + row * BRICK_HEIGHT, BRICK_WIDTH - 2, BRICK_HEIGHT - 2)
        bricks.append(brick)


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
        paddle.x -= paddle_speed
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        paddle.x += paddle_speed

    if paddle.left < 0:
        paddle.left = 0
    if paddle.right > WIDTH:
        paddle.right = WIDTH

    # ball movement
    ball_x += ball_dx
    ball_y += ball_dy

    # wall collisions
    if ball_x - BALL_RADIUS <= 0 or ball_x + BALL_RADIUS >= WIDTH:
        ball_dx = -ball_dx
    if ball_y - BALL_RADIUS <= 0:
        ball_dy = -ball_dy

    # cheat mode
    if ball_y - BALL_RADIUS > HEIGHT:
        # ball_x = WIDTH // 2
        # ball_y = HEIGHT // 2
        # ball_dx = 5
        # ball_dy = -5

        pygame.quit() # lowkey just ends the game right now lmao

    # paddle collisions
    ball_rect = pygame.Rect(ball_x - BALL_RADIUS, ball_y - BALL_RADIUS, BALL_RADIUS * 2, BALL_RADIUS * 2)
    if ball_rect.colliderect(paddle) and ball_dy > 0:
        ball_dy = -ball_dy

    # brick collisions
    hit_index = ball_rect.collidelist(bricks)
    if hit_index != -1:
        hit_brick = bricks.pop(hit_index)
        ball_dy = -ball_dy


    # building game
    screen.fill(BLACK)
    for brick in bricks:
        pygame.draw.rect(screen, BRICK_COLOR, brick)

    # draw paddle
    pygame.draw.rect(screen, PADDLE_COLOR, paddle)

    # draw ball
    pygame.draw.circle(screen, BALL_COLOR, (int(ball_x), int(ball_y)), BALL_RADIUS)


    pygame.display.flip()


pygame.quit()
