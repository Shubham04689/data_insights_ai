import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH = 800
HEIGHT = 600
SPEED = 10

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up the snake and food
snake = [(200, 200), (220, 200), (240, 200)]
food = (400, 300)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Move the snake
    head = snake[-1]
    new_head = (head[0] + SPEED, head[1] + SPEED)
    snake.append(new_head)

    # Check for collisions with food
    if new_head == food:
        print("You ate the food!")
        food = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
    else:
        snake.pop(0)

    # Draw everything
    screen.fill(BLACK)
    for pos in snake:
        pygame.draw.rect(screen, WHITE, pygame.Rect(pos[0], pos[1], 20, 20))
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(food[0], food[1], 20, 20))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)
