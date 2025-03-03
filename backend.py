import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Snake Game")

# Define colors
white = (255, 255, 255)
black = (0, 0, 0)

# Set up the snake and food
snake_block = 10
snake_speed = 15

x1 = screen_width / 2
y1 = screen_height / 2

foodx = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
foody = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0

snake_List = []
Length_of_snake = 1

clock = pygame.time.Clock()

def our_snake(snake_block, snake_List):
    for x in snake_List:
        pygame.draw.rect(screen, black, [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    mesg = font.render(msg, True, color)
    screen.blit(mesg, [screen_width / 6, screen_height / 3])

font_style = pygame.font.SysFont(None, 50)

game_over = False
game_close = False

while not game_over:

    while game_close == True:
        screen.fill(white)
        message("You Lost! Press Q-Quit or C-Play Again", black)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    game_over = True
                    game_close = False
                if event.key == pygame.K_c:
                    gameLoop()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x1_change = -snake_block
                y1_change = 0
            elif event.key == pygame.K_RIGHT:
                x1_change = snake_block
                y1_change = 0
            elif event.key == pygame.K_UP:
                y1_change = -snake_block
                x1_change = 0
            elif event.key == pygame.K_DOWN:
                y1_change = snake_block
                x1_change = 0

    if x1 >= screen_width or x1 < 0 or y1 >= screen_height or y1 < 0:
        game_close = True
    x1 += x1_change
    y1 += y1_change
    screen.fill(white)
    pygame.draw.rect(screen, green, [foodx, foody, snake_block, snake_block])
    snake_Head = []
    snake_Head.append(x1)
    snake_Head.append(y1)
    snake_List.append(snake_Head)
    if len(snake_List) > Length_of_snake:
        del snake_List[0]

    for x in snake_List[:-1]:
        if x == snake_Head:
            game_close = True

    our_snake(snake_block, snake_List)
    pygame.display.update()

    if x1 == foodx and y1 == foody:
        foodx = round(random.randrange(0, screen_width - snake_block) / 10.0) * 10.0
        foody = round(random.randrange(0, screen_height - snake_block) / 10.0) * 10.0
        Length_of_snake += 1

    clock.tick(snake_speed)

pygame.quit()
sys.exit()
