import pygame
import random
import numpy as np

GRID_SIZE = 10
CELL_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE

BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
SNAKE_COLOR = (0, 200, 100)
FOOD_COLOR = (255, 80, 80)
OBSTACLE_COLOR = (100, 100, 100)

DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # fel, le, balra, jobbra

class SnakeGame:
    def __init__(self, grid_size=GRID_SIZE, num_obstacles=5, render=True):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render
        self.score = 0

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 18)

        self.reset()

    def reset(self):
        self.snake = [self._random_empty_cell()]
        self.goal = self._random_empty_cell(exclude=self.snake)
        self.obstacles = [
            self._random_empty_cell(exclude=self.snake + [self.goal])
            for _ in range(self.num_obstacles)
        ]
        self.done = False
        self.score = 0
        return self._get_state()

    def _random_empty_cell(self, exclude=None, max_attempts=1000):
        if exclude is None:
            exclude = []
        attempts = 0
        while attempts < max_attempts:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in exclude:
                return pos
            attempts += 1
        raise RuntimeError("Nem található üres cella")

    def _get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[self.snake[0]] = 1
        grid[self.goal] = 2
        for obs in self.obstacles:
            grid[obs] = -1
        return grid.flatten()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True

        dx, dy = DIRECTIONS[action]
        x, y = self.snake[0]
        new_head = (x + dx, y + dy)

        if (
            not 0 <= new_head[0] < self.grid_size
            or not 0 <= new_head[1] < self.grid_size
            or new_head in self.obstacles
        ):
            self.done = True
            return self._get_state(), -10, True

        if new_head == self.goal:
            self.score += 10
            self.snake[0] = new_head
            self.goal = self._random_empty_cell(exclude=self.snake)
            self.obstacles = [
                self._random_empty_cell(exclude=self.snake + [self.goal])
                for _ in range(self.num_obstacles)
            ]
            return self._get_state(), 10, False

        self.snake[0] = new_head
        return self._get_state(), 0, False

    def render(self):
        if not self.render_mode:
            return

        self.screen.fill(BG_COLOR)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        for obs in self.obstacles:
            self._draw_cell(obs, OBSTACLE_COLOR)
        self._draw_cell(self.goal, FOOD_COLOR, radius=10)
        self._draw_cell(self.snake[0], SNAKE_COLOR)

        score_text = self.font.render(f"Pontszám: {self.score}", True, (200, 200, 200))
        self.screen.blit(score_text, (10, 5))

        pygame.display.flip()
        self.clock.tick(8)

    def _draw_cell(self, pos, color, radius=6):
        rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)

# Teszt: futtatás emberi vagy AI vezérlés nélkül, véletlenszerű lépésekkel
if __name__ == "__main__":
    game = SnakeGame(render=True)
    state = game.reset()
    running = True

    while running and not game.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = random.randint(0, 3)  # véletlen irány választás
        state, reward, done = game.step(action)
        game.render()

    pygame.quit()
