import pygame
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# Beállítások
GRID_SIZE = 10
CELL_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
NUM_OBSTACLES = 5

BG_COLOR = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
SNAKE_COLOR = (0, 200, 100)
FOOD_COLOR = (255, 80, 80)
OBSTACLE_COLOR = (100, 100, 100)

DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # fel, le, bal, jobb


class SnakeGame:
    def __init__(self, render=True):
        self.grid_size = GRID_SIZE
        self.num_obstacles = NUM_OBSTACLES
        self.render_mode = render
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
        self.obstacles = [self._random_empty_cell(exclude=self.snake + [self.goal]) for _ in range(self.num_obstacles)]
        self.score = 0
        self.done = False
        return self._get_state()

    def _random_empty_cell(self, exclude=None):
        if exclude is None:
            exclude = []
        while True:
            pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if pos not in exclude:
                return pos

    def _get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[self.snake[0]] = 1.0
        grid[self.goal] = 2.0
        for obs in self.obstacles:
            grid[obs] = -1.0
        return grid.flatten()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True

        dx, dy = DIRECTIONS[action]
        x, y = self.snake[0]
        new_head = (x + dx, y + dy)

        if (
            not 0 <= new_head[0] < self.grid_size or
            not 0 <= new_head[1] < self.grid_size or
            new_head in self.obstacles
        ):
            self.done = True
            return self._get_state(), -10, True

        if new_head == self.goal:
            self.score += 10
            self.snake[0] = new_head
            self.goal = self._random_empty_cell(exclude=self.snake)
            self.obstacles = [self._random_empty_cell(exclude=self.snake + [self.goal]) for _ in range(self.num_obstacles)]
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
        self.clock.tick(10)

    def _draw_cell(self, pos, color, radius=6):
        rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, color, rect, border_radius=radius)


def build_model(input_size, output_size):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_size,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_size, activation='linear'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# Q-learning beállítások
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.995
gamma = 0.99
episodes = 100

# Modell és játék inicializálás
input_size = GRID_SIZE * GRID_SIZE
output_size = 4
model = build_model(input_size, output_size)
game = SnakeGame(render=True)

# Tanítás
for ep in range(episodes):
    state = game.reset()
    state = np.array(state).reshape(1, -1)
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            q_vals = model.predict(state, verbose=0)
            action = np.argmax(q_vals[0])

        next_state, reward, done = game.step(action)
        next_state = np.array(next_state).reshape(1, -1)
        total_reward += reward

        target = reward
        if not done:
            target += gamma * np.max(model.predict(next_state, verbose=0))

        target_q = model.predict(state, verbose=0)
        target_q[0][action] = target

        model.fit(state, target_q, verbose=0)
        state = next_state

        game.render()

    print(f"Tanulási kör {ep+1}: Pontszám: {total_reward}, Felfedezési arány: {epsilon:.2f}")
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

pygame.quit()
