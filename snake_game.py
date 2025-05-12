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
    def __init__(self, grid_size=GRID_SIZE, num_obstacles=5, render=True, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render
        self.score = 0
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration-exploitation trade-off

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
        self.last_distance = self._manhattan_distance(self.snake[0], self.goal)
        self.q_table = {}
        self.prev_pos = self.snake[0]  # ÚJ: előző pozíció mentése
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

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_state(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        grid[self.snake[0]] = 1
        grid[self.goal] = 2
        for obs in self.obstacles:
            grid[obs] = -1
        return grid.flatten()

    def _choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Explore: véletlenszerű akció
        else:
            # Exploit: a legjobb akció a Q-táblázat alapján
            if state not in self.q_table:
                self.q_table[state] = [0] * 4  # Ha az állapot nem létezik a táblázatban, inicializáljuk
            q_values = self.q_table[state]
            return np.argmax(q_values)

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
            return self._get_state(), -10, True  # Halál büntetés

        if new_head == self.goal:
            self.score += 10
            self.snake[0] = new_head
            self.goal = self._random_empty_cell(exclude=self.snake)
            self.obstacles = [
                self._random_empty_cell(exclude=self.snake + [self.goal])
                for _ in range(self.num_obstacles)
            ]
            self.last_distance = self._manhattan_distance(self.snake[0], self.goal)
            self.prev_pos = self.snake[0]  # ÚJ: frissítés cél elérése után is
            return self._get_state(), 10, False  # Cél elérés jutalom

        new_distance = self._manhattan_distance(new_head, self.goal)
        distance_delta = self.last_distance - new_distance

        # alapértelmezett büntetés minden lépésre
        reward = -0.2

        # ha közelebb kerül a célhoz
        if distance_delta > 0:
            reward += 0.6  # jutalom közeledésért

        # ha távolodik
        elif distance_delta < 0:
            reward -= 0.5  # szigorúbb büntetés

        # ha ugyanannyi a távolság mint előzőleg (pl. oda-vissza lépegetés)
        elif distance_delta == 0:
            reward -= 0.3  # extra büntetés stagnálásért

        # ÚJ: ha visszalép az előző mezőre, extra büntetés
        if new_head == self.prev_pos:
            reward -= 0.6

        self.prev_pos = self.snake[0]  # jelenlegi pozíció eltárolása
        self.snake[0] = new_head
        self.last_distance = new_distance
        return self._get_state(), reward, False

    def update_q_table(self, state, action, reward, next_state):
        # Q-táblázat frissítése
        next_max = np.max(self.q_table.get(next_state, [0] * 4))
        old_q = self.q_table.get(state, [0] * 4)[action]
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        if state not in self.q_table:
            self.q_table[state] = [0] * 4
        self.q_table[state][action] = new_q

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


# Teszt: futtatás Q-learning AI vezérlésével
if __name__ == "__main__":
    game = SnakeGame(render=True)
    state = game.reset()
    running = True
    episodes = 1000  # hány epizódot futtatunk

    for episode in range(episodes):
        state = game.reset()
        game.done = False
        total_reward = 0

        while not game.done:
            action = game._choose_action(state)  # AI döntése
            next_state, reward, done = game.step(action)
            game.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            game.render()

        print(f"Epizód: {episode + 1}, Pontszám: {game.score}, Teljes jutalom: {total_reward}")

    pygame.quit()
