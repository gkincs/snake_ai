import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import pygame
import time

from snake_game import SnakeGame

class Net(nn.Module):
    def __init__(self, input_size, hidden=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4)  # 4 lehetséges akció
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_size, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, learning_rate=0.001):
        self.model = Net(state_size)
        self.target_model = Net(state_size)
        self.target_model.load_state_dict(self.model.state_dict())  # Célhálózat inicializálása

        self.memory = deque(maxlen=2000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # TensorBoard naplózó
        log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_step = 0

        self.prev_action = None  # Az előző akció tárolása

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Véletlenszerű akció
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()  # Legjobb akció a hálózattól

    def remember(self, s, a, r, s2, done):
        # Ellenőrizzük, hogy a visszalépés történt-e
        if self.prev_action is not None and self.is_backwards_move(a):
            r -= 5  # Kisebb büntetés

        self.memory.append((s, a, r, s2, done))
        self.prev_action = a  # Frissítjük az előző akciót

    def is_backwards_move(self, action):
        # Visszalépés ellenőrzése
        if self.prev_action == action:
            return False  # Ugyanabba az irányba lép, nem visszalépés
        if (self.prev_action == 0 and action == 1) or (self.prev_action == 1 and action == 0):
            return True  # Fel és le irány ellentétes mozgás
        if (self.prev_action == 2 and action == 3) or (self.prev_action == 3 and action == 2):
            return True  # Balra és jobbra irány ellentétes mozgás
        return False

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.BoolTensor(dones)

        # Q-értékek kiszámítása
        q_vals = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            q_next = self.target_model(next_states).max(1)[0]  # Célhálózat alkalmazása
        targets = rewards + self.gamma * q_next * (~dones)

        loss = self.loss_fn(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TensorBoard-ba loss naplózása
        self.writer.add_scalar("Loss", loss.item(), self.train_step)

        # Az epsilon és score naplózása
        self.writer.add_scalar("Epsilon", self.epsilon, self.train_step)
        self.train_step += 1

        # Epsilon csökkentése
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # A célhálózat frissítése gyakrabban
        self.target_model.load_state_dict(self.model.state_dict())

    def close(self):
        # TensorBoard writer bezárása
        self.writer.close()

# Játék környezet inicializálása
env = SnakeGame(render=True)
state_size = env._get_state().shape[0]
agent = DQNAgent(state_size)

# Képzés paraméterek
episodes = 1000
scores = []
target_update_interval = 10  # Célhálózat frissítési gyakoriság

# Képzés ciklus
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Pygame események kezelése, hogy a játék ne fagyjon le
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                agent.close()
                exit()

        # Akció választása és játék lépés
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        # Memória frissítése és tanulás
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        # Állapot frissítése
        state = next_state
        total_reward += reward

        # Képernyő frissítése a rendereléshez (csak ha szükséges)
        if ep % 10 == 0:  # Csak minden 10. epizódban
            env.render()

    # Az aktuális eredmény kiírása
    print(f"Tanulási kör {ep + 1}: Pontszám: {total_reward}, Felfedezési arány: {agent.epsilon:.2f}")
    scores.append(total_reward)

    # Eredmények naplózása TensorBoard-ba
    if ep % 10 == 0:  # Csak minden 10. epizódban loggolunk
        agent.writer.add_scalar('Score', total_reward, ep)
        agent.writer.add_scalar('Epsilon', agent.epsilon, ep)

    # Célhálózat frissítése minden target_update_interval epizód után
    if ep % target_update_interval == 0:
        agent.update_target_network()

# Naplózás bezárása
agent.close()
pygame.quit()
