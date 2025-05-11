from game import SnakeGame
from dqn_agent import DQNAgent
import pygame
import time

# Játék környezet inicializálása
env = SnakeGame(render=True)
state_size = env._get_state().shape[0]
agent = DQNAgent(state_size)

# Képzés paraméterek
episodes = 300
scores = []
epsilon = 1.0

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

        # Képernyő frissítése a rendereléshez
        env.render()

        time.sleep(0.01)

    # Epsilon csökkentése a felfedezés érdekében
    epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

    # Az aktuális eredmény kiírása
    print(f"Tanulási kör {ep + 1}: Pontszám: {total_reward}, Felfedezési arány: {epsilon:.2f}")
    scores.append(total_reward)

# Játék vége
pygame.quit()
