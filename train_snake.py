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
                agent.writer.close()
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

    # Az aktuális eredmény kiírása
    print(f"Tanulási kör {ep + 1}: Pontszám: {total_reward}, Felfedezési arány: {agent.epsilon:.2f}")
    scores.append(total_reward)

    # Eredmények naplózása TensorBoard-ba
    agent.writer.add_scalar('Score', total_reward, ep)
    agent.writer.add_scalar('Epsilon', agent.epsilon, ep)

# Naplózás bezárása
agent.writer.close()
pygame.quit()
