from snake_game import SnakeGame
from dqn_snake_agent import DQNAgent
import pygame

# Játék környezet inicializálása
env = SnakeGame(render=True)
state_size = env._get_state().shape[0]
agent = DQNAgent(state_size)

# Képzés paraméterek
episodes = 800
scores = []

# Célhálózat frissítése minden target_update_interval után
target_update_interval = agent.target_update_interval

# Képzés ciklus
for ep in range(episodes):
    state = env.reset()  # Reset minden epizódban új pálya
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

        # Ha cél elérése vagy akadály ütközés történt, azonnali pályaváltás
        if done:
            # Az aktuális eredmény kiírása
            print(f"Tanulási kör {ep + 1}: Pontszám: {total_reward}, Felfedezési arány: {agent.epsilon:.2f}")
            scores.append(total_reward)

            # Reset új pályára
            state = env.reset()
            total_reward = 0
            done = False
            continue  # Azonnali visszatérés a következő epizódra

        # Memória frissítése és tanulás
        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        # Állapot frissítése
        state = next_state
        total_reward += reward

        # Képernyő frissítése azonnal
        env.render()

    # Eredmények naplózása TensorBoard-ba
    if ep % 10 == 0:  # Csak minden 10. epizódban loggolunk
        agent.writer.add_scalar('Score', total_reward, ep)
        agent.writer.add_scalar('Epsilon', agent.epsilon, ep)

    # Célhálózat frissítése minden target_update_interval epizód után
    if ep % target_update_interval == 0:
        agent.update_target_network()

# Naplózás bezárása
agent.writer.close()
pygame.quit()
