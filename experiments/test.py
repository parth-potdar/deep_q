from deep_q.agent import DQNAgent
import gymnasium as gym

import torch

env = gym.make("CartPole-v1", render_mode="human") # render_mode = human -> opens a pygame window

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
agent = DQNAgent(env, 1000, 100, device)

for i in range(1):
    state, info = env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    terminated = False

    while terminated == False:
        action = agent.select_action(state, epsilon=1.0)
        print(action)
        next_state, reward, terminated, truncatated, info = env.step(action)

        state = next_state