"""Evaluate the trained Q-network"""

import torch
from deep_q.agent import DQNAgent
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human") # render_mode = human -> opens a pygame window

agent = DQNAgent(env)

agent.q_network.load_state_dict(torch.load("q_network.pth", weights_only=True))
agent.q_network.eval()

state, info = env.reset()
done = False
while done == False:
    action = agent.select_action(state, epsilon=0.05)

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    state = next_state