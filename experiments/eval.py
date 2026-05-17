"""Evaluate the trained Q-network for demonstration purposes"""

import torch
import numpy as np
from dqn.agent import DQNAgent
from dqn.utils import evaluate
import gymnasium as gym

# make new env for evaluation
env = gym.make("CartPole-v1") #render_mode="human")
agent = DQNAgent(env)

# load learned network
agent.q_network.load_state_dict(torch.load("models/clipped_error_0997_decay.pth", weights_only=True))
agent.q_network.eval()

state, info = env.reset()

mean, std = evaluate(agent, env, 100)
print(mean, std)