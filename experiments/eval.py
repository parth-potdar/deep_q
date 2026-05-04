"""Evaluate the trained Q-network for demonstration purposes"""

import torch
import numpy as np
from deep_q.agent import DQNAgent
from deep_q.utils import evaluate
import gymnasium as gym

# make new env for evaluation
env = gym.make("CartPole-v1", render_mode="human")
agent = DQNAgent(env)

# load learned network
agent.q_network.load_state_dict(torch.load("q_network_best.pth", weights_only=True))
agent.q_network.eval()

state, info = env.reset()

mean, std = evaluate(agent, env, 1)
