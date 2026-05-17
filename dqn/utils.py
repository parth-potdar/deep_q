"""Helper functions"""
import numpy as np

def evaluate(agent, env, num_eps):
    """Evaluate an agent on an evaluation environment
    -> returning mean and std over a number of episodes"""

    returns = []

    for i in range(num_eps):
        total = 0
        done = False
        state, info = env.reset()

        while not done:
            action = agent.select_action(state, epsilon=0.0) # greedy policy on learned Q network

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state

            total += reward
        returns.append(total)

    return np.mean(returns), np.std(returns)