import deep_q
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1", render_mode="human") # render_mode = human -> opens a pygame window

env.reset(seed=123, options={"low": -0.1, "high": 0.1})

print(env.action_space)
print(env.observation_space)
terminated = False
while True:
    action = env.action_space.sample()  
    next, reward, terminated, truncated, info = env.step(action)
