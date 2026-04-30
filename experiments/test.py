from deep_q.replay_buffer import ReplayBuffer
import gymnasium as gym
import matplotlib.pyplot as plt
env = gym.make("CartPole-v1", render_mode="human") # render_mode = human -> opens a pygame window

replay_buffer = ReplayBuffer(100, env.observation_space.shape, env.action_space.shape)

for i in range(10):
    state, info = env.reset(seed=123, options={"low": -0.1, "high": 0.1})
    terminated = False

    while terminated == False:
        action = env.action_space.sample()
        next_state, reward, terminated, truncatated, info = env.step(action)

        replay_buffer.store(state, action, reward, next_state, terminated)
        state = next_state

print(replay_buffer.size)

batch_size = 15
sample = replay_buffer.sample(batch_size)
