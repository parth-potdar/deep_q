from dqn.agent import DQNAgent
from dqn.utils import evaluate
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT="clipped_error_25000_capacity"
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
agent = DQNAgent(env, capacity=25000, learning_rate=1e-4, device=device)

num_episodes = 1000
rewards = []
eval_means = []
eval_stds = []

epsilon = 1.0
decay = 0.997
state, info = env.reset()

# track best evaluation every 50 episodes
eval_freq = 50
best_eval_mean = -np.inf

for i in range(num_episodes):
    print(f"Episode: {i+1}/{num_episodes}")
    state, info = env.reset()
    done = False
    episode_reward = 0 # initialise episode reward, which will increment during episode
    while done == False:
        action = agent.select_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        # store experience
        agent.buffer.store(state, action, reward, next_state, done)
        
        # update state every time
        state = next_state

        # update Q network after warmup period of 1000 experiences
        if agent.buffer.size >= 1000:
            agent.update(target_freq=1000, batch_size=64)
    
    # anneal epsilon every episode
    epsilon = max(0.01, epsilon * decay)

    # save episode reward for plotting
    rewards.append(episode_reward)

    if i % eval_freq == 0:
        mean, std = evaluate(agent, eval_env, 20) # evaluate 20 episodes with greedy policy
        print(f"Episode {i} | Eval mean: {mean:.1f} | Std: {std:.1f}")

        eval_means.append(mean)
        eval_stds.append(std)
        
        if mean > best_eval_mean:
            best_eval_mean = mean
            torch.save(agent.q_network.state_dict(), f"models/{EXPERIMENT}.pth")

# plot rewards and evaluation means and stds
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward', color=color)
ax1.plot(rewards, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Eval Mean', color=color)  # we already handled the x-label
ax2.errorbar(np.arange(0, num_episodes, eval_freq), eval_means
, yerr=eval_stds, fmt='o', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
# save figure
fig.savefig(f"results/{EXPERIMENT}.png")