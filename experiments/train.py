from deep_q.agent import DQNAgent
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1") # render_mode = human -> opens a pygame window

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
agent = DQNAgent(env, capacity=2500, learning_rate=1e-3, device=device)

num_episodes = 1000
rewards = []

epsilon = 1.0
state, info = env.reset()

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

        # update Q network
        agent.update(target_freq=1000, batch_size=32)

        state = next_state
    
    # anneal epsilon every episode
    epsilon = max(0.01, epsilon * 0.995)

    # save episode reward for plotting
    rewards.append(episode_reward)

    # save model at end of each episode
    torch.save(agent.q_network.state_dict(), "q_network.pth")

# plot rewards
plt.plot(rewards)
plt.show()