import torch
import random

from deep_q.q_network import QNetwork
from deep_q.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, env, capacity=1000, learning_rate=1e-3, device='cpu'):
        """Initialise DQN agent"""
        # get observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # initialise replay buffer
        self.buffer = ReplayBuffer(capacity, self.observation_space.shape) 

        # initialise Q and target network
        self.device = device
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)

        # set target network to Q network (copy weights)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # set target network to evaluate mode and set update frequency
        self.target_network.eval()

        # count number of updates, for updating target network
        self.update_counter = 0

        # set up optimiser
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr = learning_rate)

    def select_action(self, state, epsilon = 0.05):
        """Select action with epsilon greedy policy"""
        if random.random() < epsilon:
            action = int(self.action_space.sample())
        else:
            q_values = self.q_network(torch.as_tensor(state, dtype=torch.float32).to(self.device))
            action = int(torch.argmax(q_values).item())

        return action

    def update(self, target_freq=1000, discount=0.99, batch_size=100):
        """
        Apply Q-learning update
        -> Train Q-network from experience batch
        -> Copy weights to Target network at specified frequency    
        """

        # sample a batch of experiences from buffer as torch tensors
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # get q values for the actions taken 
        q_values = self.q_network(states)

        q_taken = torch.gather(q_values, dim=1, index=actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            # compute TD targets (accounting for terminating episodes)
            targets = rewards + discount * self.target_network(next_states).max(dim=1).values * (1 - dones)

        # use Huber loss (smooth_l1_loss) - clipped gradients between -1 and 1
        loss = torch.nn.functional.smooth_l1_loss(q_taken, targets)
        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # update target network at specified frequency
        self.update_counter += 1
        if self.update_counter % target_freq == 0:
            # set target network to Q network (copy weights)
            self.target_network.load_state_dict(self.q_network.state_dict())

        if self.update_counter % 100 == 0:
            print(f"Loss: {loss.item():.4f}  Q_mean: {q_values.mean().item():.3f}")