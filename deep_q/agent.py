import torch
import random

from deep_q.q_network import QNetwork
from deep_q.replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, env, capacity, target_freq, device='cpu'):
        """Initialise DQN agent"""
        # get observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # initialise replay buffer
        self.buffer = ReplayBuffer(capacity, self.observation_space.shape, self.action_space.shape) 

        # initialise Q and target network
        self.device = device
        self.q_network = QNetwork().to(self.device)
        self.target_network = QNetwork().to(self.device)

        # set target network to Q network (copy weights)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # set target network to evaluate mode and set update frequency
        self.target_network.eval()
        self.target_update_freq = target_freq

    def select_action(self, state, epsilon = 0.05):
        """Select action with epsilon greedy policy"""
        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            q_values = self.q_network.forward(torch.as_tensor(state).to(self.device))
            action = torch.argmax(q_values).item()
        
        return action