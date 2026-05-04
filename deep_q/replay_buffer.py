import numpy as np

class ReplayBuffer:
    """
    Experience Replay Buffer to train Q network
        Store 'experiences' (s_t, a_t, r_t, s_t+1, done) in separate numpy arrays
        Use fixed-size numpy arrays with pointer -> O(1) store method
        (using .pop() would be really slow as python has to move all pointers across)
    """
    def __init__(self, capacity, state_dim):
        self.capacity = capacity

        self.pointer = 0 # track the next write position for new experience
        self.size = 0 # current size of memory

        # memory arrays, unpack any tuple inputs using *
        self.states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int_)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *state_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from the buffer"""
        # pick random indices to sample for batch
        indices = np.random.randint(0, self.size, size=batch_size)

        # return batch of experience tuples
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
                            )
        
    def store(self, state, action, reward, next_state, done):
        """
        Store a new experience in the buffer
        -> Using pointer, we can just wrap-around using modulo to overwrite oldest experience 
        """

        # overwrite values at current write pointer
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.dones[self.pointer] = done
        
        # wrap around the pointer if capacity reached
        self.pointer = (self.pointer + 1) % self.capacity

        # update size of memory
        self.size = min(self.size + 1, self.capacity)