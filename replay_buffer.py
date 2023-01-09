import numpy as np
from collections import deque, namedtuple
import random
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed, discount_factor, n_step=1):
        self.device = device
        self.memory = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.discount_factor = discount_factor
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def calc_multistep_return(self, n_step_buffer):
        return_val = 0
        for i in range(self.n_step):
            return_val += self.discount_factor ** i * n_step_buffer[i][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], return_val, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(object):
    def __init__(self, buffer_size, batch_size, seed, discount_factor, n_step, alpha=0.6, beta_start=0.4,
                 beta_frames=100000):
        
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size)
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)
        self.seed = np.random.seed(seed)
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.discount_factor = discount_factor

        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def add(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self):
        if self.priorities.dtype != "float32":
            print("priorities dtype changed to: ", self.priorities.dtype)

        buffer_length = len(self.buffer)
        if buffer_length == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        P = (probs / probs.sum())

        indices = np.random.choice(buffer_length, self.batch_size, p=P)
        state_samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        weights = (buffer_length * P[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*state_samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def calc_multistep_return(self, n_step_buffer):
        return_val = 0
        for idx in range(self.n_step):
            return_val += self.discount_factor ** idx * n_step_buffer[idx][2]

        return n_step_buffer[0][0], n_step_buffer[0][1], return_val, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def __len__(self):
        return len(self.buffer)
