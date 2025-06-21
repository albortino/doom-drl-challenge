# agents/replay_buffer.py

import random
import numpy as np
import torch

class SumTree:
    """
    A SumTree data structure for storing priorities in the Prioritized Experience Replay buffer.
    This allows for efficient sampling of experiences based on their priorities.
    """
    write = 0

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_indices = np.zeros(capacity, dtype=int)

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, p: float, data_index: int):
        idx = self.write + self.capacity - 1
        self.data_indices[self.write] = data_index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, int]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data_indices[data_idx]


class PrioritizedReplayBuffer:
    """
    A Prioritized Experience Replay (PER) buffer.
    This buffer stores transitions and samples them based on their TD-error,
    allowing the agent to learn more from "surprising" transitions.
    While the internal storage uses NumPy for efficiency, it returns PyTorch tensors.
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity: int, device: str = "cpu"):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def __len__(self):
        return len(self.memory)

    def store(self, transition: tuple):
        """
        Stores a new transition in the buffer.
        New transitions are given max priority to ensure they are sampled at least once.
        """
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper

        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        data_index = self.position
        self.tree.add(max_p, data_index)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        """
        Samples a batch of transitions from the buffer based on their priorities.
        
        Returns:
            A tuple containing tensors for:
            - observations, actions, rewards, next_observations, dones, indices, and importance sampling weights.
        """
        pri_segment = self.tree.total() / batch_size
        priorities = []
        batch = []
        idxs = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data_idx = self.tree.get(s)
            
            priorities.append(p)
            batch.append(self.memory[data_idx])
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(len(self.memory) * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        obs, actions, rewards, next_obs, dones = zip(*batch)

        return (torch.stack(obs).to(self.device, dtype=torch.float32),
                torch.tensor(actions, device=self.device, dtype=torch.long),
                torch.tensor(rewards, device=self.device, dtype=torch.float32),
                torch.stack(next_obs).to(self.device, dtype=torch.float32),
                torch.tensor(dones, device=self.device, dtype=torch.float32),
                idxs,
                torch.tensor(is_weight, device=self.device, dtype=torch.float32))

    def batch_update(self, tree_idx: list[int], abs_errors: np.ndarray):
        """Updates the priorities of the sampled transitions."""
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

