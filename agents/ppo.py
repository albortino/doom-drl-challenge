# ppo.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
from einops import rearrange



class PPOActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_space: int):
        super().__init__()

        # Minimal encoder (extend this)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # TODO: Replace with appropriate dimensions
        self.policy_head = nn.Linear(16 * 32 * 32, action_space)
        self.value_head = nn.Linear(16 * 32 * 32, 1)

    def forward(self, obs):
        x = self.encoder(obs)
        return self.policy_head(x), self.value_head(x)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value
