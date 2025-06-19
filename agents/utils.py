from typing import Dict, List

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import math
import random


from doom_arena.render import render_episode
from IPython.display import HTML


class Downsample(nn.Module):
    def __init__(self, space: int, dim: int, downsample: int = 2):
        super().__init__()

        if space == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            downsample = (1, downsample, downsample)

        self.conv = Conv(dim, dim, downsample, downsample, bias=False)

    def forward(self, x: torch.Tensor):
        return F.silu(self.conv(x))

class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        
        return self.out(attn_output)

class ResidualBlock(nn.Module):
    def __init__(self, space: int, dim: int, act_fn: nn.Module = nn.SiLU, depth: int = 2, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super().__init__()
        Conv = nn.Conv2d if space == 2 else nn.Conv3d
        convs = []
        for d in range(depth):
            conv = nn.Sequential(
                Conv(dim, dim, kernel_size=kernel_size, padding=padding, stride=stride),
                act_fn(),
                Conv(dim, dim, kernel_size=kernel_size, padding=padding, stride=stride),
            )
            convs.append(conv)
            if d < depth - 1:
                convs.append(act_fn())

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        residual = x
        for conv in self.convs:
            x = conv(x) + residual
            residual = x
        return x


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.995):
    ema_params = OrderedDict(ema_model.named_parameters())
    if hasattr(model, "module"):
        model_params = OrderedDict(model.module.named_parameters())
    else:
        model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def batch_tree(x: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    batched = {}
    keys = x[0].keys()

    for key in keys:
        batched[key] = torch.stack([d[key] for d in x], dim=0)

    return batched


def process_observation(obs: torch.Tensor, state_dims: dict, device: str, dtype = torch.float32, permute: bool = False) -> Dict[str, torch.Tensor]:
    """Process multi-buffer observation from environment"""
    
    if isinstance(obs, torch.Tensor) and obs.size()[1] > 1:
        
        state_dim_idx = obs.ndim - 1 if permute else obs.ndim - 3 # if permute last channel, regular case: channel 0 if 3 dimensions, otherwise channel 1 as 0 is batches.
        
        # Put state channel at the end
        if permute and obs.ndim == 3:
            obs = obs.permute(1,2,0)
            
        elif permute and obs.ndim == 4:
            obs = obs.permute(0,2,3,1)
        
        obs_processed = obs.to(device, dtype=dtype).split(list(state_dims.values()), dim=state_dim_idx)
        return dict(zip(state_dims.keys(), obs_processed))
        
    else:
        # Single observation case
        return {'screen': obs.to(device, dtype=dtype)}

def epsilon_greedy_multi_buffer(env, model: nn.Module, obs: torch.Tensor, epsilon: float, device: str, state_dims: dict, dtype = torch.float32):
    """Epsilon-greedy action selection for multi-buffer observations"""

    
    # TODO: select_action method OR simplify with epsilon greedy?
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        model.eval()
        with torch.no_grad():
            processed_obs = process_observation(obs, state_dims, device, dtype, permute=False)
            # Add batch dimension
            for key in processed_obs:
                processed_obs[key] = processed_obs[key].unsqueeze(0)
            q_values = model(processed_obs)
            return q_values.argmax().item()

def hard_update_target_network(target_net, main_net):
    """Hard update of target network"""
    target_net.load_state_dict(main_net.state_dict())
    
    
    # Update Target network
def soft_update_target_network(local_model: nn.Module, target_model: nn.Module, tau: float):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    # TODO: Update target network
    
    # Store current state dicts
    local_state_dict = local_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    # Iterate over the modules and soft update the target state dict with parameter tau
    for module in local_state_dict.keys():
        target_state_dict[module] = local_state_dict[module] * tau + target_state_dict[module] * (1 - tau)
    
    # Load the soft updated values
    target_model.load_state_dict(target_state_dict)
    
    
    
def replay_episode(env, model, device, extra_state_dims, dtype):
    # ----------------------------------------------------------------
    # Hint for replay visualisation:
    # ----------------------------------------------------------------
    env.enable_replay()
    # ... run an evaluation episode ..
    
    # Tracking reward components
    eval_reward = 0.0
    play_episode_metrics = {
        'frags': 0, 'hits': 0, 'damage_taken': 0, 
        'movement': 0, 'ammo_efficiency': 0, 'survival': 0
    }
    
    # Reset environment
    eval_obs = env.reset()[0]
    eval_done = False
    model.eval().cpu()

    # # Run episode
    # while not done:
    #     act = epsilon_greedy(env, model, obs, epsilon=0.0, device=device, dtype=DTYPE)
    #     next_obs, r, done, info = env.step(act)

    #     # Accumulate reward (adapt keys depending on env)
    #     if isinstance(info, dict):  # single env
    #         for k in reward_components:
    #             reward_components[k] += info.get(k, 0.0)
    #     elif isinstance(info, list):  # vectorized env
    #         for k in reward_components:
    #             reward_components[k] += info[0].get(k, 0.0)

    #     ep_return += r[0]
    #     obs = next_obs[0]

    while not eval_done:
        eval_act = epsilon_greedy_multi_buffer(env, model, eval_obs.cpu(), 0.05, device, extra_state_dims, dtype)
        eval_obs, reward_components, eval_done, _ = env.step(eval_act)
        eval_obs = eval_obs[0]
        
        total_reward = sum(reward_components)
        eval_reward += total_reward
        
        if len(reward_components) >= 6:
            play_episode_metrics['frags'] += reward_components[0]
            play_episode_metrics['hits'] += reward_components[1]
            play_episode_metrics['damage_taken'] += reward_components[2]
            play_episode_metrics['movement'] += reward_components[3]
            play_episode_metrics['ammo_efficiency'] += reward_components[4]
            play_episode_metrics['survival'] += reward_components[5]


    print(f"Final evaluation - Total reward: {eval_reward:.1f}")
    print(f"Metrics: {play_episode_metrics}")

    # Finalize episode
    env.disable_replay()

    replays = env.get_player_replays()
    HTML(render_episode(replays, subsample=5).to_html5_video())
    
    
    model.to(device)
    #
    # Feel free to adapt or write your own GIF/MP4 export.