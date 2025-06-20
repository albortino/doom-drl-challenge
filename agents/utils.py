from typing import Dict, List

import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
import math
import random

import numpy as np

from datetime import datetime

from doom_arena.render import render_episode
from IPython.display import HTML
import os

    
from matplotlib import pyplot as plt
from matplotlib.table import Table

import pandas as pd



        
class LossLogger():
    """ Class to log the losses during training or evaluation. """
    def __init__(self):
        self.all_losses = list()
        self.loss: float = 0.0
        self.num_batches: int = 0
        
    def add(self, loss: torch.Tensor):
        """ Adds the loss tensor. """
        self.loss += loss.detach().item()
        self.num_batches += 1

    def get_loss(self)-> float:
        """ Returns the average loss as float. """
        if self.num_batches > 0:
            return self.loss / self.num_batches
        else:
            return 0.0
        
    def reset_on_epoch(self):
        """ Resets the losses after an epoch. """
        if self.num_batches > 0:
            self.all_losses.append(self.get_loss())
            self.loss = 0.0
            self.num_batches = 0


class FileLogger():
    def __init__(self, path: str, filename: str = "logs.txt", also_print: bool = False):
        """ Initialize a logger class that logs the messages to a file but is also capable of printing it.

        Args:
            path (str): Path to store the log file
            also_print (bool, optional): Whether every log should also be printed (Tipp: self.log() allows for one-time printing, too!). Defaults to False.
        """
        self.path = path
        self.filename = filename
        self.also_print = also_print
        
        self.file_path = os.path.join(self.path, self.filename)
        
        self.create_log_file()
    
    def create_log_file(self):
        # Ensure the directory exists
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            
        with open(self.file_path, "w") as f:
            f.write(F"LOGGER INITIALIZED AT {datetime.now().strftime("%Y%m%d-%H%M%S")}\n")
    
    def log(self, msg: str, print_once: bool = False, end="\n"):
        with open(self.file_path, "a") as f:
            f.write(msg + end)
        
        if self.also_print or print_once:
            print(msg)
            
            
class Parallel(nn.Module):
    def __init__(self, modules: dict[str, torch.nn.Module]):
        super().__init__()
        if not isinstance(modules, dict):
            raise ValueError("Modules are not dicts!")
        
        self.model = nn.ModuleDict(modules) # Otherwise modeules are not registered correctly

    def forward(self, inputs: dict[str, torch.Tensor]):
        return [module(inputs[name]) for name, module in self.model.items()]
    


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
        
        # Put state channel at the end for plotting
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

    num_players = len(obs)
    
    if random.random() < epsilon:
        return env.action_space.sample() # Single value or tuple

    elif num_players == 1:
        #obs = obs[0]
        
        model.eval()
        with torch.no_grad():
            processed_obs = process_observation(obs, state_dims, device, dtype, permute=False)
            # Add batch dimension
            for key in processed_obs:
                processed_obs[key] = processed_obs[key].unsqueeze(0)
            q_values = model(processed_obs)
            return q_values.argmax().item()
    
    else: # num_players > 2
        model.eval()
        with torch.no_grad():
            obs_batch = torch.stack(obs)
            processed_obs = process_observation(obs_batch, state_dims, device, dtype, permute=False)
            q_values = model(processed_obs)
            return q_values.argmax(dim=1).tolist() # return list of actions

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
    
    
def replay_episode(env, model, device, extra_state_dims, dtype, path: str = "", store: bool = False):
    # ----------------------------------------------------------------
    # Hint for replay visualisation:
    # ----------------------------------------------------------------

    env.enable_replay()

    # Tracking reward components
    eval_reward = 0.0

    # Reset environment
    eval_obs = env.reset()#[player_idx]
    #eval_ob = eval_obs[player_idx]
    eval_done = False
    model.eval().cpu()


    while not eval_done:
        eval_act = epsilon_greedy_multi_buffer(env, model, [eval_ob.cpu() for eval_ob in eval_obs], 0, "cpu", extra_state_dims, dtype)
        eval_obs, reward_components, eval_done, _ = env.step(eval_act)
        if env.num_players == 1:
            eval_obs = eval_obs[0]

        eval_reward += sum(reward_components)

    print(f"Final evaluation - Total reward: {eval_reward:.1f}")

    # Finalize episode
    env.disable_replay()

    replays = env.get_player_replays()

    # Random Player
    player_idx = np.random.randint(0, len(replays))
    player_name = list(replays.keys())[player_idx]
    one_player_replay = {player_name: replays.get(player_name)}

    if store:
        path = os.path.join(path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{eval_reward:.0f}.mp4")
        render_episode(one_player_replay, subsample=5, replay_path=path)
    else:
        HTML(render_episode(one_player_replay, subsample=5).to_html5_video())

    model.train()
    model.to(device)
    

def plot_images(obs: torch.Tensor, state_dims: dict):
    
    num_plots = len(state_dims)
    
    obs_processed = process_observation(obs, state_dims, device="cpu", permute=True)
    
    last_plot = 1
    
    if state_dims.get("screen") is not None:
        plt.subplot(100 + 10*num_plots + last_plot)
        plt.imshow(obs_processed.get("screen"))
        plt.axis("off")
        plt.title("Screen")
        last_plot += 1

    # Labels 
    if state_dims.get("labels") is not None:
        plt.subplot(100 + 10*num_plots + last_plot)
        plt.imshow(obs_processed.get("labels"), vmin=0, vmax=1)
        plt.axis("off")
        plt.title("Labels")
        last_plot += 1
    
    # Depth
    if state_dims.get("depth") is not None:
        plt.subplot(100 + 10*num_plots + last_plot)
        plt.imshow(obs_processed.get("depth"))
        plt.axis("off")
        plt.title("Depth")
        last_plot += 1



    # Automap
    if state_dims.get("automap") is not None:
        plt.subplot(100 + 10*num_plots + last_plot)
        plt.imshow(obs_processed.get("automap"))
        plt.axis("off")
        plt.title("Automap")
        

    
    plt.tight_layout()
    plt.show()
    
    
def get_average_result(episode_metrics: Dict[int, dict]) -> dict:
    """ Returns the average reward from all players for each reward type. """
    averaged_metrics = dict()

    for values in episode_metrics.values():
        for key, reward_val in values.items():
            if isinstance(averaged_metrics.get(key), list):
                averaged_metrics[key].append(reward_val)
            else:
                averaged_metrics[key] = [reward_val]
                
    return {key: np.mean(value, dtype=int) for key, value in averaged_metrics.items()}
        

def get_avg_reward(reward_history: dict, episodes: int = 1, player_idx: int=-1, round:int = 2) -> np.ndarray:
    """ Returns the average reward for the episode or player id

    Args:
        reward_history (dict): A dictionary with player id as key and the rewards as values
        episodes (int, optional): Number of last episode to extract. Defaults to 1 (=last).
        player_idx (int, optional): Player idx to filter. Defaults to -1.

    Returns:
        np.ndarray: Average rewards as an array
    """
    df_rwds = pd.DataFrame.from_dict(reward_history)
    
    if player_idx == -1:
        df_rwds = df_rwds.mean(axis=1) # Mean over columns
        
    else:
        df_rwds = df_rwds.iloc[:, player_idx]
        
    if episodes > 1:
        df_rwds = df_rwds[-episodes:].mean(axis=0) # Mean over columns
    else:
        df_rwds = df_rwds.iloc[episodes]
        
    return np.round(df_rwds, round)
        
    