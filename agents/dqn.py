# dqn.py

import torch
import torch.nn as nn
from collections import OrderedDict
import os
import random
from typing import Dict
from agents.modules import ResidualBlock, Downsample, Parallel, OwnModule
from agents.helpers import ActionCounter, ExtraStates, EnvActions
import numpy as np

@torch.no_grad()
def process_observation(obs: torch.Tensor, state_dims: dict = {}, device: str = "cpu", dtype = torch.float32, permute: bool = False) -> torch.Tensor|Dict[str, torch.Tensor]:
    """Process multi-buffer observation from environment, for training and for plotting. Returns a dictionary with the state as key and data as value. """
    """ NOTE: state dims is only for splitting it -> depreceated as model now does the transformation itself!"""
    if isinstance(obs, torch.Tensor) and obs.size()[1] > 1:
                
        # Put state channel at the end for plotting (color detection)
        if permute and obs.ndim == 3:
            obs = obs.permute(1,2,0)
            
        elif permute and obs.ndim == 4:
            obs = obs.permute(0,2,3,1)
        
        obs = obs.to(device, dtype=dtype)
        
        if state_dims:
            state_dim_idx = obs.ndim - 1 if permute else obs.ndim - 3 # if permute last channel, regular case: channel 0 if 3 dimensions, otherwise channel 1 as 0 is batches.
            obs_states = obs.split(list(state_dims.values()), dim=state_dim_idx)
            return dict(zip(state_dims.keys(), obs_states))
        
        return obs
        
    else:
        # Fallback: Single observation case
        return {'screen': obs.to(device, dtype=dtype)}


@torch.no_grad()
def epsilon_greedy(env, model: nn.Module, obs: list, epsilon: float, env_actions: EnvActions, device: str = "cpu", dtype = torch.float32, action_counter: ActionCounter = None, debug: bool = False):
    """Epsilon-greedy action selection for multi-buffer observations"""

    num_players = env.num_players
    
    if random.random() < epsilon:
       
        # One different action (1-7) per player
        chosen_actions = env_actions.get_random_action(n=num_players, use_proba=True)
        print("WITHIN EPSILON:", chosen_actions) if debug else None
        
    else:
        model.eval()
        obs_batch = torch.stack(obs)
        processed_obs = process_observation(obs_batch, device=device, dtype=dtype, permute=False)
        q_values = model(processed_obs)
        
        # Indices are necessary if buttons don't start at 0 but at 1 (e.g., if there wouldn't be a noop option)
    
        if num_players == 1:
            chosen_actions_idx = q_values.argmax().item() # Single action
            print("DEBUG EPSILON NUM PLAYERS=1:", chosen_actions_idx, q_values, end="")  if debug else None
        else:
            chosen_actions_idx = q_values.argmax(dim=1).tolist() # list of actions
            print("DEBUG EPSILON NUM PLAYERS>1:", chosen_actions_idx, q_values, end="")  if debug else None
        
        chosen_actions = env_actions.get_action_value(chosen_actions_idx) 
        print(chosen_actions) if debug else None
        
    # Add to action counter
    if action_counter:
        if isinstance(chosen_actions, list):
            for action in chosen_actions:
                action_counter.add(action)
        else: # single action
            action_counter.add(chosen_actions)

    return chosen_actions

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
    

class EfficientDQN(OwnModule):
    """
    Efficient DQN with multi-buffer visual processing and attention mechanisms
    
    Architecture:
    TODO
    """
    
    def __init__(self, obs_state_infos: ExtraStates, action_space: int, feature_dim_cnns: int = 128, hidden_dim_heads: int = 1024, phi: nn.Module = nn.ELU()):
    
        super().__init__()
        
        self.action_space = action_space
        self.hidden_dim_heads = hidden_dim_heads
        self.phi = phi
        
        # Feature dimension after encoding
        self.feature_dim_cnns = feature_dim_cnns

        # Observation states info
        self.obs_state_dims = obs_state_infos.get_dims(return_dict=False) # Indices to split the observation
        self.obs_states_num = obs_state_infos.num_states
        
        # Separate encoders for each buffer type
        self.encoder = Parallel([self._build_encoder(dim) for dim in self.obs_state_dims])
        
        self.head_first = nn.Sequential(
            nn.Linear(self.feature_dim_cnns * self.obs_states_num, self.hidden_dim_heads),
            self.phi,
            nn.Dropout(0.3))
        
        # Dueling network heads
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim_heads, self.hidden_dim_heads // 4),
            nn.Dropout(0.2),
            self.phi,
            nn.Linear(self.hidden_dim_heads // 4, 32), # one value prediction
            self.phi,
            nn.Linear(32, 1) # one value prediction
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_dim_heads, self.hidden_dim_heads // 4),
            nn.Dropout(0.2),
            self.phi,
            nn.Linear(self.hidden_dim_heads // 4, action_space * self.obs_states_num),
            self.phi,
            nn.Linear(action_space * self.obs_states_num, action_space) # one prediction per action
        )
        
        params_encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        params_head_first = sum(p.numel() for p in self.head_first.parameters() if p.requires_grad)
        params_advantage_head = sum(p.numel() for p in self.advantage_head.parameters() if p.requires_grad)
        params_value_head = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        
        self.init_weights()
        
        print(f"Initialized model with {params_encoder + params_head_first + params_advantage_head + params_value_head} parameters!")
    
    def _build_encoder(self, input_channels: int) -> nn.Module:
        """Build CNN encoder with residual blocks for visual input"""
        
        first_channel_out = 16 if input_channels == 3 else 4
        
        return nn.Sequential(
            # Initial convolution - reduce spatial dimensions significantly
            nn.Conv2d(input_channels, first_channel_out, 8, stride=4, padding=2),
            self.phi,  # Output [N, C=16/4, WH=32]
            
            # First residual stage
            ResidualBlock(space=2, dim=first_channel_out, act_fn=self.phi, depth=1),
            Downsample(space=2, dim=first_channel_out, downsample=2),  # [N, C, 16]
            
            # Second residual stage - double channels
            nn.Conv2d(first_channel_out, first_channel_out * 2, 1, bias=False),
            ResidualBlock(space=2, dim=first_channel_out * 2, act_fn=self.phi, depth=1),
            Downsample(space=2, dim=first_channel_out * 2, downsample=2),  # [N, 2C, 8]
            
            # Third residual stage - double channels again, with kernel
            nn.Conv2d(first_channel_out * 2, first_channel_out * 4, 3, bias=False),
            ResidualBlock(space=2, dim=first_channel_out * 4, act_fn=self.phi, depth=2), # [N, 4C, 6]
            
            # Channel reduction with residual connection
            ResidualBlock(space=2, dim=first_channel_out * 4, act_fn=self.phi, depth=1),
            
            # Final spatial reduction and channel adjustment
            nn.Conv2d(first_channel_out * 4, first_channel_out * 2, 3, stride=3, padding=1), # [N, C, 2]
            self.phi,
            
            # Global average pooling -> one pixel per channel
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Output projection
            nn.Linear(first_channel_out * 2, self.feature_dim_cnns),
            self.phi
        )
    
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-buffer DQN
        
        Args:
            observations: Tensor for "screen", "depth", "labels", "automap"
        """                
        # Encode each visual buffer and split the observations
        features = self.encoder(observations.split(self.obs_state_dims, dim=1))
        
        # Use one head for the dueling network                
        head_logits = self.head_first(features)
        
        # Dueling network computation
        value = self.value_head(head_logits)
        advantage = self.advantage_head(head_logits)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def save_model(self, path: str = "", filename: str = "model.pt"):
        """ Saves the model's state dict at the provided path in a subfolder based on the date. Name: "model.pt"

        Args:
            path (str, optional): Path where the model should be stored. Defaults to "".
        """
        
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__)) if path == "" else path     
            os.makedirs(dir_path, exist_ok=True)
            
            file_path = os.path.join(dir_path, filename)
            
            checkpoint = dict(
                architecture= dict(
                    input_channels_dict=self.input_channels_dict,
                    action_space=self.action_space,
                    feature_dim_cnns = self.feature_dim_cnns,
                    hidden_dim_heads = self.hidden_dim_heads,
                    phi = self.phi,
                ),
                state_dict = self.state_dict()
            )
            
            torch.save(checkpoint, file_path)
            
            print(f"Successfully stored the model: {file_path}")
        
        except Exception as e:
            print(f"Failed to store the model: {e}")
        
    
    @classmethod  
    def load_model(cls, path: str = ""):
        """ Loads an instance of this model class from the provided path. 

        Args:
            path (str, optional): Path to the state dicht. Defaults to "".

        Returns:
            PM_Model: An instance of the model class.
        """
        try:
            checkpoint = torch.load(path, weights_only=False)
                        
            arch_params = checkpoint.get("architecture")
            state_dict = checkpoint.get("state_dict")

            if arch_params is None or state_dict is None:
                print(f"Failed to load model: Checkpoint file {path} is missing 'architecture' or 'state_dict'.")
                return None

            loaded_model = cls(**arch_params)
            loaded_model.load_state_dict(state_dict)
            
            return loaded_model
        
        except Exception as e:
            print(f"Failed to load the model: {e}")

    
if __name__ == "__main__":
    obs_states = ExtraStates(["screen", "depth", "labels", "automap"], 1)
    model = EfficientDQN(obs_states, 8)
    
     