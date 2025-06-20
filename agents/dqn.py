# dqn.py

import torch
import torch.nn as nn
from einops import rearrange
from gym import Env
from collections import OrderedDict
import os
import random
from typing import Dict
from agents.utils import ResidualBlock, Downsample, Parallel, process_observation
from agents.my_modules import *

@torch.no_grad
def epsilon_greedy(
    env: Env,
    model: nn.Module,
    obs: torch.Tensor,
    epsilon: float,
    device: torch.device,
    dtype: torch.dtype,
    state_dims: dict,
):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        #obs = obs.to(device, dtype=dtype).unsqueeze(0)
        processed_obs = process_observation(obs, state_dims, device, dtype)
        for key in processed_obs:
                processed_obs[key] = processed_obs[key].unsqueeze(0)
        
        return model(processed_obs).argmax().item()


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.995):
    """Exponential moving average model updates."""
    ema_params = OrderedDict(ema_model.named_parameters())
    if hasattr(model, "module"):
        model_params = OrderedDict(model.module.named_parameters())
    else:
        model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        

class EfficientDQN(OwnModule):
    """
    Efficient DQN with multi-buffer visual processing and attention mechanisms
    
    Architecture:
    TODO
    """
    
    def __init__(self, input_channels_dict: Dict[str, int], action_space: int, feature_dim_cnns: int = 128, hidden_dim_heads: int = 1024, phi: nn.Module = nn.ELU()):
    
        super().__init__()
        
        self.action_space = action_space
        self.hidden_dim_heads = hidden_dim_heads
        self.phi = phi
        
        # Feature dimension after encoding
        self.feature_dim_cnns = feature_dim_cnns
        
        self.input_channels_dict = input_channels_dict
        self.num_buffers = len(input_channels_dict)
        
        # Separate encoders for each buffer type
        encoder_dict = {key: self._build_encoder(dims) for key, dims in input_channels_dict.items()}
        self.encoder = Parallel(encoder_dict)
        
        self.head_first = nn.Sequential(
            nn.Linear(self.feature_dim_cnns * self.num_buffers, self.hidden_dim_heads),
            self.phi,
            nn.Dropout(0.3),
        )
        
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
            nn.Linear(self.hidden_dim_heads // 4, action_space * self.num_buffers),
            self.phi,
            nn.Linear(action_space * self.num_buffers, action_space) # one prediction per action
        )
        
        params_encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        params_head_first = sum(p.numel() for p in self.head_first.parameters() if p.requires_grad)
        params_advantage_head = sum(p.numel() for p in self.advantage_head.parameters() if p.requires_grad)
        params_value_head = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
        
        print(f"Initialized model with {params_encoder + params_head_first + params_advantage_head + params_value_head} parameters!")
        
    def _build_encoder_old(self, input_channels: int) -> nn.Module:
        """Build CNN encoder for visual input"""
        
        first_channel_out = 16 if input_channels == 3 else 4
        
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, first_channel_out, 8, stride=4, padding=2),
            #nn.BatchNorm2d(32),
            self.phi, # Output [N, C=16/4, WH=32]
            
            nn.Conv2d(first_channel_out, first_channel_out * 2, 4, stride=2, padding=1), # [N, C, 16]
            self.phi,
            
            nn.Conv2d(first_channel_out * 2, first_channel_out * 4, 2, stride=2, padding=0), # [N, C, 8]
            self.phi,

            DepthwiseConvBlock(first_channel_out * 4, phi=self.phi, kernel_size=2), # returns c
            PointwiseConvBlock(first_channel_out * 4, out_channels=first_channel_out * 4//self.r, phi=self.phi), # returns c//r
            
            SqueezeExcitationBlock(first_channel_out*4//self.r, init_weights=False),
            
            # Global average pooling -> one pixel per channel
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Output projection
            nn.Linear(first_channel_out*4//self.r, self.feature_dim_cnns),
            self.phi
        )

    
    def _build_encoder(self, input_channels: int) -> nn.Module:
        """Build CNN encoder with residual blocks for visual input"""
        
        first_channel_out = 16 if input_channels == 3 else 4
        
        return nn.Sequential(
            # Initial convolution - reduce spatial dimensions significantly
            nn.Conv2d(input_channels, first_channel_out, 8, stride=4, padding=2),
            self.phi,  # Output [N, C=16/4, WH=32]
            
            # First residual stage
            ResidualBlock(space=2, dim=first_channel_out, act_fn=type(self.phi), depth=1),
            Downsample(space=2, dim=first_channel_out, downsample=2),  # [N, C, 16]
            
            # Second residual stage - double channels
            nn.Conv2d(first_channel_out, first_channel_out * 2, 1, bias=False),
            ResidualBlock(space=2, dim=first_channel_out * 2, act_fn=type(self.phi), depth=1),
            Downsample(space=2, dim=first_channel_out * 2, downsample=2),  # [N, 2C, 8]
            
            # Third residual stage - double channels again, with kernel
            nn.Conv2d(first_channel_out * 2, first_channel_out * 4, 3, bias=False),
            ResidualBlock(space=2, dim=first_channel_out * 4, act_fn=type(self.phi), depth=2), # [N, 4C, 6]
            
            # Channel reduction with residual connection
            ResidualBlock(space=2, dim=first_channel_out * 4, act_fn=type(self.phi), depth=1),
            
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
    
    
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multi-buffer DQN
        
        Args:
            observations: Tensor for "screen", "depth", "labels", "automap"
        """
        features = []
                
        # Encode each visual buffer
        features = self.encoder(observations)
        
        # Stack features for attention
        if len(features) > 1:
            fused_features = torch.cat(features, dim=1)  # (batch, feature_dim * num_buffers)
        
        else:
            fused_features = features[0]
        
        head_logits = self.head_first(fused_features)
        
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
    model = EfficientDQN({'screen': 3, 'labels': 1, 'depth': 1, 'automap': 3}, 8)
    
     