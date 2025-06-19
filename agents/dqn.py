# dqn.py

import torch
import torch.nn as nn
from einops import rearrange
from gym import Env
import random
from typing import Dict
from collections import OrderedDict
from agents.utils import MultiHeadAttention, ResidualBlock, process_observation
from agents.my_modules import *



class DQN(nn.Module):
    """Basic Deep Q network."""

    def __init__(self, input_dim: int = 3, action_space: int = 8, hidden: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.action_space = action_space
        self.hidden= hidden
        
        self.encoder = nn. Sequential(
            nn. Conv2d (input_dim, 32, 8, stride=4, padding=2), nn.ReLU(), # 32
            nn. Conv2d (32, 64, 4, stride=2, padding=1), nn.ReLU(), # 16
            nn. Conv2d(64, 128, 3, stride=2), nn.ReLU(), #7
            nn. Conv2d (128, 128, 3, stride=2), nn.ReLU(), #3
        )
        
        self.head = nn. Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden), nn.ReLU(),
            #nn. Linear(64 * 3 * 3, hidden), nn.ReLU()
            nn.Linear(hidden, action_space**2), nn.ReLU(),
            nn.Linear(action_space**2, action_space),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = self.encoder(frame)
        x = self.head(x)
        return x



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


class EfficientDQN(nn.Module):
    """
    Efficient DQN with multi-buffer visual processing and attention mechanisms
    
    Architecture:
    TODO
    """
    
    def __init__(self, input_channels_dict: Dict[str, int], action_space: int, feature_dim_cnns: int = 64, hidden_dim_heads: int = 512, phi: nn.Module = nn.ReLU(), r: int = 4):
    
        super().__init__()
        
        self.action_space = action_space
        self.hidden_dim_heads = hidden_dim_heads
        self.phi = phi
        self.r = r
        
        # Feature dimension after encoding
        self.feature_dim_cnns = feature_dim_cnns
        
        self.num_buffers = len(input_channels_dict)
        
        # Separate encoders for each buffer type
        self.screen_encoder = self._build_encoder(input_channels_dict.get("screen", 3))
        self.depth_encoder = self._build_encoder(input_channels_dict.get("depth", 1))
        self.labels_encoder = self._build_encoder(input_channels_dict.get("labels", 1))
        self.automap_encoder = self._build_encoder(input_channels_dict.get("automap", 3))
        
        
        # Dueling network heads
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_dim_cnns * self.num_buffers, self.hidden_dim_heads),
            self.phi,
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim_heads, self.hidden_dim_heads // 4),
            self.phi,
            nn.Linear(self.hidden_dim_heads // 4, 1) # one value prediction
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(self.feature_dim_cnns* self.num_buffers, self.hidden_dim_heads),
            self.phi,
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim_heads, self.hidden_dim_heads // 4),
            self.phi,
            nn.Linear(self.hidden_dim_heads // 4, action_space) # one prediction per action
        )
        
        params_screen_encoder = sum(p.numel() for p in self.screen_encoder.parameters() if p.requires_grad)
        params_depth_encoder = sum(p.numel() for p in self.depth_encoder.parameters() if p.requires_grad)
        params_labels_encoder = sum(p.numel() for p in self.labels_encoder.parameters() if p.requires_grad)
        params_automap_encoder = sum(p.numel() for p in self.automap_encoder.parameters() if p.requires_grad)
        params_advantage_head = sum(p.numel() for p in self.advantage_head.parameters() if p.requires_grad)
        params_value_head = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
    
        
        print(f"Initialized model with {params_screen_encoder + params_depth_encoder + params_labels_encoder + params_automap_encoder + params_advantage_head + params_value_head} parameters!")
        
    def _build_encoder(self, input_channels: int) -> nn.Module:
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
    
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multi-buffer DQN
        
        Args:
            observations: Tensor for "screen", "depth", "labels", "automap"
        """
        features = []
        
        observation_keys = observations.keys()
        
        # Encode each visual buffer
        if "screen" in observation_keys:
            #print("model: screen was found")
            screen_feat = self.screen_encoder(observations["screen"])
            features.append(screen_feat)
            
        if "depth" in observation_keys:
            #print("model: depth was found")
            depth_feat = self.depth_encoder(observations["depth"])
            features.append(depth_feat)
            
        if "labels" in observation_keys:
            #print("model: labels was found")
            labels_feat = self.labels_encoder(observations["labels"])
            features.append(labels_feat)
            
        if "automap" in observation_keys:
            #print("model: automap was found")
            automap_feat = self.automap_encoder(observations["automap"])
            features.append(automap_feat)
        
        # Stack features for attention
        if len(features) > 1:
            fused_features = torch.cat(features, dim=1)  # (batch, feature_dim * num_buffers)
        
        else:
            fused_features = features[0]
        
        # Dueling network computation
        value = self.value_head(fused_features)
        advantage = self.advantage_head(fused_features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    

class EnhancedDQN(nn.Module):
    """
    Enhanced DQN with multi-buffer visual processing and attention mechanisms
    
    Architecture:
    - Separate encoders for each visual buffer (Screen, Depth, Labels, Automap)
    - Multi-head attention for feature fusion
    - Residual connections and batch normalization
    - Dueling network head
    """
    
    def __init__(self, input_channels_dict: Dict[str, int], action_space: int, hidden_dim_heads: int = 512):
        super().__init__()
        
        self.action_space = action_space
        self.hidden_dim_heads = hidden_dim_heads
        
        # Feature dimension after encoding
        self.feature_dim = 256
        
        # Separate encoders for each buffer type
        self.screen_encoder = self._build_encoder(input_channels_dict.get("screen", 3))
        self.depth_encoder = self._build_encoder(input_channels_dict.get("depth", 1))
        self.labels_encoder = self._build_encoder(input_channels_dict.get("labels", 1))
        self.automap_encoder = self._build_encoder(input_channels_dict.get("automap", 3))
        
        # Multi-head attention for feature fusion
        self.attention = MultiHeadAttention(self.feature_dim, num_heads=8)
        
        # Dueling network heads
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim_heads),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_heads, hidden_dim_heads // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim_heads // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim_heads),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim_heads, hidden_dim_heads // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim_heads // 2, action_space)
        )
        
    def _build_encoder(self, input_channels: int) -> nn.Module:
        """Build CNN encoder for visual input"""
        return nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, 32, 8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Residual blocks
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Output projection
            nn.Linear(256, self.feature_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through multi-buffer DQN
        
        Args:
            observations: Tensor for "screen", "depth", "labels", "automap"
        """
        features = []
        
        observation_keys = observations.keys()
        
        # Encode each visual buffer
        if "screen" in observation_keys:
            print("model: screen was found")
            screen_feat = self.screen_encoder(observations["screen"])
            features.append(screen_feat)
            
        if "depth" in observation_keys:
            print("model: depth was found")
            depth_feat = self.depth_encoder(observations["depth"])
            features.append(depth_feat)
            
        if "labels" in observation_keys:
            print("model: labels was found")
            labels_feat = self.labels_encoder(observations["labels"])
            features.append(labels_feat)
            
        if "automap" in observation_keys:
            print("model: automap was found")
            automap_feat = self.automap_encoder(observations["automap"])
            features.append(automap_feat)
        
        # Stack features for attention
        if len(features) > 1:
            stacked_features = torch.stack(features, dim=1)  # (batch, num_buffers, feature_dim)
            # Apply attention
            attended_features = self.attention(stacked_features)
            # Global average pooling over buffer dimension
            fused_features = attended_features.mean(dim=1)
        else:
            fused_features = features[0]
        
        # Dueling network computation
        value = self.value_head(fused_features)
        advantage = self.advantage_head(fused_features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    
if __name__ == "__main__":
    model = EfficientDQN({'screen': 3, 'labels': 1, 'depth': 1, 'automap': 3}, 8)
    
     