import torch
import torch.nn as nn
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
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.stack(obs) # stack players
        
        processed_obs = process_observation(obs, device=device, dtype=dtype, permute=False)
        q_values = model(processed_obs)
        
        # Indices are necessary if buttons don't start at 0 but at 1 (e.g., if there wouldn't be a noop option)
        if num_players == 1:
            chosen_actions_idx = q_values.argmax().to(dtype=int).item() # Single action
            print("DEBUG EPSILON NUM PLAYERS=1:", chosen_actions_idx, q_values, end="")  if debug else None
        else:
            chosen_actions_idx = q_values.argmax(dim=1).to(dtype=int).tolist() # list of actions
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

def hard_update_target_network(target_net: nn.Module, main_net: nn.Module):
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
    
    # Store current state dicts
    local_state_dict = local_model.state_dict()
    target_state_dict = target_model.state_dict()
    
    # Iterate over the modules and soft update the target state dict with parameter tau
    for module in local_state_dict.keys():
        target_state_dict[module] = local_state_dict[module] * tau + target_state_dict[module] * (1 - tau)
    
    # Load the soft updated values
    target_model.load_state_dict(target_state_dict)


class EfficientDQN(OwnModule):
    """  EfficientDQN with multi-buffer visual processing  """
    
    def __init__(self, input_dim: int, action_space: int, obs_state_infos: ExtraStates, feature_dim_cnns: int = 256, hidden_dim_heads: int = 1024, phi: nn.Module = nn.LeakyReLU(), dropout_p:float = 0.15):
        #obs_state_infos: ExtraStates = None, 
        super().__init__()
        self.input_dim = input_dim # not used
        self.obs_state_infos = obs_state_infos
        self.action_space = action_space
        self.hidden_dim_heads = hidden_dim_heads
        self.phi = phi
        self.dropout_p = dropout_p
        
        # Feature dimension after encoding
        self.feature_dim_cnns = feature_dim_cnns

        # Observation states info
        self.obs_state_dims = obs_state_infos.get_dims(return_dict=False) # Indices to split the observation
        self.obs_states_num = obs_state_infos.num_states
        
        # Separate encoders for each buffer type
        self.encoders = Parallel([self._build_encoder(dim) for dim in self.obs_state_dims])
        
        self.head_first = nn.Sequential(
            #nn.Linear(self.feature_dim_cnns * self.obs_states_num, self.hidden_dim_heads),
            nn.Linear(np.array([2*2*4 * [32 if dim == 3 else 16 for dim in self.obs_state_dims]]).sum(), self.hidden_dim_heads),
            self.phi,
            nn.BatchNorm1d(self.hidden_dim_heads),
            nn.Dropout(self.dropout_p),
        )
        
        # Dueling network heads
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim_heads, self.hidden_dim_heads // 8),
            self.phi,
            nn.BatchNorm1d(self.hidden_dim_heads // 8),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim_heads // 8, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_dim_heads, self.hidden_dim_heads // 8),
            self.phi,
            nn.BatchNorm1d(self.hidden_dim_heads // 8),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim_heads // 8, action_space) 
        )
        
        params_encoder = sum(p.numel() for p in self.encoders.parameters() if p.requires_grad)
        params_head_first = sum(p.numel() for p in self.head_first.parameters() if p.requires_grad)
        params_advantage_head = sum(p.numel() for p in self.advantage_head.parameters() if p.requires_grad)
        params_value_head = sum(p.numel() for p in self.value_head.parameters() if p.requires_grad)
                
        self.init_weights(bias=False, debug=True)
        
        print(f"Initialized model with {params_encoder  + params_head_first + params_advantage_head + params_value_head} parameters!")
    
    def _build_encoder(self, input_channels: int) -> nn.Module:
        """Build CNN encoder with residual blocks for visual input"""
        
        first_channel_out = 32 if input_channels == 3 else 16
        
        return nn.Sequential(
            # Initial convolution - reduce spatial dimensions significantly
            nn.Conv2d(input_channels, first_channel_out, 8, stride=4, padding=2),
            self.phi,  # Output [N, C=16/4, WH=32
            nn.BatchNorm2d(first_channel_out),

            # Second CNN stage - double channels
            nn.Conv2d(first_channel_out, first_channel_out * 2, 4, stride=3, padding=0), # [N, 2C, 10]
            self.phi,
            nn.BatchNorm2d(first_channel_out * 2),
            
            # Third residual stage - double channels again, with kernel
            nn.Conv2d(first_channel_out * 2, first_channel_out * 4, 4, stride=2, padding=0), # [N, 4C, 4]
            self.phi,
            nn.BatchNorm2d(first_channel_out * 4),

            # Fourth residual stage - double channels again, with kernel
            nn.Conv2d(first_channel_out * 4, first_channel_out * 4, 2, stride=2), # [N, 4C, 4]
            self.phi,
            nn.BatchNorm2d(first_channel_out * 4),
        
            nn.Flatten(),
        )

    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-buffer DQN
        
        Args:
            observations: Tensor for "screen", "depth", "labels", "automap"
        """                
        # Encode each visual buffer and split the observations
        features = self.encoders(observations.split(self.obs_state_dims, dim=1))

        # Use one head for the dueling network                
        head_logits = self.head_first(features)

        # Dueling network computation
        value = self.value_head(head_logits)
        advantage = self.advantage_head(head_logits)

        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    

if __name__ == "__main__":
    obs_states = ExtraStates(["screen", "depth", "labels", "automap"], 1)
    model = EfficientDQN(1, 8, ExtraStates(["screen", "depth", "labels", "automap"], 1))