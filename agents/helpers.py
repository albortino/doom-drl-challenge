  
import torch 
import os
from datetime import datetime
import numpy as np
from doom_arena.reward import VizDoomReward
from collections import defaultdict
from functools import singledispatchmethod
from tqdm.notebook import trange

class TqdmProgress:
    """A tqdm progress bar wrapper for training loops."""

    def __init__(self, total: int, desc: str = "Training", unit: str = "episode"):
        """
        Initializes the progress bar.

        Args:
            total (int): The total number of iterations (e.g., episodes).
            desc (str): A description for the progress bar.
            unit (str): The unit for one iteration.
        """
        self._iterable_pbar = trange(total, total=total, desc=desc, unit=f" {unit}")
        self.pbar = self._iterable_pbar
        self._total_steps = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __iter__(self):
        yield from self._iterable_pbar

    def update_step_count(self, steps: int = 1):
        """
        Increments the total step counter.

        Args:
            steps (int): The number of steps to add.
        """
        self._total_steps += steps

    def set_description(self, current_episode: int):
        """
        Updates the description of the progress bar.

        Args:
            current_episode (int): The current episode number (0-indexed).
        """
        desc = f"Episode {current_episode + 1}/{self.pbar.total} | Total Training Steps: {self._total_steps:,}"
        self.pbar.set_description(desc)

    def set_postfix(self, stats: dict):
        """
        Updates the postfix of the progress bar with training statistics.
        """
        self.pbar.set_postfix(stats)

    def close(self):
        """Closes the progress bar."""
        self.pbar.close()



class YourReward(VizDoomReward):
    def __init__(self, num_players: int):
        super().__init__(num_players)
        self.prev_ammo = {}
        self.prev_health = {}
        self.prev_position = {}
        self.survival_bonus = 0
        
    def __call__(self, vizdoom_reward: float, game_var: dict[str, float], game_var_old: dict[str, float], player_id: int) -> tuple:
        """
        Custom reward functions
        * +100 for frags (kills)
        * +10 for hits
        * -2 for damage taken
        * +3 for movement (exploration)
        * +1 for ammo efficiency, +2 if ammo pickup
        * +0.05 survival bonus per step, -20 if dead
        * +2 if health pickup
        """
        
        """
            {'HEALTH': 100.0,
            'AMMO3': 0.0,
            'FRAGCOUNT': 0.0,
            'ARMOR': 0.0,
            'HITCOUNT': 0.0,
            'HITS_TAKEN': 0.0,
            'DEAD': 0.0,
            'DEATHCOUNT': 0.0,
            'DAMAGECOUNT': 0.0,
            'DAMAGE_TAKEN': 0.0,
            'KILLCOUNT': 0.0,
            'SELECTED_WEAPON': 2.0,
            'SELECTED_WEAPON_AMMO': 94.0,
            'POSITION_X': 389.96946716308594,
            'POSITION_Y': 274.2670135498047}
        """
        self._step += 1
        _ = vizdoom_reward, player_id  # unused
        
        def calc_movement(pos_x, pos_y, pos_x_old, pos_y_old):
            return np.sqrt((pos_x - pos_x_old)**2 + (pos_y - pos_y_old)**2).item() # Euclidean distance

        # Combat reward from hits
        rwd_frag = 100.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])
        rwd_hit = 1.0 * (game_var["DAMAGECOUNT"] - game_var_old["DAMAGECOUNT"])
        rwd_hit_taken = -2.0 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
        
        # Movement reward
        pos_x, pos_y = game_var.get("POSITION_X", 0), game_var.get("POSITION_Y", 0)
        pos_x_old, pos_y_old = game_var_old.get("POSITION_X", 0), game_var_old.get("POSITION_Y", 0)
                
        movement_dist = calc_movement(pos_x, pos_y, pos_x_old, pos_y_old)
        #rwd_movement = 0.5 * min((movement_dist if movement_dist else -5e-2) / 100.0, 1.0) # Max movement factor is 1, miniumum is -0.05 (slight punishment if standing still)
        #rwd_movement = 0.25 * min(movement_dist / 100.0, 1.0) # Max movement factor is 1
        rwd_movement = 0.1 * np.tanh(movement_dist / 50.0)  # Smooth saturation
        #if self._step%30 == 0:
        #    self.prev_position[player_id] = (pos_x, pos_y)
        
        # Obstacle detection: Subtract movement if standing still for a long time
        #if self._step > 100 and self._step %30 == 0:
        #    prev_x, prev_y = self.prev_position.get(player_id, (pos_x, pos_y)) ## Use current pos if not in dict
        #    movement_dist = calc_movement(pos_x, pos_y, prev_x, prev_y)
        #    
        #    if movement_dist < 1:
        #        rwd_movement -= 0.5
            
        # Ammo efficiency
        ammo_used = game_var_old.get("SELECTED_WEAPON_AMMO", 0) - game_var.get("SELECTED_WEAPON_AMMO", 0)
        hits_made = game_var["HITCOUNT"] - game_var_old["HITCOUNT"]
        
        if ammo_used > 0: # Shots fired
            accuracy = min(hits_made / ammo_used, 1.0)  # Cap at 100%
            rwd_ammo_efficiency = 2.0 * accuracy
    
        elif ammo_used < 0: # Picked up ammunition
            rwd_ammo_efficiency = 2.0
            
        else:
            rwd_ammo_efficiency = 0.0
            
        # Survival bonus
        #rwd_survival = 0.01 if game_var["HEALTH"] > 0 else -15.0 #1e-3 if game_var["HEALTH"] > 0 else -10.0 #-20 # Moving or surviving improves score slightly
        health_ratio = game_var["HEALTH"] / 100.0
        is_alive = game_var["HEALTH"] > 0
        rwd_survival = (0.01 + 0.005 * health_ratio) if is_alive else -20.0 # THe higher the health percentage the higher the reward, not linearly

        # Bonus point if a reward is picked up
        rwd_health_pickup = +5.0 if game_var["HEALTH"] > game_var_old["HEALTH"] else 0.0
        
        return rwd_frag, rwd_hit, rwd_hit_taken, rwd_movement, rwd_ammo_efficiency, rwd_survival, rwd_health_pickup

class ExtraStates():
    """ Class to store the state informations and calculations. """
    # States in correct order
    EXTRA_STATE_INFOS = {
        "screen": {"dim": 3, "index": 0},
        "labels": {"dim": 1, "index": 3},
        "depth": {"dim": 1, "index": 4},
        "automap": {"dim": 3, "index": 5}
        }
    
    def __init__(self, selected_states: list, num_frames: int = 1):
        self.states = [key for key in self.EXTRA_STATE_INFOS.keys() if key in selected_states]
        self.states_infos = {key: values for key, values in self.EXTRA_STATE_INFOS.items() if key in self.states}
        self.num_states = len(self.states)
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.states)
    
    def get_state_info(self, info: str, return_dict: bool = True) -> dict|list:
        if isinstance(info, str):
            filtered_infos_dict = {key: infos.get(info, None) for key, infos in self.states_infos.items()}
            
            if return_dict:
                return filtered_infos_dict
        
            else:
                return list(filtered_infos_dict.values())
    
    def get_dims(self, return_dict: bool = True) -> dict|list:
        return self.get_state_info("dim", return_dict)
    
    def get_indices(self, return_dict: bool = True) -> dict|list:
        return self.get_state_info("index", return_dict)
    
    def get_channels(self, num_frames: int = None):
        """ TODO: input_channels per state. Currently only for model initialization. """
        if num_frames is not None:
            self.num_frames = num_frames
            
        return {k: v * self.num_frames for k, v in self.get_dims().items()}

class EnvActions():
    action_weights = {
            'Noop': 0.05,
            'Move Forward': 0.18,
            'Attack': 0.2,
            'Move Left': 0.10,
            'Move Right': 0.10,
            'Turn Left': 0.15,
            'Turn Right': 0.15,
            'Jump': 0.05}
    
    def __init__(self, env, seed: int = 149, rng = None) -> None:
        self.set_actions_from_env(env)
        self.action_space = len(self)
        
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        
    def __len__(self) -> int:
        return len(self.actions)
    
    def set_actions_from_env(self, env) -> None:
        self.actions = {0: "Noop"}

        for player_env in env.envs:
            for idx, action in enumerate(player_env.game.get_available_buttons()):
                action_name = str(action).split(".")[1].split(":")[0].replace("_", " ")
                action_val = idx + 1
                
                self.actions[action_val] = action_name.title()
    
    def get_actions(self, return_vals_list: bool = False) -> dict|list:
        if return_vals_list:
            return list(self.actions.values())
        
        return self.actions
    
    def get_action_name(self, action_num: int) -> str:
        return self.actions.get(action_num, "Unknown Action")
    
    @singledispatchmethod
    def get_action_value(self, arg1, arg2):
        """Gets the action value(s) for the given index or list of indices."""
        raise NotImplementedError(f"Cannot get action value for types {type(arg1)} and {type(arg2)}")

    @get_action_value.register
    def _(self, index: int, num=1) -> int|list:    
        all_buttons = list(self.actions.keys())
    
        if num == 1:
            return all_buttons[index]
        
        return [all_buttons[index] for _ in range(num)]
    
    @get_action_value.register
    def _(self, indices: list, num=1) -> list:
        return [self.get_action_value(index) for index in indices]
    
    def get_action_proba(self) -> np.ndarray:
        # Calculate the probability of each action
        action_weight_vals = np.array(list(self.action_weights.values()))
        action_proba = action_weight_vals / action_weight_vals.sum()
        
        return action_proba
        
    def get_random_action(self, n: int = 1, use_proba: bool = True) -> int | list:
        if n <= 0:
            return []

        if use_proba:
            action_proba = self.get_action_proba()
            selection = self.rng.choice(list(self.actions.keys()), p=action_proba, size=n)
        else:
            selection = self.rng.choice(list(self.actions.keys()), size=n)

        if n == 1:
            return selection.item()
        
        return selection.tolist()
    
class ActionCounter:
    """ Class to count the occurrences of each action. """
    def __init__(self):
        self.counts = defaultdict(int)

    def add(self, action: int):
        """ Increments the count for a given action. """
        self.counts[action] += 1

    def get_counts(self) -> dict[int, int]:
        """ Returns the current action counts. """
        return dict(self.counts) # Return a regular dict

    def reset(self):
        """ Resets all action counts. """
        self.counts.clear()
        
    def get_name_counts(self, env_actions: EnvActions) -> dict[str, int]:
        """ Returns the current action counts and names. """
        return {env_actions.get_action_name(key): self.counts.get(key, 0) for key in sorted(self.counts.keys())}
        

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


class Logger():
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
    
    def log(self, msg: str, print_once: bool = False, improve_file_output: bool = False, end="\n"):
        
        if self.also_print or print_once:
            print(msg)
            
        with open(self.file_path, "a") as f:
            if improve_file_output:
                msg = msg.replace("|", ",")
                msg = msg.replace("\t", "")
                msg = msg.replace("  ", "")
                msg = msg.strip()
            
            f.write(msg + end)

    def create_log_file(self):
        # Ensure the directory exists
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
            
        with open(self.file_path, "w") as f:
            f.write(F"LOGGER INITIALIZED AT {datetime.now().strftime('%Y%m%d-%H%M%S')}\n")
             
            
class ActivationLogger(Logger):
    def __init__(self, path: str, filename: str = "activations.txt", also_print: bool = False):
        super().__init__(path, filename, also_print)
        
    def analyze_activations(self, activations: torch.Tensor, episode: int = -1, title: str = "", print_once: bool = False) -> str:
        return f"Episode {episode} | {title:<15}| Shape: {list(activations.shape)},\tMean: {activations.mean().item():.2f},\tStd: {activations.std().item():.2f},\tNorm: {torch.norm(activations).item():.2f}"
            
    @torch.no_grad()
    def log_model_activations(self, obs:tuple[torch.Tensor], model: torch.nn.Module, model_sequence: list = [None, 0, 1, 1], episode: int = -1, print_once: bool = False, return_activations_from_idx:  int = -1):       
        
        orig_device = next(model.parameters()).device
        model.eval().cpu()
        if isinstance(obs, tuple):
            obs = tuple([o.cpu() for o in obs])
        
        elif isinstance(obs, list):
            obs = [o.cpu() for o in obs]
            
        elif isinstance(obs, torch.Tensor):
            obs = obs.cpu()
            
        # Get all modules ot the model except for activation functions        
        all_modules = [name for name, _ in model.named_children() if name != "phi"]
        
        # Create a dictionary with relevant information that stores all data
        module_info = {idx: {"name": module, 
                             "sequence": sequence, 
                             "logits": None} 
                       for idx, (module, sequence) in enumerate(zip(all_modules, model_sequence))}
        
        # Store all texts at one place
        log_str = ""
        
        # Iterate over provided sequences and log 
        for module_idx, module_vals in module_info.items():
            name = module_vals.get("name", "Unknown Module") # Get the name
            sequence = module_vals.get("sequence") # Get the sequence, that is order where to retrieve from
            module: torch.nn.Module = getattr(model, name) # Get the module as instance

            if sequence is None:
                module_input = obs
            else:
                module_input = module_info.get(sequence).get("logits") # If module needs logits form previously get them
            
            module_logits = module.forward(module_input) # Make forward pass
            module_info[module_idx]["logits"] = module_logits # Append forward pass to module info dictionary
            
            log_str += self.analyze_activations(module_logits, episode, name, print_once) + "\n" # Analyze the logits
            
        # Log all values together
        self.log(log_str, print_once, improve_file_output=True)

        # Put model on original device
        model.to(orig_device)
        
        # Return the last X activations if necessary
        if return_activations_from_idx is not None:
            return_idx = len(all_modules) + return_activations_from_idx if return_activations_from_idx < 0 else return_activations_from_idx
            
            all_indices = list(module_info.keys())
            selected_indices = all_indices[return_idx:]
            
            return [value.get("logits") for key, value in module_info.items() if key in selected_indices]
        
    def analyze_weights(self, weights: torch.Tensor, episode: int = -1, title: str = "", layer_type: str = "unknown", print_once: bool = False) -> str:
        """
        Analyze weight statistics for a given layer
        
        Args:
            weights: Weight tensor to analyze
            episode: Episode number
            title: Layer name/title
            layer_type: Type of layer (conv, linear, etc.)
            print_once: Whether to print once or always
        
        Returns:
            Formatted string with weight statistics
        """
        # Basic statistics
        mean_val = weights.mean().item()
        std_val = weights.std().item()
        norm_val = torch.norm(weights).item()
        
        # Advanced statistics for training monitoring
        min_val = weights.min().item()
        max_val = weights.max().item()
        abs_mean = weights.abs().mean().item()
        
        # Gradient flow indicators
        near_zero_ratio = (weights.abs() < 1e-6).float().mean().item()
        large_weight_ratio = (weights.abs() > 1.0).float().mean().item()
        
        # Weight distribution analysis
        q25 = torch.quantile(weights.flatten(), 0.25).item()
        q75 = torch.quantile(weights.flatten(), 0.75).item()
        
        # Sparsity measure (useful for detecting dead neurons)
        sparsity = (weights == 0).float().mean().item()
        
        return f"Episode {episode} | {title:<15}| Type: {layer_type:<6}| Shape: {list(weights.shape)}, Mean: {mean_val:.4f}, Std: {std_val:.4f}, Norm: {norm_val:.2f}, Range: [{min_val:.4f}, {max_val:.4f}], AbsMean: {abs_mean:.4f}, NearZero%: {near_zero_ratio:.2%}, Large%: {large_weight_ratio:.2%}, Q25/75: [{q25:.4f}, {q75:.4f}], Sparsity: {sparsity:.2%}"

        
    @torch.no_grad()
    def log_model_weights(self, model: torch.nn.Module, episode: int = -1, print_once: bool = False, include_bias: bool = True):
        """
        Log weight statistics for all layers in the model
        
        Args:
            model: PyTorch model to analyze
            episode: Episode number
            print_once: Whether to print once or always
            include_bias: Whether to include bias terms in analysis
        """
        orig_device = next(model.parameters()).device
        model.cpu()
        
         # Get all modules ot the model except for activation functions        
        all_modules = [name for name, _ in model.named_children() if name != "phi"]
        
        log_str = ""
        
        for module_name in all_modules:
            model_module: torch.nn.Module = getattr(model, module_name)
            for enc_idx, encoder in enumerate(model_module.modules_list):
                for name, sub_module in encoder.named_modules():
                    if len(list(sub_module.parameters())) > 0:  # Only modules with parameters
                        for param_name, param in sub_module.named_parameters():
                            if 'weight' in param_name or (include_bias and 'bias' in param_name):
                                layer_type = type(sub_module).__name__.lower()
                                full_name = f"enc{enc_idx}_{name}_{param_name}" if name else f"enc{enc_idx}_{param_name}"
                                log_str += self.analyze_weights(param.data, episode, full_name, layer_type, print_once) + "\n"
        
        
        all_weights = torch.cat([p.data.flatten() for p in model.parameters() if p.requires_grad])
        log_str += self.analyze_weights(all_weights, episode, "ALL_WEIGHTS", "global", print_once) + "\n"
        
        log_str += f"{'='*120}\n"
        
        # Log everything
        self.log(log_str, print_once, improve_file_output=True)
        
        # Restore original device
        model.to(orig_device)

                
        