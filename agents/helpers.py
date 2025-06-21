  
import torch 
import os
from datetime import datetime
import numpy as np
from typing import Dict, Tuple
from doom_arena.reward import VizDoomReward
from collections import defaultdict
from functools import singledispatchmethod

class YourReward(VizDoomReward):
    def __init__(self, num_players: int):
        super().__init__(num_players)
        self.prev_ammo = {}
        self.prev_health = {}
        self.prev_position = {}
        self.survival_bonus = 0
        
    def __call__(self, vizdoom_reward: float, game_var: Dict[str, float], game_var_old: Dict[str, float], player_id: int) -> Tuple:
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
        rwd_hit = 2.0 * (game_var["HITCOUNT"] - game_var_old["HITCOUNT"])
        rwd_hit_taken = -0.5 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
        
        # Movement reward
        pos_x = game_var.get("POSITION_X", 0)
        pos_y = game_var.get("POSITION_Y", 0)
        pos_x_old = game_var_old.get("POSITION_X", 0)
        pos_y_old = game_var_old.get("POSITION_Y", 0)
                
        movement_dist = calc_movement(pos_x, pos_y, pos_x_old, pos_y_old)
        rwd_movement = 1 * min((movement_dist if movement_dist else -5e-1) / 10.0, 1.0) # Max movement factor is 1, miniumum is -0.05 (slight punishment if standing still)
        
        if self._step%30 == 0:
            self.prev_position[player_id] = (pos_x, pos_y)
        
        # Obstacle detection: Subtract movement if standing still for a long time
        if self._step > 100 and self._step %30 == 0:
            prev_x, prev_y = self.prev_position.get(player_id)
            movement_dist = calc_movement(pos_x, pos_y, prev_x, prev_y)
            
            if movement_dist < 1:
                rwd_movement -= 0.1
            
        # Ammo efficiency
        ammo_used = game_var_old.get("SELECTED_WEAPON_AMMO", 0) - game_var.get("SELECTED_WEAPON_AMMO", 0)
        hits_made = game_var["HITCOUNT"] - game_var_old["HITCOUNT"]
        
        if ammo_used > 0: # Shots fired
            accuracy = hits_made / ammo_used
            rwd_ammo_efficiency = 1 * accuracy
            
        elif ammo_used < 0: # Picked up ammunition
            rwd_ammo_efficiency = 2.0
            
        else:
            rwd_ammo_efficiency = 0.0
            
        # Survival bonus
        rwd_survival = -1e-3 if game_var["HEALTH"] > 0 else -10.0 #-20 # Moving AND surviving improves score slightly
        
        # Bonus point if a reward is picked up
        rwd_health_pickup = +3.0 if game_var["HEALTH"] > game_var_old["HEALTH"] else 0.0
        
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
            'Move Forward': 0.1,
            'Attack': 0.2,
            'Move Left': 0.15,
            'Move Right': 0.15,
            'Turn Left': 0.2,
            'Turn Right': 0.2,
            'Jump': 0.1}
    
    def __init__(self, env, seed: int = 149, rng: np.random.default_rng = None) -> None:
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
        return self.actions.get(action_num)
    
    @singledispatchmethod
    def get_action_value(self, arg):
        """Gets the action value(s) for the given index or list of indices."""
        raise NotImplementedError(f"Cannot get action value for type {type(arg)}")

    @get_action_value.register
    def _(self, index: int) -> int:    
        all_buttons = list(self.actions.keys())
        return all_buttons[index]

    @get_action_value.register
    def _(self, indices: list) -> list:
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

    def get_counts(self) -> Dict[int, int]:
        """ Returns the current action counts. """
        return dict(self.counts) # Return a regular dict

    def reset(self):
        """ Resets all action counts. """
        self.counts.clear()

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