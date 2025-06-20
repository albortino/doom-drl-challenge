  
import torch 
import os
from datetime import datetime
import numpy as np
from typing import Dict, Tuple
from doom_arena.reward import VizDoomReward

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
        * +2 for movement (exploration)
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

        # Combat reward from hits
        rwd_frag = 100.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])
        rwd_hit = 10.0 * (game_var["HITCOUNT"] - game_var_old["HITCOUNT"])
        rwd_hit_taken = -2 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
        
        # Movement reward
        pos_x = game_var.get("POSITION_X", 0)
        pos_y = game_var.get("POSITION_Y", 0)
        pos_x_old = game_var_old.get("POSITION_X", 0)
        pos_y_old = game_var_old.get("POSITION_Y", 0)
                
        movement_dist = np.sqrt((pos_x - pos_x_old)**2 + (pos_y - pos_y_old)**2) # Euclidean distance
        rwd_movement = 2 * min((movement_dist if movement_dist else -5e-2) / 100.0, 1.0) # Max movement factor is 1, miniumum is -0.05 (slight punishment if standing still)
        
        # Ammo efficiency
        ammo_used = game_var_old.get("SELECTED_WEAPON_AMMO", 0) - game_var.get("SELECTED_WEAPON_AMMO", 0)
        hits_made = game_var["HITCOUNT"] - game_var_old["HITCOUNT"]
        
        if ammo_used > 0: # Shots fired
            accuracy = hits_made / ammo_used
            rwd_ammo_efficiency = 1.0 * accuracy
            
        elif ammo_used < 0: # Picked up ammunition
            rwd_ammo_efficiency = 2.0
            
        else:
            rwd_ammo_efficiency = 0.0
            
        # Survival bonus
        rwd_survival = 3e-2 if game_var["HEALTH"] > 0 else 0 #-20 # Moving AND surviving improves score slightly
        
        # Bonus point if a reward is picked up
        rwd_health_pickup = +2.0 if game_var["HEALTH"] > game_var_old["HEALTH"] else 0.0
        
        return rwd_frag, rwd_hit, rwd_hit_taken, rwd_movement, rwd_ammo_efficiency, rwd_survival, rwd_health_pickup
    

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