from typing import Dict, List
import torch
from collections import OrderedDict
import numpy as np
from datetime import datetime
from doom_arena.render import render_episode
from IPython.display import HTML
import os
import pandas as pd
import contextlib
from agents.dqn import epsilon_greedy
from agents.helpers import EnvActions

# def get_buttons(env, return_vals_list: bool = False) -> dict|list:
#     """ Returns a dictionary or list with the button key and description """
#     buttons = {0: "Noop"}

#     for player_env in env.envs:
#         for idx, button in enumerate(player_env.game.get_available_buttons()):
#             button_name = str(button).split(".")[1].split(":")[0].replace("_", " ")
#             button_val = idx + 1
            
#             buttons[button_val] = button_name.title()
        
#     if return_vals_list:
#         return list(buttons.keys())
    
#     return buttons
    
def replay_episode(env, model, device, dtype, path: str = "", store: bool = False, random_player: bool = True):
    # ----------------------------------------------------------------
    # Hint for replay visualisation:
    # ----------------------------------------------------------------

    env.enable_replay()

    # Tracking reward components
    eval_reward = 0.0

    # Reset environment
    with suppress_output():
        eval_obs = env.reset()
    
    eval_done = False
    model.cpu()

    while not eval_done:
        env_actions = EnvActions(env)
        
        eval_act = epsilon_greedy(env, model, eval_obs, 0, env_actions, "cpu", dtype=dtype)
        eval_obs, reward_components, eval_done, _ = env.step(eval_act)
        eval_reward += sum(reward_components)

    print(f"Final evaluation - Total reward: {eval_reward:.1f}")

    # Finalize episode
    env.disable_replay()

    replays = env.get_player_replays()

    if random_player:
        # Random Player
        player_idx = np.random.randint(0, len(replays))
        player_name = list(replays.keys())[player_idx]
        replays = {player_name: replays.get(player_name)}
    
    if store:
        path = os.path.join(path, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{eval_reward:.0f}.mp4")
        render_episode(replays, subsample=5, replay_path=path)
    else:
        HTML(render_episode(replays, subsample=5).to_html5_video())

    model.train()
    model.to(device)
    
    
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
        

@contextlib.contextmanager
def suppress_output():
    """Suppress both stdout and stderr, including output from C extensions."""
    with open(os.devnull, 'w') as devnull:
        # Save original file descriptors
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)

        try:
            # Redirect stdout and stderr to devnull
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)

            # Also suppress Python-level stdout/stderr
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            # Restore original file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
    