from typing import Dict
import torch
import numpy as np
from datetime import datetime
from doom_arena.render import render_episode
from IPython.display import HTML
import os
import pandas as pd
import contextlib
from agents.dqn import epsilon_greedy
from agents.helpers import EnvActions, ActivationLogger, ExtraStates
import onnx
import json

def onnx_dump(env, model, config, filename: str):
    # dummy state
    with suppress_output():
        init_state = env.reset()[0].unsqueeze(0)

    orig_device = next(model.parameters()).device
    
    model.cpu()
    
    # Export to ONNX
    torch.onnx.export(
        model.cpu(),
        args=init_state,
        f=filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    onnx_model = onnx.load(filename)

    meta = onnx_model.metadata_props.add()
    meta.key = "config"
    meta.value = json.dumps(config)

    onnx.save(onnx_model, filename)
    
    model.to(orig_device)
    
    

def analyze_model_tensors(logger: ActivationLogger, episode: int, obs_clean: tuple[torch.Tensor], model: torch.nn.Module, split_dims: list, model_sequence: list = [None, 0, 1, 1], print_once: bool = False, dtype=torch.float32):    
    rand_tensor = torch.rand(size=torch.stack(obs_clean).shape).to("cpu").split(split_dims, dim=1)
    
    value, advantage = logger.log_model_activations(rand_tensor, model, model_sequence, episode=episode, return_activations_from_idx=-2, print_once=print_once)
    q_values = (value + advantage - advantage.mean(dim=1, keepdim=True)).to(dtype=dtype)
    q_actions = q_values.argmax(dim=1).to(dtype=dtype)
    
    log_qvals = logger.analyze_activations(q_values, episode, title="Qvalues", print_once=print_once)
    log_qacts = logger.analyze_activations(q_actions, episode, title="Actions", print_once=print_once)
    logger.log(log_qvals + "\n" + log_qacts, improve_file_output=True)
    
    
def replay_episode(env, model, device, dtype, path: str = "", store: bool = False, random_player: bool = True):
    # ----------------------------------------------------------------
    # Hint for replay visualisation:
    # ----------------------------------------------------------------

    env.enable_replay()
    model.eval()

    # Tracking reward components
    eval_reward = 0.0

    # Reset environment
    with suppress_output():
        eval_obs = env.reset()
    
    eval_dones = [False]

    while not all(eval_dones):
        env_actions = EnvActions(env)
        
        eval_act = epsilon_greedy(env, model, eval_obs, 0, env_actions, device, dtype=dtype)
        eval_obs, reward_components, eval_dones, _ = env.step(eval_act)
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
        _ = render_episode(replays, subsample=5, replay_path=path)
    else:
        return HTML(render_episode(replays, subsample=5).to_html5_video())

    model.train()
    
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
    