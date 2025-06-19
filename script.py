# %% [markdown]
# # Deep Reinforcement Learning — Doom Agent (SS2025)
# 
# Welcome to the last assignment for the **Deep Reinforcement Learning** course (SS2025). In this notebook, you'll implement and train a reinforcement learning agent to play **Doom**.
# 
# You will:
# - Set up a custom VizDoom environment with shaped rewards
# - Train an agent using an approach of your choice
# - Track reward components across episodes
# - Evaluate the best model
# - Visualize performance with replays and GIFs
# - Export the trained agent to ONNX to submit to the evaluation server

# %%
# Clone repo
#!git clone https://$token@github.com/gerkone/jku.wad.git
#%cd jku.wad

# %%
# Install the dependencies
#!pip install torch numpy matplotlib vizdoom portpicker gym onnx

# %%
from typing import Dict, Sequence

import torch
from collections import deque, OrderedDict
from copy import deepcopy
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import vizdoom as vzd
from vizdoom import ScreenFormat

from gym import Env
from torch import nn
from einops import rearrange

from doom_arena import VizdoomMPEnv
from doom_arena.reward import VizDoomReward
from doom_arena.render import render_episode
from IPython.display import HTML
from typing import Dict, Tuple
from doom_arena.reward import VizDoomReward

# %%
from agents.dqn import epsilon_greedy, update_ema

# %% [markdown]
# ## Environment configuration
# 
# ViZDoom supports multiple visual buffers that can be used as input for training agents. Each buffer provides different information about the game environment, as seen from left to right:
# 
# 
# Screen
# - The default first-person RGB view seen by the agent.
# 
# Labels
# - A semantic map where each pixel is tagged with an object ID (e.g., enemy, item, wall).
# 
# Depth
# - A grayscale map showing the distance from the agent to surfaces in the scene.
# 
# Automap
# - A top-down schematic view of the map, useful for global navigation tasks.
# 
# ![buffers gif](https://vizdoom.farama.org/_images/vizdoom-demo.gif)

# %%
USE_GRAYSCALE = False  # ← flip to False for RGB

PLAYER_CONFIG = {
    # NOTE: "algo_type" defaults to POLICY in evaluation script!
    "algo_type": "QVALUE",  # OPTIONAL, change to POLICY if using policy-based (eg PPO)
    "n_stack_frames": 1,
    "extra_state": ["depth"],
    "hud": "none",
    "crosshair": True,
    "screen_format": 8 if USE_GRAYSCALE else 0,
}

# %%
# TODO: environment training paramters
N_STACK_FRAMES = 1
NUM_BOTS = 4
EPISODE_TIMEOUT = 1000
# TODO: model hyperparams
GAMMA = 0.95
EPISODES = 100
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
N_EPOCHS = 50

# %% [markdown]
# ## Reward function
# In this task, you will define a reward function to guide the agent's learning. The function is called at every step and receives the current and previous game variables (e.g., number of frags, hits taken, health).
# 
# Your goal is to combine these into a meaningful reward, encouraging desirable behavior, such as:
# 
# - Rewarding frags (enemy kills)
# 
# - Rewarding accuracy (hitting enemies)
# 
# - Penalizing damage taken
# 
# - (Optional) Encouraging survival, ammo efficiency, etc.
# 
# You can return multiple reward components, which are summed during training. Consider the class below as a great starting point!

# %%
class YourReward(VizDoomReward):
    def __init__(self, num_players: int):
        super().__init__(num_players)

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple[float, float, float]:
        """
        Custom reward function used by both training and evaluation.
        *  +100  for every new frag
        *  +2    for every hit landed
        *  -0.1  for every hit taken
        """
        self._step += 1
        _ = vizdoom_reward, player_id  # unused

        rwd_hit = 2.0 * (game_var["HITCOUNT"] - game_var_old["HITCOUNT"])
        rwd_hit_taken = -0.1 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
        rwd_frag = 100.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])

        return rwd_hit, rwd_hit_taken, rwd_frag

# %%
device = "mps" #"cuda"
DTYPE = torch.float32

reward_fn = YourReward(num_players=1)

env = VizdoomMPEnv(
    num_players=1,
    num_bots=NUM_BOTS,
    bot_skill=0,
    doom_map="ROOM",  # NOTE simple, small map; other options: TRNM, TRNMBIG
    extra_state=PLAYER_CONFIG[
        "extra_state"
    ],  # see info about states at the beginning of 'Environment configuration' above
    episode_timeout=EPISODE_TIMEOUT,
    n_stack_frames=PLAYER_CONFIG["n_stack_frames"],
    crosshair=PLAYER_CONFIG["crosshair"],
    hud=PLAYER_CONFIG["hud"],
    screen_format=PLAYER_CONFIG["screen_format"],
    reward_fn=reward_fn,
)

# %% [markdown]
# ## Agent
# 
# Implement **your own agent** in the code cell that follows.
# 
# * In `agents/dqn.py` and `agents/ppo.py` you’ll find very small **skeletons**—they compile but are meant only as reference or quick tests.  
#   Feel free to open them, borrow ideas, extend them, or ignore them entirely.
# * The notebook does **not** import those files automatically; whatever class you define in the next cell is the one that will be trained.
# * You may keep the DQN interface, switch to PPO, or try something else.
# * Tweak any hyper-parameters (`PLAYER_CONFIG`, ε-schedule, optimiser, etc.) and document what you tried.
# 

# %%
# ================================================================
# DQN — design your network here
# ================================================================


class DQN(nn.Module):
    """
    Deep-Q Network template.

    Expected behaviour
    ------------------
    forward(frame)      # frame: (B, C, H, W)  →  Q-values: (B, num_actions)

    What to add / change
    --------------------
    • Replace the two `NotImplementedError` lines.
    • Build an encoder (Conv2d / Conv3d) + a head (MLP or duelling).
    • Feel free to use residual blocks from `agents/utils.py` or any design you like.
    """

    def __init__(self, input_dim: int, action_space: int, hidden: int = 128):
        super().__init__()

        # -------- TODO: define your layers ------------------------
        # Example (very small) baseline — delete or improve:
        #
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, 8, stride=4, padding=2), nn.ReLU(), # 32
            nn.Conv2d(32, 64, 4, stride=2, padding=1),       nn.ReLU(), # 16
            nn.Conv2d(64, 128, 3, stride=2),       nn.ReLU(), #7
            nn.Conv2d(128, 128, 3, stride=2),       nn.ReLU(), #3
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden), nn.ReLU(),
            #nn.Linear(64 * 3 * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, action_space**2), nn.ReLU(),
            nn.Linear(action_space**2, action_space),
        )
        #
        # -----------------------------------------------------------
        #raise NotImplementedError("Define DQN layers")

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        # -------- TODO: implement forward -------------------------
        #
        x = self.encoder(frame)
        x = self.head(x)
        return x
        #
        # -----------------------------------------------------------
        #raise NotImplementedError("Implement forward pass")

# %%
# ================================================================
# Initialise your networks and training utilities
# ================================================================

# main Q-network
in_channels = env.observation_space.shape[0]  # 1 if grayscale, else 3/4
model = DQN(
    input_dim=in_channels,
    action_space=env.action_space.n,
    hidden=2048,  # change or ignore
).to(device, dtype=DTYPE)

# TODO ------------------------------------------------------------
# 1. Create a target network (hard-copy or EMA)
# 2. Choose an optimiser + learning-rate schedule
# 3. Instantiate a replay buffer and set the initial epsilon value
#
# Hints:
#   model_tgt  = deepcopy(model).to(device)
#   optimiser  = torch.optim.Adam(...)
#   scheduler  = torch.optim.lr_scheduler.ExponentialLR(...)
#   replay_buf = collections.deque(maxlen=...)
# ---------------------------------------------------------------


model_tgt  = deepcopy(model).to(device)
optimizer  = torch.optim.AdamW(model_tgt.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-10)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

#raise NotImplementedError("Create target net, optimiser, scheduler, replay buffer")

# %% [markdown]
# ## Training loop

# %%
# ---------------------  TRAINING LOOP  ----------------------
# Feel free to change EVERYTHING below:
#   • choose your own reward function
#   • track different episode statistics in `ep_metrics`
#   • switch optimiser, scheduler, update rules, etc.

reward_list, q_loss_list = [], []
best_eval_return, best_model = float("-inf"), None

epsilon = EPSILON_START

for episode in range(EPISODES):
    ep_metrics = {"custom_reward": 0.0}  # ← add or replace keys as you like
    obs = env.reset()[0]
    done, ep_return = False, 0.0
    model.eval()

    # ───────── rollout ─────────────────────────────────────────────
    while not done:
        act = epsilon_greedy(env, model, obs, epsilon, device, DTYPE)
        next_obs, rwd_raw, done, _ = env.step(act)

        # ----- reward definition (EDIT here) ----------------
        custom_rwd = float(rwd_raw[0])  # default: raw env reward
        # Example: access game variables for more detailed reward engineering
        # gv, gv_pre = env.envs[0].unwrapped._game_vars, env.envs[0].unwrapped._game_vars_pre
        # custom_rwd = your_function(gv, gv_pre)

        ep_metrics["custom_reward"] += custom_rwd

        replay_buffer.append((obs, act, custom_rwd, next_obs[0], done))
        obs, ep_return = next_obs[0], ep_return + custom_rwd
    reward_list.append(ep_return)

    # ───────── learning step (experience replay) ──────────────────
    if len(replay_buffer) >= BATCH_SIZE:
        model.train()
        for _ in range(N_EPOCHS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            s, a, r, s2, d = zip(*batch)

            s = torch.stack(s).to(device, dtype=DTYPE)
            s2 = torch.stack(s2).to(device, dtype=DTYPE)
            a = torch.tensor(a, device=device)
            r = torch.tensor(r, device=device, dtype=torch.float32)
            d = torch.tensor(d, device=device, dtype=torch.float32)

            q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q2 = model_tgt(s2).max(1).values
                tgt = r + GAMMA * q2 * (1 - d)
            loss = F.mse_loss(q, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            q_loss_list.append(loss.item())
        update_ema(model_tgt, model)

    scheduler.step()
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Ep {episode+1:03}: return {ep_return:6.1f}  |  ε {epsilon:.3f}")

    # ───────── quick evaluation for best-model tracking ───────────
    eval_obs, done, eval_return = env.reset()[0], False, 0.0
    model.eval()
    while not done:
        act = epsilon_greedy(env, model, eval_obs, 0.05, device, DTYPE)
        eval_obs_n, r, done, _ = env.step(act)
        eval_obs = eval_obs_n[0]
        eval_return += r[0]
    if eval_return > best_eval_return:
        best_eval_return, best_model = eval_return, deepcopy(model)

# ---------------------  SAVE / EXPORT ---------------------------------------
final_model = best_model if best_model is not None else model  # choose best

# %% [markdown]
# ## Dump to ONNX

# %%
import onnx
import json


def onnx_dump(env, model, config, filename: str):
    # dummy state
    init_state = env.reset()[0].unsqueeze(0)

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


onnx_dump(env, final_model, PLAYER_CONFIG, filename="model.onnx")
print("Best network exported to model.onnx")

# %% [markdown]
# ### Evaluation and Visualization
# 
# In this final section, you can evaluate your trained agent, inspect its performance visually, and analyze reward components over time.
# 

# %%
# ---------------------------------------------------------------
#  Reward-plot helper  (feel free to edit / extend)
# ---------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


def plot_reward_components(reward_log, smooth_window: int = 5):
    """
    Plot raw and smoothed episode-level reward components.

    Parameters
    ----------
    reward_log : list[dict]
        Append a dict for each episode, e.g. {"frag": …, "hit": …, "hittaken": …}
    smooth_window : int
        Rolling-mean window size for the smoothed curve.
    """
    if not reward_log:
        print("reward_log is empty – nothing to plot.")
        return

    df = pd.DataFrame(reward_log)
    df_smooth = df.rolling(window=smooth_window, min_periods=1).mean()

    # raw
    plt.figure(figsize=(12, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.title("Raw episode reward components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # smoothed
    plt.figure(figsize=(12, 5))
    for col in df.columns:
        plt.plot(df.index, df_smooth[col], label=f"{col} (avg)")
    plt.title(f"Smoothed (window={smooth_window})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------
# Hint for replay visualisation:
# ----------------------------------------------------------------
env.enable_replay()
# ... run an evaluation episode ..

reward_log = list()
# Reset environment
obs = env.reset()[0]
done = False
model.eval()

# Tracking reward components
reward_components = {"frag": 0.0, "hit": 0.0, "hittaken": 0.0}  # Add/remove keys as needed
ep_return = 0.0

# Run episode
while not done:
    act = epsilon_greedy(env, model, obs, epsilon=0.0, device=device, dtype=DTYPE)
    next_obs, r, done, info = env.step(act)

    # Accumulate reward (adapt keys depending on env)
    if isinstance(info, dict):  # single env
        for k in reward_components:
            reward_components[k] += info.get(k, 0.0)
    elif isinstance(info, list):  # vectorized env
        for k in reward_components:
            reward_components[k] += info[0].get(k, 0.0)

    ep_return += r[0]
    obs = next_obs[0]

# Finalize episode
env.disable_replay()
reward_log.append(reward_components)

from doom_arena.render import render_episode

from IPython.display import HTML
replays = env.get_player_replays()
HTML(render_episode(replays, subsample=5).to_html5_video())
#
# Feel free to adapt or write your own GIF/MP4 export.


