import os
import re
import glob
from datetime import datetime

import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import send_from_directory

# --- CONFIGURATION ---
# The root folder where all your training runs are stored.
RUNS_BASE_FOLDER = 'runs'
# How often the app should refresh, in milliseconds.
REFRESH_INTERVAL_MS = 60000  # 60 seconds

# --- HELPER FUNCTIONS ---

def find_latest_run_folder(base_folder):
    """Finds the most recent directory in the base_folder."""
    try:
        # List all entries in the base folder that are directories
        all_runs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        # Sort them, the default string sort will work for YYYYMMDD-HHMMSS format
        all_runs.sort(reverse=True)
        if all_runs:
            return os.path.join(base_folder, all_runs[0])
    except FileNotFoundError:
        return None
    return None

def parse_log_file(log_file_path):
    """
    Parses the logs.txt file to extract structured data.
    This version is designed to be robust against changes in log entry order
    and multi-line log messages.
    Returns a pandas DataFrame.
    """
    if not os.path.exists(log_file_path):
        return pd.DataFrame()

    # Regex to split the log file into individual, multi-line entries.
    # Each entry starts with a timestamp (HH:MM:SS |). We use a positive
    # lookahead to split *before* the timestamp, keeping it with the entry.
    log_splitter_pattern = re.compile(r'(?=\d{2}:\d{2}:\d{2} \|)')

    # Regex for entries containing step and rollout buffer information.
    steps_pattern = re.compile(
        r"Episode: (\d+).*?Steps done: (\d+).*?Gathering rollout.*?currently (\d+)"
    )

    # Regex for the main multi-line entry containing rewards, loss, and metrics.
    reward_pattern = re.compile(
        r"Episode: (\d+).*?Rewards:.*?"  # Start of the rewards block
        r"Reward: \[(.*?)\]\s*\|\s*"  # Individual rewards
        r"Avg Reward: ([\d\.-]+)\s*\|\s*"  # Average reward
        r"Loss: ([\d\.-]+)\s*\|\s*"  # Loss
        r"ε: ([\d\.-]+)\s*\|\s*"  # Epsilon
        r"LR: ([\d\.e\+-]+).*"  # Learning Rate
        r"Metrics - \[(.*?)\]",  # Metrics list
        re.DOTALL  # Use DOTALL since the reward entry spans multiple lines
    )

    # Use a dictionary keyed by episode number to aggregate data from different log lines.
    data = {}

    with open(log_file_path, 'r') as f:
        content = f.read()

    # Split the entire log content into individual event entries.
    log_entries = log_splitter_pattern.split(content)

    for entry in log_entries:
        if not entry.strip():
            continue

        # Try to match each type of pattern on the current entry.
        steps_match = steps_pattern.search(entry)
        reward_match = reward_pattern.search(entry)

        if steps_match:
            episode = int(steps_match.group(1))
            if episode not in data:
                data[episode] = {'Episode': episode}
            data[episode]['Steps Done'] = int(steps_match.group(2))
            data[episode]['Rollout Buffer'] = int(steps_match.group(3))

        elif reward_match:
            episode = int(reward_match.group(1))
            if episode not in data:
                data[episode] = {'Episode': episode}

            # Unpack all captured groups from the reward_match regex
            reward_str = reward_match.group(2)
            avg_reward = float(reward_match.group(3))
            loss = float(reward_match.group(4))
            epsilon = float(reward_match.group(5))
            lr = float(reward_match.group(6))
            metrics_str = reward_match.group(7)

            data[episode].update({
                'Average Reward (All Players)': avg_reward,
                'Loss': loss,
                'Epsilon': epsilon,
                'Learning Rate': lr
            })

            # Parse the list of individual player rewards
            rewards = [float(r) for r in re.findall(r'[\d\.-]+', reward_str)]
            for i, r in enumerate(rewards):
                data[episode][f'Player {i+1} Reward'] = r

            # Parse the string containing all other metrics
            metrics_pattern = re.compile(r"'([^:]+): ([^']+)'")
            for m_key, m_val in metrics_pattern.findall(metrics_str):
                try:
                    # Convert metrics to float if possible, otherwise keep as string
                    data[episode][m_key.strip()] = float(m_val)
                except ValueError:
                    data[episode][m_key.strip()] = m_val

    if not data:
        return pd.DataFrame()

    # Convert the dictionary of episode data into a list, then a DataFrame
    data_list = sorted(list(data.values()), key=lambda x: x.get('Episode', 0))
    df = pd.DataFrame(data_list)

    # Clean up the DataFrame
    if 'Episode' in df.columns:
        df = df.set_index('Episode')

    # Forward-fill missing 'Steps Done' and 'Rollout Buffer' values. This handles
    # cases where a reward entry for an episode appears before the next steps entry.
    if 'Steps Done' in df.columns:
        df['Steps Done'] = df['Steps Done'].fillna(method='ffill')
    if 'Rollout Buffer' in df.columns:
        df['Rollout Buffer'] = df['Rollout Buffer'].fillna(method='ffill')

    # Drop rows that don't have the core reward data, as they are not plottable
    df = df.dropna(subset=['Average Reward (All Players)'])
    if 'Steps Done' in df.columns: # Ensure integer formatting for display
        df['Steps Done'] = df['Steps Done'].astype(int)
        df['Rollout Buffer'] = df['Rollout Buffer'].astype(int)


    return df.reset_index()


def find_latest_video(folder_path):
    """Finds the most recent video file (.mp4) in a folder."""
    videos = glob.glob(os.path.join(folder_path, '*.mp4'))
    if not videos:
        return None, None
    latest_video = max(videos, key=os.path.getctime)
    return os.path.basename(latest_video), latest_video

def count_pt_files(folder_path):
    """Counts the number of .pt (PyTorch model) files in a folder."""
    return len(glob.glob(os.path.join(folder_path, '*.pt')))


# --- DASH APP INITIALIZATION ---
app = dash.Dash(__name__, title="DRL Training Dashboard")

# The folder for the currently displayed run. This is a global variable
# that will be updated periodically by our update callback.
CURRENT_RUN_FOLDER = find_latest_run_folder(RUNS_BASE_FOLDER)
print(CURRENT_RUN_FOLDER)

# --- APP LAYOUT ---
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f4f4', 'padding': '20px'}, children=[
    html.H1("DRL Training Progress Dashboard", style={'textAlign': 'center', 'color': '#333'}),

    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL_MS,
        n_intervals=0
    ),

    html.Div(id='latest-info-container', style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginBottom': '20px'}),

    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        html.Div(style={'flex': 1, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
            html.H3("Latest Evaluation Video", style={'color': '#555'}),
            html.Div(id='video-container')
        ]),

        html.Div(style={'flex': 1, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
            html.H3("Run Summary", style={'color': '#555'}),
            html.P(id='current-run-folder-text'),
            html.P(id='pt-file-count-text')
        ]),
    ]),

    html.Div(style={'marginTop': '20px'}, children=[
        dcc.Graph(id='avg-reward-chart'),
        dcc.Graph(id='individual-rewards-chart'),
        dcc.Graph(id='metrics-chart'),
        dcc.Graph(id='lr-chart'),
    ])
])


# --- CALLBACKS TO UPDATE THE APP ---

@app.callback(
    [Output('latest-info-container', 'children'),
     Output('video-container', 'children'),
     Output('current-run-folder-text', 'children'),
     Output('pt-file-count-text', 'children'),
     Output('avg-reward-chart', 'figure'),
     Output('individual-rewards-chart', 'figure'),
     Output('metrics-chart', 'figure'),
     Output('lr-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """
    This function is triggered by the interval component and updates all
    the dynamic elements of the dashboard.
    """
    global CURRENT_RUN_FOLDER
    CURRENT_RUN_FOLDER = find_latest_run_folder(RUNS_BASE_FOLDER)

    if not CURRENT_RUN_FOLDER:
        no_data_msg = "No run folder found."
        empty_figure = {'data': [], 'layout': {'title': no_data_msg}}
        video_player = html.Video(controls=True, style={'width': '100%'})
        return (
            [html.H4(no_data_msg)],
            video_player,
            "Current Run: N/A",
            "Checkpoints: N/A",
            empty_figure, empty_figure, empty_figure, empty_figure
        )

    # --- 1. Parse Data and Get File Info ---
    df = parse_log_file(os.path.join(CURRENT_RUN_FOLDER, 'logs.txt'))
    latest_video_name, _ = find_latest_video(CURRENT_RUN_FOLDER)
    pt_file_count = count_pt_files(CURRENT_RUN_FOLDER)

    # Handle case where log file is empty or not yet created
    if df.empty:
        no_data_msg = "Waiting for log data..."
        empty_figure = {'data': [], 'layout': {'title': no_data_msg}}
        video_player = html.Video(controls=True, style={'width': '100%'})
        return (
            [html.H4(no_data_msg)],
            video_player,
            f"Current Run: {os.path.basename(CURRENT_RUN_FOLDER)}",
            f"Found {pt_file_count} model checkpoints (.pt files).",
            empty_figure, empty_figure, empty_figure, empty_figure
        )

    # Get the very last row of data for the summary stats
    latest_data = df.iloc[-1]

    # --- 2. Create Components to Return ---

    # Summary Info
    steps_done_val = latest_data.get('Steps Done', 'N/A')
    steps_done = f"Steps Done: {steps_done_val:,.0f}" if isinstance(steps_done_val, (int, float)) else f"Steps Done: {steps_done_val}"
    avg_reward = f"Avg. Reward: {latest_data.get('Average Reward (All Players)', 0):.2f}"
    rollout_buffer_val = latest_data.get('Rollout Buffer', 0)
    rollout_buffer = f"Rollout Buffer: {rollout_buffer_val:,.0f}" if isinstance(rollout_buffer_val, (int, float)) else f"Rollout Buffer: {rollout_buffer_val}"

    latest_info_children = [html.Div(html.H4(val, style={'textAlign': 'center'})) for val in [steps_done, avg_reward, rollout_buffer]]

    # Video Player
    video_player = html.Video(
        controls=True,
        id='movie_player',
        src=f"/videos/{latest_video_name}" if latest_video_name else "",
        autoPlay=True,
        muted=True,
        style={'width': '100%'}
    )

    # Summary Text
    current_run_text = f"Current Run: {os.path.basename(CURRENT_RUN_FOLDER)}"
    pt_file_count_text = f"Found {pt_file_count} model checkpoints (.pt files)."

    # --- 3. Create Figures ---

    # Average Reward Chart
    avg_reward_fig = px.line(df, x='Episode', y='Average Reward (All Players)', title='Average Reward (All Players) Over Time')
    avg_reward_fig.update_traces(mode='lines+markers')

    # Individual Rewards Chart
    reward_cols = [col for col in df.columns if 'Player' in col and 'Reward' in col]
    individual_rewards_fig = px.line(df, x='Episode', y=reward_cols, title='Individual Player Rewards Over Time')
    individual_rewards_fig.update_traces(mode='lines+markers')

    # Metrics Chart
    metric_cols = ['hits', 'damage_taken', 'movement', 'ammo_efficiency', 'survival', 'health_pickup']
    available_metrics = [m for m in metric_cols if m in df.columns]
    metrics_fig = px.line(df, x='Episode', y=available_metrics, title='Metrics Over Time')
    metrics_fig.update_traces(mode='lines+markers')

    # Learning Rate Chart
    lr_fig = px.line(df, x='Episode', y='Learning Rate', title='Learning Rate Schedule')
    lr_fig.update_traces(mode='lines+markers')

    return (
        latest_info_children,
        video_player,
        current_run_text,
        pt_file_count_text,
        avg_reward_fig,
        individual_rewards_fig,
        metrics_fig,
        lr_fig
    )

# --- Special Route to Serve Video Files ---
@app.server.route('/videos/<video_name>')
def serve_video(video_name):
    """Allows the web app to play videos from the run folder."""
    if CURRENT_RUN_FOLDER:
        return send_from_directory(CURRENT_RUN_FOLDER, video_name)
    return "Error: No run folder specified", 404

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    app.run(debug=True, port=8051)