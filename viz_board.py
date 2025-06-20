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
RUNS_BASE_FOLDER = '/runs'
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
    Returns a pandas DataFrame.
    """
    if not os.path.exists(log_file_path):
        return pd.DataFrame()

    # Regex to capture the key information from each episode log entry
    episode_pattern = re.compile(
        r"Episode: (\d+).*?Steps done: (\d+).*?currently (\d+).*?Rewards:*?\s*Reward: \[(.*?)\].*?Avg Reward: ([\d\.]+).*?Loss: ([\d\.]+).*?ε: ([\d\.]+).*?LR: ([\d\.e\-]+).*?Metrics - \[(.*?)\]",
        re.DOTALL
    )

    data = []
    with open(log_file_path, 'r') as f:
        content = f.read()
        matches = episode_pattern.finditer(content)

        for match in matches:
            episode_data = {
                'Episode': int(match.group(1)),
                'Steps Done': int(match.group(2)),
                'Rollout Buffer': int(match.group(3)),
                'Average Reward (All Players)': float(match.group(5)),
                'Loss': float(match.group(6)),
                'Epsilon': float(match.group(7)),
                'Learning Rate': float(match.group(8)),
            }
            
            # Parse reward lists
            reward_str = match.group(4)
            rewards = [float(r) for r in re.findall(r'[\d\.]+', reward_str)]
            for i, r in enumerate(rewards):
                episode_data[f'Player {i+1} Reward'] = r
            
            # Parse metrics
            metrics_str = match.group(9)
            metrics_pattern = re.compile(r"'([^:]+): ([^']+)'")
            for m_key, m_val in metrics_pattern.findall(metrics_str):
                try:
                    episode_data[m_key.strip()] = float(m_val)
                except ValueError:
                    episode_data[m_key.strip()] = m_val # Keep as string if not a number

            data.append(episode_data)
    
    return pd.DataFrame(data)

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
    
    # This component will trigger the update function every N milliseconds
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL_MS,
        n_intervals=0
    ),

    # Section for Key Metrics
    html.Div(id='latest-info-container', style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginBottom': '20px'}),
    
    # Section for Video and Plots
    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        # Left side: Video Player
        html.Div(style={'flex': 1, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
            html.H3("Latest Evaluation Video", style={'color': '#555'}),
            html.Div(id='video-container')
        ]),
        
        # Right side: Summary and File Count
        html.Div(style={'flex': 1, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
            html.H3("Run Summary", style={'color': '#555'}),
            html.P(id='current-run-folder-text'),
            html.P(id='pt-file-count-text')
        ]),
    ]),
    
    # Section for Graphs
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
        # Default empty state if no run folder is found
        no_data_msg = [html.P("No run folder found.")] * 4
        empty_figure = {'data': [], 'layout': {'title': 'No data'}}
        return no_data_msg + [empty_figure] * 4

    # --- 1. Parse Data and Get File Info ---
    df = parse_log_file(os.path.join(CURRENT_RUN_FOLDER, 'logs.txt'))
    latest_video_name, _ = find_latest_video(CURRENT_RUN_FOLDER)
    pt_file_count = count_pt_files(CURRENT_RUN_FOLDER)
    
    # Handle case where log file is empty or not yet created
    if df.empty:
        no_log_msg = [html.P("Waiting for log data...")] * 4
        empty_figure = {'data': [], 'layout': {'title': 'Waiting for log data...'}}
        return no_log_msg + [empty_figure] * 4

    # Get the very last row of data for the summary stats
    latest_data = df.iloc[-1]
    
    # --- 2. Create Components to Return ---
    
    # Summary Info
    steps_done = f"Steps Done: {latest_data.get('Steps Done', 'N/A'):,}"
    avg_reward = f"Avg. Reward: {latest_data.get('Average Reward (All Players)', 0):.2f}"
    rollout_buffer = f"Rollout Buffer: {latest_data.get('Rollout Buffer', 0):,}" # From log, this seems to be the same as steps done
    
    latest_info_children = [html.Div(html.H4(val, style={'textAlign': 'center'})) for val in [steps_done, avg_reward, rollout_buffer]]

    # Video Player
    video_player = html.Video(
        controls=True,
        id='movie_player',
        # The source points to a special URL that our app will handle
        src=f"/videos/{latest_video_name}" if latest_video_name else "",
        autoPlay=True,
        muted=True, # Autoplay often requires the video to be muted
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
    # Filter for columns that actually exist in the dataframe
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
# This is needed because web browsers cannot directly access local file paths.
@app.server.route('/videos/<video_name>')
def serve_video(video_name):
    """Allows the web app to play videos from the run folder."""
    if CURRENT_RUN_FOLDER:
        return send_from_directory(CURRENT_RUN_FOLDER, video_name)
    return "Error: No run folder specified", 404

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Run the Dash app
    # debug=True allows for hot-reloading when you save the script,
    # but it's recommended to turn it off for a stable deployment.
    app.run(debug=True)