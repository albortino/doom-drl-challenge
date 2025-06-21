import os
import re
import glob
from datetime import datetime

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from flask import send_from_directory

# --- CONFIGURATION ---
# The root folder where all your training runs are stored.
RUNS_BASE_FOLDER = 'runs'
# How often the app should automatically refresh data, in milliseconds.
REFRESH_INTERVAL_MS = 30000  # 30 seconds

# --- DATA PARSING HELPER FUNCTIONS ---

def get_run_folders(base_folder):
    """Finds all valid run directories, sorted from newest to oldest."""
    try:
        all_runs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        all_runs.sort(reverse=True)
        return all_runs
    except FileNotFoundError:
        return []

def get_run_duration(log_file_path):
    """Parses a log file to find the start time and total duration of the run."""
    if not os.path.exists(log_file_path):
        return "N/A", "N/A"

    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2})')
    timestamps = []
    with open(log_file_path, 'r') as f:
        for line in f:
            match = time_pattern.search(line)
            if match:
                timestamps.append(datetime.strptime(match.group(1), '%H:%M:%S'))

    if not timestamps:
        return "N/A", "N/A"

    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration = end_time - start_time
    
    return start_time.strftime('%H:%M:%S'), str(duration)


def parse_log_file(log_file_path):
    """
    Parses the logs.txt file to extract structured training data into a DataFrame.
    This function handles the complex, multi-line log entry format.
    """
    if not os.path.exists(log_file_path):
        return pd.DataFrame()

    # Regex to split the file content into individual log entries, starting with a timestamp.
    log_splitter_pattern = re.compile(r'(?=\d{2}:\d{2}:\d{2} \|)')
    
    # Regex to find lines indicating the start of a new episode's data gathering phase.
    steps_pattern = re.compile(r"Episode: (\d+).*?Steps done: (\d+).*?currently (\d+)")
    
    # A comprehensive regex to capture the main block of training results for an episode.
    # It uses re.DOTALL to allow '.' to match newlines within the block.
    reward_pattern = re.compile(
        r"Episode: (\d+).*?Rewards:.*?"
        # Capture Reward: [values] -> Extracts the numbers inside the brackets.
        r"Reward: \[(.*?)\]\s*\|\s*"
        # Capture Avg Reward: value
        r"Avg Reward: ([\d\.-]+)\s*\|\s*"
        # Capture Loss: value
        r"Loss: ([\d\.-]+)\s*\|\s*"
        # Capture ε (Epsilon): value
        r"ε: ([\d\.-]+)\s*\|\s*"
        # Capture LR (Learning Rate): value (handles scientific notation)
        r"LR: ([\d\.e\+-]+).*?"
        # Capture Metrics: [list of 'key: value']
        r"Metrics - \[(.*?)\]\s*"
        # Capture Actions - key: value, ...
        r"Actions - (.*)",
        re.DOTALL
    )

    data = {}
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
    except IOError:
        return pd.DataFrame()

    log_entries = log_splitter_pattern.split(content)
    for entry in log_entries:
        if not entry.strip():
            continue

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
            
            (reward_str, avg_reward, loss, epsilon, lr,
             metrics_str, action_counts_str) = reward_match.groups()[1:]

            data[episode].update({
                'Average Reward': float(avg_reward),
                'Loss': float(loss),
                'Epsilon': float(epsilon),
                'Learning Rate': float(lr)
            })

            # Parse individual player rewards
            rewards = [float(r) for r in re.findall(r'[\d\.-]+', reward_str)]
            for i, r in enumerate(rewards):
                data[episode][f'Player {i+1} Reward'] = r

            # Parse metrics
            metrics_pattern = re.compile(r"'([^:]+): ([^']+)'")
            for m_key, m_val in metrics_pattern.findall(metrics_str):
                try:
                    data[episode][m_key.strip()] = float(m_val)
                except ValueError:
                    data[episode][m_key.strip()] = m_val

            # Parse action counts
            if action_counts_str:
                for item in action_counts_str.split(','):
                    if ':' in item:
                        action_id, count = item.split(':')
                        data[episode][f'Action_{action_id.strip()}'] = int(count)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(sorted(list(data.values()), key=lambda x: x.get('Episode', 0)))
    
    # Forward-fill steps data and drop rows without reward info to clean the dataset
    if 'Steps Done' in df.columns:
        df['Steps Done'] = df['Steps Done'].ffill().astype(int)
    if 'Rollout Buffer' in df.columns:
        df['Rollout Buffer'] = df['Rollout Buffer'].ffill().astype(int)
    df = df.dropna(subset=['Average Reward'])

    return df.reset_index(drop=True)


def parse_activations_file(activations_file_path):
    """
    Parses the activations.txt file to extract layer statistics using regex.
    Returns a pandas DataFrame with columns: Episode, Layer, Mean, Std, Norm.
    """
    if not os.path.exists(activations_file_path):
        return pd.DataFrame()

    # Regex to capture the required fields from each line.
    # Example line: "Episode 0 , encoders , Shape: [2, 256],Mean: -0.00,Std: 0.07,Norm: 1.55"
    pattern = re.compile(
        r"Episode\s+(\d+)\s+,"      # Capture Episode number
        r"\s+([\w_]+)\s+,"         # Capture Layer name (e.g., "encoders", "Qvalues")
        r".*?Mean:\s+([-\d\.]+)"   # Capture Mean value
        r",Std:\s+([-\d\.]+)"      # Capture Std value
        r",Norm:\s+([-\d\.]+)"     # Capture Norm value
    )
    
    data = []
    try:
        with open(activations_file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    episode, layer, mean, std, norm = match.groups()
                    data.append({
                        'Episode': int(episode),
                        'Layer': layer.strip(),
                        'Mean': float(mean),
                        'Std': float(std),
                        'Norm': float(norm)
                    })
    except (IOError, ValueError):
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    # Aggregate stats by Episode and Layer to reduce noise
    df = pd.DataFrame(data)
    return df.groupby(['Episode', 'Layer']).mean().reset_index()


def find_all_videos(run_folder_path):
    """Finds all video files (.mp4) in a folder, sorted from oldest to newest."""
    if not run_folder_path or not os.path.isdir(run_folder_path):
        return []
    videos = glob.glob(os.path.join(run_folder_path, '*.mp4'))
    videos.sort(key=os.path.getmtime)
    return [os.path.basename(v) for v in videos]

# --- DASH APP ---
app = dash.Dash(__name__, title="DRL Training Dashboard")
server = app.server

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f2f5', 'padding': '20px'}, children=[
    # Stores for holding state without global variables
    dcc.Store(id='selected-run-store'),
    dcc.Store(id='video-state-store'),
    
    # --- HEADER ---
    html.H1("DRL Training Progress Dashboard", style={'textAlign': 'center', 'color': '#1f2937'}),
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MS, n_intervals=0),

    # --- RUN SELECTOR ---
    html.Div([
        html.Label("Select Training Run:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(id='run-selector-dropdown', placeholder="Select a run...", style={'flex': '1'}),
        html.Button("Refresh List", id='refresh-runs-btn', style={'marginLeft': '10px'})
    ], style={'display': 'flex', 'alignItems': 'center', 'maxWidth': '800px', 'margin': '20px auto'}),
    
    # --- MAIN CONTENT AREA ---
    html.Div(id='main-content-area', children=[
        # This area will be populated by the callback
    ])
])

# --- CALLBACKS ---

@app.callback(
    Output('run-selector-dropdown', 'options'),
    Output('run-selector-dropdown', 'value'),
    Input('refresh-runs-btn', 'n_clicks')
)
def update_run_list(_):
    """Populates the dropdown with available run folders."""
    run_folders = get_run_folders(RUNS_BASE_FOLDER)
    options = [{'label': folder, 'value': folder} for folder in run_folders]
    # Set the default value to the most recent run
    default_value = run_folders[0] if run_folders else None
    return options, default_value

@app.callback(
    Output('selected-run-store', 'data'),
    Input('run-selector-dropdown', 'value')
)
def store_selected_run(selected_run_name):
    """Stores the selected run name for other callbacks to use."""
    if not selected_run_name:
        return None
    return {'run_folder': os.path.join(RUNS_BASE_FOLDER, selected_run_name)}

@app.callback(
    Output('main-content-area', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('selected-run-store', 'data')
)
def render_main_content(n, stored_run_data):
    """Renders the entire dashboard layout below the dropdown."""
    if not stored_run_data or not stored_run_data.get('run_folder'):
        return html.H3("Please select a training run to view its data.", style={'textAlign': 'center'})

    # This layout is re-rendered to ensure all components are fresh
    return html.Div([
        # --- TOP INFO CARDS ---
        html.Div(id='latest-info-container', style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        
        # --- SUMMARY AND VIDEO PLAYER ---
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}, children=[
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
                html.H3("Run Summary", style={'color': '#374151', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),
                html.P(id='run-name-text'),
                html.P(id='pt-file-count-text'),
                html.P(id='run-duration-text'),
            ]),
            html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}, children=[
                html.H3("Evaluation Videos", style={'color': '#374151', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'gap': '15px', 'marginBottom': '10px'}, children=[
                    html.Button('⬅️ Previous', id='prev-video-btn', n_clicks=0),
                    html.P(id='video-info-text', style={'fontWeight': 'bold', 'margin': '0'}),
                    html.Button('Next ➡️', id='next-video-btn', n_clicks=0),
                ]),
                html.Div(id='video-container')
            ]),
        ]),
        
        # --- GRAPHS ---
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            dcc.Graph(id='avg-reward-chart'),
            dcc.Graph(id='lr-epsilon-chart'),
            dcc.Graph(id='individual-rewards-chart'),
            dcc.Graph(id='metrics-chart'),
        ]),
        html.H3("Action Distribution", style={'textAlign': 'center', 'marginTop': '30px'}),
        dcc.Graph(id='action-distribution-chart'),
        
        html.H3("Activation Statistics", style={'textAlign': 'center', 'marginTop': '30px'}),
        dcc.Graph(id='activation-mean-chart'),
        dcc.Graph(id='activation-norm-chart'),
        dcc.Graph(id='activation-action-chart'),
    ])

@app.callback(
    # --- Top Info Cards ---
    Output('latest-info-container', 'children'),
    # --- Run Summary ---
    Output('run-name-text', 'children'),
    Output('pt-file-count-text', 'children'),
    Output('run-duration-text', 'children'),
    # --- Figures ---
    Output('avg-reward-chart', 'figure'),
    Output('lr-epsilon-chart', 'figure'),
    Output('individual-rewards-chart', 'figure'),
    Output('metrics-chart', 'figure'),
    Output('action-distribution-chart', 'figure'),
    Output('activation-mean-chart', 'figure'),
    Output('activation-norm-chart', 'figure'),
    Output('activation-action-chart', 'figure'),
    # --- Video Player ---
    Output('video-container', 'children'),
    Output('video-info-text', 'children'),
    Output('video-state-store', 'data'),
    # --- Inputs ---
    Input('interval-component', 'n_intervals'),
    Input('prev-video-btn', 'n_clicks'),
    Input('next-video-btn', 'n_clicks'),
    State('selected-run-store', 'data'),
    State('video-state-store', 'data')
)
def update_dashboard_data(_, prev_clicks, next_clicks, run_data, video_state):
    if not run_data or not run_data.get('run_folder'):
        return [no_update] * 15 # Must match number of outputs

    run_folder_path = run_data['run_folder']
    log_file = os.path.join(run_folder_path, 'logs.txt')
    activations_file = os.path.join(run_folder_path, 'activations.txt')

    # --- Determine Trigger ---
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'interval-component'

    # --- Load Data ---
    df_logs = parse_log_file(log_file)
    df_activations = parse_activations_file(activations_file)
    
    # --- Default Empty States ---
    empty_fig = {'data': [], 'layout': {'title': "Waiting for data...", 'paper_bgcolor': '#ffffff', 'plot_bgcolor': '#ffffff'}}
    latest_info = [html.H4("Waiting for log data...")]
    run_name = f"Run: {os.path.basename(run_folder_path)}"
    
    # --- Summary Info ---
    pt_file_count = len(glob.glob(os.path.join(run_folder_path, '*.pt')))
    pt_count_text = f"Checkpoints Found: {pt_file_count} (.pt files)"
    start_time, duration = get_run_duration(log_file)
    duration_text = f"Run Started: {start_time} | Total Time Logged: {duration}"

    # --- Initialize Figures ---
    (avg_reward_fig, lr_epsilon_fig, ind_rewards_fig, metrics_fig,
     action_dist_fig, act_mean_fig, act_norm_fig, act_action_fig) = [empty_fig] * 8
    
    # --- Populate Figures with Log Data ---
    if not df_logs.empty:
        latest_data = df_logs.iloc[-1]
        info_episode = f"Current Episode: {latest_data.get('Episode', 'N/A')}"
        info_steps = f"Steps Done: {latest_data.get('Steps Done', 0):,}"
        info_reward = f"Latest Avg. Reward: {latest_data.get('Average Reward', 0):.2f}"
        latest_info = [html.Div(html.H4(val, style={'textAlign': 'center'})) for val in [info_episode, info_steps, info_reward]]

        avg_reward_fig = px.line(df_logs, x='Episode', y='Average Reward', title='Average Reward Over Time', markers=True)
        
        # Individual Rewards
        reward_cols = [col for col in df_logs.columns if 'Player' in col and 'Reward' in col]
        ind_rewards_fig = px.line(df_logs, x='Episode', y=reward_cols, title='Individual Player Rewards')
        
        # Metrics
        metric_cols = ['hits', 'damage_taken', 'movement', 'ammo_efficiency', 'survival', 'health_pickup', 'frags']
        available_metrics = [m for m in metric_cols if m in df_logs.columns]
        metrics_fig = px.line(df_logs, x='Episode', y=available_metrics, title='Game Metrics Over Time') if available_metrics else empty_fig

        # Action Distribution
        action_cols = sorted([col for col in df_logs.columns if col.startswith('Action_')])
        if action_cols:
            action_df = df_logs[['Episode'] + action_cols].copy()
            action_dist_fig = px.area(action_df, x='Episode', y=action_cols, title='Action Selection Distribution (Stacked)', groupnorm='percent')
            action_dist_fig.update_layout(yaxis_title="Percentage of Actions")
        
        # LR & Epsilon Chart
        lr_epsilon_fig = make_subplots(specs=[[{"secondary_y": True}]])
        lr_epsilon_fig.add_trace(go.Scatter(x=df_logs['Episode'], y=df_logs['Learning Rate'], name='Learning Rate'), secondary_y=False)
        lr_epsilon_fig.add_trace(go.Scatter(x=df_logs['Episode'], y=df_logs['Epsilon'], name='Epsilon'), secondary_y=True)
        lr_epsilon_fig.update_layout(title_text="Learning Rate and Epsilon")
        lr_epsilon_fig.update_yaxes(title_text="Learning Rate", secondary_y=False)
        lr_epsilon_fig.update_yaxes(title_text="Epsilon", secondary_y=True)

    # --- Populate Figures with Activation Data ---
    if not df_activations.empty:
        # 1. Mean of all non-action layers
        df_other_layers = df_activations[df_activations['Layer'] != 'Actions']
        if not df_other_layers.empty:
            act_mean_fig = px.line(df_other_layers, x='Episode', y='Mean', color='Layer', title='Activation Mean (Excluding Actions)')
        
        # 2. Norm of all non-action layers
        if not df_other_layers.empty:
            act_norm_fig = px.line(df_other_layers, x='Episode', y='Norm', color='Layer', title='Activation Norm (Excluding Actions)')
            
        # 3. Mean and Std of 'Actions' layer
        df_actions_layer = df_activations[df_activations['Layer'] == 'Actions']
        if not df_actions_layer.empty:
            melted_actions = df_actions_layer.melt(id_vars=['Episode'], value_vars=['Mean', 'Std'], var_name='Statistic', value_name='Value')
            act_action_fig = px.line(melted_actions, x='Episode', y='Value', color='Statistic', title='Actions Layer Statistics')

    # --- Handle Video Navigation ---
    video_state = video_state or {'videos': [], 'current_index': -1}
    if trigger_id == 'interval-component':
        # Only check for new videos on interval refresh
        new_video_list = find_all_videos(run_folder_path)
        if video_state['videos'] != new_video_list:
            video_state['videos'] = new_video_list
            video_state['current_index'] = len(new_video_list) - 1 # Default to latest
    
    if trigger_id == 'prev-video-btn':
        video_state['current_index'] = max(0, video_state['current_index'] - 1)
    elif trigger_id == 'next-video-btn':
        video_state['current_index'] = min(len(video_state['videos']) - 1, video_state['current_index'] + 1)
    
    video_player = html.Div("No videos found.")
    video_info_text = "Video 0 of 0"
    if video_state['videos'] and video_state['current_index'] != -1:
        current_video_name = video_state['videos'][video_state['current_index']]
        video_player = html.Video(
            controls=True, id='movie_player', src=f"/videos/{current_video_name}",
            autoPlay=(trigger_id != 'interval-component'), muted=True, style={'width': '100%', 'borderRadius': '6px'}
        )
        video_info_text = f"Video {video_state['current_index'] + 1} of {len(video_state['videos'])}"

    return (latest_info, run_name, pt_count_text, duration_text,
            avg_reward_fig, lr_epsilon_fig, ind_rewards_fig, metrics_fig,
            action_dist_fig, act_mean_fig, act_norm_fig, act_action_fig,
            video_player, video_info_text, video_state)

# --- Special Route to Serve Video Files from the selected run ---
@server.route('/videos/<video_name>')
def serve_video(video_name):
    # This is a bit of a trick to get the currently selected run folder.
    # It relies on the file system and is not ideal for multi-user scenarios,
    # but works well for a local dashboard. A better way would involve a DB or more complex state.
    run_folders = get_run_folders(RUNS_BASE_FOLDER)
    for folder in run_folders:
        video_path = os.path.join(RUNS_BASE_FOLDER, folder, video_name)
        if os.path.exists(video_path):
            return send_from_directory(os.path.join(RUNS_BASE_FOLDER, folder), video_name)
    return "Video not found", 404


if __name__ == '__main__':
    app.run(debug=True, port=8050)