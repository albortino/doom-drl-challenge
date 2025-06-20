import os
import re
import glob
from datetime import datetime

import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from flask import send_from_directory

# conda install dash pandas plotly

# --- CONFIGURATION ---
# The root folder where all your training runs are stored.
RUNS_BASE_FOLDER = 'runs'
# How often the app should refresh, in milliseconds.
REFRESH_INTERVAL_MS = 60000  # 60 seconds

# --- HELPER FUNCTIONS ---

def find_latest_run_folder(base_folder):
    """Finds the most recent directory in the base_folder."""
    try:
        all_runs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        all_runs.sort(reverse=True)
        if all_runs:
            return os.path.join(base_folder, all_runs[0])
    except FileNotFoundError:
        return None
    return None

def find_all_videos(folder_path):
    """Finds all video files (.mp4) in a folder, sorted from oldest to newest."""
    if not folder_path or not os.path.isdir(folder_path):
        return []
    videos = glob.glob(os.path.join(folder_path, '*.mp4'))
    # Sort by creation time (oldest first) to make navigation intuitive
    videos.sort(key=os.path.getctime)
    return [os.path.basename(v) for v in videos]

def parse_log_file(log_file_path):
    """
    Parses the logs.txt file to extract structured data.
    This version is designed to be robust against changes in log entry order
    and multi-line log messages.
    Returns a pandas DataFrame.
    """
    if not os.path.exists(log_file_path):
        return pd.DataFrame()

    log_splitter_pattern = re.compile(r'(?=\d{2}:\d{2}:\d{2} \|)')
    steps_pattern = re.compile(
        r"Episode: (\d+).*?Steps done: (\d+).*?Gathering rollout.*?currently (\d+)"
    )
    reward_pattern = re.compile(
        r"Episode: (\d+).*?Rewards:.*?"
        r"Reward: \[(.*?)\]\s*\|\s*"
        r"Avg Reward: ([\d\.-]+)\s*\|\s*"
        r"Loss: ([\d\.-]+)\s*\|\s*"
        r"ε: ([\d\.-]+)\s*\|\s*"
        r"LR: ([\d\.e\+-]+).*?"
        r"Metrics - \[(.*?)\]\s*"
        r"Actions - (.*)",
        re.DOTALL
    )

    data = {}
    with open(log_file_path, 'r') as f:
        content = f.read()

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
                'Average Reward (All Players)': float(avg_reward),
                'Loss': float(loss),
                'Epsilon': float(epsilon),
                'Learning Rate': float(lr)
            })

            rewards = [float(r) for r in re.findall(r'[\d\.-]+', reward_str)]
            for i, r in enumerate(rewards):
                data[episode][f'Player {i+1} Reward'] = r

            metrics_pattern = re.compile(r"'([^:]+): ([^']+)'")
            for m_key, m_val in metrics_pattern.findall(metrics_str):
                try:
                    data[episode][m_key.strip()] = float(m_val)
                except ValueError:
                    data[episode][m_key.strip()] = m_val

            if action_counts_str:
                action_counts_dict = {}
                for item in action_counts_str.split(','):
                    if ':' in item:
                        try:
                            action_id, count = item.split(':')
                            action_counts_dict[f'Action_{action_id.strip()}'] = int(count)
                        except ValueError:
                            pass
                data[episode].update(action_counts_dict)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(sorted(list(data.values()), key=lambda x: x.get('Episode', 0)))
    if 'Episode' in df.columns:
        df = df.set_index('Episode')
    if 'Steps Done' in df.columns:
        df['Steps Done'] = df['Steps Done'].fillna(method='ffill')
    if 'Rollout Buffer' in df.columns:
        df['Rollout Buffer'] = df['Rollout Buffer'].fillna(method='ffill')
    df = df.dropna(subset=['Average Reward (All Players)'])
    if 'Steps Done' in df.columns:
        df['Steps Done'] = df['Steps Done'].astype(int)
        df['Rollout Buffer'] = df['Rollout Buffer'].astype(int)

    return df.reset_index()

def count_pt_files(folder_path):
    """Counts the number of .pt (PyTorch model) files in a folder."""
    return len(glob.glob(os.path.join(folder_path, '*.pt')))


# --- DASH APP INITIALIZATION ---
app = dash.Dash(__name__, title="DRL Training Dashboard")
CURRENT_RUN_FOLDER = find_latest_run_folder(RUNS_BASE_FOLDER)
print(f"Monitoring run folder: {CURRENT_RUN_FOLDER}")

# --- APP LAYOUT ---
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f4f4', 'padding': '20px'}, children=[
    # NEW: dcc.Store to hold video player state
    dcc.Store(id='video-state-store'),

    html.H1("DRL Training Progress Dashboard", style={'textAlign': 'center', 'color': '#333'}),

    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MS, n_intervals=0),

    html.Div(id='latest-info-container', style={'display': 'flex', 'justifyContent': 'space-around', 'padding': '20px', 'backgroundColor': 'white', 'borderRadius': '8px', 'marginBottom': '20px'}),

    html.Div(style={'display': 'flex', 'gap': '20px'}, children=[
        html.Div(style={'flex': 1, 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px'}, children=[
            html.H3("Evaluation Videos", style={'color': '#555'}),
            # NEW: Video navigation controls
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'gap': '15px', 'marginBottom': '10px'}, children=[
                html.Button('⬅️ Previous', id='prev-video-btn', n_clicks=0),
                html.P(id='video-info-text', style={'fontWeight': 'bold', 'margin': '0'}),
                html.Button('Next ➡️', id='next-video-btn', n_clicks=0),
            ]),
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
        dcc.Graph(id='action-distribution-chart'),
        dcc.Graph(id='lr-chart'),
    ])
])


# --- CALLBACKS TO UPDATE THE APP ---

@app.callback(
    [Output('latest-info-container', 'children'),
     Output('current-run-folder-text', 'children'),
     Output('pt-file-count-text', 'children'),
     Output('avg-reward-chart', 'figure'),
     Output('individual-rewards-chart', 'figure'),
     Output('metrics-chart', 'figure'),
     Output('lr-chart', 'figure'),
     Output('action-distribution-chart', 'figure'),
     Output('video-container', 'children'),
     Output('video-info-text', 'children'),
     Output('video-state-store', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('prev-video-btn', 'n_clicks'),
     Input('next-video-btn', 'n_clicks')],
    [State('video-state-store', 'data')]
)
def update_dashboard(interval_tick, prev_clicks, next_clicks, video_state):
    """
    This function is triggered by the interval OR button clicks and updates the dashboard.
    """
    global CURRENT_RUN_FOLDER
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'interval-component'

    # Initialize or update the run folder only on interval tick
    if trigger_id == 'interval-component':
        CURRENT_RUN_FOLDER = find_latest_run_folder(RUNS_BASE_FOLDER)

    if not CURRENT_RUN_FOLDER:
        no_data_msg = "No run folder found."
        empty_figure = {'data': [], 'layout': {'title': no_data_msg}}
        return ([html.H4(no_data_msg)], "Current Run: N/A", "Checkpoints: N/A",
                empty_figure, empty_figure, empty_figure, empty_figure, empty_figure,
                None, "No Videos Found", None)

    # --- Handle Video Navigation ---
    video_state = video_state or {'videos': [], 'current_index': -1}
    new_video_list = find_all_videos(CURRENT_RUN_FOLDER)
    
    # If the list of videos has changed, reset the state
    if video_state['videos'] != new_video_list:
        video_state['videos'] = new_video_list
        video_state['current_index'] = len(new_video_list) - 1 # Start at the latest video

    # Handle button clicks
    if trigger_id == 'prev-video-btn':
        video_state['current_index'] = max(0, video_state['current_index'] - 1)
    elif trigger_id == 'next-video-btn':
        video_state['current_index'] = min(len(video_state['videos']) - 1, video_state['current_index'] + 1)
        
    # --- Generate Video Player and Info Text ---
    video_player = html.Div("No videos found in this run folder.")
    video_info_text = "Video 0 of 0"
    if video_state['videos'] and video_state['current_index'] != -1:
        current_video_name = video_state['videos'][video_state['current_index']]
        video_player = html.Video(
            controls=True, id='movie_player',
            src=f"/videos/{current_video_name}",
            autoPlay=(trigger_id != 'interval-component'), # Autoplay only when navigating
            muted=True, style={'width': '100%'}
        )
        video_info_text = f"Video {video_state['current_index'] + 1} of {len(video_state['videos'])}"

    # --- Optimize: Only update graphs on interval ---
    if trigger_id != 'interval-component':
        # If a button was clicked, don't re-calculate all the graphs
        return (no_update, no_update, no_update, no_update, no_update, no_update,
                no_update, no_update, video_player, video_info_text, video_state)

    # --- Full Update (triggered by interval) ---
    df = parse_log_file(os.path.join(CURRENT_RUN_FOLDER, 'logs.txt'))
    pt_file_count = count_pt_files(CURRENT_RUN_FOLDER)
    run_name_text = f"Current Run: {os.path.basename(CURRENT_RUN_FOLDER)}"
    pt_count_text = f"Found {pt_file_count} model checkpoints (.pt files)."
    empty_figure = {'data': [], 'layout': {'title': "Waiting for log data..."}}

    if df.empty:
        return ([html.H4("Waiting for log data...")], run_name_text, pt_count_text,
                empty_figure, empty_figure, empty_figure, empty_figure, empty_figure,
                video_player, video_info_text, video_state)

    latest_data = df.iloc[-1]
    steps_done = f"Steps Done: {latest_data.get('Steps Done', 0):,.0f}"
    avg_reward = f"Avg. Reward: {latest_data.get('Average Reward (All Players)', 0):.2f}"
    rollout_buffer = f"Rollout Buffer: {latest_data.get('Rollout Buffer', 0):,.0f}"
    latest_info_children = [html.Div(html.H4(val, style={'textAlign': 'center'})) for val in [steps_done, avg_reward, rollout_buffer]]

    avg_reward_fig = px.line(df, x='Episode', y='Average Reward (All Players)', title='Average Reward (All Players) Over Time')
    reward_cols = [col for col in df.columns if 'Player' in col and 'Reward' in col]
    individual_rewards_fig = px.line(df, x='Episode', y=reward_cols, title='Individual Player Rewards Over Time')
    metric_cols = ['hits', 'damage_taken', 'movement', 'ammo_efficiency', 'survival', 'health_pickup']
    available_metrics = [m for m in metric_cols if m in df.columns]
    metrics_fig = px.line(df, x='Episode', y=available_metrics, title='Metrics Over Time')
    lr_fig = px.line(df, x='Episode', y='Learning Rate', title='Learning Rate Schedule')

    action_labels = {'Action_0': 'Nothing', 'Action_1': 'Forward', 'Action_2': 'Fire Weapon', 'Action_3': 'Move Left', 'Action_4': 'Move Right', 'Action_5': 'Turn Left', 'Action_6': 'Turn Right', 'Action_7': 'Jump Forward'}
    action_cols = sorted([col for col in df.columns if col.startswith('Action_')])
    action_dist_fig = empty_figure
    if action_cols:
        plot_df = df[['Episode'] + action_cols].copy().rename(columns=action_labels)
        y_cols_labeled = [action_labels.get(c, c) for c in action_cols]
        action_dist_fig = px.area(plot_df, x='Episode', y=y_cols_labeled, title='Action Distribution (Percentage)', groupnorm='percent')
        action_dist_fig.update_layout(yaxis_title="Percentage of Actions")
    else:
        action_dist_fig['layout']['title'] = "No Action Data Found"

    return (latest_info_children, run_name_text, pt_count_text, avg_reward_fig,
            individual_rewards_fig, metrics_fig, lr_fig, action_dist_fig,
            video_player, video_info_text, video_state)

# --- Special Route to Serve Video Files ---
@app.server.route('/videos/<video_name>')
def serve_video(video_name):
    if CURRENT_RUN_FOLDER:
        return send_from_directory(CURRENT_RUN_FOLDER, video_name)
    return "Error: No run folder specified", 404

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    app.run(debug=True, port=8051)