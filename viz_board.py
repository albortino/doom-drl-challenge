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
RUNS_BASE_FOLDER = 'runs'
REFRESH_INTERVAL_MS = 60000  # 60 seconds

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
    if not os.path.exists(log_file_path): return "N/A", "N/A"
    time_pattern = re.compile(r'(\d{2}:\d{2}:\d{2})')
    timestamps = [datetime.strptime(match.group(1), '%H:%M:%S') for match in (time_pattern.search(line) for line in open(log_file_path, 'r')) if match]
    if not timestamps: return "N/A", "N/A"
    return timestamps[0].strftime('%H:%M:%S'), str(timestamps[-1] - timestamps[0])

def parse_hyperparameters(log_file_path):
    """Parses the initial block of hyperparameters from the log file."""
    if not os.path.exists(log_file_path): return []
    params = {}
    param_pattern = re.compile(r"(\w+): ([\w\.\-]+)")
    try:
        with open(log_file_path, 'r') as f:
            for _ in range(50):
                line = f.readline()
                if not line: break
                for key, value in param_pattern.findall(line): params[key] = value
    except IOError: return []
    if not params: return [html.P("No hyperparameters found.")]
    return [html.Li(f"{key}: {value}") for key, value in params.items()]

def parse_log_file(log_file_path):
    """Parses logs.txt to extract structured training data into a DataFrame."""
    if not os.path.exists(log_file_path): return pd.DataFrame()
    log_splitter = re.compile(r'(?=\d{2}:\d{2}:\d{2} \|)')
    steps_pattern = re.compile(r"Episode: (\d+).*?Steps done: (\d+).*?currently (\d+)")
    reward_pattern = re.compile(
        r"Episode: (\d+).*?Rewards:.*?"
        r"Reward: \[(.*?)\]\s*\|\s*"
        r"Avg Reward: ([\d\.-]+)\s*\|\s*"
        r"Loss: ([\d\.-]+)\s*\|\s*"
        r"ε: ([\d\.-]+)\s*\|\s*"
        r"LR: ([\d\.e\+-]+).*?"
        r"Metrics - \[(.*?)\]\s*"
        r"Actions - (.*)", re.DOTALL)

    data = {}
    try:
        with open(log_file_path, 'r') as f: content = f.read()
    except IOError: return pd.DataFrame()

    for entry in log_splitter.split(content):
        if not entry.strip(): continue
        steps_match, reward_match = steps_pattern.search(entry), reward_pattern.search(entry)
        if steps_match:
            episode = int(steps_match.group(1))
            if episode not in data: data[episode] = {'Episode': episode}
            data[episode]['Steps Done'], data[episode]['Rollout Buffer'] = map(int, steps_match.groups()[1:])
        elif reward_match:
            episode, reward_str, avg_reward, loss, epsilon, lr, metrics_str, actions_str = reward_match.groups()
            episode = int(episode)
            if episode not in data: data[episode] = {'Episode': episode}
            data[episode].update({'Average Reward': float(avg_reward), 'Loss': float(loss), 'Epsilon': float(epsilon), 'Learning Rate': float(lr)})
            for i, r in enumerate(re.findall(r'[\d\.-]+', reward_str)): data[episode][f'Player {i+1} Reward'] = float(r)
            for m_key, m_val in re.findall(r"'([^:]+): ([^']+)'", metrics_str):
                try: data[episode][m_key.strip()] = float(m_val)
                except ValueError: data[episode][m_key.strip()] = m_val
            # FIX: Handle named actions (e.g., "Move Forward") and numeric actions
            for item in actions_str.split(','):
                if ':' in item:
                    action_id, count = item.split(':')
                    safe_action_name = action_id.strip().replace(' ', '_')
                    data[episode][f'Action_{safe_action_name}'] = int(count)

    if not data: return pd.DataFrame()
    df = pd.DataFrame(sorted(data.values(), key=lambda x: x.get('Episode', 0)))
    if 'Steps Done' in df.columns: df['Steps Done'] = df['Steps Done'].ffill().astype(int)
    if 'Rollout Buffer' in df.columns: df['Rollout Buffer'] = df['Rollout Buffer'].ffill().astype(int)
    return df.dropna(subset=['Average Reward']).reset_index(drop=True)

def parse_activations_file(activations_file_path):
    """Parses activations.txt to extract layer statistics using regex."""
    if not os.path.exists(activations_file_path): return pd.DataFrame()
    pattern = re.compile(r"Episode\s+(\d+)\s*,\s*([\w_]+)\s*,.+?Mean:\s+([-\d\.]+),Std:\s+([-\d\.]+),Norm:\s+([-\d\.]+)")
    data = []
    try:
        with open(activations_file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    ep, layer, mean, std, norm = match.groups()
                    data.append({'Episode': int(ep), 'Layer': layer, 'Mean': float(mean), 'Std': float(std), 'Norm': float(norm)})
    except (IOError, ValueError): return pd.DataFrame()
    if not data: return pd.DataFrame()
    return pd.DataFrame(data).groupby(['Episode', 'Layer']).mean().reset_index()

def parse_episode_times(log_file_path):
    """Parses log file to calculate the time spent in each phase of training per episode."""
    if not os.path.exists(log_file_path): return pd.DataFrame()
    
    line_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}) \| Episode: (\d+) \| (.*)')
    event_markers = {
        "Gathering rollout": "Rollout", "Training for": "Training",
        "Updating target network": "Updating", "Replaying animation": "Animation",
        "Running quick evaluation": "Evaluation"}
    
    episodes_events = {}
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                match = line_pattern.match(line)
                if not match: continue
                time_str, ep_str, msg = match.groups()
                ep_num = int(ep_str)
                if ep_num not in episodes_events: episodes_events[ep_num] = []
                event_time = datetime.strptime(time_str, '%H:%M:%S')
                for marker, event_type in event_markers.items():
                    if marker in msg:
                        episodes_events[ep_num].append({'time': event_time, 'type': event_type})
                        break
    except IOError: return pd.DataFrame()

    durations_list = []
    sorted_episodes = sorted(episodes_events.keys())
    for i, ep_num in enumerate(sorted_episodes):
        events = sorted(episodes_events[ep_num], key=lambda x: x['time'])
        if not events: continue
        
        # Determine the total time for this episode
        ep_start_time = events[0]['time']
        next_ep_start_time = None
        if i + 1 < len(sorted_episodes) and episodes_events[sorted_episodes[i+1]]:
             next_ep_start_time = sorted(episodes_events[sorted_episodes[i+1]], key=lambda x: x['time'])[0]['time']
        else:
             next_ep_start_time = events[-1]['time'] # Fallback for the last episode
        
        total_ep_duration = (next_ep_start_time - ep_start_time).total_seconds()
        
        # Calculate duration of each phase
        durations = {'Episode': ep_num}
        for j, event in enumerate(events):
            start = event['time']
            end = events[j+1]['time'] if j + 1 < len(events) else next_ep_start_time
            duration = (end - start).total_seconds()
            durations[event['type']] = durations.get(event['type'], 0) + duration

        # Calculate "Rest" time
        sum_explicit_durations = sum(v for k, v in durations.items() if k != 'Episode')
        durations['Rest'] = max(0.0, total_ep_duration - sum_explicit_durations)
        durations_list.append(durations)

    return pd.DataFrame(durations_list).fillna(0)

def find_all_videos(run_folder_path):
    """Finds all video files (.mp4) in a folder, sorted from oldest to newest."""
    if not run_folder_path or not os.path.isdir(run_folder_path): return []
    videos = glob.glob(os.path.join(run_folder_path, '*.mp4'))
    videos.sort(key=os.path.getmtime)
    return [os.path.basename(v) for v in videos]

# --- DASH APP ---
app = dash.Dash(__name__, title="DRL Training Dashboard")
server = app.server

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f0f2f5', 'padding': '20px'}, children=[
    dcc.Store(id='selected-run-store'), dcc.Store(id='video-state-store'),
    html.H1("DRL Training Progress Dashboard", style={'textAlign': 'center', 'color': '#1f2937'}),
    dcc.Interval(id='interval-component', interval=REFRESH_INTERVAL_MS, n_intervals=0),
    html.Div([
        html.Label("Select Training Run:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
        dcc.Dropdown(id='run-selector-dropdown', placeholder="Select a run...", style={'flex': '1'}),
        html.Button("Refresh List", id='refresh-runs-btn', style={'marginLeft': '10px'})
    ], style={'display': 'flex', 'alignItems': 'center', 'maxWidth': '800px', 'margin': '20px auto'}),
    html.Div(id='main-content-area', children=[])
])

# --- CALLBACKS ---
@app.callback(
    Output('run-selector-dropdown', 'options'), Output('run-selector-dropdown', 'value'),
    Input('refresh-runs-btn', 'n_clicks'))
def update_run_list(_):
    run_folders = get_run_folders(RUNS_BASE_FOLDER)
    options = [{'label': folder, 'value': folder} for folder in run_folders]
    return options, run_folders[0] if run_folders else None

@app.callback(
    Output('selected-run-store', 'data'), Input('run-selector-dropdown', 'value'))
def store_selected_run(name):
    if not name: return None
    return {'run_name': name, 'run_folder_path': os.path.join(RUNS_BASE_FOLDER, name)}

@app.callback(
    Output('main-content-area', 'children'), Input('selected-run-store', 'data'))
def render_main_content(stored_run_data):
    if not stored_run_data: return html.H3("Please select a run.", style={'textAlign': 'center'})
    card_style = {'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '8px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}
    return html.Div([
        html.Div(id='latest-info-container', style={**card_style, 'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '20px'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginBottom': '20px'}, children=[
            html.Div(style=card_style, children=[
                html.H3("Run Summary", style={'color': '#374151', 'borderBottom': '1px solid #e5e7eb', 'paddingBottom': '10px'}),
                html.P(id='run-name-text'), html.P(id='pt-file-count-text'), html.P(id='run-duration-text'),
                html.Details([html.Summary("Hyperparameters"), html.Ul(id='hyperparameters-display')])
            ]),
            html.Div(style=card_style, children=[
                html.H3("Evaluation Videos"),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'}, children=[
                    html.Button('⬅️ Prev', id='prev-video-btn'), html.P(id='video-info-text'), html.Button('Next ➡️', id='next-video-btn')]),
                html.Div(id='video-container', style={'marginTop': '10px'})
            ])
        ]),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            dcc.Graph(id='avg-reward-chart'), dcc.Graph(id='loss-chart'),
            dcc.Graph(id='individual-rewards-chart'), dcc.Graph(id='lr-epsilon-chart'),
            dcc.Graph(id='metrics-chart'), dcc.Graph(id='time-distribution-chart') # New chart
        ]),
        html.H3("Action & Activation Statistics", style={'textAlign': 'center', 'marginTop': '30px'}),
        dcc.Graph(id='action-distribution-chart'), dcc.Graph(id='activation-mean-chart'),
        dcc.Graph(id='activation-norm-chart'), dcc.Graph(id='activation-action-chart'),
    ])

@app.callback(
    [Output('latest-info-container', 'children'), Output('run-name-text', 'children'),
     Output('pt-file-count-text', 'children'), Output('run-duration-text', 'children'),
     Output('hyperparameters-display', 'children'), Output('avg-reward-chart', 'figure'),
     Output('loss-chart', 'figure'), Output('lr-epsilon-chart', 'figure'),
     Output('individual-rewards-chart', 'figure'), Output('metrics-chart', 'figure'),
     Output('time-distribution-chart', 'figure'), Output('action-distribution-chart', 'figure'),
     Output('activation-mean-chart', 'figure'), Output('activation-norm-chart', 'figure'),
     Output('activation-action-chart', 'figure'), Output('video-container', 'children'),
     Output('video-info-text', 'children'), Output('video-state-store', 'data')],
    [Input('interval-component', 'n_intervals'), Input('prev-video-btn', 'n_clicks'), Input('next-video-btn', 'n_clicks')],
    [State('selected-run-store', 'data'), State('video-state-store', 'data')]
)
def update_dashboard_data(_, prev_clicks, next_clicks, run_data, video_state):
    if not run_data: return [no_update] * 18

    run_name, run_path = run_data['run_name'], run_data['run_folder_path']
    log_file, act_file = os.path.join(run_path, 'logs.txt'), os.path.join(run_path, 'activations.txt')
    
    df_logs, df_activations, df_times = parse_log_file(log_file), parse_activations_file(act_file), parse_episode_times(log_file)
    empty_fig = {'data': [], 'layout': {'title': "Waiting for data..."}}
    
    # --- Summary Info ---
    run_name_text = f"Run: {run_name}"
    pt_count_text = f"Checkpoints: {len(glob.glob(os.path.join(run_path, '*.pt')))}"
    start_time, duration = get_run_duration(log_file)
    duration_text = f"Started: {start_time} | Logged Time: {duration}"
    hyperparams_list = parse_hyperparameters(log_file)
    
    # --- Figures ---
    figs = {k: empty_fig for k in ['avg_reward', 'loss', 'lr_epsilon', 'ind_rewards', 'metrics', 'time_dist', 'action_dist', 'act_mean', 'act_norm', 'act_action']}
    if not df_logs.empty:
        latest = df_logs.iloc[-1]
        latest_info = [html.H4(f"Ep: {latest.get('Episode', 'N/A')}"), html.H4(f"Steps: {latest.get('Steps Done', 0):,}"), html.H4(f"Avg Reward: {latest.get('Average Reward', 0):.2f}")]
        figs['avg_reward'] = px.line(df_logs, x='Episode', y='Average Reward', title='Average Reward')
        figs['loss'] = px.line(df_logs, x='Episode', y='Loss', title='Training Loss')
        reward_cols = [c for c in df_logs.columns if 'Player' in c and 'Reward' in c]
        figs['ind_rewards'] = px.line(df_logs, x='Episode', y=reward_cols, title='Individual Player Rewards')
        metric_cols = ['hits', 'damage_taken', 'movement', 'ammo_efficiency', 'survival', 'health_pickup', 'frags']
        figs['metrics'] = px.line(df_logs, x='Episode', y=[m for m in metric_cols if m in df_logs.columns], title='Game Metrics')
        action_cols = sorted([c for c in df_logs.columns if c.startswith('Action_')])
        if action_cols: figs['action_dist'] = px.area(df_logs, x='Episode', y=action_cols, title='Action Distribution', groupnorm='percent')
        
        lr_fig = make_subplots(specs=[[{"secondary_y": True}]])
        lr_fig.add_trace(go.Scatter(x=df_logs['Episode'], y=df_logs['Learning Rate'], name='LR'), secondary_y=False)
        lr_fig.add_trace(go.Scatter(x=df_logs['Episode'], y=df_logs['Epsilon'], name='Epsilon'), secondary_y=True)
        lr_fig.update_layout(title_text="LR & Epsilon").update_yaxes(type='log', secondary_y=False)
        figs['lr_epsilon'] = lr_fig
    else: latest_info = [html.H4("Waiting for log data...")]

    if not df_times.empty:
        time_cols = [c for c in ['Rollout', 'Training', 'Updating', 'Animation', 'Evaluation', 'Rest'] if c in df_times.columns]
        figs['time_dist'] = px.area(df_times, x='Episode', y=time_cols, title='Time Distribution of Phases', groupnorm='percent')

    if not df_activations.empty:
        other_layers = df_activations[df_activations['Layer'] != 'Actions']
        action_layer = df_activations[df_activations['Layer'] == 'Actions']
        if not other_layers.empty:
            figs['act_mean'] = px.line(other_layers, x='Episode', y='Mean', color='Layer', title='Activation Mean')
            figs['act_norm'] = px.line(other_layers, x='Episode', y='Norm', color='Layer', title='Activation Norm')
        if not action_layer.empty:
            melted = action_layer.melt(id_vars=['Episode'], value_vars=['Mean', 'Std'], var_name='Stat', value_name='Value')
            figs['act_action'] = px.line(melted, x='Episode', y='Value', color='Stat', title='Actions Layer Stats')

    # --- Video Navigation ---
    ctx = dash.callback_context.triggered[0]
    video_state = video_state or {'videos': [], 'idx': -1}
    if ctx['prop_id'] == 'interval-component.n_intervals':
        new_videos = find_all_videos(run_path)
        if video_state.get('videos') != new_videos: video_state.update({'videos': new_videos, 'idx': len(new_videos) - 1})
    elif ctx['prop_id'] == 'prev-video-btn.n_clicks': video_state['idx'] = max(0, video_state.get('idx', 0) - 1)
    elif ctx['prop_id'] == 'next-video-btn.n_clicks': video_state['idx'] = min(len(video_state.get('videos',[]))-1, video_state.get('idx',-1)+1)
    
    video_player, video_info = html.Div("No videos."), "Video 0 of 0"
    if video_state.get('videos'):
        idx, total = video_state['idx'], len(video_state['videos'])
        video_info = f"Video {idx + 1} of {total}"
        video_player = html.Video(controls=True, src=f"/videos/{run_name}/{video_state['videos'][idx]}", style={'width': '100%'})

    return (latest_info, run_name_text, pt_count_text, duration_text, hyperparams_list,
            figs['avg_reward'], figs['loss'], figs['lr_epsilon'], figs['ind_rewards'], figs['metrics'],
            figs['time_dist'], figs['action_dist'], figs['act_mean'], figs['act_norm'], figs['act_action'],
            video_player, video_info, video_state)

@server.route('/videos/<run_folder>/<video_name>')
def serve_video(run_folder, video_name):
    return send_from_directory(os.path.join(RUNS_BASE_FOLDER, run_folder), video_name)

if __name__ == '__main__':
    app.run(debug=True, port=8051)