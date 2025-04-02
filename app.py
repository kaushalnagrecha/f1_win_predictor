import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Enable FastF1 Caching for performance
fastf1.Cache.enable_cache('f1_cache')  

# Load available events
def get_available_events(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule[['EventName', 'RoundNumber']]

# Load session data
def get_fastest_lap_data(year, event):
    session = fastf1.get_session(year, event, 'R')  # 'R' for Race Session
    session.load()
    
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_telemetry()
    
    return telemetry

# Initialize Dash App
app = dash.Dash(__name__)
app.title = "F1 Ideal Racing Line Dashboard"

# Layout
app.layout = html.Div([
    html.Div([
        html.H2("F1 Ideal Racing Line"),
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(y), 'value': y} for y in range(2018, 2025)],
            value=2023,  # Default Year
            clearable=False
        ),
        html.Label("Select Event:"),
        dcc.Dropdown(id='event-dropdown', clearable=False),
    ], style={'width': '20%', 'display': 'inline-block', 'padding': '20px', 'verticalAlign': 'top'}),
    
    html.Div([
        dcc.Graph(id='speed-map'),
        dcc.Graph(id='gear-map'),
        dcc.Graph(id='drs-map'),
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
])

# Callback to update event dropdown based on selected year
@app.callback(
    Output('event-dropdown', 'options'),
    Input('year-dropdown', 'value')
)
def update_event_dropdown(year):
    events = get_available_events(year)
    return [{'label': name, 'value': round_num} for name, round_num in zip(events['EventName'], events['RoundNumber'])]

# Callback to update the track maps
@app.callback(
    [Output('speed-map', 'figure'),
     Output('gear-map', 'figure'),
     Output('drs-map', 'figure')],
    [Input('year-dropdown', 'value'),
     Input('event-dropdown', 'value')]
)
def update_charts(year, event):
    if not event:
        return dash.no_update, dash.no_update, dash.no_update

    telemetry = get_fastest_lap_data(year, event)

    # Speed Map
    speed_fig = px.scatter(
        telemetry,
        x='X', y='Y', color='Speed',
        color_continuous_scale='turbo',
        title='Ideal Racing Line (Speed-Based)',
        labels={'Speed': 'Speed (km/h)'}
    )

    # Gear Map
    gear_fig = px.scatter(
        telemetry,
        x='X', y='Y', color='nGear',
        color_continuous_scale='viridis',
        title='Ideal Racing Line (Gear Shifts)',
        labels={'nGear': 'Gear'}
    )

    # DRS Activation Map
    telemetry['DRS_Active'] = telemetry['DRS'].apply(lambda x: 'DRS On' if x > 0 else 'DRS Off')
    drs_fig = px.scatter(
        telemetry,
        x='X', y='Y', color='DRS_Active',
        title='DRS Activation Zones',
        labels={'DRS_Active': 'DRS Status'}
    )

    return speed_fig, gear_fig, drs_fig

# Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)
