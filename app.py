import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import datetime


# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

# Streamlit Page Setup
st.set_page_config(page_title="F1 Racing Line", layout="wide")
st.title("🏎️ F1 Ideal Racing Line Dashboard")

# Sidebar Controls
current_year = datetime.datetime.now().year
year_range = list(range(1950, current_year + 1))
year = st.sidebar.selectbox("Select Year", year_range, index= len(year_range) - 1)  # Default 2023
event_data = fastf1.get_event_schedule(year)[['EventName', 'RoundNumber']]
event_dict = dict(zip(event_data['EventName'], event_data['RoundNumber']))
event = st.sidebar.selectbox("Select Event", list(event_dict.keys()))

# Function to Get Data
@st.cache_data
def get_fastest_lap_data(year, event):
    session = fastf1.get_session(year, event_dict[event], 'R')
    session.load()
    fastest_lap = session.laps.pick_fastest()
    return fastest_lap.get_telemetry()

# Load Data
if event:
    telemetry = get_fastest_lap_data(year, event)

    # Speed Map
    speed_fig = px.scatter(
        telemetry, x='X', y='Y', color='Speed',
        color_continuous_scale='turbo', title='Ideal Racing Line (Speed)',
        labels={'Speed': 'Speed (km/h)'}
    )
    speed_fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False
    )
    
    # Gear Shift Map
    gear_fig = px.scatter(
        telemetry, x='X', y='Y', color='nGear',
        color_continuous_scale='viridis', title='Ideal Racing Line (Gear Shifts)',
        labels={'nGear': 'Gear'}
    )
    gear_fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False
    )

    # DRS Map
    telemetry['DRS_Active'] = telemetry['DRS'].apply(lambda x: 'DRS On' if x > 0 else 'DRS Off')
    drs_fig = px.scatter(
        telemetry, x='X', y='Y', color='DRS_Active',
        title='DRS Activation Zones', labels={'DRS_Active': 'DRS Status'}
    )
    drs_fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False
    )

    # Display Graphs
    col1, col2, col3 = st.columns(3)
    with col1: st.plotly_chart(speed_fig, use_container_width=True)
    with col2: st.plotly_chart(gear_fig, use_container_width=True)
    with col3: st.plotly_chart(drs_fig, use_container_width=True)
