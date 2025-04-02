import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import requests
from datetime import datetime

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

# Streamlit Page Setup
st.set_page_config(page_title="F1 Ideal Racing Line", layout="wide")
st.title("🏎️ F1 Ideal Racing Line Prediction for the Upcoming Season")

# Sidebar Controls
year = st.sidebar.selectbox("Select Year", list(range(1950, datetime.now().year + 1)), index=5)
event_data = fastf1.get_event_schedule(year)[['EventName', 'RoundNumber']]
event_dict = dict(zip(event_data['EventName'], event_data['RoundNumber']))
event = st.sidebar.selectbox("Select Event", list(event_dict.keys()))

# Function to Get Data
@st.cache_data
def get_fastest_lap_data(year, event):
    try:
        session = fastf1.get_session(year, event_dict[event], 'R')
        session.load()
        fastest_lap = session.laps.pick_fastest()
        return fastest_lap.get_telemetry()
    except Exception as e:
        st.error(f"Error loading session data: {e}")
        return None


# Predict Ideal Racing Line Based on Historical Data
def predict_ideal_line(telemetry_data, weather_data=None):
    if telemetry_data is None or len(telemetry_data) == 0:
        st.warning("Sufficient data not available.")
        return None

    # Example of using speed and gear data for prediction (simplified)
    # Adjust this logic based on more advanced prediction methods
    avg_speed = telemetry_data['Speed'].mean()
    avg_gear = telemetry_data['nGear'].mode()[0]

    # Create the predicted racing line
    predicted_line = telemetry_data.copy()
    predicted_line['PredictedSpeed'] = avg_speed
    predicted_line['PredictedGear'] = avg_gear
    
    return predicted_line

# Load Data
if event:
    telemetry = get_fastest_lap_data(year, event)

    # Predict the ideal racing line
    predicted_racing_line = predict_ideal_line(telemetry, weather_data)

    if predicted_racing_line is not None:
        # Speed Map for Predicted Line
        speed_fig = px.scatter(
            predicted_racing_line, x='X', y='Y', color='PredictedSpeed',
            color_continuous_scale='turbo', title='Predicted Ideal Racing Line (Speed)',
            labels={'PredictedSpeed': 'Speed (km/h)'}
        )
        speed_fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False
        )

        # Gear Map for Predicted Line
        gear_fig = px.scatter(
            predicted_racing_line, x='X', y='Y', color='PredictedGear',
            color_continuous_scale='viridis', title='Predicted Ideal Racing Line (Gear Shifts)',
            labels={'PredictedGear': 'Gear'}
        )
        gear_fig.update_layout(
            xaxis_visible=False,
            yaxis_visible=False
        )

        # Display Graphs
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(speed_fig, use_container_width=True)
        with col2: st.plotly_chart(gear_fig, use_container_width=True)
    else:
        st.warning("Not enough data available to predict the racing line.")
