import fastf1
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Enable FastF1 Cache
fastf1.Cache.enable_cache('f1_cache')

# Streamlit Page Setup
st.set_page_config(page_title="F1 Racing Line Predictor", layout="wide")
st.title("🏎️ F1 Ideal Racing Line Predictor")

# Sidebar Controls
current_year = datetime.now().year
year_range = list(range(1950, current_year + 1))
year = st.sidebar.selectbox("Select Year", year_range, index = len(year_range) - 1)
event_data = fastf1.get_event_schedule(year)[['EventName', 'RoundNumber']]
event_dict = dict(zip(event_data['EventName'], event_data['RoundNumber']))
event = st.sidebar.selectbox("Select Event", list(event_dict.keys()))

# Cache fastest lap telemetry
@st.cache_data
def get_fastest_lap_data(year, event):
    try:
        session = fastf1.get_session(year, event_dict[event], 'R')
        session.load()
        fastest_lap = session.laps.pick_fastest()
        return fastest_lap.get_telemetry()
    except Exception:
        return None

# Cache all historical data for a given track
@st.cache_data
def get_historical_data(track_name):
    all_data = []
    for hist_year in range(1950, current_year):
        try:
            event_data = fastf1.get_event_schedule(hist_year)
            round_number = event_data[event_data['EventName'] == track_name]['RoundNumber'].values[0]
            session = fastf1.get_session(hist_year, round_number, 'R')
            session.load()
            fastest_lap = session.laps.pick_fastest()
            telemetry = fastest_lap.get_telemetry()
            telemetry['Year'] = hist_year
            all_data.append(telemetry)
        except:
            continue
    return pd.concat(all_data, ignore_index=True) if all_data else None

# Predict racing line using a Machine Learning model
@st.cache_data
def train_racing_line_model(track_data):
    if track_data is None or track_data.empty:
        return None

    # Features and Target Variables
    features = track_data[['X', 'Y', 'nGear']]
    targets = track_data[['Speed']]

    # Train Random Forest Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, targets.values.ravel())

    return model

# Predict ideal racing line based on historical data
@st.cache_data
def predict_ideal_racing_line(model, track_data):
    if model is None or track_data is None or track_data.empty:
        return None
    
    predicted_data = track_data.copy()
    predicted_data['PredictedSpeed'] = model.predict(track_data[['X', 'Y', 'nGear']])
    return predicted_data

# Load Data
if event:
    telemetry = get_fastest_lap_data(year, event)
    is_past_race = year < current_year

    if is_past_race and telemetry is not None:
        # Display actual fastest lap and predicted ideal lap
        historical_data = get_historical_data(event)
        ml_model = train_racing_line_model(historical_data)
        predicted_racing_line = predict_ideal_racing_line(ml_model, historical_data)

        if predicted_racing_line is not None:
            # Speed Map for Actual Fastest Lap
            actual_speed_fig = px.scatter(
                telemetry, x='X', y='Y', color='Speed',
                color_continuous_scale='turbo', title='Actual Fastest Lap (Speed)',
                labels={'Speed': 'Speed (km/h)'}
            )
            actual_speed_fig.update_layout(xaxis_visible=False, yaxis_visible=False)

            # Speed Map for Predicted Ideal Line
            predicted_speed_fig = px.scatter(
                predicted_racing_line, x='X', y='Y', color='PredictedSpeed',
                color_continuous_scale='turbo', title='Predicted Ideal Racing Line (Speed)',
                labels={'PredictedSpeed': 'Speed (km/h)'}
            )
            predicted_speed_fig.update_layout(xaxis_visible=False, yaxis_visible=False)

            # Display Graphs
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(actual_speed_fig, use_container_width=True)
            with col2: st.plotly_chart(predicted_speed_fig, use_container_width=True)
        else:
            st.warning("Not enough historical data available for prediction.")

    elif not is_past_race:
        # Predict only ideal racing line for future races
        historical_data = get_historical_data(event)
        ml_model = train_racing_line_model(historical_data)
        predicted_racing_line = predict_ideal_racing_line(ml_model, historical_data)

        if predicted_racing_line is not None:
            predicted_speed_fig = px.scatter(
                predicted_racing_line, x='X', y='Y', color='PredictedSpeed',
                color_continuous_scale='turbo', title='Predicted Ideal Racing Line (Speed)',
                labels={'PredictedSpeed': 'Speed (km/h)'}
            )
            predicted_speed_fig.update_layout(xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(predicted_speed_fig, use_container_width=True)
        else:
            st.warning("Not enough historical data available for prediction.")
