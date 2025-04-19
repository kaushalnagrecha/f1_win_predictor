import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache')  # You can specify a custom path if desired

def build_dataset(start_year, end_year, round_number):
    """
    Builds a dataset of qualifying and race results from start_year to end_year.
    """
    try:
        # Load Qualifier session data
        qual_session = fastf1.get_session(end_year, round_number, 'Q')
        qual_session.load()
    
        # Get best time using priority: q3 > q2 > q1
        qual_session.results['best_time'] = qual_session.results[['Q3', 'Q2', 'Q1']].bfill(axis=1).iloc[:, 0]
    
        # Convert best_time to seconds
        qual_session.results['best_time_seconds'] = qual_session.results['best_time'].dt.total_seconds()
        qual_session.results['best_time_seconds'].fillna(qual_session.results['best_time_seconds'].max() + 100, inplace=True)
    
        # Load Race session data
        race_session = fastf1.get_session(start_year, round_number, 'R')
        race_session.load()
        laps_2024 = race_session.laps[["Driver", "LapTime"]].copy()
        laps_2024.dropna(subset=["LapTime"], inplace=True)
        laps_2024["LapTime (s)"] = laps_2024["LapTime"].dt.total_seconds()
    
        merged_results = laps_2024.merge(qual_session.results, left_on='Driver', right_on='Abbreviation')
        return merged_results, qual_session.results
    except Exception as e:
        st.warning(body = 'Something is not right!', icon = '⚠️')
        return pd.DataFrame()

def train_models(X_train, y_train):
    """
    Trains 3 models - GradientBoostingRegressor, RandomForestRegressor, XGBoostRegressor
    """
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=35)
    gb_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=35)
    rf_model.fit(X_train, y_train)

    xgboost_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=35)
    xgboost_model.fit(X_train, y_train)
    return gb_model, rf_model, xgboost_model

def predict(gb_model, rf_model, xgboost_model, qualifying_results):
    gb_y_pred = gb_model.predict(qualifying_results[['best_time_seconds']])
    rf_y_pred = rf_model.predict(qualifying_results[['best_time_seconds']])
    xgb_y_pred = xgboost_model.predict(qualifying_results[['best_time_seconds']])
    final_df = pd.DataFrame({
        'Driver': qualifying_results['FullName'],
        'Qualifying Time (s)': qualifying_results['best_time_seconds'],
        'GB Time (s)': gb_y_pred,
        'RF Time (s)': rf_y_pred,
        'XGB Time (s)': xgb_y_pred
        })
    return final_df

def evaluate_models(gb_model, rf_model, xgboost_model, X_test, y_test):
    gb_y_pred = gb_model.predict(X_test)
    rf_y_pred = rf_model.predict(X_test)
    xgb_y_pred = xgboost_model.predict(X_test)
    evaluation_dict = {
        'Gradient Boosting': mean_absolute_percentage_error(y_test, gb_y_pred) * 100,
        'Random Forest': mean_absolute_percentage_error(y_test, rf_y_pred) * 100, 
        'XGBoost': mean_absolute_percentage_error(y_test, xgb_y_pred) * 100
        }
    return evaluation_dict

def create_gauge(title, value):
    upper_limit = int(value + 2)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, upper_limit]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 10], 'color': 'green'},
                {'range': [10, 15], 'color': 'yellow'},
                {'range': [15, 25], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
    return fig

def plot(final_df, evaluation_df):
    st.subheader('‼️ Mean Absolute Percentage Error ‼️')
    col1, col2, col3 = st.columns(3)
    # Render each gauge in its own column
    with col1:
        st.plotly_chart(create_gauge("Gradient Boosting", evaluation_df['Gradient Boosting']), use_container_width=True)

    with col2:
        st.plotly_chart(create_gauge("Random Forest", evaluation_df['Random Forest']), use_container_width=True)

    with col3:
        st.plotly_chart(create_gauge("XGBoost", evaluation_df['XGBoost']), use_container_width=True)
    
    st.subheader('🔮 Race Predictions')
    # Render Predicted DataFrame
    st.dataframe(final_df, height=300)

def main():

    st.title("🏎️ F1 Race Prediction App")

    st.sidebar.header("Race Settings")
    year = st.sidebar.number_input("Year", disabled=True, value=datetime.now().year)
    events = fastf1.get_event_schedule(year)
    events = events[events['EventName'] != 'Pre-Season Testing']
    # Filter for completed qualifying sessions
    now = datetime.now(timezone.utc)
    # Convert 'Session4DateUtc' column to timezone-aware datetime objects
    events['Session4DateUtc'] = pd.to_datetime(events['Session4DateUtc'], utc=True) 
    completed_qualis = events[events['Session4DateUtc'] < now]

    # Extract event names
    qualifying_done_events = completed_qualis['EventName'].tolist()
    round_number = st.sidebar.selectbox("Select Race", options=qualifying_done_events, index = 0)

    # Build dataset from 2019 to 2024
    data, qualifying_results = build_dataset(year - 1, year, round_number)
    
    if data.empty:
        print("No data available.")
        return
    
    X = data[['best_time_seconds']]
    y = data['LapTime (s)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
    # Train models
    gb_model, rf_model, xgboost_model = train_models(X_train, y_train)

    # Predict
    final_df = predict(gb_model, rf_model, xgboost_model, qualifying_results)
    
    # Evaluate models
    evaluation_df = evaluate_models(gb_model, rf_model, xgboost_model, X_test, y_test)
    
    # Plot correlation (using data from 2018 onwards)
    plot(final_df, evaluation_df)

if __name__ == "__main__":
    main()
