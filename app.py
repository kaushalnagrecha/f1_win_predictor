import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone

# Suppress the messy FastF1 diagnostic logs in the terminal
fastf1.set_log_level('ERROR')

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_cache') 

@st.cache_data(show_spinner=False)
def build_dataset(current_year, event_name):
    """
    Builds a robust training dataset by fetching the last 3 occurrences of the event.
    Target variable: Driver's Fastest Race Lap Time.
    """
    historical_data = []
    
    # 1. Search backwards to find up to 3 past occurrences of the event
    past_years = []
    for y in range(current_year - 1, 2017, -1):
        try:
            schedule = fastf1.get_event_schedule(y)
            if event_name in schedule['EventName'].values:
                event_info = schedule[schedule['EventName'] == event_name].iloc[0]
                if event_info['EventFormat'] != 'testing':
                    past_years.append(y)
            if len(past_years) == 3:
                break
        except Exception:
            continue
            
    if not past_years:
        return pd.DataFrame(), pd.DataFrame(), None

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, year in enumerate(past_years):
        status_text.text(f"Fetching {year} historical data for training...")
        try:
            # Q Session (Qualifying Pace)
            # Setting telemetry/weather/messages to False makes fetching lightning fast!
            q_session = fastf1.get_session(year, event_name, 'Q')
            q_session.load(telemetry=False, weather=False, messages=False)
            q_results = q_session.results.copy()
            q_results['best_time'] = q_results[['Q3', 'Q2', 'Q1']].bfill(axis=1).iloc[:, 0]
            q_results['Qualifying Time (s)'] = q_results['best_time'].dt.total_seconds()
            
            # Fill missing quali times
            max_q = q_results['Qualifying Time (s)'].max() + 10
            q_results['Qualifying Time (s)'] = q_results['Qualifying Time (s)'].fillna(max_q)
            
            # R Session (Fastest Race Laps)
            r_session = fastf1.get_session(year, event_name, 'R')
            r_session.load(telemetry=False, weather=False, messages=False)
            
            # Extract the absolute fastest lap per driver
            laps = r_session.laps[["Driver", "LapTime"]].copy()
            laps.dropna(subset=["LapTime"], inplace=True)
            laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()
            fastest_laps = laps.groupby("Driver")["LapTime (s)"].min().reset_index()
            fastest_laps.rename(columns={"LapTime (s)": "Fastest Race Lap (s)"}, inplace=True)
            
            # Merge
            merged = fastest_laps.merge(q_results[['Abbreviation', 'Qualifying Time (s)']], left_on='Driver', right_on='Abbreviation')
            historical_data.append(merged)
        except Exception:
            pass # Skip if that specific year is missing data
            
        progress_bar.progress((i + 1) / len(past_years))
    
    status_text.text("Fetching current year's qualifying data...")
    try:
        qual_session_current = fastf1.get_session(current_year, event_name, 'Q')
        qual_session_current.load(telemetry=False, weather=False, messages=False)
        curr_results = qual_session_current.results.copy()
        curr_results['best_time'] = curr_results[['Q3', 'Q2', 'Q1']].bfill(axis=1).iloc[:, 0]
        curr_results['Qualifying Time (s)'] = curr_results['best_time'].dt.total_seconds()
        max_time_curr = curr_results['Qualifying Time (s)'].max() + 10
        curr_results['Qualifying Time (s)'] = curr_results['Qualifying Time (s)'].fillna(max_time_curr)
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        st.warning(f"Failed to load current qualifying data: {e}")
        return pd.DataFrame(), pd.DataFrame(), None
        
    status_text.empty()
    progress_bar.empty()

    if not historical_data:
        return pd.DataFrame(), pd.DataFrame(), None
        
    final_training_data = pd.concat(historical_data, ignore_index=True)
    return final_training_data, curr_results, past_years

def train_models(X_train, y_train):
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=35)
    gb_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=35)
    rf_model.fit(X_train, y_train)

    xgboost_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=35)
    xgboost_model.fit(X_train, y_train)
    return gb_model, rf_model, xgboost_model

def predict(gb_model, rf_model, xgboost_model, qualifying_results):
    features = qualifying_results[['Qualifying Time (s)']]
    gb_y_pred = gb_model.predict(features)
    rf_y_pred = rf_model.predict(features)
    xgb_y_pred = xgboost_model.predict(features)
    
    final_df = pd.DataFrame({
        'Driver': qualifying_results['FullName'],
        'Qualifying Time (s)': qualifying_results['Qualifying Time (s)'],
        'GB Predicted Fastest Lap (s)': gb_y_pred,
        'RF Predicted Fastest Lap (s)': rf_y_pred,
        'XGB Predicted Fastest Lap (s)': xgb_y_pred
    })
    
    # Sort by XGB predicted time by default, so the driver predicted to have the fastest lap is at the top
    final_df = final_df.sort_values('XGB Predicted Fastest Lap (s)').reset_index(drop=True)
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
    upper_limit = int(value + 2) if not np.isnan(value) else 100
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
    
    st.subheader('🔮 Fastest Race Lap Predictions')
    # Render Predicted DataFrame
    st.dataframe(final_df, height=300)

def main():

    st.title("🏎️ F1 Race Prediction App")

    st.sidebar.header("Race Settings")
    year = st.sidebar.number_input("Year", disabled=True, value=datetime.now().year)
    events = fastf1.get_event_schedule(year)
    
    # Use .copy() here to avoid the Pandas chained assignment warning later
    events = events[events['EventName'] != 'Pre-Season Testing'].copy()
    
    # Filter for completed qualifying sessions
    now = datetime.now(timezone.utc)
    events['Session4DateUtc'] = pd.to_datetime(events['Session4DateUtc'], utc=True) 
    completed_qualis = events[events['Session4DateUtc'] < now]

    # Extract event names
    qualifying_done_events = completed_qualis['EventName'].tolist()
    if not qualifying_done_events:
        st.warning("No completed qualifying sessions found yet.")
        return
        
    event_name = st.sidebar.selectbox("Select Race", options=qualifying_done_events, index=0)

    # Build robust historical dataset (now fetches multiple years)
    data, qualifying_results, training_years = build_dataset(year, event_name)
    
    if data.empty or qualifying_results.empty:
        st.error("Data is currently unavailable for this session. It's possible FastF1 hasn't published the race telemetry yet.")
        return
        
    years_str = ", ".join(map(str, training_years))
    st.sidebar.success(f"Model trained on robust historical data from: **{years_str}**.")
    
    X = data[['Qualifying Time (s)']]
    y = data['Fastest Race Lap (s)']
    
    if len(data) > 5:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)
        
        # Train models
        gb_model, rf_model, xgboost_model = train_models(X_train, y_train)

        # Predict
        final_df = predict(gb_model, rf_model, xgboost_model, qualifying_results)
        
        # Evaluate models
        evaluation_df = evaluate_models(gb_model, rf_model, xgboost_model, X_test, y_test)
        
        # Plot
        plot(final_df, evaluation_df)
    else:
        st.error("Not enough historical data available to accurately train the models.")

if __name__ == "__main__":
    main()
