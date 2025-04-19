## 🏁 F1 Race Prediction App — Real-Time Machine Learning on Formula 1 Data

Welcome to a project that merges **machine learning**, **sports analytics**, and **real-time data engineering** into a sleek and interactive dashboard.

This application predicts **Formula 1 race performance** using actual qualifying data from the FastF1 API. It trains multiple regression models on historical data to estimate fastest race laps for each driver and presents the output in a visually intuitive Streamlit interface.

### [View my Dashboard](https://kn-f1-dashboard.streamlit.app/)

---

### 🚀 Live Features

✅ Predicts race **fastest lap times** for each driver based on qualifying  
✅ Trains **Gradient Boosting**, **Random Forest**, and **XGBoost** regressors  
✅ Fetches and cleans real F1 race and qualifying data from [FastF1](https://theoehrly.github.io/Fast-F1/)  
✅ **MAPE (Mean Absolute Percentage Error)** evaluation for model comparison  
✅ Streamlit-based UI with **interactive gauge charts** and **scrollable DataFrames**  
✅ Automatically filters races with **completed qualifying sessions only**

---

### 🎯 Use Case

> “How can qualifying performance help predict race pace?”

This app investigates that relationship by analyzing data across multiple seasons (since 2018), learning from it, and then applying the insights to **real-time qualifying data** to predict fastest lap performance for a new race.

---

### 🧠 Prediction Logic

1. 🔄 Pull **race data from previous year**, and **qualifying data from current year**
2. 📊 Extract best qualifying time per driver (Q3 > Q2 > Q1)
3. 🏎️ Extract fastest lap time per driver in the race
4. 🎓 Train ML models using qualifying times as inputs
5. 🔮 Predict each driver’s **fastest lap** using their current qualifying time
6. 📉 Evaluate all models using **MAPE**

---

### ✅ MAPE Evaluation

Three gauge charts showing model performance side by side:

- Gradient Boosting
- Random Forest
- XGBoost

### 🔮 Race Prediction Table

| Driver           | Qualifying Time (s) | GB Time (s) | RF Time (s) | XGB Time (s) |
|------------------|---------------------|-------------|-------------|--------------|
| Max Verstappen   | 92.845              | 97.32       | 97.20       | 97.28        |
| Lewis Hamilton   | 93.130              | 97.56       | 97.39       | 97.45        |
| Charles Leclerc  | 93.210              | 97.61       | 97.42       | 97.50        |

---

### 📦 Tech Stack

- 📡 **FastF1 API** – official F1 timing data
- 🧪 **scikit-learn**, **XGBoost** – machine learning models
- 📊 **Plotly**, **Streamlit** – interactive dashboards and visualizations
- 🧹 **pandas**, **numpy** – data manipulation
- 🧠 **MAPE** – model evaluation metric

---

### 🛠️ Getting Started

#### ✅ Prerequisites

```bash
pip install fastf1 streamlit pandas numpy scikit-learn xgboost plotly
```

#### ▶️ Run the App

```bash
streamlit run app.py
```

---

### 🧠 Behind the Scenes

- Uses `Q3 > Q2 > Q1` priority logic to determine each driver's best qualifying time  
- Filters for **completed qualifying sessions** only using `Session4DateUtc` from the event schedule  
- Trains models on qualifying time → race lap time mapping  
- Compares predictions with actual race lap times using MAPE  
- Handles FastF1 fetch errors gracefully with fallback to empty DataFrame

---

### 🎯 Future Improvements

- Include predicted race finishing positions  
- Extend prediction logic using car/team telemetry  
- Plot actual vs. predicted lap times on interactive charts  
- Allow users to select historical races for comparative analysis  

---

### 📂 Project Structure

```
├── app.py               # Streamlit app with full pipeline
├── f1_cache/            # FastF1 cache directory
├── README.md            # This file
```

---

### 👋 About Me

This project was built as part of a **Data Science internship application**, designed to showcase:

- ML proficiency  
- Real-world data wrangling  
- API consumption  
- Real-time dashboard creation

If you'd like to collaborate, extend the project, or just talk F1 + data science — let’s connect!

