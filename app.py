
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Label encodings (based on training)
day_mapping = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}

traffic_mapping = {0: "heavy", 1: "high", 2: "low", 3: "normal"}

st.title("🚦 Traffic Situation Predictor")

# Input fields
time = st.time_input("Select Time")
date = st.date_input("Select Date")
day_of_week = st.selectbox("Day of the Week", list(day_mapping.keys()))
car_count = st.number_input("Car Count", min_value=0, value=10)
bike_count = st.number_input("Bike Count", min_value=0, value=5)
bus_count = st.number_input("Bus Count", min_value=0, value=2)
truck_count = st.number_input("Truck Count", min_value=0, value=3)

if st.button("Predict Traffic"):
    # Convert inputs to model-ready format
    time_minutes = time.hour * 60 + time.minute
    days_since_ref = (date - datetime(2023, 10, 9).date()).days
    day_encoded = day_mapping[day_of_week]

    features = np.array([[time_minutes, days_since_ref, day_encoded,
                          car_count, bike_count, bus_count, truck_count]])
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0]

    # Display results
    st.success(f"Predicted Traffic Situation: **{traffic_mapping[prediction]}**")
    st.info(f"Prediction Confidence: {round(np.max(proba) * 100, 2)}%")
