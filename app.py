
import streamlit as st
import numpy as np
import joblib
from datetime import datetime

st.title("🚦 Traffic Situation Predictor")

# Input fields
date = st.date_input("Date")
time_input = st.time_input("Time")
day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
car = st.number_input("Car Count", min_value=0)
bike = st.number_input("Bike Count", min_value=0)
bus = st.number_input("Bus Count", min_value=0)
truck = st.number_input("Truck Count", min_value=0)

submitted = st.button("Predict Traffic")

if submitted:
    try:
        # Encode day of week
        day_mapping = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }
        day_encoded = day_mapping[day_of_week]

        # Calculate total vehicles
        total = car + bike + bus + truck

        # Prepare features (must match the 6 used in training)
        features = np.array([[day_encoded, car, bike, bus, truck, total]])

        # Load scaler and model
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("random_forest_model.pkl")

        # Scale and predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]

        st.success(f"🚗 Predicted Traffic Situation: **{prediction}**")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

