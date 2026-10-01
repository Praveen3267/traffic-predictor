import streamlit as st
import numpy as np
import joblib


st.set_page_config(
    page_title="Traffic Situation Predictor",
    page_icon="🚦",
    layout="centered"
)


@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("random_forest_model.pkl")
    return model, scaler


model, scaler = load_models()


st.title("🚦 Traffic Situation Predictor")
st.write(
    "Predict traffic conditions using vehicle counts and day of the week."
)


day_mapping = {
    "Friday": 0,
    "Monday": 1,
    "Saturday": 2,
    "Sunday": 3,
    "Thursday": 4,
    "Tuesday": 5,
    "Wednesday": 6
}


day_of_week = st.selectbox(
    "Day of the Week",
    list(day_mapping.keys())
)

car = st.number_input(
    "Car Count",
    min_value=0,
    value=10,
    step=1
)

bike = st.number_input(
    "Bike Count",
    min_value=0,
    value=5,
    step=1
)

bus = st.number_input(
    "Bus Count",
    min_value=0,
    value=2,
    step=1
)

truck = st.number_input(
    "Truck Count",
    min_value=0,
    value=3,
    step=1
)


if st.button("Predict Traffic"):
    try:
        day_encoded = day_mapping[day_of_week]

        total = car + bike + bus + truck

        features = np.array([[
            day_encoded,
            car,
            bike,
            bus,
            truck,
            total
        ]])

        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        confidence = float(np.max(probabilities) * 100)

        st.success(
            f"🚗 Predicted Traffic Situation: **{prediction}**"
        )

        st.info(
            f"Prediction confidence: **{confidence:.2f}%**"
        )

        st.caption(
            f"Total vehicles: {total}"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")