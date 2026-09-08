# Traffic Situation Predictor

A machine learning web application that predicts traffic conditions from vehicle-count and time-related inputs using a Random Forest classifier.

## Overview

This project demonstrates an end-to-end machine learning workflow for traffic situation prediction.

The Streamlit application accepts:

- Date
- Time
- Day of the week
- Car count
- Bike count
- Bus count
- Truck count

The application calculates the total vehicle count, applies the saved preprocessing scaler, and generates a traffic-situation prediction using a trained Random Forest model.

## Features

- Interactive Streamlit web interface
- Traffic prediction using Random Forest
- Vehicle-count based feature inputs
- Day-of-week encoding
- Automatic total vehicle calculation
- Saved model and scaler artifacts
- Prediction result displayed directly in the application

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Workflow

1. Load and prepare traffic data
2. Prepare traffic-related features
3. Encode the day of the week
4. Calculate total vehicle count
5. Scale the input features
6. Load the trained Random Forest model
7. Generate the traffic-situation prediction

## Project Structure

```text
traffic-predictor/
├── TrafficDataset.csv
├── Untitled.ipynb
├── app.py
├── prediction_result.csv
├── random_forest_model.pkl
├── scaler.pkl
└── requirements.txt
