import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Loading saved objects
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("numeric_features.pkl", "rb") as f:
    numeric_features = pickle.load(f)

st.title("Term Deposit Subscription Predictor")

# Defining by prefilling with top 10 features
prefill = {
    'age': 51,
    'duration': 687,
    'pdays': 0,
    'previous': 1
}

# This is the user inout form
def user_input_form(prefill):
    age = st.number_input("Age", 18, 100, prefill['age'])
    duration = st.number_input("Last contact duration (seconds)", 0, 10000, prefill['duration'])
    pdays = st.number_input("Days since last contact", -1, 999, prefill['pdays'])
    previous = st.number_input("Number of previous contacts", 0, 100, prefill['previous'])

    input_dict = {
        'age': age,
        'duration': duration,
        'pdays': pdays,
        'previous': previous
    }

    return pd.DataFrame([input_dict])

# Getting user input
input_df = user_input_form(prefill)

# The process input
# One-hot encode input to match training set
input_encoded = pd.get_dummies(input_df)

# Adding any missing columns (from training)
missing_cols = set(feature_names) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0

# ensuring column order matches training
input_encoded = input_encoded[feature_names]

# This is for the scale numeric features
input_encoded[numeric_features] = scaler.transform(input_encoded[numeric_features])

# For the prediction
if st.button("🔍 Predict"):
    proba_yes = model.predict_proba(input_encoded)[0][1]
    threshold = 0.3
    prediction = "YES ✅" if proba_yes >= threshold else "NO ❌"

    st.write(f"Debug - Raw Probability of Yes: {proba_yes:.4f}")
    st.subheader(f"Prediction: {prediction}")
    st.write(f"Probability of Subscription: **{proba_yes:.2%}**")
    st.info(f"Custom threshold used: {threshold}")
