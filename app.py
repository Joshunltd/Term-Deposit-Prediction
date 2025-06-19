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
# Define eager profile values
prefill = {
    'age': 35,
    'duration': 620,
    'campaign': 1,
    'pdays': 0,
    'previous': 2,
    'emp.var.rate': -1.1,
    'cons.price.idx': 93.0,
    'cons.conf.idx': -45.0,
    'euribor3m': 1.0,
    'nr.employed': 5000.0
}

# This is the user inout form
def user_input_form(prefill):
    age = st.number_input("Age", 18, 100, prefill['age'])
    duration = st.number_input("Last contact duration (seconds)", 0, 10000, prefill['duration'])
    campaign = st.number_input("Number of contacts in campaign", 1, 50, prefill['campaign'])
    pdays = st.number_input("Days since last contact", -1, 999, prefill['pdays'])
    previous = st.number_input("Number of previous contacts", 0, 100, prefill['previous'])
    emp_var_rate = st.number_input("Employment variation rate", -3.0, 3.0, prefill['emp.var.rate'])
    cons_price_idx = st.number_input("Consumer price index", 90.0, 100.0, prefill['cons.price.idx'])
    cons_conf_idx = st.number_input("Consumer confidence index", -60.0, 0.0, prefill['cons.conf.idx'])
    euribor3m = st.number_input("Euribor 3 month rate", 0.0, 6.0, prefill['euribor3m'])
    nr_employed = st.number_input("Number of employees", 4000.0, 5500.0, prefill['nr.employed'])

    input_dict = {
        'age': age,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }

    return pd.DataFrame([input_dict])

# Getting user input
input_df = user_input_form(prefill)

# The process input
# One-hot encode input to match training set
input_encoded = pd.get_dummies(input_df)

# adding any missing columns (from training)
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
    threshold = 0.29
    prediction = "YES ✅" if proba_yes >= threshold else "NO ❌"

    st.write(f"Debug - Raw Probability of Yes: {proba_yes:.4f}")
    st.subheader(f"Prediction: {prediction}")
    st.write(f"Probability of Subscription: **{proba_yes:.2%}**")
    st.info(f"Custom threshold used: {threshold}")
