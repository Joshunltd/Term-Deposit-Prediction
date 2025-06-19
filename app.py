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
eager_profile = {
    'age': 41,
    'duration': 1575,
    'campaign': 1,
    'pdays': 999,
    'previous': 0,
    'emp.var.rate': 1.1,
    'cons.price.idx': 93.9,
    'cons.conf.idx': -36.4,
    'euribor3m': 4.8,
    'nr.employed': 5191
}

# This is the user input form
def user_input_form():
    age = st.number_input("Age", 18, 100, int(eager_profile['age']))
    duration = st.number_input("Last contact duration (seconds)", 0, 10000, int(eager_profile['duration']))
    campaign = st.number_input("Number of contacts in campaign", 1, 50, int(eager_profile['campaign']))
    pdays = st.number_input("Days since last contact", -1, 999, int(eager_profile['pdays']))
    previous = st.number_input("Number of previous contacts", 0, 100, int(eager_profile['previous']))
    emp_var_rate = st.number_input("Employment variation rate", -3.0, 3.0, float(eager_profile['emp.var.rate']))
    cons_price_idx = st.number_input("Consumer price index", 90.0, 100.0, float(eager_profile['cons.price.idx']))
    cons_conf_idx = st.number_input("Consumer confidence index", -60.0, 0.0, float(eager_profile['cons.conf.idx']))
    euribor3m = st.number_input("Euribor 3 month rate", 0.0, 6.0, float(eager_profile['euribor3m']))
    nr_employed = st.number_input("Number of employees", 4000.0, 5500.0, float(eager_profile['nr.employed']))

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
input_df = user_input_form()

# The process input
# One-hot encode input to match training set
input_encoded = pd.get_dummies(input_df)

# Adding any missing columns (from training)
missing_cols = set(feature_names) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0

# Ensuring column order matches training
input_encoded = input_encoded[feature_names]

# This is for the scale numeric features
input_encoded[numeric_features] = scaler.transform(input_encoded[numeric_features])

# For the prediction
if st.button("Predict"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]
    threshold = 0.2

    result = "YES ✅" if pred == 1 else "NO ❌"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Subscription: **{prob:.2%}**")
    st.info(f"Using adjusted threshold: {threshold}")
   
