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

# This is the user inout form
def user_input_form():
    age = st.number_input("Age", 18, 100, 30)
    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
        'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'
    ])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y',
                                           'high.school', 'illiterate', 'professional.course',
                                           'university.degree', 'unknown'])
    default = st.selectbox("Has credit in default?", ['no', 'yes', 'unknown'])
    housing = st.selectbox("Has housing loan?", ['no', 'yes', 'unknown'])
    loan = st.selectbox("Has personal loan?", ['no', 'yes', 'unknown'])
    contact = st.selectbox("Contact communication type", ['cellular', 'telephone'])
    month = st.selectbox("Last contact month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox("Last contact day", ['mon', 'tue', 'wed', 'thu', 'fri'])

    duration = st.number_input("Last contact duration (seconds)", 0, 10000, 300)
    campaign = st.number_input("Number of contacts in campaign", 1, 50, 1)
    pdays = st.number_input("Days since last contact", -1, 999, -1)
    previous = st.number_input("Number of previous contacts", 0, 100, 0)
    emp_var_rate = st.number_input("Employment variation rate", -3.0, 3.0, 1.1)
    cons_price_idx = st.number_input("Consumer price index", 90.0, 100.0, 93.0)
    cons_conf_idx = st.number_input("Consumer confidence index", -60.0, 0.0, -40.0)
    euribor3m = st.number_input("Euribor 3 month rate", 0.0, 6.0, 4.5)
    nr_employed = st.number_input("Number of employees", 4000.0, 5500.0, 5200.0)

    input_dict = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
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

# adding any missing columns (from training)
missing_cols = set(feature_names) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0

# ensuring column order matches training
input_encoded = input_encoded[feature_names]

# This is for the scale numeric features
input_encoded[numeric_features] = scaler.transform(input_encoded[numeric_features])

# For the prediction
if st.button("Predict"):
    pred = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    result = "YES ✅" if pred == 1 else "NO ❌"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Subscription: **{prob:.2%}**")