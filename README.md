
# Term Deposit Subscription Prediction

This project uses machine learning to predict whether a client will subscribe to a term deposit based on a Portuguese bank’s marketing data. It includes full exploratory data analysis, feature engineering, model development, evaluation, and deployment via a Streamlit web app.

---

## Project Overview

- **Objective**: Predict client subscription to a term deposit ('yes' or 'no') using features such as call duration, employment status, and economic indicators.
- **Dataset**: Bank Marketing Data from a Portuguese financial institution (`bank-additional-full.csv`)
- **Target Variable**: 'y' (subscription to a term deposit)

---

## Machine Learning Pipeline

1. **Exploratory Data Analysis (EDA)**
   - Data profiling, class distribution, correlation matrix, and mutual information ranking
2. **Preprocessing**
   - One-hot encoding for categorical features
   - Standardization of numerical features
   - Addressing class imbalance using **SMOTE**
3. **Modeling**
   - **Random Forest Classifier** with balanced class weights
   - Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
4. **Feature Importance**
   - Analyzed top contributing features such as `duration`, `euribor3m`, and `nr.employed`
5. **Deployment**
   - Streamlit app for real-time predictions

---

## Performance Metrics

- **Accuracy**: ~90%
- **Precision**: High
- **Recall**: High
- **F1 Score**: Balanced and strong
- Model is well-generalized and effective on imbalanced data due to SMOTE.

---

## 💻 Streamlit App

An interactive Streamlit application is included for end users to test the model in real time by entering client attributes and receiving a prediction.

```bash
# Run the app locally
streamlit run app.py
```

---

## 🗂️ Project Structure

```
├── data/
│   └── bank-additional-full.csv
├── notebooks/
│   └── analysis.ipynb
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── numeric_features.pkl
    └── feature_names.pkl
├── app.py               # Streamlit app
├── requirements.txt     # Python dependencies
├── README.md            # Project overview
```

---

## ⚙️ Requirements

- Python 3.13+
- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn
- streamlit

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📌 Key Insights

- Call duration and economic variables are the strongest predictors.
- Clients contacted in specific months or under certain economic conditions are more likely to subscribe.
- SMOTE significantly improved model sensitivity to the minority class.

---

## 📫 Contact

For questions or feedback, feel free to reach out!
