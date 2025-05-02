# app/dashboard.py

import streamlit as st
import pandas as pd
import requests

st.title("🧠 Customer Churn Prediction Dashboard")

# Define UI inputs for each feature
feature_inputs = {}
feature_list_int = ['Age', 'Subscription_Length_Months','Monthly_Bill', 'Total_Usage_GB']
feature_list_str = ['Gender', 'Location', 'Model']
feature_list = ['Age', 'Gender', 'Location', 'Subscription_Length_Months',
       'Monthly_Bill', 'Total_Usage_GB','Model']
    
for feature in feature_list:
    if feature in feature_list_int:
        feature_inputs[feature] = st.number_input(f"{feature}", value=0.0)
    else:
        feature_inputs[feature] = st.text_input(f"{feature}", value='')

if st.button("Predict Churn"):
    response = requests.post("http://localhost:8000/predict", json=feature_inputs)
    if response.status_code == 200:
        result = response.json()
        st.success(f"Churn Prediction: {result['Churn prediction']}")
    else:
        st.error("Failed to get prediction from API.")
