import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

st.title("🛍️ Customer Segmentation using K-Means")

st.write("Enter customer details to predict the segment")

# User inputs
annual_income = st.number_input("Annual Income (k$)", min_value=0.0, step=1.0)
spending_score = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict Cluster"):
    input_data = np.array([[annual_income, spending_score]])
    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"✅ This customer belongs to **Cluster {cluster}**")
