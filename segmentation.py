import streamlit as st
import pandas as pd
import joblib

# --- Title and description ---
st.title("🧭 Customer Segmentation Prediction App")
st.write("Enter customer details below to predict which segment they belong to.")

# --- Load trained models safely ---
try:
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    st.success("Models loaded successfully ✅")
except FileNotFoundError:
    st.error("Model files not found. Please place 'kmeans_model.pkl' and 'scaler.pkl' in this folder.")
    st.stop()

# --- User inputs ---
age = st.number_input("Age", min_value=18, max_value=100, value=35)
income = st.number_input("Annual Income (€)", min_value=0, max_value=200000, value=50000)
total_spending = st.number_input("Total Spending", min_value=0, max_value=50000, value=2000)
num_web_purchases = st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)
num_store_purchases = st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=5)
num_web_visits = st.number_input("Number of Web Visits per Month", min_value=0, max_value=100, value=8)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

# --- Prepare input data ---
input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_spending': [total_spending],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [num_web_visits],
    'Recency': [recency]
})

# --- Scale and predict ---
if st.button("Predict Segment"):
    try:
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)
        st.success(f"🎯 The predicted customer segment is: **Cluster {int(cluster[0])}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
