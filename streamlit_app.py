import streamlit as st
import requests
import numpy as np

st.title("Model Predictions")

# Collect user input
feature1 = st.text_input("Feature 1")
# Add more input fields for each feature

if st.button('Predict'):
    # Prepare data for the POST request
    data = {
        'feature1': feature1,
        # Add all features
    }
    response = requests.post('http://localhost:5000/predict', data=data)
    prediction = response.json()

    st.write("### Logistic Regression Prediction")
    st.write("Prediction: ", prediction['logistic_result'])
    st.write("Probability: ", prediction['logistic_prob'])

    st.write("### DNN Prediction")
    st.write("Prediction: ", prediction['dnn_result'])
    st.write("Probability: ", prediction['dnn_prob'])
