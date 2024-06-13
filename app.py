from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models
logistic_model = pickle.load(open(r'C:\Users\USER\OneDrive\Desktop\churn\model\logistic_regression.pkl', 'rb'))
dnn_model = load_model(r'C:\Users\USER\OneDrive\Desktop\churn\model\dnn_model.h5')
scaler = pickle.load(open(r'C:\Users\USER\OneDrive\Desktop\churn\model\scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form['age'])
    duration_of_pitch = float(request.form['duration_of_pitch'])
    monthly_income = float(request.form['monthly_income'])
    number_of_persons = float(request.form['number_of_persons'])
    preferred_service_star = float(request.form['preferred_service_star'])
    
    # Create feature array
    features = np.array([[age, duration_of_pitch, monthly_income, number_of_persons, preferred_service_star]])
    
    # Scale the input
    features_scaled = scaler.transform(features)
    
    # Logistic Regression Prediction
    logistic_prediction = logistic_model.predict(features_scaled)
    logistic_prob = logistic_model.predict_proba(features_scaled)[0][1]
    
    # DNN Prediction
    dnn_prediction = dnn_model.predict(features_scaled)
    dnn_prob = dnn_prediction[0][0]
    
    # Prepare the response
    response = {
        'logistic_result': int(logistic_prediction[0]),
        'logistic_prob': logistic_prob,
        'dnn_result': int(dnn_prob > 0.5),
        'dnn_prob': dnn_prob
    }
    
    return render_template('index.html', prediction=response)

if __name__ == "__main__":
    app.run(debug=True)
