from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load models
logistic_model = pickle.load(open(r'C:\Users\USER\OneDrive\Desktop\churn\model\logistic_regression.pkl', 'rb'))
dnn_model = load_model(r'C:\Users\USER\OneDrive\Desktop\churn\model\dnn_model.h5')

# Load scaler
scaler = pickle.load(open(r'C:\Users\USER\OneDrive\Desktop\churn\model\scaler.pkl, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    
    # Scale the input
    final_features_scaled = scaler.transform(final_features)
    
    # Logistic Regression Prediction
    logistic_prediction = logistic_model.predict(final_features_scaled)
    logistic_prob = logistic_model.predict_proba(final_features_scaled)[0][1]
    
    # DNN Prediction
    dnn_prediction = dnn_model.predict(final_features_scaled)
    dnn_prob = dnn_prediction[0][0]
    
    # Prepare the response
    response = {
        'logistic_result': logistic_prediction[0],
        'logistic_prob': logistic_prob,
        'dnn_result': int(dnn_prob > 0.5),
        'dnn_prob': dnn_prob
    }
    
    return render_template('index.html', prediction=response)

if __name__ == "__main__":
    app.run(debug=True)
