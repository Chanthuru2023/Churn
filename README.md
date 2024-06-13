# Customer Plan Prediction

This project predicts whether a customer will take a plan based on key features using both Logistic Regression and Deep Neural Network (DNN) models. The application is built using Flask for the backend and provides an interface for users to input customer data and get predictions.

## Features

- **Logistic Regression Model**: Predicts the likelihood of a customer taking the plan.
- **DNN Model**: Provides another prediction of the likelihood with potentially better accuracy.
- **Flask Backend**: Handles the model predictions and serves the web application.
- **Simple Web Interface**: Allows users to input key customer features and see the predictions.

## Key Features Used

- Age
- Duration of Pitch
- Monthly Income
- Number of Persons
- Preferred Service Star

## Setup

### Prerequisites

- Python 3.x
- Flask
- NumPy
- Pandas
- scikit-learn
- TensorFlow

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/customer-plan-prediction.git
   cd customer-plan-prediction
