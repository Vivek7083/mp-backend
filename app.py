from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask_cors import CORS
import os
import psutil
from waitress import serve
import tensorflow as tf
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
tf.config.set_soft_device_placement(True)  # Better CPU fallback

app = Flask(__name__)
CORS(app)

# Load models and scalers
wind_model = load_model('models/wind_model.h5', custom_objects={'mse': 'mse'})
wind_scaler = joblib.load('models/wind_scaler.pkl')
electric_model = joblib.load('models/electric_load_model.pkl')

# Add memory check middleware
@app.before_request
def check_memory():
    if psutil.virtual_memory().percent > 90:
        return jsonify({"error": "Server overloaded"}), 503

# Modify the main block
if __name__ == '__main__':
    # Production settings
    port = int(os.environ.get("PORT", 10000))
    serve(app, host='0.0.0.0', port=port)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Predict Wind Power
    wind_data = np.array(data['wind_input'])  # Shape: (24, 2) [WindSpeed, ActivePower]
    wind_scaled = wind_scaler.transform(wind_data)
    wind_input = wind_scaled.reshape(1, 24, 2)
    wind_pred = wind_model.predict(wind_input)
    wind_pred_actual = wind_scaler.inverse_transform(
        np.concatenate([wind_pred, np.zeros((1, 1))], axis=1)
    )[0][0]  # Inverse scaling

    # Build DataFrame for Electric Load prediction
    electric_input = pd.DataFrame([{
        'irradiance': data['irradiance'],
        'temperature': data['temperature'],
        'dewpoint': data['dewpoint'],
        'specific humidity': data['humidity'],      # Match training data's column name
        'wind speed': data.get('wind_speed', 0),   # Add missing feature
        'Load_Lag_24': data['load_lag_24'],
        'Temp_Lag_24': data.get('temp_lag_24', 0), # Add missing feature
        'Hour': data['hour']
    }])

    # ─── BEGIN ADDED: Ensure all feature columns are numeric ─────────────────────
    numeric_cols = [
        'irradiance',
        'temperature',
        'dewpoint',
        'specific humidity',
        'wind speed',
        'Load_Lag_24',
        'Temp_Lag_24',
        'Hour'
    ]
    # Convert each column to numeric, raising an error if conversion fails
    electric_input[numeric_cols] = electric_input[numeric_cols].apply(
        lambda col: pd.to_numeric(col, errors='raise')
    )
    # ─── END ADDED ───────────────────────────────────────────────────────────────

    # Predict Electric Load
    electric_pred = electric_model.predict(electric_input)[0]

    return jsonify({
        'wind_power': round(float(wind_pred_actual), 2),
        'electric_load': round(float(electric_pred), 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
