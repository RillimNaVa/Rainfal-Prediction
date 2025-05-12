from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import json

app = Flask(__name__)

# Load all models
models = {}
try:
    models = joblib.load('all_models.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html', models=list(models.keys()))

@app.route('/model_results')
def get_model_results():
    try:
        with open('model_results.json', 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({'error': 'Models not loaded'})
    
    try:
        # Get values from the form
        data = request.get_json()
        features = [
            float(data['temperature']),
            float(data['humidity']),
            float(data['pressure']),
            float(data['wind_speed']),
            float(data['wind_direction'])
        ]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Get selected model
        selected_model = data.get('model', 'Random Forest')
        if selected_model not in models:
            selected_model = 'Random Forest'
        
        model = models[selected_model]
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        return jsonify({
            'prediction': 'Rain' if prediction == 1 else 'No Rain',
            'probability': f'{probability:.2%}',
            'model_used': selected_model
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 