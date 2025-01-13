from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Cargar el modelo
try:
    model = joblib.load('backend/models/stroke_model.joblib')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json
        features = np.array([[
            data['age'],
            data['hypertension'],
            data['heart_disease'],
            data['avg_glucose_level'],
            data['bmi']
        ]])
        
        prediction = model.predict_proba(features)[0][1]
        return jsonify({"probability": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Vercel requires a handler
handler = app 