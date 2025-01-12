from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from joblib import load
import os

app = Flask(__name__)
CORS(app)

# Columnas esperadas por el modelo
EXPECTED_COLUMNS = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# Cargar el modelo
model_path = os.path.join(os.path.dirname(__file__), 'models', 'stroke_prediction_model.joblib')
model = load(model_path) if os.path.exists(model_path) else None

def prepare_input_data(data):
    """Prepara los datos de entrada de forma más eficiente"""
    processed_data = []
    for col in EXPECTED_COLUMNS:
        processed_data.append(data.get(col, '0'))
    return np.array([processed_data])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not model:
            return jsonify({'error': 'Model not loaded'}), 500

        input_data = prepare_input_data(data)
        prediction = float(model.predict_proba(input_data)[0][1])

        return jsonify({
            'prediction': prediction,
            'risk_factors': {
                'age': float(data.get('age', 0)),
                'hypertension': data.get('hypertension') == '1',
                'heart_disease': data.get('heart_disease') == '1',
                'avg_glucose_level': float(data.get('avg_glucose_level', 0)),
                'bmi': float(data.get('bmi', 0))
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Stroke Prediction API"

if __name__ == "__main__":
    app.run(debug=True)