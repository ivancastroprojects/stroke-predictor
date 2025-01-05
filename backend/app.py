from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"],
        "max_age": 3600
    }
})

def find_model_file():
    """Busca el archivo del modelo en la carpeta models"""
    models_dir = './models'
    for file in os.listdir(models_dir):
        if file.startswith('stroke_prediction_model') and file.endswith('.joblib'):
            return os.path.join(models_dir, file)
    return None

# Cargar el modelo y la importancia de características
try:
    model_path = find_model_file()
    if model_path is None:
        raise FileNotFoundError("No se encontró ningún archivo de modelo en la carpeta models")
    
    model = load(model_path)
    feature_importance = load('./models/feature_importance.joblib')
    print(f"Modelo cargado desde: {model_path}")
    
except Exception as e:
    print(f"Error cargando modelos: {e}")
    model = None
    # Proporcionar datos de ejemplo si los archivos no existen
    feature_importance = {
        'Edad': 0,
        'Hipertensión': 0,
        'Nivel de Glucosa': 0,
        'IMC': 0,
        'Enf. Cardíacas': 0,
        'Tabaquismo': 0,
        'Otros Factores': 0
    }

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if model is None:  # Verificar si el modelo se cargó correctamente
        return jsonify({"error": "Modelo no disponible"}), 500

    try:
        data = request.json
        print("Received data:", data)  # Debug print
        
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        print(f"Prediction: {prediction}")  # Debug print
        
        return jsonify({"prediction": int(prediction)}), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

@app.route('/feature-importance', methods=['GET', 'OPTIONS'])
def get_feature_importance():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        return jsonify(feature_importance)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)