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

# Definir las columnas esperadas por el modelo
EXPECTED_COLUMNS = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

def find_model_file():
    """Busca el archivo del modelo en la carpeta models"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        print(f"Directorio de modelos no encontrado: {models_dir}")
        return None
        
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
    feature_importance_path = os.path.join(os.path.dirname(__file__), 'models', 'feature_importance.joblib')
    feature_importance = load(feature_importance_path)
    print(f"Modelo cargado desde: {model_path}")
    print(f"Importancia de características cargada desde: {feature_importance_path}")
    
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

def prepare_input_data(data):
    """Prepara los datos de entrada asegurando que todas las columnas necesarias estén presentes"""
    # Convertir valores a minúsculas para consistencia
    data = {k.lower(): v for k, v in data.items()}
    
    # Asegurar que residence_type esté presente y con la primera letra en mayúscula
    if 'residence_type' in data:
        data['Residence_type'] = data.pop('residence_type')
    elif 'Residence_type' not in data:
        data['Residence_type'] = 'Urban'  # valor por defecto
    
    # Crear un diccionario con valores por defecto para columnas faltantes
    default_values = {
        'gender': 'Male',
        'age': '45',
        'hypertension': '0',
        'heart_disease': '0',
        'ever_married': 'No',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': '100',
        'bmi': '25',
        'smoking_status': 'never smoked'
    }
    
    # Completar valores faltantes
    for col in EXPECTED_COLUMNS:
        if col not in data:
            data[col] = default_values[col]
    
    # Crear DataFrame asegurando el orden correcto de las columnas
    df = pd.DataFrame([data])[EXPECTED_COLUMNS]
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Datos recibidos:", data)  # Debug
        
        # Preparar datos asegurando todas las columnas necesarias
        input_data = prepare_input_data(data)
        print("Datos preparados:", input_data)  # Debug
        
        # Usar el modelo ya cargado globalmente
        if model is None:
            raise Exception("El modelo no está cargado correctamente")
            
        # Realizar predicción
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # Usar feature_importance ya cargado globalmente
        # Calcular contribuciones específicas para este caso
        feature_contributions = {}
        for feature, base_importance in feature_importance.items():
            # Ajustar importancia según los valores del paciente
            if feature == 'Edad' and float(data['age']) > 60:
                contribution = base_importance
            elif feature == 'Hipertensión' and data['hypertension'] == '1':
                contribution = base_importance
            elif feature == 'Nivel de Glucosa' and float(data['avg_glucose_level']) > 125:
                contribution = base_importance
            elif feature == 'IMC' and float(data['bmi']) > 30:
                contribution = base_importance
            elif feature == 'Enf. Cardíacas' and data['heart_disease'] == '1':
                contribution = base_importance
            else:
                contribution = base_importance * 0.5
            
            feature_contributions[feature] = contribution
        
        # Normalizar las contribuciones
        total_contribution = sum(feature_contributions.values())
        feature_contributions = {k: (v/total_contribution)*100 
                               for k, v in feature_contributions.items()}
        
        response = {
            'prediction': float(prediction_proba),
            'feature_contributions': feature_contributions,
            'risk_factors': {
                'age': float(data['age']),
                'hypertension': data['hypertension'] == '1',
                'heart_disease': data['heart_disease'] == '1',
                'avg_glucose_level': float(data['avg_glucose_level']),
                'bmi': float(data['bmi'])
            },
            'feature_importance': feature_importance
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error en predicción: {str(e)}")  # Debug
        return jsonify({'error': str(e)}), 500

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