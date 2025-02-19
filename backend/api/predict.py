from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS, cross_origin
import joblib
import numpy as np
import pandas as pd
import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear Blueprint
predict_bp = Blueprint('predict', __name__)

# Configurar CORS para el blueprint
CORS(predict_bp, resources={
    r"/api/predict": {
        "origins": ["http://localhost:3000", "https://*.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Cargar el modelo solo cuando se necesite (lazy loading)
model = None

def load_model():
    global model
    if model is None:
        try:
            # Construir la ruta al modelo
            backend_dir = Path(__file__).parent.parent
            model_path = backend_dir / 'models' / 'stroke_prediction_model.joblib'
            logger.info(f"Buscando modelo en: {model_path}")

            if model_path.exists():
                model = joblib.load(str(model_path))
                logger.info(f"Modelo cargado exitosamente desde: {model_path}")
                return True
            else:
                logger.error(f"El archivo del modelo no existe en: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error cargando el modelo: {str(e)}")
            return False
    return True

def validate_input(data):
    required_fields = [
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    errors = []
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Campo requerido: {field}")
            continue
        
        try:
            if field in ['age', 'avg_glucose_level', 'bmi']:
                value = float(data[field])
                if field == 'age' and (value < 0 or value > 120):
                    errors.append("La edad debe estar entre 0 y 120 años")
                elif field == 'avg_glucose_level' and (value < 0 or value > 500):
                    errors.append("El nivel de glucosa debe estar entre 0 y 500")
                elif field == 'bmi' and (value < 10 or value > 100):
                    errors.append("El BMI debe estar entre 10 y 100")
            elif field in ['hypertension', 'heart_disease']:
                value = int(data[field])
                if value not in [0, 1]:
                    errors.append(f"{field} debe ser 0 o 1")
            elif field == 'gender':
                if data[field] not in ['Male', 'Female', 'Other']:
                    errors.append("Género debe ser 'Male', 'Female' u 'Other'")
            elif field == 'ever_married':
                if data[field] not in ['Yes', 'No']:
                    errors.append("ever_married debe ser 'Yes' o 'No'")
            elif field == 'work_type':
                if data[field] not in ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']:
                    errors.append("work_type inválido")
            elif field == 'Residence_type':
                if data[field] not in ['Urban', 'Rural']:
                    errors.append("Residence_type debe ser 'Urban' o 'Rural'")
            elif field == 'smoking_status':
                if data[field] not in ['formerly smoked', 'never smoked', 'smokes', 'Unknown']:
                    errors.append("smoking_status inválido")
        except ValueError:
            errors.append(f"Valor inválido para {field}")
    
    return errors

@predict_bp.route('/api/predict', methods=['POST', 'OPTIONS'])
@cross_origin()
def predict():
    if request.method == 'OPTIONS':
        # Manejar la solicitud preflight
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    if not load_model():
        return jsonify({
            'error': 'Error al cargar el modelo',
            'status': 503
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No se proporcionaron datos',
                'status': 400
            }), 400

        # Validar input
        errors = validate_input(data)
        if errors:
            return jsonify({
                'error': 'Datos inválidos',
                'details': errors,
                'status': 400
            }), 400

        # Preparar datos para predicción
        features_df = pd.DataFrame([{
            'gender': data['gender'],
            'age': float(data['age']),
            'hypertension': float(data['hypertension']),
            'heart_disease': float(data['heart_disease']),
            'ever_married': data['ever_married'],
            'work_type': data['work_type'],
            'Residence_type': data['Residence_type'],
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data['bmi']),
            'smoking_status': data['smoking_status']
        }])

        # Realizar predicción
        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)[0][1]

        logger.info(f"Datos de entrada: {features_df.to_dict('records')[0]}")
        logger.info(f"Predicción: {prediction[0]}, Probabilidad: {probability}")

        # Calcular las contribuciones de las características
        feature_importance = {
            'age': 0.25,
            'hypertension': 0.20,
            'heart_disease': 0.15,
            'avg_glucose_level': 0.15,
            'bmi': 0.10,
            'smoking_status': 0.08,
            'work_type': 0.07
        }

        # Analizar factores de riesgo específicos
        risk_factors = []
        if float(features_df['age'].iloc[0]) > 65:
            risk_factors.append({'factor': 'Age', 'message': 'La edad superior a 65 años aumenta significativamente el riesgo de accidente cerebrovascular'})
        if features_df['hypertension'].iloc[0] == 1:
            risk_factors.append({'factor': 'Hipertensión', 'message': 'El historial de presión arterial alta es un factor de riesgo importante'})
        if features_df['heart_disease'].iloc[0] == 1:
            risk_factors.append({'factor': 'Enfermedad Cardíaca', 'message': 'Las condiciones cardíacas existentes aumentan el riesgo de accidente cerebrovascular'})
        if float(features_df['avg_glucose_level'].iloc[0]) > 140:
            risk_factors.append({'factor': 'Glucosa Alta', 'message': 'Los niveles elevados de glucosa aumentan el riesgo'})
        if float(features_df['bmi'].iloc[0]) > 25:
            risk_factors.append({'factor': 'IMC', 'message': 'Un IMC por encima del rango normal aumenta el riesgo'})
        if features_df['smoking_status'].iloc[0] in ['formerly smoked', 'smokes']:
            risk_factors.append({'factor': 'Tabaquismo', 'message': 'El historial de tabaquismo aumenta los riesgos cardiovasculares'})

        # Calcular contribuciones específicas
        feature_contributions = {}
        for feature, importance in feature_importance.items():
            value = features_df[feature].iloc[0]
            if feature == 'age':
                contribution = min(1.0, float(value) / 100) * importance
            elif feature in ['hypertension', 'heart_disease']:
                contribution = float(value) * importance
            elif feature == 'avg_glucose_level':
                contribution = min(1.0, float(value) / 200) * importance
            elif feature == 'bmi':
                contribution = min(1.0, (float(value) - 18.5) / 30) * importance
            elif feature == 'smoking_status':
                contribution = (value in ['formerly smoked', 'smokes']) * importance
            else:
                contribution = 0.5 * importance
            feature_contributions[feature] = contribution

        logger.info(f"Feature contributions: {feature_contributions}")
        logger.info(f"Risk factors: {risk_factors}")

        response_data = {
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'feature_contributions': feature_contributions,
            'risk_factors': risk_factors,
            'feature_importance': feature_importance,
            'status': 'success'
        }

        logger.info(f"Enviando respuesta: {response_data}")
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error en la predicción: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'status': 500
        }), 500

@predict_bp.route('/api/predict/test', methods=['GET'])
@cross_origin()
def test_predict():
    """Endpoint de prueba con datos de ejemplo"""
    if not load_model():
        return jsonify({
            'error': 'Servicio no disponible temporalmente',
            'status': 503
        }), 503

    test_data = {
        'gender': 'Male',
        'age': 65,
        'hypertension': 1,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 169.5,
        'bmi': 35.5,
        'smoking_status': 'formerly smoked'
    }

    try:
        features_df = pd.DataFrame([{
            'gender': test_data['gender'],
            'age': float(test_data['age']),
            'hypertension': float(test_data['hypertension']),
            'heart_disease': float(test_data['heart_disease']),
            'ever_married': test_data['ever_married'],
            'work_type': test_data['work_type'],
            'Residence_type': test_data['Residence_type'],
            'avg_glucose_level': float(test_data['avg_glucose_level']),
            'bmi': float(test_data['bmi']),
            'smoking_status': test_data['smoking_status']
        }])

        prediction = model.predict(features_df)
        probability = model.predict_proba(features_df)[0][1]

        return jsonify({
            'test_data': test_data,
            'prediction': int(prediction[0]),
            'probability': float(probability),
            'status': 'success'
        })

    except Exception as e:
        logger.error(f"Error en la predicción de prueba: {e}")
        return jsonify({
            'error': 'Error en la predicción de prueba',
            'status': 500
        }), 500

if __name__ == '__main__':
    # Crear la aplicación Flask solo si se ejecuta directamente este archivo
    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(predict_bp)
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f'Iniciando servidor en puerto {port}, debug={debug}')
    app.run(host='0.0.0.0', port=port, debug=debug) 