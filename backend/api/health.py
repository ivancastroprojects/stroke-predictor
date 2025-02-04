from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
import psutil
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def get_system_health():
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        return {
            "memory_usage": f"{memory.percent}%",
            "disk_usage": f"{disk.percent}%",
            "cpu_usage": f"{psutil.cpu_percent()}%"
        }
    except Exception as e:
        logger.error(f"Error obteniendo métricas del sistema: {str(e)}")
        return {}

@app.route('/', methods=['GET'])
def health_check():
    try:
        response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "environment": os.environ.get('VERCEL_ENV', 'development'),
            "python_version": os.environ.get('PYTHON_VERSION', '3.9')
        }
        
        # Solo incluir métricas del sistema en desarrollo
        if os.environ.get('VERCEL_ENV') != 'production':
            response.update(get_system_health())
            
        logger.info(f"Health check exitoso: {response}")
        return jsonify(response)
    except Exception as e:
        error_msg = f"Error en health check: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/detailed', methods=['GET'])
def detailed_health():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'stroke_prediction_model.joblib')
        response = {
            "api_status": "operational",
            "model_exists": os.path.exists(model_path),
            "model_path": model_path,
            "environment": os.environ.get('VERCEL_ENV', 'development'),
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Detailed health check: {response}")
        return jsonify(response)
    except Exception as e:
        error_msg = f"Error en detailed health check: {str(e)}"
        logger.error(error_msg)
        return jsonify({"status": "error", "error": error_msg}), 500

# Vercel handler
app.debug = False
handler = app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 