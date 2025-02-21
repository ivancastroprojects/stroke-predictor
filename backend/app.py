from flask import Flask, jsonify, request
from flask_cors import CORS
from api.predict import predict_bp
import os
import logging
import sys
from pathlib import Path
from werkzeug.middleware.proxy_fix import ProxyFix

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asegurarse de que el modelo esté en el PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
model_path = os.path.join(current_dir, 'models')
sys.path.append(model_path)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configuración de CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "https://*.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Configuración de la aplicación
app.config.update(
    ENV='development',
    DEBUG=True,
    PROPAGATE_EXCEPTIONS=True
)

# Registrar el blueprint de predicción
app.register_blueprint(predict_bp, url_prefix='/api')

@app.route('/api', methods=['GET'])
def home():
    return jsonify({"message": "Stroke Prediction API"})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "/api/health": "Estado del servicio",
            "/api/predict": "Predicción de stroke (POST)"
        }
    })

# Manejador específico para OPTIONS
@app.route('/api/predict', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

# Manejador de errores global
@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Error no manejado: {str(error)}", exc_info=True)
    return jsonify({
        "error": "Error interno del servidor",
        "message": str(error)
    }), 500

# Vercel serverless handler
def handler(request):
    if request.path.startswith('/api/'):
        with app.request_context(request):
            return app.full_dispatch_request()
    return jsonify({"error": "Invalid endpoint"}), 404

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f'Iniciando servidor en puerto {port}, debug={debug}')
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f'Error al iniciar el servidor: {e}')