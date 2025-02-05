from flask import Flask, jsonify, request
from flask_cors import CORS
from api.predict import predict_bp
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuración de CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://*.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuración de la aplicación
app.config.update(
    ENV='development',
    DEBUG=True,
    PROPAGATE_EXCEPTIONS=True
)

# Registrar el blueprint de predicción
app.register_blueprint(predict_bp)

@app.route('/api', methods=['GET'])
def home():
    return jsonify({"message": "Stroke Prediction API"})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

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
    app.run(host='0.0.0.0', port=port, debug=debug)