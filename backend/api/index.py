import os
import sys
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Asegurarse de que el modelo esté en el PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
model_path = os.path.join(parent_dir, 'models')
sys.path.append(model_path)

# Crear la aplicación Flask
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configuración de CORS basada en el entorno
if os.environ.get('FLASK_ENV') == 'development':
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "expose_headers": ["Content-Type"]
        }
    })
else:
    CORS(app, resources={
        r"/api/*": {
            "origins": ["*"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "expose_headers": ["Content-Type"]
        }
    })

# Importar blueprints después de crear la aplicación
from api.predict import predict_bp

# Registrar blueprints
app.register_blueprint(predict_bp)

@app.route('/')
def home():
    return jsonify({
        "status": "API is running",
        "version": "1.0.0"
    })

@app.route('/api/health')
def health():
    logger.info('Verificación de salud solicitada')
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    logger.error(f'Ruta no encontrada: {error}')
    return jsonify({
        'error': 'Ruta no encontrada',
        'message': str(error)
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Error interno del servidor: {error}')
    return jsonify({
        'error': 'Error interno del servidor',
        'status': 500
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f'Iniciando servidor en puerto {port}, debug={debug}')
    try:
        app.run(host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        logger.error(f'Error al iniciar el servidor: {e}') 