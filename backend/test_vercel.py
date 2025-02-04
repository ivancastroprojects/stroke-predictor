from flask import Flask, request, send_from_directory, jsonify
import os
from predict import app as predict_app
from health import app as health_app
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obtener rutas absolutas
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..', 'frontend', 'build'))
MODEL_DIR = os.path.join(BACKEND_DIR, 'models')

logger.info(f"Backend directory: {BACKEND_DIR}")
logger.info(f"Frontend directory: {FRONTEND_DIR}")
logger.info(f"Model directory: {MODEL_DIR}")

app = Flask(__name__)

# Simular el comportamiento de Vercel
@app.route('/api/predict', methods=['GET', 'POST'])
def predict_handler():
    try:
        with predict_app.request_context(request):
            return predict_app.full_dispatch_request()
    except Exception as e:
        logger.error(f"Error en predict_handler: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_handler():
    try:
        with health_app.request_context(request):
            return health_app.full_dispatch_request()
    except Exception as e:
        logger.error(f"Error en health_handler: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Servir archivos estáticos
@app.route('/static/<path:path>')
def serve_static(path):
    try:
        static_dir = os.path.join(FRONTEND_DIR, 'static')
        if os.path.exists(os.path.join(static_dir, path)):
            return send_from_directory(static_dir, path)
        logger.error(f"Archivo estático no encontrado: {path}")
        return jsonify({"error": "Archivo no encontrado"}), 404
    except Exception as e:
        logger.error(f"Error sirviendo archivo estático {path}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Servir archivos del frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    try:
        # Primero intentar servir el archivo directamente si existe
        if path and os.path.exists(os.path.join(FRONTEND_DIR, path)):
            return send_from_directory(FRONTEND_DIR, path)
        
        # Si no existe, servir index.html para rutas del frontend
        index_path = os.path.join(FRONTEND_DIR, 'index.html')
        if not os.path.exists(index_path):
            logger.error(f"No se encuentra index.html en: {index_path}")
            return jsonify({"error": "Frontend no construido correctamente"}), 404
            
        return send_from_directory(FRONTEND_DIR, 'index.html')
    except Exception as e:
        logger.error(f"Error sirviendo frontend: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        # Verificar directorios críticos
        for dir_path in [MODEL_DIR, FRONTEND_DIR]:
            if not os.path.exists(dir_path):
                logger.error(f"Directorio no encontrado: {dir_path}")
            else:
                logger.info(f"Directorio encontrado: {dir_path}")
                if dir_path == MODEL_DIR:
                    logger.info(f"Contenido de {dir_path}: {os.listdir(dir_path)}")
                elif dir_path == FRONTEND_DIR:
                    static_dir = os.path.join(dir_path, 'static')
                    if os.path.exists(static_dir):
                        logger.info(f"Contenido de static: {os.listdir(static_dir)}")
                    else:
                        logger.error(f"Directorio static no encontrado en: {static_dir}")
                
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Iniciando servidor en puerto {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logger.error(f"Error iniciando el servidor: {str(e)}") 