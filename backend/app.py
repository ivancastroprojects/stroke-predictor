from flask import Flask, jsonify, request
from flask_cors import CORS
from api.predict import predict_bp
import os

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://*.vercel.app"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Registrar el blueprint de predicción
app.register_blueprint(predict_bp)

@app.route('/api', methods=['GET'])
def home():
    return jsonify({"message": "Stroke Prediction API"})

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

# Vercel serverless handler
def handler(request):
    if request.path.startswith('/api/'):
        with app.request_context(request):
            return app.full_dispatch_request()
    return jsonify({"error": "Invalid endpoint"}), 404

# Local development server
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)