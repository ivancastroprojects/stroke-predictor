from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['GET'])
def home():
    return jsonify({"message": "Stroke Prediction API"})

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"status": "API is working"})

# Vercel serverless handler
def handler(request):
    if request.path.startswith('/api/'):
        with app.request_context(request):
            return app.full_dispatch_request()
    return jsonify({"error": "Invalid endpoint"}), 404

# Local development server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)