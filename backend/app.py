from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Stroke Prediction API"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "API is working"})

if __name__ == "__main__":
    app.run()