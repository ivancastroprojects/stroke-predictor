from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from joblib import load

# Initialize the Flask app
app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://192.168.56.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load the trained model
model = load('./stroke_prediction_model.joblib')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    try:
        data = request.json
        print("Received data:", data)  # Debug print
        
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        print(f"Prediction: {prediction}")  # Debug print
        
        return jsonify({"prediction": int(prediction)}), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Stroke Prediction API"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)