from flask import Flask, request, jsonify
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route("/")
def home():
    if model is None or scaler is None:
        return "API is running, but model/scaler failed to load.", 500
    return "Fraud Detection API is running and model is loaded!"

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.json
    
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key in JSON payload"}), 400
        
    try:
        # Extract features and convert to numpy array
        features = np.array(data["features"])
        
        # Check if it's a 1D array (single sample) and reshape to 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # In our supervised model, 1 is typically Fraud and 0 is Not Fraud
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        
        return jsonify({
            "prediction": result,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc(),
            "status": "failed"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
