from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "Fraud Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])

    result = "Fraud" if prediction[0] == -1 else "Not Fraud"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
