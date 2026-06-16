# 🚀 Credit Card Fraud Detection API

A production-ready Flask API that detects fraudulent credit card transactions in real-time using a supervised **Random Forest Classifier** model trained on PCA-transformed transaction features.

---

## 🧠 Tech Stack
- **Language:** Python
- **Machine Learning:** Scikit-learn (`RandomForestClassifier`, `StandardScaler`)
- **API Framework:** Flask
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib
- **Production Deployment:** AWS EC2, Gunicorn (WSGI Server), Nginx (Reverse Proxy)

---

## 📊 Model Performance
The model was evaluated using a stratified 80/20 train/test split. Here are the evaluation metrics on the test dataset:

- **Accuracy:** `99.96%`
- **Precision (Fraud Class):** `94%` (Low rate of false positives)
- **Recall (Fraud Class):** `81%` (Detects 81% of all actual fraud)
- **F1-Score (Fraud Class):** `87%`

---

## ⚙️ Features
✔ Real-time fraud prediction  
✔ REST API endpoint (`POST /predict`)  
✔ Robust input validation and feature scaling  
✔ Scalable cloud deployment architecture  

---

## 🌐 Live API
👉 http://43.204.35.234/

---

## 🔌 API Usage

### Endpoint
`POST /predict`

### Headers
```http
Content-Type: application/json
```

### Request Payload (29 Features)
The input expects a JSON object with a `"features"` key containing a list of exactly 29 numeric values (representing PCA features `V1` to `V28` and the transaction `Amount`):

```json
{
  "features": [
    -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 
    0.462388, 0.239599, 0.098698, 0.363787, 0.090794, 
    -0.551600, -0.617801, -0.991390, -0.311169, 1.468177, 
    -0.470401, 0.207971, 0.025791, 0.403993, 0.251412, 
    -0.018307, 0.277838, -0.110474, 0.066928, 0.128539, 
    -0.189115, 0.133558, -0.021053, 149.62
  ]
}
```

### Response Payload
```json
{
  "prediction": "Not Fraud",
  "status": "success"
}
```

---

## 🛠️ Local Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/Syedayan23/fraud-detection-api.git
cd fraud-detection-api
```

### 2. Download the dataset
Due to GitHub's file size limits, the raw dataset (`data.csv`) is not included in the repository. 
* Download the **Credit Card Fraud Detection dataset** from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
* Place the downloaded `data.csv` file into the root folder of the project.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model & Scaler
Run the training script to split the dataset, fit the feature scaler, train the Random Forest classifier, and save the serialized models:
```bash
python model.py
```

### 5. Run the Flask API Server
Start the local Flask development server (runs on port `5001`):
```bash
python app.py
```
