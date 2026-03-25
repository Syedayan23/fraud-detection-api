import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load dataset
df = pd.read_csv("data.csv")

# Drop Time column
df = df.drop(['Time'], axis=1)

# Features
X = df.drop(['Class'], axis=1)

# Train model
model = IsolationForest(contamination=0.01)
model.fit(X)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved!")