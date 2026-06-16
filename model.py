import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
print("Loading data...")
df = pd.read_csv("data.csv")

# Drop Time column as it's not usually a robust predictive feature by itself
df = df.drop(['Time'], axis=1)

# Features and target
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split data into training and testing sets
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features (important for Amount and any other non-PCA features)
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train supervised model
print("Training Random Forest model (this may take a minute)...")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Save model and scaler
print("Saving model and scaler...")
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Success! Model and Scaler trained and saved.")