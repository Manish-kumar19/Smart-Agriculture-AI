
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# 2. Select input features
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

X = data[features]
y = data["label"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create machine learning pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# 5. Train model
print("Training Crop Recommendation Model...")

model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Crop Model Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# 7. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/crop_model.pkl")

print("Crop model saved successfully!")

# 8. Test prediction
sample = pd.DataFrame(
    [[90, 42, 43, 20.8, 82, 6.5, 202]],
    columns=features
)

result = model.predict(sample)

print("Recommended Crop:", result[0])
