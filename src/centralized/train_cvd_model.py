import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve
)

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../../data/heart.csv")
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print(df.head())

# Separate features and label
X = df.drop("target", axis=1)
y = df["target"]

# Convert ALL categorical columns automatically
X = pd.get_dummies(X)

print("\nColumns after encoding:")
print(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nModel Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ========================
# ROC Curve Plot
# ========================

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="Centralized Model")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Centralized Model")
plt.legend()
plt.show()

# ========================
# Save Model
# ========================

model_dir = os.path.join(os.path.dirname(__file__), "../models")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "centralized_model.pkl")
joblib.dump(model, model_path)

print("\nModel saved at:", model_path)