import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../../data/heart.csv")
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

# Encode categorical
X = pd.get_dummies(X)

# Split into 3 hospitals
X1, X_temp, y1, y_temp = train_test_split(X, y, test_size=0.66, random_state=42)
X2, X3, y2, y3 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

hospitals = [(X1, y1), (X2, y2), (X3, y3)]

local_models = []

print("Training local hospital models...\n")

for i, (Xi, yi) in enumerate(hospitals):
    scaler = StandardScaler()
    Xi_scaled = scaler.fit_transform(Xi)

    model = LogisticRegression(max_iter=2000)
    model.fit(Xi_scaled, yi)

    local_models.append((model, scaler))
    print(f"Hospital {i+1} trained.")

# Federated Averaging (manual)
print("\nPerforming Federated Averaging...")

coefs = np.mean([model.coef_ for model, _ in local_models], axis=0)
intercepts = np.mean([model.intercept_ for model, _ in local_models], axis=0)

# Create global model
global_model = LogisticRegression(max_iter=2000)
global_model.coef_ = coefs
global_model.intercept_ = intercepts
global_model.classes_ = np.array([0,1])

# Evaluate global model on full dataset
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)

y_pred = global_model.predict(X_scaled)
y_prob = global_model.predict_proba(X_scaled)[:,1]

print("\nGlobal Federated Model Performance")
print("Accuracy:", accuracy_score(y, y_pred))
print("AUC:", roc_auc_score(y, y_prob))