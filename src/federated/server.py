import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from src.models.heart_model import HeartModel

# Load full dataset
data_path = os.path.join(os.path.dirname(__file__), "../../data/heart.csv")
df = pd.read_csv(data_path)

X = df.drop("target", axis=1)
y = df["target"]

X = pd.get_dummies(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1,1)

input_dim = X_tensor.shape[1]

def evaluate_global(server_round, parameters, config):
    model = HeartModel(input_dim)

    # Load parameters into model
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param)

    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = (outputs > 0.5).float()

    acc = accuracy_score(y_tensor.numpy(), predictions.numpy())
    auc = roc_auc_score(y_tensor.numpy(), outputs.numpy())

    print(f"\n[ROUND {server_round}] Global Accuracy: {acc:.4f}")
    print(f"[ROUND {server_round}] Global AUC: {auc:.4f}\n")

    return 0.0, {"accuracy": acc, "auc": auc}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_fn=evaluate_global,
)

fl.server.start_server(
    server_address="127.0.0.1:9090",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)