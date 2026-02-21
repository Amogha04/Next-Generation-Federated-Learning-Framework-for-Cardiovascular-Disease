import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from src.models.heart_model import HeartModel
from src.utils.data_loader import load_heart_data, load_cancer_data

# Load dataset
# Choose disease
disease = "cancer"  # change to "heart" or "cancer"
if disease == "heart":
    X, y = load_heart_data()
else:
    X, y = load_cancer_data()
# Choose client id manually
client_id = 0

# Split into 3 hospitals
X_split = np.array_split(X, 3)
y_split = np.array_split(y, 3)

X_local = X_split[client_id]
y_local = y_split[client_id]

scaler = StandardScaler()
X_local = scaler.fit_transform(X_local)

X_tensor = torch.tensor(X_local, dtype=torch.float32)
y_tensor = torch.tensor(y_local, dtype=torch.float32).view(-1,1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X_tensor.shape[1]
model = HeartModel(input_dim)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Add Differential Privacy
privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    noise_multiplier=0.5,
    max_grad_norm=1.0,
)

class DPClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [val.detach().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()
        for epoch in range(5):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
        return float(loss), len(dataset), {}

fl.client.start_numpy_client(
    server_address="127.0.0.1:9090",
    client=DPClient(),
)