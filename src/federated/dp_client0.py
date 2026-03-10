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
from src.utils.data_partitioner import non_iid_partition

# Load dataset
disease = "cancer"

if disease == "heart":
    X, y = load_heart_data()
else:
    X, y = load_cancer_data()

client_id = 0

# Non-IID partition
partitions = non_iid_partition(X, y, num_clients=3)
X_local, y_local = partitions[client_id]

# Print dataset details
print("Client", client_id, "dataset size:", len(y_local))
print("Label distribution:", np.bincount(y_local.astype(int)))

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

        # FedProx sends proximal_mu in fit config; for other strategies this is 0.0
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Snapshot global parameters once per round for proximal term
        global_params = None
        if proximal_mu > 0.0:
            global_params = [p.detach().clone() for p in model.parameters()]

        for epoch in range(5):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

                # FedProx regularization: (mu/2) * ||w - w_global||^2
                if proximal_mu > 0.0:
                    prox_term = torch.zeros(1, device=loss.device, dtype=loss.dtype)
                    for p, g in zip(model.parameters(), global_params):
                        prox_term += torch.sum((p - g) ** 2)
                    loss = loss + (proximal_mu / 2.0) * prox_term

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