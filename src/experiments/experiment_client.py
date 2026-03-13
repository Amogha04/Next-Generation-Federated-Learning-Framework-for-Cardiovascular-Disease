import os

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.heart_model import HeartModel
from src.utils.data_loader import load_heart_data, load_cancer_data
from src.utils.data_partitioner import non_iid_partition, dirichlet_partition


CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
FL_NUM_CLIENTS = int(os.getenv("FL_NUM_CLIENTS", "3"))
FL_DATASET = os.getenv("FL_DATASET", "heart").strip().lower()
FL_DISTRIBUTION = os.getenv("FL_DISTRIBUTION", "iid").strip().lower()
FL_DP_ENABLED = os.getenv("FL_DP_ENABLED", "true").strip().lower() in {"true", "1", "yes"}
FL_RANDOM_SEED = int(os.getenv("FL_RANDOM_SEED", "42"))
FL_DIRICHLET_ALPHA = float(os.getenv("FL_DIRICHLET_ALPHA", "1.0"))


def load_dataset():
    if FL_DATASET == "heart":
        return load_heart_data()
    if FL_DATASET == "cancer":
        return load_cancer_data()
    raise ValueError(f"Unsupported FL_DATASET: {FL_DATASET}")


def iid_partition(X, y, num_clients, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)

    X_np = np.asarray(X)
    y_np = np.asarray(y)
    return [(X_np[idx], y_np[idx]) for idx in splits]


def build_local_data():
    X, y = load_dataset()

    if FL_DISTRIBUTION in {"noniid", "non-iid", "non_iid"}:
        partitions = non_iid_partition(X, y, num_clients=FL_NUM_CLIENTS)
    elif FL_DISTRIBUTION == "iid":
        partitions = iid_partition(X, y, num_clients=FL_NUM_CLIENTS, seed=FL_RANDOM_SEED)
    elif FL_DISTRIBUTION == "dirichlet":
        print(f"[CLIENT] Dirichlet partition alpha={FL_DIRICHLET_ALPHA}")
        partitions = dirichlet_partition(
            X,
            y,
            num_clients=FL_NUM_CLIENTS,
            alpha=FL_DIRICHLET_ALPHA,
        )
    else:
        raise ValueError("FL_DISTRIBUTION must be iid, noniid, or dirichlet")

    X_local, y_local = partitions[CLIENT_ID]

    scaler = StandardScaler()
    X_local = scaler.fit_transform(X_local)

    X_tensor = torch.tensor(X_local, dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y_local), dtype=torch.float32).view(-1, 1)

    return X_tensor, y_tensor


X_tensor, y_tensor = build_local_data()
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X_tensor.shape[1]
model = HeartModel(input_dim).float()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


if FL_DP_ENABLED:
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=0.5,
        max_grad_norm=1.0,
    )


class ExperimentClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        model.train()

        proximal_mu = float(config.get("proximal_mu", 0.0))
        global_params = None

        if proximal_mu > 0.0:
            global_params = [p.detach().clone() for p in model.parameters()]

        for _ in range(5):
            for x_batch, y_batch in loader:
                x_batch = x_batch.float()
                y_batch = y_batch.float()

                optimizer.zero_grad()

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

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
            outputs = model(X_tensor.float())
            loss = criterion(outputs, y_tensor.float())

        return float(loss), len(dataset), {}


if __name__ == "__main__":
    print(
        f"[CLIENT {CLIENT_ID}] dataset={FL_DATASET}, distribution={FL_DISTRIBUTION}, dp={FL_DP_ENABLED}"
    )

    fl.client.start_numpy_client(
        server_address="127.0.0.1:9090",
        client=ExperimentClient(),
    )
