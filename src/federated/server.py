import os

import flwr as fl
import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.heart_model import HeartModel
from src.utils.data_loader import load_heart_data, load_cancer_data


# =========================
# Choose disease
# =========================

disease = "cancer"  # change to "heart" or "cancer"

if disease == "heart":
    X, y = load_heart_data()
else:
    X, y = load_cancer_data()


# =========================
# Preprocessing
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

input_dim = X_tensor.shape[1]


# =========================
# FL Algorithm Config
# =========================

# Select one: "fedavg", "fedprox", "fedadam"
FL_ALGORITHM = os.getenv("FL_ALGORITHM", "fedavg").strip().lower()

# FedProx config
FEDPROX_MU = float(os.getenv("FEDPROX_MU", "0.01"))

# FedAdam config
FEDADAM_ETA = float(os.getenv("FEDADAM_ETA", "0.1"))
FEDADAM_ETA_L = float(os.getenv("FEDADAM_ETA_L", "0.01"))
FEDADAM_BETA_1 = float(os.getenv("FEDADAM_BETA_1", "0.9"))
FEDADAM_BETA_2 = float(os.getenv("FEDADAM_BETA_2", "0.99"))
FEDADAM_TAU = float(os.getenv("FEDADAM_TAU", "1e-9"))


# =========================
# Global Evaluation Function
# =========================

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


def create_strategy(algorithm: str):
    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=evaluate_global,
    )

    if algorithm == "fedavg":
        print("[SERVER] Using algorithm: FedAvg")
        return fl.server.strategy.FedAvg(**common_kwargs)

    if algorithm == "fedprox":
        print(f"[SERVER] Using algorithm: FedProx (mu={FEDPROX_MU})")
        return fl.server.strategy.FedProx(
            proximal_mu=FEDPROX_MU,
            **common_kwargs,
        )

    if algorithm == "fedadam":
        print(
            "[SERVER] Using algorithm: FedAdam "
            f"(eta={FEDADAM_ETA}, eta_l={FEDADAM_ETA_L}, "
            f"beta_1={FEDADAM_BETA_1}, beta_2={FEDADAM_BETA_2}, tau={FEDADAM_TAU})"
        )
        return fl.server.strategy.FedAdam(
            eta=FEDADAM_ETA,
            eta_l=FEDADAM_ETA_L,
            beta_1=FEDADAM_BETA_1,
            beta_2=FEDADAM_BETA_2,
            tau=FEDADAM_TAU,
            **common_kwargs,
        )

    raise ValueError(
        f"Unsupported FL_ALGORITHM='{algorithm}'. Use fedavg, fedprox, or fedadam."
    )


# =========================
# Federated Strategy
# =========================

strategy = create_strategy(FL_ALGORITHM)


# =========================
# Start Server
# =========================

fl.server.start_server(
    server_address="127.0.0.1:9090",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
