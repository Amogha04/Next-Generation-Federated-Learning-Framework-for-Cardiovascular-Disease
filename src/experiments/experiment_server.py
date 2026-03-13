import csv
import json
import os
import random
from typing import Dict, Any, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.models.heart_model import HeartModel
from src.utils.data_loader import load_heart_data, load_cancer_data


FL_ALGORITHM = os.getenv("FL_ALGORITHM", "fedavg").strip().lower()
FL_DATASET = os.getenv("FL_DATASET", "heart").strip().lower()
FL_DISTRIBUTION = os.getenv("FL_DISTRIBUTION", "iid").strip().lower()
FL_NUM_ROUNDS = int(os.getenv("FL_NUM_ROUNDS", "10"))
FL_NUM_CLIENTS = int(os.getenv("FL_NUM_CLIENTS", "3"))

FL_CLIENT_DROPOUT_PROB = float(os.getenv("FL_CLIENT_DROPOUT_PROB", "0.0"))
FL_RANDOM_SEED = int(os.getenv("FL_RANDOM_SEED", "42"))

FL_METRICS_OUT = os.getenv("FL_METRICS_OUT", "results/tmp/last_metrics.json")

FEDPROX_MU = float(os.getenv("FEDPROX_MU", "0.01"))
FEDADAM_ETA = float(os.getenv("FEDADAM_ETA", "0.1"))
FEDADAM_ETA_L = float(os.getenv("FEDADAM_ETA_L", "0.01"))
FEDADAM_BETA_1 = float(os.getenv("FEDADAM_BETA_1", "0.9"))
FEDADAM_BETA_2 = float(os.getenv("FEDADAM_BETA_2", "0.99"))
FEDADAM_TAU = float(os.getenv("FEDADAM_TAU", "1e-9"))

STRATEGY_NAME = FL_ALGORITHM
DATASET_NAME = FL_DATASET
DATA_DISTRIBUTION = FL_DISTRIBUTION


def load_dataset():
    if FL_DATASET == "heart":
        return load_heart_data()
    if FL_DATASET == "cancer":
        return load_cancer_data()
    raise ValueError(f"Unsupported FL_DATASET: {FL_DATASET}")


# ---------------- DATA ----------------

X, y = load_dataset()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

input_dim = X_tensor.shape[1]
criterion = nn.BCELoss()

state: Dict[str, Any] = {
    "algorithm": FL_ALGORITHM,
    "dataset": FL_DATASET,
    "distribution": FL_DISTRIBUTION,
    "training_rounds": FL_NUM_ROUNDS,
    "client_dropout_prob": FL_CLIENT_DROPOUT_PROB,
    "final_accuracy": None,
    "final_auc": None,
    "last_round": 0,
}


# ---------------- CONVERGENCE LOGGING ----------------

# This helper appends one row per communication round.
# A separate CSV is created per strategy/dataset/distribution combination so
# convergence curves can be plotted directly for research comparisons.
def log_convergence(round_num, loss, accuracy, strategy, dataset, distribution):
    os.makedirs("results/convergence", exist_ok=True)

    file_path = f"results/convergence/{strategy}_{dataset}_{distribution}.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["round", "loss", "accuracy"])

        writer.writerow([round_num, loss, accuracy])


# ---------------- EVALUATION ----------------

def evaluate_global(server_round, parameters, config):
    model = HeartModel(input_dim).float()

    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor.float())
        predictions = (outputs > 0.5).float()
        loss = criterion(outputs, y_tensor.float())

    acc = accuracy_score(y_tensor.numpy(), predictions.numpy())
    auc = roc_auc_score(y_tensor.numpy(), outputs.numpy())

    state["final_accuracy"] = float(acc)
    state["final_auc"] = float(auc)
    state["last_round"] = int(server_round)

    metrics = {"accuracy": float(acc), "auc": float(auc)}

    # Safe fallback in case accuracy is absent or malformed in future changes.
    accuracy_for_log = metrics.get("accuracy", None)
    if accuracy_for_log is None:
        accuracy_for_log = 0.0

    log_convergence(
        server_round,
        float(loss),
        float(accuracy_for_log),
        STRATEGY_NAME,
        DATASET_NAME,
        DATA_DISTRIBUTION,
    )

    print(f"[ROUND {server_round}] Global Loss: {float(loss):.4f}")
    print(f"[ROUND {server_round}] Global Accuracy: {acc:.4f}")
    print(f"[ROUND {server_round}] Global AUC: {auc:.4f}")

    return float(loss), metrics


# ---------------- CLIENT FORMAT ----------------

def _cid_sort_key(cid: str):
    return (0, int(cid)) if str(cid).isdigit() else (1, str(cid))


def _format_client_ids(
    pairs: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]
):
    ids = [str(client.cid) for client, _ in pairs]
    ids = sorted(ids, key=_cid_sort_key)
    out = [int(cid) if cid.isdigit() else cid for cid in ids]
    return out


# ---------------- DROPOUT STRATEGIES ----------------

class DropoutFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, dropout_prob: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob = dropout_prob
        self.rng = random.Random(seed)

    def configure_fit(self, server_round, parameters, client_manager):
        base_pairs = super().configure_fit(server_round, parameters, client_manager)

        if not base_pairs:
            return base_pairs

        active_pairs = [
            p for p in base_pairs if self.rng.random() >= self.dropout_prob
        ]

        if not active_pairs:
            active_pairs = [self.rng.choice(base_pairs)]

        print(f"\nROUND {server_round}")
        print(f"Active clients: {_format_client_ids(active_pairs)}\n")

        return active_pairs


class DropoutFedProx(fl.server.strategy.FedProx):
    def __init__(self, dropout_prob: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob = dropout_prob
        self.rng = random.Random(seed)

    def configure_fit(self, server_round, parameters, client_manager):
        base_pairs = super().configure_fit(server_round, parameters, client_manager)

        if not base_pairs:
            return base_pairs

        active_pairs = [
            p for p in base_pairs if self.rng.random() >= self.dropout_prob
        ]

        if not active_pairs:
            active_pairs = [self.rng.choice(base_pairs)]

        print(f"\nROUND {server_round}")
        print(f"Active clients: {_format_client_ids(active_pairs)}\n")

        return active_pairs


class DropoutFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, dropout_prob: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.dropout_prob = dropout_prob
        self.rng = random.Random(seed)

    def configure_fit(self, server_round, parameters, client_manager):
        base_pairs = super().configure_fit(server_round, parameters, client_manager)

        if not base_pairs:
            return base_pairs

        active_pairs = [
            p for p in base_pairs if self.rng.random() >= self.dropout_prob
        ]

        if not active_pairs:
            active_pairs = [self.rng.choice(base_pairs)]

        print(f"\nROUND {server_round}")
        print(f"Active clients: {_format_client_ids(active_pairs)}\n")

        return active_pairs


# ---------------- INITIAL PARAMETERS ----------------

def get_initial_parameters():
    model = HeartModel(input_dim)

    return fl.common.ndarrays_to_parameters(
        [val.detach().cpu().numpy() for val in model.parameters()]
    )


# ---------------- STRATEGY ----------------

def create_strategy():
    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_evaluate_clients=0,
        min_available_clients=1,
        evaluate_fn=evaluate_global,
    )

    if FL_ALGORITHM == "fedavg":
        print(f"[SERVER] Using FedAvg | dropout={FL_CLIENT_DROPOUT_PROB}")

        return DropoutFedAvg(
            dropout_prob=FL_CLIENT_DROPOUT_PROB,
            seed=FL_RANDOM_SEED,
            **common_kwargs,
        )

    if FL_ALGORITHM == "fedprox":
        print(f"[SERVER] Using FedProx | dropout={FL_CLIENT_DROPOUT_PROB}")

        return DropoutFedProx(
            proximal_mu=FEDPROX_MU,
            dropout_prob=FL_CLIENT_DROPOUT_PROB,
            seed=FL_RANDOM_SEED,
            **common_kwargs,
        )

    if FL_ALGORITHM == "fedadam":
        print(f"[SERVER] Using FedAdam | dropout={FL_CLIENT_DROPOUT_PROB}")

        return DropoutFedAdam(
            eta=FEDADAM_ETA,
            eta_l=FEDADAM_ETA_L,
            beta_1=FEDADAM_BETA_1,
            beta_2=FEDADAM_BETA_2,
            tau=FEDADAM_TAU,
            initial_parameters=get_initial_parameters(),
            dropout_prob=FL_CLIENT_DROPOUT_PROB,
            seed=FL_RANDOM_SEED,
            **common_kwargs,
        )

    raise ValueError("FL_ALGORITHM must be one of: fedavg, fedprox, fedadam")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    strategy = create_strategy()

    fl.server.start_server(
        server_address="127.0.0.1:9090",
        config=fl.server.ServerConfig(num_rounds=FL_NUM_ROUNDS),
        strategy=strategy,
    )

    os.makedirs(os.path.dirname(FL_METRICS_OUT), exist_ok=True)

    with open(FL_METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
