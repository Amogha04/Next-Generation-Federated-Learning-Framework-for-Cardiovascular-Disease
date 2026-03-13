import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import yaml


# Debug toggle
TEST_MODE = False

CONFIG_PATH = "experiments_config.yaml"

REQUIRED_CONFIG_FIELDS = [
    "algorithms",
    "datasets",
    "distributions",
    "client_counts",
    "dirichlet_alphas",
    "dropout_levels",
    "differential_privacy",
    "training_rounds",
]


@dataclass
class ExperimentConfig:
    experiment_id: str
    algorithm: str
    dataset: str
    distribution: str
    differential_privacy: bool
    client_dropout_prob: float
    training_rounds: int
    num_clients: int
    dirichlet_alpha: Optional[float] = None


def load_experiment_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiment config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Experiment config file is empty.")

    missing_fields = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required config fields: {missing_fields}")

    print("\n[RUNNER] Loaded experiment configuration:")
    print(json.dumps(config, indent=2))

    return config


def build_experiments(config: Dict[str, Any]) -> List[ExperimentConfig]:
    algorithms = config["algorithms"]
    datasets = config["datasets"]
    distributions = config["distributions"]
    dp_options = config["differential_privacy"]
    dropout_levels = config["dropout_levels"]
    client_counts = config["client_counts"]
    dirichlet_alphas = config["dirichlet_alphas"]
    training_rounds = int(config["training_rounds"])

    configs = []
    exp_num = 1

    for algo, ds, dist, dp, drop, num_clients in itertools.product(
        algorithms, datasets, distributions, dp_options, dropout_levels, client_counts
    ):
        if dist == "dirichlet":
            for alpha in dirichlet_alphas:
                configs.append(
                    ExperimentConfig(
                        experiment_id=f"exp_{exp_num:03d}",
                        algorithm=algo,
                        dataset=ds,
                        distribution=dist,
                        differential_privacy=bool(dp),
                        client_dropout_prob=float(drop),
                        training_rounds=training_rounds,
                        num_clients=int(num_clients),
                        dirichlet_alpha=float(alpha),
                    )
                )
                exp_num += 1
        else:
            configs.append(
                ExperimentConfig(
                    experiment_id=f"exp_{exp_num:03d}",
                    algorithm=algo,
                    dataset=ds,
                    distribution=dist,
                    differential_privacy=bool(dp),
                    client_dropout_prob=float(drop),
                    training_rounds=training_rounds,
                    num_clients=int(num_clients),
                    dirichlet_alpha=None,
                )
            )
            exp_num += 1

    return configs


def run_single_experiment(cfg: ExperimentConfig, results_dir: str) -> Dict[str, Any]:
    metrics_path = os.path.join(results_dir, "tmp", f"{cfg.experiment_id}.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    env = os.environ.copy()
    env["FL_ALGORITHM"] = cfg.algorithm
    env["FL_DATASET"] = cfg.dataset
    env["FL_DISTRIBUTION"] = cfg.distribution
    env["FL_DP_ENABLED"] = "true" if cfg.differential_privacy else "false"
    env["FL_NUM_ROUNDS"] = str(cfg.training_rounds)
    env["FL_NUM_CLIENTS"] = str(cfg.num_clients)
    env["FL_CLIENT_DROPOUT_PROB"] = str(cfg.client_dropout_prob)
    env["FL_METRICS_OUT"] = metrics_path

    if cfg.distribution == "dirichlet" and cfg.dirichlet_alpha is not None:
        env["FL_DIRICHLET_ALPHA"] = str(cfg.dirichlet_alpha)

    server_cmd = [sys.executable, "-m", "src.experiments.experiment_server"]
    client_cmd = [sys.executable, "-m", "src.experiments.experiment_client"]

    print("\n==============================")
    print(f"Experiment ID        : {cfg.experiment_id}")
    print(f"Algorithm            : {cfg.algorithm}")
    print(f"Dataset              : {cfg.dataset}")
    print(f"Distribution         : {cfg.distribution}")
    print(f"Differential Privacy : {cfg.differential_privacy}")
    print(f"Client Dropout Prob  : {cfg.client_dropout_prob}")
    print(f"Num Clients          : {cfg.num_clients}")
    print(f"Dirichlet Alpha      : {cfg.dirichlet_alpha}")
    print("==============================")

    server_proc = subprocess.Popen(server_cmd, env=env)
    time.sleep(3)

    client_procs = []
    try:
        for cid in range(cfg.num_clients):
            cenv = env.copy()
            cenv["CLIENT_ID"] = str(cid)
            client_procs.append(subprocess.Popen(client_cmd, env=cenv))

        server_proc.wait(timeout=60 * 60)
    finally:
        for p in client_procs:
            if p.poll() is None:
                p.terminate()
        for p in client_procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    final_accuracy = None
    final_auc = None
    rounds = cfg.training_rounds

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        final_accuracy = m.get("final_accuracy")
        final_auc = m.get("final_auc")
        rounds = m.get("last_round", cfg.training_rounds)

    return {
        "experiment_id": cfg.experiment_id,
        "algorithm": cfg.algorithm,
        "dataset": cfg.dataset,
        "distribution": cfg.distribution,
        "num_clients": cfg.num_clients,
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "differential_privacy": str(cfg.differential_privacy).lower(),
        "client_dropout_prob": cfg.client_dropout_prob,
        "final_accuracy": final_accuracy,
        "final_auc": final_auc,
        "training_rounds": rounds,
    }


def run_all_experiments(output_csv: str = "results/experiment_results.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    loaded_config = load_experiment_config()
    configs = build_experiments(loaded_config)

    if TEST_MODE:
        print("[RUNNER] TEST_MODE=True -> running only first experiment config")
        configs = configs[:1]
    else:
        print(f"[RUNNER] TEST_MODE=False -> running all {len(configs)} experiments")

    rows = []
    for cfg in configs:
        rows.append(run_single_experiment(cfg, os.path.dirname(output_csv)))

    fieldnames = [
        "experiment_id",
        "algorithm",
        "dataset",
        "distribution",
        "num_clients",
        "dirichlet_alpha",
        "differential_privacy",
        "client_dropout_prob",
        "final_accuracy",
        "final_auc",
        "training_rounds",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results: {output_csv}")


if __name__ == "__main__":
    run_all_experiments()
