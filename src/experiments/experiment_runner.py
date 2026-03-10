import csv
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Any


# Debug toggle
TEST_MODE = False


@dataclass
class ExperimentConfig:
    experiment_id: str
    algorithm: str
    dataset: str
    distribution: str
    differential_privacy: bool
    client_dropout_prob: float
    training_rounds: int


def build_experiments() -> List[ExperimentConfig]:
    algorithms = ["fedavg", "fedprox", "fedadam"]
    datasets = ["heart", "cancer"]
    distributions = ["iid", "noniid"]
    dp_options = [False, True]
    dropout_levels = [0.0, 0.3, 0.5]
    training_rounds = 10

    configs = []
    exp_num = 1
    for algo, ds, dist, dp, drop in itertools.product(
        algorithms, datasets, distributions, dp_options, dropout_levels
    ):
        configs.append(
            ExperimentConfig(
                experiment_id=f"exp_{exp_num:03d}",
                algorithm=algo,
                dataset=ds,
                distribution=dist,
                differential_privacy=dp,
                client_dropout_prob=drop,
                training_rounds=training_rounds,
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
    env["FL_NUM_CLIENTS"] = "3"
    env["FL_CLIENT_DROPOUT_PROB"] = str(cfg.client_dropout_prob)
    env["FL_METRICS_OUT"] = metrics_path

    server_cmd = [sys.executable, "-m", "src.experiments.experiment_server"]
    client_cmd = [sys.executable, "-m", "src.experiments.experiment_client"]

    print("\n==============================")
    print(f"Experiment ID        : {cfg.experiment_id}")
    print(f"Algorithm            : {cfg.algorithm}")
    print(f"Dataset              : {cfg.dataset}")
    print(f"Distribution         : {cfg.distribution}")
    print(f"Differential Privacy : {cfg.differential_privacy}")
    print(f"Client Dropout Prob  : {cfg.client_dropout_prob}")
    print("==============================")

    server_proc = subprocess.Popen(server_cmd, env=env)
    time.sleep(3)

    client_procs = []
    try:
        for cid in range(3):
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
        "differential_privacy": str(cfg.differential_privacy).lower(),
        "client_dropout_prob": cfg.client_dropout_prob,
        "final_accuracy": final_accuracy,
        "final_auc": final_auc,
        "training_rounds": rounds,
    }


def run_all_experiments(output_csv: str = "results/experiment_results.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    configs = build_experiments()
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
