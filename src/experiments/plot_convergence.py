import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


def format_algorithm_name(name: str) -> str:
    mapping = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedadam": "FedAdam",
    }
    return mapping.get(name.lower(), name)


def parse_filename(filename: str):
    """
    Expected format:
    strategy_dataset_distribution.csv

    Example:
    fedavg_heart_iid.csv
    """
    if not filename.endswith(".csv"):
        return None

    base_name = os.path.splitext(filename)[0]
    parts = base_name.split("_")

    if len(parts) < 3:
        return None

    strategy = parts[0]
    dataset = parts[1]
    distribution = "_".join(parts[2:])

    return strategy, dataset, distribution


def load_convergence_data(convergence_dir: str):
    grouped_data = defaultdict(list)

    if not os.path.exists(convergence_dir):
        os.makedirs(convergence_dir, exist_ok=True)
        return grouped_data

    for filename in os.listdir(convergence_dir):
        parsed = parse_filename(filename)
        if parsed is None:
            continue

        strategy, dataset, distribution = parsed
        file_path = os.path.join(convergence_dir, filename)

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"Skipping {filename}: failed to read CSV ({exc})")
            continue

        required_columns = {"round", "loss", "accuracy"}
        if not required_columns.issubset(df.columns):
            print(f"Skipping {filename}: missing required columns")
            continue

        df = df.copy()
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df = df.dropna(subset=["round", "loss", "accuracy"]).sort_values("round")

        if df.empty:
            print(f"Skipping {filename}: no valid rows found")
            continue

        grouped_data[(dataset, distribution)].append((strategy, df))

    return grouped_data


def create_plot(entries, metric: str, dataset: str, distribution: str, output_path: str):
    plt.figure(figsize=(10, 6))

    for strategy, df in sorted(entries, key=lambda item: item[0]):
        plt.plot(
            df["round"],
            df[metric],
            marker="o",
            linewidth=2,
            markersize=5,
            label=format_algorithm_name(strategy),
        )

    plt.title(f"{dataset.upper()} | {distribution.upper()} | {metric.capitalize()} Convergence")
    plt.xlabel("Communication Round")
    plt.ylabel(metric.capitalize())
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    convergence_dir = os.path.join(project_root, "src", "experiments", "results", "convergence")
    plots_dir = os.path.join(project_root, "results", "plots")

    os.makedirs(convergence_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    grouped_data = load_convergence_data(convergence_dir)
    generated_plots = []

    for (dataset, distribution), entries in sorted(grouped_data.items()):
        accuracy_plot_path = os.path.join(
            plots_dir,
            f"{dataset}_{distribution}_accuracy_convergence.png",
        )
        loss_plot_path = os.path.join(
            plots_dir,
            f"{dataset}_{distribution}_loss_convergence.png",
        )

        create_plot(entries, "accuracy", dataset, distribution, accuracy_plot_path)
        create_plot(entries, "loss", dataset, distribution, loss_plot_path)

        generated_plots.append(accuracy_plot_path)
        generated_plots.append(loss_plot_path)

    if generated_plots:
        print("Generated plots:")
        for plot_path in generated_plots:
            print(plot_path)
    else:
        print(f"No valid convergence CSV files found in: {convergence_dir}")


if __name__ == "__main__":
    main()
