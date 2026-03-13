import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(rounds, accuracy, auc, title, save_path):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(rounds, accuracy)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Rounds")

    plt.subplot(1, 2, 2)
    plt.plot(rounds, auc)
    plt.xlabel("Rounds")
    plt.ylabel("AUC")
    plt.title("AUC vs Rounds")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_accuracy_vs_dirichlet_alpha():
    results_path = os.path.join("results", "experiment_results.csv")
    plots_dir = os.path.join("results", "plots")
    save_path = os.path.join(plots_dir, "accuracy_vs_alpha.png")

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    required_columns = {"distribution", "algorithm", "dirichlet_alpha", "final_accuracy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[df["distribution"].astype(str).str.lower() == "dirichlet"].copy()

    if df.empty:
        print("No Dirichlet experiment rows found in results/experiment_results.csv")
        return

    df["dirichlet_alpha"] = pd.to_numeric(df["dirichlet_alpha"], errors="coerce")
    df["final_accuracy"] = pd.to_numeric(df["final_accuracy"], errors="coerce")
    df = df.dropna(subset=["dirichlet_alpha", "final_accuracy"])

    if df.empty:
        print("No valid Dirichlet alpha / accuracy values found for plotting.")
        return

    grouped = (
        df.groupby(["algorithm", "dirichlet_alpha"], as_index=False)["final_accuracy"]
        .mean()
        .rename(columns={"final_accuracy": "mean_accuracy"})
    )

    alpha_order = [10, 1, 0.5, 0.1]
    algorithm_labels = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedadam": "FedAdam",
    }

    plt.figure(figsize=(10, 6))

    for algorithm in sorted(grouped["algorithm"].unique()):
        algo_df = grouped[grouped["algorithm"] == algorithm].copy()
        algo_df["alpha_sort"] = algo_df["dirichlet_alpha"].apply(
            lambda x: alpha_order.index(x) if x in alpha_order else len(alpha_order)
        )
        algo_df = algo_df.sort_values("alpha_sort")

        plt.plot(
            algo_df["dirichlet_alpha"],
            algo_df["mean_accuracy"],
            marker="o",
            linewidth=2,
            label=algorithm_labels.get(algorithm.lower(), algorithm),
        )

    plt.xlabel("Dirichlet Alpha")
    plt.ylabel("Mean Accuracy")
    plt.title("Impact of Data Heterogeneity (Dirichlet Alpha) on Federated Learning Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xticks(alpha_order, [str(alpha) for alpha in alpha_order])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Saved plot: {save_path}")


def plot_accuracy_vs_clients():
    results_path = os.path.join("results", "experiment_results.csv")
    plots_dir = os.path.join("results", "plots")
    save_path = os.path.join(plots_dir, "accuracy_vs_clients.png")

    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    required_columns = {"algorithm", "num_clients", "final_accuracy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["num_clients"] = pd.to_numeric(df["num_clients"], errors="coerce")
    df["final_accuracy"] = pd.to_numeric(df["final_accuracy"], errors="coerce")
    df = df.dropna(subset=["num_clients", "final_accuracy"])

    if df.empty:
        print("No valid client scalability rows found in results/experiment_results.csv")
        return

    grouped = (
        df.groupby(["algorithm", "num_clients"], as_index=False)["final_accuracy"]
        .mean()
        .rename(columns={"final_accuracy": "mean_accuracy"})
    )

    algorithm_labels = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedadam": "FedAdam",
    }

    plt.figure(figsize=(10, 6))

    for algorithm in sorted(grouped["algorithm"].unique()):
        algo_df = grouped[grouped["algorithm"] == algorithm].copy()
        algo_df = algo_df.sort_values("num_clients")

        plt.plot(
            algo_df["num_clients"],
            algo_df["mean_accuracy"],
            marker="o",
            linewidth=2,
            label=algorithm_labels.get(algorithm.lower(), algorithm),
        )

    plt.xlabel("Number of Clients")
    plt.ylabel("Mean Accuracy")
    plt.title("Federated Learning Scalability: Accuracy vs Number of Clients")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(sorted(grouped["num_clients"].unique()))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    plot_accuracy_vs_dirichlet_alpha()
    plot_accuracy_vs_clients()
