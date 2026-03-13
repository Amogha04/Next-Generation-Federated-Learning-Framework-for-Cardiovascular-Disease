import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


RESULTS_PATH = os.path.join("results", "experiment_results.csv")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        st.error(f"Results file not found: {RESULTS_PATH}")
        st.stop()

    df = pd.read_csv(RESULTS_PATH)

    if "algorithm" in df.columns:
        df["algorithm"] = df["algorithm"].astype(str).str.lower()

    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.lower()

    if "distribution" in df.columns:
        df["distribution"] = df["distribution"].astype(str).str.lower()

    if "differential_privacy" in df.columns:
        df["differential_privacy"] = df["differential_privacy"].astype(str).str.lower()

    numeric_cols = ["final_accuracy", "dirichlet_alpha", "num_clients"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def format_algorithm_name(name: str) -> str:
    mapping = {
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
        "fedadam": "FedAdam",
    }
    return mapping.get(name, name)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    algorithm_options = ["All"] + sorted(df["algorithm"].dropna().unique().tolist()) if "algorithm" in df.columns else ["All"]
    dataset_options = ["All"] + sorted(df["dataset"].dropna().unique().tolist()) if "dataset" in df.columns else ["All"]
    distribution_options = ["All"] + sorted(df["distribution"].dropna().unique().tolist()) if "distribution" in df.columns else ["All"]
    dp_options = ["All"] + sorted(df["differential_privacy"].dropna().unique().tolist()) if "differential_privacy" in df.columns else ["All"]

    selected_algorithm = st.sidebar.selectbox("Algorithm", algorithm_options)
    selected_dataset = st.sidebar.selectbox("Dataset", dataset_options)
    selected_distribution = st.sidebar.selectbox("Distribution", distribution_options)
    selected_dp = st.sidebar.selectbox("Differential Privacy", dp_options)

    filtered = df.copy()

    if selected_algorithm != "All":
        filtered = filtered[filtered["algorithm"] == selected_algorithm]

    if selected_dataset != "All":
        filtered = filtered[filtered["dataset"] == selected_dataset]

    if selected_distribution != "All":
        filtered = filtered[filtered["distribution"] == selected_distribution]

    if selected_dp != "All":
        filtered = filtered[filtered["differential_privacy"] == selected_dp]

    return filtered


def plot_algorithm_comparison(df: pd.DataFrame):
    st.subheader("Algorithm Comparison")

    required = {"algorithm", "final_accuracy"}
    if not required.issubset(df.columns):
        st.warning("Missing columns required for algorithm comparison.")
        return

    plot_df = (
        df.dropna(subset=["algorithm", "final_accuracy"])
        .groupby("algorithm", as_index=False)["final_accuracy"]
        .mean()
        .rename(columns={"final_accuracy": "mean_accuracy"})
    )

    if plot_df.empty:
        st.info("No data available for Algorithm Comparison.")
        return

    plot_df["algorithm_label"] = plot_df["algorithm"].apply(format_algorithm_name)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["algorithm_label"], plot_df["mean_accuracy"], color=["#4E79A7", "#F28E2B", "#59A14F"][:len(plot_df)])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Mean Accuracy by Algorithm")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)


def plot_dirichlet_heterogeneity(df: pd.DataFrame):
    st.subheader("Dirichlet Heterogeneity")

    required = {"distribution", "algorithm", "dirichlet_alpha", "final_accuracy"}
    if not required.issubset(df.columns):
        st.warning("Missing columns required for Dirichlet heterogeneity plot.")
        return

    plot_df = df[df["distribution"] == "dirichlet"].copy()
    plot_df = plot_df.dropna(subset=["algorithm", "dirichlet_alpha", "final_accuracy"])

    if plot_df.empty:
        st.info("No Dirichlet experiment data available.")
        return

    grouped = (
        plot_df.groupby(["algorithm", "dirichlet_alpha"], as_index=False)["final_accuracy"]
        .mean()
        .rename(columns={"final_accuracy": "mean_accuracy"})
    )

    alpha_order = [10, 1, 0.5, 0.1]

    fig, ax = plt.subplots(figsize=(9, 5))

    for algorithm in sorted(grouped["algorithm"].unique()):
        algo_df = grouped[grouped["algorithm"] == algorithm].copy()
        algo_df["alpha_sort"] = algo_df["dirichlet_alpha"].apply(
            lambda x: alpha_order.index(x) if x in alpha_order else len(alpha_order)
        )
        algo_df = algo_df.sort_values("alpha_sort")

        ax.plot(
            algo_df["dirichlet_alpha"],
            algo_df["mean_accuracy"],
            marker="o",
            linewidth=2,
            label=format_algorithm_name(algorithm),
        )

    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(alpha_order)
    ax.set_xticklabels([str(alpha) for alpha in alpha_order])
    ax.set_xlabel("Dirichlet Alpha")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Dirichlet Alpha")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)


def plot_client_scalability(df: pd.DataFrame):
    st.subheader("Client Scalability")

    required = {"algorithm", "num_clients", "final_accuracy"}
    if not required.issubset(df.columns):
        st.warning("Missing columns required for client scalability plot.")
        return

    plot_df = df.dropna(subset=["algorithm", "num_clients", "final_accuracy"]).copy()

    if plot_df.empty:
        st.info("No client scalability data available.")
        return

    grouped = (
        plot_df.groupby(["algorithm", "num_clients"], as_index=False)["final_accuracy"]
        .mean()
        .rename(columns={"final_accuracy": "mean_accuracy"})
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for algorithm in sorted(grouped["algorithm"].unique()):
        algo_df = grouped[grouped["algorithm"] == algorithm].sort_values("num_clients")
        ax.plot(
            algo_df["num_clients"],
            algo_df["mean_accuracy"],
            marker="o",
            linewidth=2,
            label=format_algorithm_name(algorithm),
        )

    ax.set_xlabel("Number of Clients")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy vs Number of Clients")
    ax.set_xticks(sorted(grouped["num_clients"].dropna().unique()))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)


def plot_dataset_comparison(df: pd.DataFrame):
    st.subheader("Dataset Comparison")

    required = {"dataset", "final_accuracy"}
    if not required.issubset(df.columns):
        st.warning("Missing columns required for dataset comparison.")
        return

    plot_df = (
        df.dropna(subset=["dataset", "final_accuracy"])
        .groupby("dataset", as_index=False)["final_accuracy"]
        .mean()
        .rename(columns={"final_accuracy": "mean_accuracy"})
    )

    if plot_df.empty:
        st.info("No dataset comparison data available.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_df["dataset"].str.upper(), plot_df["mean_accuracy"], color=["#E15759", "#76B7B2"][:len(plot_df)])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Heart vs Cancer Dataset Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Federated Learning Experiment Dashboard", layout="wide")
    st.title("Federated Learning Experiment Dashboard")

    df = load_results()
    filtered_df = apply_filters(df)

    st.write(f"Filtered experiments: {len(filtered_df)}")

    if filtered_df.empty:
        st.warning("No experiments match the selected filters.")
        return

    plot_algorithm_comparison(filtered_df)
    plot_dirichlet_heterogeneity(filtered_df)
    plot_client_scalability(filtered_df)
    plot_dataset_comparison(filtered_df)


if __name__ == "__main__":
    main()
