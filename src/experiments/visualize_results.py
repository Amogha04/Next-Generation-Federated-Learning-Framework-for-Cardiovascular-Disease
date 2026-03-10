import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _to_bool_str(series: pd.Series) -> pd.Series:
    """Normalize DP column values to 'true'/'false' strings."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"1": "true", "0": "false", "yes": "true", "no": "false"})
    )


def _save_barplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: str) -> None:
    # Aggregate to mean for stable comparison across repeated experiments
    plot_df = (
        df.groupby(x_col, dropna=False, as_index=False)[y_col]
        .mean()
        .sort_values(by=x_col)
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x=x_col, y=y_col, palette="viridis")
    plt.title(title)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(y_col.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_csv = os.path.join(project_root, "results", "experiment_results.csv")
    plots_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)

    required_cols = {
        "algorithm",
        "dataset",
        "distribution",
        "differential_privacy",
        "final_accuracy",
        "final_auc",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Clean numeric fields
    df["final_accuracy"] = pd.to_numeric(df["final_accuracy"], errors="coerce")
    df["final_auc"] = pd.to_numeric(df["final_auc"], errors="coerce")
    df = df.dropna(subset=["final_accuracy", "final_auc"])

    # Normalize categorical fields
    df["algorithm"] = df["algorithm"].astype(str).str.strip().str.lower()
    df["dataset"] = df["dataset"].astype(str).str.strip().str.lower()
    df["distribution"] = df["distribution"].astype(str).str.strip().str.lower()
    df["differential_privacy"] = _to_bool_str(df["differential_privacy"])

    # A) Algorithm comparison (accuracy)
    _save_barplot(
        df=df,
        x_col="algorithm",
        y_col="final_accuracy",
        title="Final Accuracy by Algorithm",
        out_path=os.path.join(plots_dir, "accuracy_by_algorithm.png"),
    )

    # B) Differential Privacy comparison (accuracy)
    _save_barplot(
        df=df,
        x_col="differential_privacy",
        y_col="final_accuracy",
        title="Final Accuracy: DP vs No-DP",
        out_path=os.path.join(plots_dir, "accuracy_dp_comparison.png"),
    )

    # C) Dataset comparison (accuracy)
    _save_barplot(
        df=df,
        x_col="dataset",
        y_col="final_accuracy",
        title="Final Accuracy by Dataset",
        out_path=os.path.join(plots_dir, "accuracy_by_dataset.png"),
    )

    # D) Distribution comparison (accuracy)
    _save_barplot(
        df=df,
        x_col="distribution",
        y_col="final_accuracy",
        title="Final Accuracy by Distribution",
        out_path=os.path.join(plots_dir, "accuracy_by_distribution.png"),
    )

    # E) AUC comparison (algorithm)
    _save_barplot(
        df=df,
        x_col="algorithm",
        y_col="final_auc",
        title="Final AUC by Algorithm",
        out_path=os.path.join(plots_dir, "auc_by_algorithm.png"),
    )

    print(f"Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
