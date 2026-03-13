
# Run Project Guide

This document explains how teammates can set up, run, and capture outputs from the federated learning project.

---

## 1. Clone the Repository

```bash
git clone https://github.com/Amogha04/Next-Generation-Federated-Learning-Framework-for-Cardiovascular-Disease
cd Next-Generation-Federated-Learning-Framework-for-Cardiovascular-Disease
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate the Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run Federated Experiments

```bash
python src/experiments/experiment_runner.py
```

This command runs the automated federated learning experiment pipeline.

It may take several hours to complete because the framework can run multiple experiment combinations involving:
- different federated learning algorithms
- different datasets
- IID, Non-IID, and Dirichlet distributions
- multiple client counts
- client dropout settings
- differential privacy settings

Please allow sufficient time for the experiments to finish fully before generating plots.

---

## 6. Generate Plots

```bash
python src/utils/plot_results.py
```

This command generates visualization outputs from the experiment results stored in:

```text
results/experiment_results.csv
```

Generated plots are typically saved inside:

```text
results/plots/
```

---

## 7. Launch Dashboard

```bash
streamlit run dashboard.py
```

This command launches the Streamlit dashboard for interactive experiment visualization.

After running it, the dashboard should automatically open in your browser.  
If it does not open automatically, Streamlit will display a local URL in the terminal that you can open manually.

---

# Screenshots to Capture for Report / README

Please capture screenshots of the following outputs after the project runs successfully.

## 1. Convergence Plots

Capture the convergence plots generated from:

```text
results/plots/convergence plots
```

These should show training behavior across communication rounds.

## 2. Accuracy vs Dirichlet Alpha Plot

Capture the plot located at:

```text
results/plots/accuracy_vs_alpha.png
```

This plot shows the effect of Dirichlet heterogeneity on model accuracy.

## 3. Accuracy vs Clients Plot

Capture the plot located at:

```text
results/plots/accuracy_vs_clients.png
```

This plot shows how model accuracy changes as the number of clients increases.

## 4. Streamlit Dashboard Main Screen

Capture the main dashboard page after launching:

```bash
streamlit run dashboard.py
```

Include the visible charts and sidebar filters if possible.

## 5. Experiment Runner Terminal Output

Capture the terminal while experiments are running using:

```bash
python src/experiments/experiment_runner.py
```

The screenshot should show experiment progress, configurations, or logging output.

## 6. Final Results CSV File

Capture the contents of the final results file:

```text
results/experiment_results.csv
```

Make sure the screenshot clearly shows the experiment result columns and sample rows.

---

## Where to Save Screenshots

Save all screenshots inside this folder in the repository:

```text
images/
```

Recommended filenames:
- `images/convergence.png`
- `images/alpha_plot.png`
- `images/clients_plot.png`
- `images/dashboard.png`
- `images/runner_terminal.png`
- `images/results_csv.png`

---

## Notes

- Ensure all experiments finish before generating final plots.
- If running the full experiment suite, expect long execution time.
- Keep screenshots clear and readable for use in the final report and repository README.