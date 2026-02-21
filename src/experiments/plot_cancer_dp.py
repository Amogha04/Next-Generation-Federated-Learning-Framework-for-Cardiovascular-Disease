from src.utils.plot_results import plot_metrics

rounds = list(range(0, 11))

accuracy = [
    0.5272, 0.7574, 0.8699, 0.9103, 0.9086,
    0.9121, 0.9191, 0.9226, 0.9244, 0.9314, 0.9314
]

auc = [
    0.7074, 0.8845, 0.9310, 0.9495, 0.9594,
    0.9648, 0.9682, 0.9706, 0.9728, 0.9746, 0.9763
]

plot_metrics(rounds, accuracy, auc,
             title="DP Federated Learning - Cancer Dataset",
             save_path="cancer_dp_convergence.png")