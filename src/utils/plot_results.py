import matplotlib.pyplot as plt

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