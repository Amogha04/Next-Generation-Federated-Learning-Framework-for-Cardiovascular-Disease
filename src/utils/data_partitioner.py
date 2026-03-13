import numpy as np


def non_iid_partition(X, y, num_clients=3):
    """
    Simulate non-IID hospital datasets.
    Each hospital receives different label distributions.
    """

    idx_class0 = np.where(y == 0)[0]
    idx_class1 = np.where(y == 1)[0]

    np.random.shuffle(idx_class0)
    np.random.shuffle(idx_class1)

    split1 = int(0.7 * len(idx_class0))
    split2 = int(0.7 * len(idx_class1))

    hospital1_idx = np.concatenate([idx_class0[:split1], idx_class1[:int(0.2 * len(idx_class1))]])
    hospital2_idx = np.concatenate([idx_class0[split1:], idx_class1[split2:]])
    hospital3_idx = np.concatenate([idx_class1[int(0.2 * len(idx_class1)):split2]])

    partitions = [
        (X[hospital1_idx], y[hospital1_idx]),
        (X[hospital2_idx], y[hospital2_idx]),
        (X[hospital3_idx], y[hospital3_idx]),
    ]

    return partitions


def _slice_features(X, indices):
    """Support both NumPy arrays and pandas DataFrames."""
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def dirichlet_partition(X, y, num_clients, alpha):
    """
    Partition a dataset across clients using a Dirichlet distribution.

    Dirichlet partitioning is widely used in federated learning to simulate
    non-IID client datasets. Each class is processed separately, and a
    Dirichlet distribution determines what fraction of that class goes to each
    client.

    Lower alpha values create stronger heterogeneity, so clients tend to have
    very different class distributions. Higher alpha values produce more
    balanced splits across clients.

    This function works for binary classification datasets and also generalizes
    to multi-class settings.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be greater than 0.")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0.")

    y_array = np.asarray(y)
    unique_classes = np.unique(y_array)
    client_indices = [[] for _ in range(num_clients)]

    for class_label in unique_classes:
        class_indices = np.where(y_array == class_label)[0]
        np.random.shuffle(class_indices)

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(class_splits):
            client_indices[client_id].extend(split.tolist())

    client_datasets = []
    for indices in client_indices:
        indices = np.array(indices, dtype=int)
        np.random.shuffle(indices)
        client_datasets.append((_slice_features(X, indices), y_array[indices]))

    return client_datasets
