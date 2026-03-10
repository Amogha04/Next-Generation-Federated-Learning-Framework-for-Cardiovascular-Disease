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

    hospital1_idx = np.concatenate([idx_class0[:split1], idx_class1[:int(0.2*len(idx_class1))]])
    hospital2_idx = np.concatenate([idx_class0[split1:], idx_class1[split2:]])
    hospital3_idx = np.concatenate([idx_class1[int(0.2*len(idx_class1)):split2]])

    partitions = [
        (X[hospital1_idx], y[hospital1_idx]),
        (X[hospital2_idx], y[hospital2_idx]),
        (X[hospital3_idx], y[hospital3_idx]),
    ]

    return partitions