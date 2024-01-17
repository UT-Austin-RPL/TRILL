import numpy as np


def weighted_pinv(A, W, rcond=1e-15):
    return np.dot(
        W,
        np.dot(
            A.transpose(), np.linalg.pinv(np.dot(np.dot(A, W), A.transpose()), rcond)
        ),
    )
