import numpy as np


def linear_regression(X, y, rcond=1e-15):
    """
    Input:
            X: pre-int. donor data (#pre-int. samples x #donor units)
            y: pre-int. target data (#pre-int. samples x 1)

    Output:
            synthetic control (regression coefficients)
    """
    return np.linalg.pinv(X, rcond=rcond).dot(y)
