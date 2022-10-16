import numpy as np
import scipy.optimize


def ConvexRegression(X, y):
    """
    Convex regression.


    minimize    ||y - Xw||^2_2
    s.t. w>=0 & \sum_i(w_i)=1

    Parameters
    ----------
    X: shape (p,n)
    y: shape (p)

    """
    p, n = X.shape

    # Objective function
    def f(w):
        return ((np.dot(X, w) - y) ** 2).sum()

    def jac_f(w):
        return -(2 * ((y - np.dot(X, w)).T).dot(X))

    # Defining constraints
    def sum_con(w):
        return np.ones((n)).dot(w) - 1

    dic_sum_con = {"type": "eq", "fun": sum_con}

    def positive_con(w):
        return w

    dic_positive_con = {"type": "ineq", "fun": positive_con}

    cons = [dic_sum_con, dic_positive_con]

    # Scipy optimization
    result = scipy.optimize.minimize(
        f, np.ones(n) / n, jac=jac_f, constraints=cons, method="SLSQP"
    )

    return result
