# import json
# from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

#############################
# Load mSSA functions
#############################


def matrixFromSVD(s, U, V, k):
    Uk = U[:, :k]
    Vk = V[:k, :]
    sk = s[:k]

    return np.dot(Uk, np.dot(np.diag(sk), Vk))


def pInverseMatrixFromSVD(sk, Uk, Vk, soft_threshold=0, probability=1.0):
    s = np.array(sk)
    s = s - soft_threshold

    for i in range(0, len(s)):
        if s[i] > 0.0:
            s[i] = 1.0 / s[i]

    p = 1.0 / probability
    return np.dot(Uk, np.dot(np.diag(s), Vk))


def rmse(y, y_h):
    return np.sqrt(np.mean(np.square(y_h - y)))


def hankelize_ts(ts, L):
    T = ts.shape[0]
    hankel = np.zeros([L, T - L + 1])
    for i in range(T - L + 1):
        hankel[:, i] = ts[i : i + L]
    return hankel


class mSSA(object):
    """docstring for mSSA"""

    def __init__(self, L, rank=None, p=None, normalize=False, ts_names=None):
        super(mSSA, self).__init__()
        self.rank = rank
        self.L = L
        self.p = p
        self.normalize = normalize
        self.cov_beta = None
        self.ts_names = ts_names

    def fit(self, ts, train_points, validation_points=None, Hankel=False):
        """
        MSSA Algorithm:
        ts: (numpy array) Time series array with shape (T: number of observations X N: number of time series)
        l: (int) Number of rows in the Page matrix
        k: (int) Number of retained singular values (When None(default), chosen automatically by the threshold in
        "The Optimal Hard Threshold for Singular Values is 4/sqrt(3), Matan Gavish, David L. Donoho"
        """

        ts_validation = ts[train_points:, :]
        ts = ts[:train_points, :]
        T = ts.shape[0]
        m = ts.shape[1]
        N = T // self.L
        ts = ts[: self.L * N, :]
        ###### Normalize #########
        self.scaler = StandardScaler()
        self.scaler.fit(ts)
        if self.normalize:
            ts_n = self.scaler.fit_transform(ts)
        else:
            ts_n = np.array(ts)
        ##########################

        if self.p is None:
            self.p = 1.0 - np.isnan(ts_n).sum() / ts_n.size
        ts_n[np.isnan(ts_n)] = 0
        page_matrix = ts_n.reshape([self.L, N * m], order="f")
        u, s, v = np.linalg.svd(page_matrix)

        if self.rank is None:
            b = page_matrix.shape[0] / page_matrix.shape[1]
            omega = 0.56 * b**3 - 0.95 * b**2 + 1.43 + 1.82 * b
            thre = omega * np.median(s)
            k = len(s[s > thre])
        else:
            k = self.rank

        M_hat = matrixFromSVD(s, u, v, k) / self.p
        denoised_array = M_hat.flatten("f")
        denoised_array = denoised_array.reshape([self.L * N, m], order="f")
        # estimate coefficients
        if Hankel:
            T = self.L * N
            page_matrix = np.zeros([self.L, (T - self.L + 1) * m])
            for j in range(m):
                page_matrix[
                    :, j * (T - self.L + 1) : (j + 1) * (T - self.L + 1)
                ] = hankelize_ts(ts[:, j], self.L)
        page_matrix_ = page_matrix[:-1, :]
        last_row = page_matrix[-1, :]
        u, s, v = np.linalg.svd(page_matrix_)
        Uk_multi = u[:, :k]
        Vk = v[:k, :]
        sk = s[:k]
        X_hat = matrixFromSVD(s, u, v, k)
        weights = np.dot(pInverseMatrixFromSVD(sk, Uk_multi, Vk) / self.p, last_row)
        project_matrix_multi = np.dot(Uk_multi, Uk_multi.T)
        S = np.diag(sk ** (-2))
        sigma_1 = np.mean(np.square(X_hat - page_matrix_))
        self.cov_beta = sigma_1 * np.dot(np.dot(Uk_multi, S), Uk_multi.T)
        ###### unnormalize #########
        if self.normalize:
            denoised_array = self.scaler.inverse_transform(denoised_array)
        self.imputed = denoised_array
        self.weights = weights
        self.project_matrix = project_matrix_multi
        self.validation_score = np.zeros(ts_validation.shape[1])
        if ts_validation.shape[0] > 0:
            y_hat_val = self.scaler.transform(
                self.forecast(ts, ts_validation.shape[0])[0]
            )
            ts_validation = self.scaler.transform(ts_validation)
            for i in range(ts_validation.shape[1]):
                self.validation_score[i] = rmse(ts_validation[:, i], y_hat_val[:, i])

    def forecast(self, data, points_ahead=1):
        if self.weights is None:
            raise Exception("use fit first")
        """
        forecast several points ahead using the linear coefficients (weights)
        """
        ##### Normlaize data ########
        if self.normalize:
            data = self.scaler.transform(data)

        L = len(self.weights)
        assert data.shape[0] >= L
        forecast = np.zeros([data.shape[0] + points_ahead, data.shape[1]])
        std = np.zeros([data.shape[0] + points_ahead, data.shape[1]])

        forecast[: data.shape[0]] = data
        for i in range(data.shape[0], data.shape[0] + points_ahead):
            forecast[i, :] = np.dot(forecast[i - L : i, :].T, self.weights)
            for j in range(data.shape[1]):
                std[i, j] = np.dot(
                    forecast[i - L : i, j].T,
                    np.dot(self.cov_beta, forecast[i - L : i, j]),
                )
        ##### Unnormalize results ####
        if self.normalize:
            forecast[data.shape[0] :, :] = self.scaler.inverse_transform(
                forecast[data.shape[0] :, :]
            )
            std[data.shape[0] :, :] = self.scaler.inverse_transform(
                std[data.shape[0] :, :]
            )

        return forecast[data.shape[0] :, :], np.sqrt(std[data.shape[0] :, :])
