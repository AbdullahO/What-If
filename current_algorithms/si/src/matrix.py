import numpy as np

# compute spectral energy (squared singular values)
def spectral_energy(s):
    return 100 * (s**2).cumsum() / (s**2).sum()


# center columns of data
def center_data(X):
    a = X.mean(axis=0)
    return X - a, a


# compute approximate rank of matrix
def approximate_rank(X, t=0.99):
    """
    Input:
        X: donor data (#samples x #donor units)
        t: percentage of spectral energy to retain

    Output:
        rank: approximate rank of X at t% of spectral energy
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    total_energy = (s**2).cumsum() / (s**2).sum()
    rank = list((total_energy > t)).index(True) + 1
    return rank


def approximate_rank2(X, t=0):
    """
    Input:
        X: donor data (#samples x #donor units)
        t: percentage of spectral energy to retain

    Output:
        rank: approximate rank of X
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    b = X.shape[1] / X.shape[0]
    omega = 0.56 * b**3 - 0.95 * b**2 + 1.43 + 1.82 * b
    thre = omega * np.median(s)
    rank = len(s[s > thre])
    return rank


# hard singular value thresholding (hsvt)
def hsvt(X, rank=2, return_all=False):
    """
    Input:
        X: donor data (#samples x #donor units)
        rank: #singular values of X to retain

    Output:
        low rank approximateion of X
    """
    if rank is None:
        return X
    u, s, v = np.linalg.svd(X, full_matrices=False)
    s[rank:].fill(0)
    if return_all:
        return np.dot(u * s, v), u[:, :rank], s[:rank], v[:rank, :]
    return np.dot(u * s, v)
