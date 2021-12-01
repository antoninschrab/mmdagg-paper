import numpy as np
from numba import njit


def compute_median_bandwidth_subset(seed, X, Y, max_samples=2000, min_value = 0.0001):
    """
    Compute the median distance in each dimension between all the points in X and Y
    using at most max_samples samples and using a threshold value min_value.
    inputs: seed: random seed
            X: (m,d) array of samples
            Y: (n,d) array of samples
            max_samples: number of samples used to compute the median (int or None)
    output: (d,) array: median of absolute difference in each component
    """
    if max_samples != None:
        rs = np.random.RandomState(seed)
        pX = rs.choice(X.shape[0], min(max_samples // 2, X.shape[0]), replace=False)
        pY = rs.choice(Y.shape[0], min(max_samples // 2, Y.shape[0]), replace=False)
        Z = np.concatenate((X[pX], Y[pY]))
    else:
        Z = np.concatenate((X, Y))
    median_bandwidth = compute_median_bandwidth(Z)
    return np.maximum(median_bandwidth, min_value)
   

@njit
def compute_median_bandwidth(Z):
    """
    Compute the median distance in each dimension between all the points in Z.
    input: Z: (m+n,d) array of pooled samples  
    output: (d,) array: median of absolute different in each component
    """
    mn, d = Z.shape
    diff = np.zeros((d, int((mn ** 2 - mn) / 2)))
    output = np.zeros(d)
    for u in range(d):
        k = 0
        for i in range(mn - 1):
            for j in range(i + 1, mn):
                diff[u, k] = np.abs(Z[i, u] - Z[j, u])
                k += 1
        output[u] = np.median(diff[u])
    return output
    