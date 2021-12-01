import numpy as np
from numba import njit


@njit
def pairwise_square_l2_distance(Z):
    """
    Compute the pairwise L^2-distance matrix between all points in Z.
    inputs: Z is (mn,d) array
    output: (mn,mn) array of pairwise squared distances (L^2)
    https://stackoverflow.com/questions/53376686/what-is-the-most-efficient-way-to-compute-the-square-euclidean-distance-between/53380192#53380192
    faster than scipy.spatial.distance.cdist(Z,Z,'sqeuclidean')
    """
    mn, d = Z.shape
    dist = np.dot(Z, Z.T)  
    TMP = np.empty(mn, dtype=Z.dtype)
    for i in range(mn):
        sum_Zi = 0.0
        for j in range(d):
            sum_Zi += Z[i, j] ** 2
        TMP[i] = sum_Zi
    for i in range(mn):
        for j in range(mn):
            dist[i, j] = -2.0 * dist[i, j] + TMP[i] + TMP[j]
    return dist


@njit 
def pairwise_l1_distance(Z):
    """
    Compute the pairwise L^1-distance matrix between all points in Z.
    inputs: Z is (mn,d) array
    output: (mn,mn) array of pairwise squared distances (L^1)
    """
    mn, d = Z.shape
    output = np.zeros((mn, mn))
    for i in range(mn):
        for j in range(mn):
            temp = 0.0
            for u in range(d):
                temp += np.abs(Z[i, u] - Z[j, u])
            output[i, j] = temp
    return output


@njit
def kernel_matrices(X, Y, kernel_type, bandwidth, bandwidth_multipliers):
    """
    Compute kernel matrices for several bandwidths.
    inputs: kernel_type: "gaussian" or "laplace"
            X is (m,d) array (m d-dimensional points)
            Y is (n,d) array (n d-dimensional points)
            bandwidth is (d,) array
            bandwidth_multipliers is (N,) array such that: 
                collection_bandwidths = [c*bandwidth for c in bandwidth_multipliers]
            kernel_type: "gaussian" or "laplace" (as defined in Section 5.3 of our paper)
    outputs: list of N kernel matrices for the pooled sample with the N bandwidths
    """
    m, d = X.shape
    Z = np.concatenate((X / bandwidth, Y / bandwidth))
    if kernel_type == "gaussian":
        pairwise_sq_l2_dists = pairwise_square_l2_distance(Z) 
        prod = np.prod(bandwidth)
        output_list = []
        for c in bandwidth_multipliers:
            output_list.append(np.exp(-pairwise_sq_l2_dists / (c ** 2))) 
        return output_list
    elif kernel_type == "laplace":
        pairwise_l1_dists = pairwise_l1_distance(Z) 
        prod = np.prod(bandwidth)
        output_list = []
        for c in bandwidth_multipliers:
            output_list.append(np.exp(-pairwise_l1_dists / c)) 
        return output_list
    else:
        raise ValueError(
            'The value of kernel_type should be either "gaussian" or "laplace"'
        )


@njit
def mutate_K(K, approx_type):
    """
    Mutate the kernel matrix K depending on the type of approximation.
    inputs: K: kernel matrix of size (m+n,m+n) consisting of 
               four matrices of sizes (m,m), (m,n), (n,m) and (n,n)
               m and n are the numbers of samples from p and q respectively
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            
    output: if approx_type is "permutation" then the estimate is MMD_a (Eq. (3)) and 
               the matrix K is mutated to have zero diagonal entries
            if approx_type is "wild bootstrap" then the estimate is MMD_b (Eq. (6)),
               we have m = n and the matrix K is mutated so that the four matrices 
               have zero diagonal entries
    """
    if approx_type == "permutation":
        for i in range(K.shape[0]):
            K[i, i] = 0      
    if approx_type == "wild bootstrap":
        m = int(K.shape[0] / 2)  # m = n
        for i in range(m):
            K[i, i] = 0
            K[m + i, m + i] = 0
            K[i, m + i] = 0 
            K[m + i, i] = 0
