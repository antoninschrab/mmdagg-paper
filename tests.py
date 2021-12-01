import numpy as np
from median import compute_median_bandwidth_subset
from weights import create_weights
from kernel import kernel_matrices, mutate_K


def mmdagg(
    seed, X, Y, alpha, kernel_type, approx_type, weights_type, l_minus, l_plus, B1, B2, B3
):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper using the collection of
    bandwidths defined in Eq. (16) and the weighting strategies proposed in Section 5.1.
    inputs: seed: integer random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            weights_type: "uniform", "decreasing", "increasing" or "centred" (Section 5.1 of our paper)
            l_minus: integer (for collection of bandwidths Eq. (16) in our paper)
            l_plus: integer (for collection of bandwidths Eq. (16) in our paper)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    assert 0 < alpha  and alpha < 1
    assert kernel_type in ["gaussian", "laplace"]
    assert approx_type in ["permutation", "wild bootstrap"]
    assert weights_type in ["uniform", "decreasing", "increasing", "centred"]
    assert l_plus >= l_minus

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
    
    # define bandwidth_multipliers and weights
    bandwidth_multipliers = np.array([2 ** i for i in range(l_minus, l_plus + 1)])
    N = bandwidth_multipliers.shape[0]  # N = 1 + l_plus - l_minus
    weights = create_weights(N, weights_type)
    
    # compute the kernel matrices
    kernel_matrices_list = kernel_matrices(
        X, Y, kernel_type, median_bandwidth, bandwidth_multipliers
    ) 

    return mmdagg_custom(
        seed, 
        kernel_matrices_list, 
        weights, 
        m, 
        alpha, 
        approx_type, 
        B1, 
        B2, 
        B3,
    )


def mmdagg_custom(
    seed, kernel_matrices_list, weights, m, alpha, approx_type, B1, B2, B3
):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper with custom kernel matrices
    and weights.
    inputs: seed: integer random seed
            kernel_matrices_list: list of N kernel matrices
                these can correspond to kernel matrices obtained by considering
                different bandwidths of a fixed kernel as we consider in our paper
                but one can also use N fundamentally different kernels.
                It is assumed that the kernel matrices are of shape (m+n,m+n) with
                the top left (m,m) submatrix corresponding to samples from X and 
                the bottom right (n,n) submatrix corresponding to samples from Y
            weights: array of shape (N,) consisting of positive entries summing to 1
            m: the number of samples from X used to create the kernel matrices
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    n = kernel_matrices_list[0].shape[0] - m
    mn = m + n
    N = len(kernel_matrices_list)
    assert len(kernel_matrices_list) == weights.shape[0]
    assert n >= 2 and m >= 2
    assert 0 < alpha  and alpha < 1
    assert approx_type in ["permutation", "wild bootstrap"]
    
    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    M  = np.zeros((N, B1 + B2 + 1))  
    rs = np.random.RandomState(seed)
    if approx_type == "permutation":
        idx = rs.rand(B1 + B2 + 1, m + n).argsort(axis=1)  # (B1+B2+1, m+n): rows of permuted indices
        #11
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        V11i = np.tile(v11, (B1 + B2 + 1, 1))  # (B1+B2+1, m+n)
        V11 = np.take_along_axis(V11i, idx, axis=1)  # (B1+B2+1, m+n): permute the entries of the rows
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V11 = V11.transpose()  # (m+n, B1+B2+1)
        #10
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        V10i = np.tile(v10, (B1 + B2 + 1, 1))
        V10 = np.take_along_axis(V10i, idx, axis=1)
        V10[B1] = v10
        V10 = V10.transpose() 
        #01
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V01i = np.tile(v01, (B1 + B2 + 1, 1))
        V01 = np.take_along_axis(V01i, idx, axis=1)
        V01[B1] = v01
        V01 = V01.transpose() 
        for i in range(N):
            K = kernel_matrices_list[i]
            mutate_K(K, approx_type)
            M[i] = (
                np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                + np.sum(V11 * (K @ V11), 0) / (m * n)
            )  # (B1+B2+1, ) permuted MMD estimates
    elif approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1) 
        for i in range(N):
            K = kernel_matrices_list[i]
            mutate_K(K, approx_type)
            M[i] = np.sum(R * (K @ R) , 0) /(n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)
    
    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for i in range(N):
            quantiles[i] = M1_sorted[
                i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
        
    # Step 3: output test result
    for i in range(N):
        if ( MMD_original[i] 
            > M1_sorted[i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1]
        ):
            return 1
    return 0 


def mmd_median_test(
    seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multiplier=1
):
    """
    Compute MMD test using kernel with bandwidth the median bandwidth multiplied by bandwidth_multiplier.
    This test has been proposed by 
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf and Alexander Smola
        A Kernel Two-Sample Test
        Journal of Machine Learning Research 2012
        https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    inputs: seed: random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            bandwidth_multiplier: multiplicative factor for the median bandwidth 
    output: result of the MMD test with median bandwidth multiplied by bandwidth_multiplier
            (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    assert 0 < alpha  and alpha < 1
    assert kernel_type in ["gaussian", "laplace"]
    assert approx_type in ["permutation", "wild bootstrap"]

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
                           
    # compute all simulated MMD estimates (efficient)
    K = kernel_matrices(
        X, Y, kernel_type, median_bandwidth, np.array([bandwidth_multiplier])
    )[0]
    mutate_K(K, approx_type)  
    rs = np.random.RandomState(seed)
    if approx_type == "permutation":
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V11 = np.tile(v11, (B1 + 1, 1))  # (B1+1, m+n)
        V10 = np.tile(v10, (B1 + 1, 1))
        V01 = np.tile(v01, (B1 + 1, 1))
        idx = rs.rand(*V11.shape).argsort(axis=1)  # (B1+1, m+n): rows of permuted indices
        V11 = np.take_along_axis(V11, idx, axis=1)  # (B1+1, m+n): permute the entries of the rows
        V10 = np.take_along_axis(V10, idx, axis=1)
        V01 = np.take_along_axis(V01, idx, axis=1)
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V10[B1] = v10
        V01[B1] = v01
        V11 = V11.transpose()  # (m+n, B1+1)
        V10 = V10.transpose() 
        V01 = V01.transpose() 
        M1 = (
            np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
            + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
            + np.sum(V11 * (K @ V11), 0) / (m * n)
        )  # (B1+1, ) permuted MMD estimates
    elif approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+1) 
        M1 = np.sum(R * (K @ R) , 0) /(n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    MMD_original = M1[B1]
    M1_sorted = np.sort(M1) 
    
    # output test result
    if MMD_original > M1_sorted[int(np.ceil((B1 + 1) * (1 - alpha))) - 1]:
        return 1
    return 0 


def ratio_mmd_stdev(K, approx_type, regulariser=10**(-8)):
    """
    Compute the estimated ratio of the MMD to the asymptotic standard deviation under the alternative.
    This is stated in Eq. (15) in our paper, it originally comes from Eq. (3) in:
        F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland
        Learning deep kernels for non-parametric two-sample tests
        International Conference on Machine Learning, 2020
        http://proceedings.mlr.press/v119/liu20m/liu20m.pdf
    assumption: m = n: equal number of samples in X and Y
    inputs: K: (m+n, m+n) kernel matrix for pooled sample WITH diagonal 
               (K has NOT been mutated by mutate_K function)
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            regulariser: small positive number (we use 10**(-8) as done by Liu et al.)
            m: number of samples (d-dimensional points) in X 
            K: (m+n, m+n) kernel matrix for pooled sample WITH diagonal        
    output: estimate of criterion J which is the ratio of MMD^2 and of the variance under the H_a
    warning: this function mutates K using the mutate_K function
             there is no approximation but approx_type is required to determine whether to use
             MMD_a estimate as in Eq. (3) or MMD_b estimate as in Eq. (6)
    """ 
    n = int(K.shape[0]/2)

    # compute variance
    Kxx = K[:n, :n]
    Kxy = K[:n, n:]
    Kyx = K[n:, :n]
    Kyy = K[n:, n:]
    H_column_sum = (
        np.sum(Kxx, axis=1)
        + np.sum(Kyy, axis=1)
        - np.sum(Kxy, axis=1)
        - np.sum(Kyx, axis=1)
    )
    var = (
        4 / n ** 3 * np.sum(H_column_sum ** 2)
        - 4 / n ** 4 * np.sum(H_column_sum) ** 2
        + regulariser
    )
    # we should obtain var > 0, if var <= 0 then we discard the corresponding
    # bandwidth by returning a large negative value so that we do not select
    # the corresponding bandwidth when selecting the maximum of the outputs
    # of ratio_mmd_stdev for bandwidths in the collection
    if not var > 0:
        return -1e10 

    # compute original MMD estimate
    mutate_K(K, approx_type)
    if approx_type == "permutation":
        # compute MMD_a estimate
        Kxx = K[:n, :n]
        Kxy = K[:n, n:]
        Kyy = K[n:, n:]
        s = np.ones(n)
        mmd = (
            s @ Kxx @ s / (n * (n - 1))
            + s @ Kyy @ s / (n * (n - 1))
            - 2 * s @ Kxy @ s / (n ** 2)
        )
    elif approx_type == "wild bootstrap":
        # compute MMD_b estimate
        v = np.concatenate((np.ones(n), -np.ones(n)))
        mmd = v @ K @ v / (n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    return mmd / np.sqrt(var)


def mmd_split_test(
    seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multipliers, proportion=0.5
):
    """
    Split data in equal halves. Select 'optimal' bandwidth using first half (in the sense 
    that it maximizes ratio_mmd_stdev) and run the MMD test with the selected bandwidth on 
    the second half. This was first proposed by Gretton et al. (2012) for the linear-time 
    MMD estimate and generalised by Liu et al. (2020) to the quadratic-time MMD estimate.
            Arthur Gretton, Bharath Sriperumbudur, Dino Sejdinovic, Heiko Strathmann,
        Sivaraman Balakrishnan, Massimiliano Pontil and Kenji Fukumizu
        Optimal kernel choice for large-scale two-sample tests
        Advances in Neural Information Processing Systems 2012
        https://papers.nips.cc/paper/2012/file/dbe272bab69f8e13f14b405e038deb64-Paper.pdf
            F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland
        Learning deep kernels for non-parametric two-sample tests
        International Conference on Machine Learning, 2020
        http://proceedings.mlr.press/v119/liu20m/liu20m.pdf
    inputs: seed: integer random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            bandwidth_multipliers: array such that the 'optimal' bandwidth is selected from
                                   collection_bandwidths = [c*median_bandwidth for c in bandwidth_multipliers]
            proportion: proportion of data used to select the bandwidth 
    output: result of MMD test run on half the data with the bandwidth from collection_bandwidths which is 
            'optimal' in the sense that it maximizes ratio_mmd_stdev on the other half of the data
            (REJECT H_0 = 1, FAIL TO REJECT H_0 = 0)
    """
    assert X.shape == Y.shape
    n, d = X.shape 
    
    split_size = int(n * proportion) 
    
    rs = np.random.RandomState(seed)
    pX = rs.permutation(n)
    pY = rs.permutation(n)
    X1 = X[pX][:split_size]
    X2 = X[pX][split_size:]
    Y1 = Y[pY][:split_size]
    Y2 = Y[pY][split_size:]  
    
    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)

    # select bandwidth which maximizes criterion J using X1 and Y1
    kernel_matrices_list = kernel_matrices(
        X1, Y1, kernel_type, median_bandwidth, bandwidth_multipliers
    )
    ratio_values = []
    for i in range(len(kernel_matrices_list)):
        K = kernel_matrices_list[i]
        ratio_values.append(ratio_mmd_stdev(K, approx_type))
    selected_multiplier = bandwidth_multipliers[np.argmax(ratio_values)]
    
    # run MMD test on X2 and Y2 with the selected bandwidth
    return mmd_median_test(
        seed, X2, Y2, alpha, kernel_type, approx_type, B1, selected_multiplier
    )
