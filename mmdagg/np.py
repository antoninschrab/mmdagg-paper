import numpy as np
import scipy.spatial

def mmdagg(
    X,
    Y,
    alpha=0.05,
    kernel="laplace_gaussian",
    number_bandwidths=10,
    weights_type="uniform", 
    B1=2000, 
    B2=2000, 
    B3=50,
    seed=42,
    return_dictionary=False,
    permutations_same_sample_size=False,
):
    """
    Two-Sample MMDAgg test.
    
    Given data from one distribution and data from another distribution,
    return 0 if the test fails to reject the null 
    (i.e. data comes from the same distribution), 
    or return 1 if the test rejects the null 
    (i.e. data comes from different distributions).
    
    Parameters
    ----------
    X : array_like
        The shape of X must be of the form (m, d) where m is the number
        of samples and d is the dimension.
    Y: array_like
        The shape of X must be of the form (n, d) where m is the number
        of samples and d is the dimension.
    alpha: scalar
        The value of alpha must be between 0 and 1.
    kernel: str
        The value of kernel must be "gaussian", "laplace", "imq", "matern_0.5_l1",
        "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", 
        "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", 
        "matern_4.5_l2", "all_matern_l1", "all_matern_l2", "all_matern_l1_l2", 
        "all", "laplace_gaussian", or "gaussian_laplace". 
    number_bandwidths: int
        The number of bandwidths per kernel to include in the collection.
    weights_type: str
        Must be "uniform", "centred", "increasing", or "decreasing".
    B1: int
        Number of simulated test statistics (through a wild bootstrap or permutations) 
        to approximate the quantiles.
    B2: int
        Number of simulated test statistics (through a wild bootstrap or permutations) 
        to approximate the level correction.
    B3: int
        Number of steps of bissection method to perform to estimate the level correction.
    seed: int 
        Random seed used for the randomness of the wild bootstrap or permutations.
    return_dictionary: bool
        If true, a dictionary is returned containing for each single test: 
        the test output, the kernel, the bandwidth, the MMD value, the MMD quantile value, 
        the p-value and the p-value threshold value.      
    permutations_same_sample_size: bool 
        If the sample sizes are different, permutations are used.
        If the sample sizes are equal, a wild bootstrap is used by default, 
        if permutations_same_sample_size is true then permutations are used instead.
    bandwidths: array_like or None
        If bandwidths is None, the bandwidths for each kernel are computed automatically. 
        If bandwidths is array_like of one dimension, the bandwidths provided are used
        for each kernel.
        Note that number_bandwidths is overwritten by the length of bandwidths.
        
    Returns
    -------
    output : int
        0 if the aggregated MMDAgg test fails to reject the null 
            (i.e. data comes from the same distribution)
        1 if the aggregated MMDAgg test rejects the null 
            (i.e. data comes from different distributions)
    dictionary: dict
        Returned only if return_dictionary is True.
        Dictionary containing the overall output of the MMDAgg test, and for each single test: 
        the test output, the kernel, the bandwidth, the MMD value, the MMD quantile value, 
        the p-value and the p-value threshold value.
    
    Examples
    --------
    # import modules
    >>> import numpy as np
    >>> from mmdagg.np import mmdagg

    # generate data for two-sample test
    >>> rs = np.random.RandomState(0)
    >>> X = rs.uniform(0, 1, (500, 10))
    >>> Y = rs.uniform(0, 1, (500, 10)) + 1

    # run MMDAgg test
    >>> output = mmdagg(X, Y)
    >>> output
    1
    >>> output, dictionary = mmdagg(X, Y, return_dictionary=True)
    >>> output
    1
    >>> dictionary
    {'MMDAgg test reject': True,
     'Single test 1.1': {'Reject': True,
      'Kernel laplace': True,
      'Bandwidth': 2.42038079739479,
      'MMD': 122196.9529089949,
      'MMD quantile': 741.5697172538066,
      'p-value': 0.0004997501249375312,
      'p-value threshold': 0.04147926036981442},
      ...
    }
    """    
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    if m != n or permutations_same_sample_size:
        approx_type ="permutations"
    else:
        approx_type = "wild bootstrap"
    assert 0 < alpha  and alpha < 1
    assert kernel in (
        "gaussian", 
        "laplace", 
        "imq", 
        "matern_0.5_l1", 
        "matern_1.5_l1", 
        "matern_2.5_l1", 
        "matern_3.5_l1", 
        "matern_4.5_l1", 
        "matern_0.5_l2", 
        "matern_1.5_l2", 
        "matern_2.5_l2", 
        "matern_3.5_l2", 
        "matern_4.5_l2", 
        "all_matern_l1", 
        "all_matern_l2", 
        "all_matern_l1_l2", 
        "all", 
        "laplace_gaussian", 
        "gaussian_laplace", 
    )
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert weights_type in ("uniform", "decreasing", "increasing", "centred")
    assert B1 > 0 and type(B1) == int
    assert B2 > 0 and type(B2) == int
    assert B3 > 0 and type(B3) == int

    # Collection of bandwidths 
    # lambda_min / 2 * C^r for r = 0, ..., number_bandwidths -1
    # where C is such that lambda_max * 2 = lambda_min / 2 * C^(number_bandwidths - 1)
    def compute_bandwidths(distances, number_bandwidths):    
        if np.min(distances) < 10 ** (-1):
            d = np.sort(distances)
            lambda_min = np.maximum(d[int(np.floor(len(d) * 0.05))], 10 ** (-1))
        else:
            lambda_min = np.min(distances)
        lambda_min = lambda_min / 2
        lambda_max = np.maximum(np.max(distances), 3 * 10 ** (-1))
        lambda_max = lambda_max * 2
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = np.array([power ** i * lambda_min for i in range(number_bandwidths)])
        return bandwidths
    max_samples = 500
    # bandwidths L1 for laplace, matern_0.5_l1, matern_1.5_l1, matern_2.5_l1, matern_3.5_l1, matern_4.5_l1
    distances_l1 = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "cityblock").reshape(-1)
    bandwidths_l1 = compute_bandwidths(distances_l1, number_bandwidths)
    # bandwidths L2 for gaussian, imq, matern_0.5_l2, matern_1.5_l2, matern_2.5_l2, matern_3.5_l2, matern_4.5_l2
    distances_l2 = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "euclidean").reshape(-1)
    bandwidths_l2 = compute_bandwidths(distances_l2, number_bandwidths)
    
    # Kernel and bandwidths list (order: "l1" first, "l2" second)
    if kernel in ( 
        "laplace", 
        "matern_0.5_l1", 
        "matern_1.5_l1", 
        "matern_2.5_l1", 
        "matern_3.5_l1", 
        "matern_4.5_l1", 
    ):
        kernel_bandwidths_l_list = [(kernel, bandwidths_l1, "l1"), ]
    elif kernel in (
        "gaussian", 
        "imq", 
        "matern_0.5_l2", 
        "matern_1.5_l2", 
        "matern_2.5_l2", 
        "matern_3.5_l2", 
        "matern_4.5_l2", 
    ):
        kernel_bandwidths_l_list = [(kernel, bandwidths_l2, "l2"), ]
    elif kernel in ("laplace_gaussian", "gaussian_laplace"):
        kernel_bandwidths_l_list = [("laplace", bandwidths_l1, "l1"), ("gaussian", bandwidths_l2, "l2")]
    elif kernel == "all_matern_l1":
        kernel_list = ["matern_" + str(i) + ".5_l1" for i in range(5)]
        kernel_bandwidths_l_list = [(kernel, bandwidths_l1, "l1") for kernel in kernel_list]
    elif kernel == "all_matern_l2":
        kernel_list = ["matern_" + str(i) + ".5_l2" for i in range(5)]
        kernel_bandwidths_l_list = [(kernel, bandwidths_l2, "l2") for kernel in kernel_list]
    elif kernel == "all_matern_l1_l2":
        kernel_list = [
            "matern_" + str(i) + ".5_l" + str(j) for j in (1, 2) for i in range(5) 
        ]
        bandwidths_list = [bandwidths_l1, ] * 5 + [bandwidths_l2, ] * 5
        l_list = ["l1", ] * 5 + ["l2", ] * 5
        kernel_bandwidths_l_list = [
            (kernel_list[i], bandwidths_list[i], l_list[i]) for i in range(10)
        ]
    elif kernel == "all":
        kernel_list = [
            "matern_" + str(i) + ".5_l" + str(j) for j in (1, 2) for i in range(5) 
        ] + ["gaussian", "imq"] 
        bandwidths_list = [] + [bandwidths_l1, ] * 5 + [bandwidths_l2, ] * 7
        l_list = ["l1", ] * 5 + ["l2", ] * 7
        kernel_bandwidths_l_list = [
            (kernel_list[i], bandwidths_list[i], l_list[i]) for i in range(12)
        ]
    else:
        raise ValueError("Kernel not defined.")
    
    # Weights 
    weights = create_weights(number_bandwidths, weights_type) / len(
        kernel_bandwidths_l_list
    )
    
    # Setup for wild bootstrap or permutations (efficient as in Appendix C in our paper)
    rs = np.random.RandomState(seed)
    if approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1) 
    elif approx_type == "permutations":
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
    else:
        raise ValueError("Approximation type not defined.")
        
    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    N = number_bandwidths * len(kernel_bandwidths_l_list)
    M = np.zeros((N, B1 + B2 + 1))  
    last_l_pairwise_matrix_computed = ""
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, l = kernel_bandwidths_l_list[j]
        # since kernel_bandwidths_l_list is ordered "l1" first, "l2" second
        # compute pairwise matrices the minimum amount of time
        # store only one pairwise matrix at once
        if l != last_l_pairwise_matrix_computed:
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            last_l_pairwise_matrix_computed = l
        for i in range(number_bandwidths):
            bandwidth = bandwidths[i]
            K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
            if approx_type == "wild bootstrap": 
                # set diagonal elements of all four submatrices to zero
                np.fill_diagonal(K, 0)
                np.fill_diagonal(K[:n, n:], 0)
                np.fill_diagonal(K[n:, :n], 0)
                # compute MMD bootstrapped values
                M[number_bandwidths * j + i] = np.sum(R * (K @ R), 0)
            elif approx_type == "permutations": 
                # set diagonal elements to zero
                np.fill_diagonal(K, 0)
                # compute MMD permuted values
                M[number_bandwidths * j + i] = (
                    np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                    + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                    + np.sum(V11 * (K @ V11), 0) / (m * n)
                )  
            else:
                raise ValueError("Approximation type not defined.")           
    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)

    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0 # or alpha
    u_max = np.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for j in range(len(kernel_bandwidths_l_list)):
            for i in range(number_bandwidths):
                quantiles[number_bandwidths * j + i] = M1_sorted[
                    number_bandwidths * j + i, 
                    int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
                ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min

    # Step 3: output test result
    p_vals = np.mean((M1_sorted - MMD_original.reshape(-1, 1) >= 0), -1)
    all_weights = np.zeros(p_vals.shape)
    for j in range(len(kernel_bandwidths_l_list)):
         for i in range(number_bandwidths):
            all_weights[number_bandwidths * j + i] = weights[i]
    thresholds = u * all_weights
    # reject if p_val <= threshold
    reject_p_vals = p_vals <= thresholds

    mmd_vals = MMD_original
    quantiles = quantiles.reshape(-1)
    # reject if mmd_val > quantile
    reject_mmd_vals = mmd_vals > quantiles
    
    # assert both rejection methods are equivalent
    np.testing.assert_array_equal(reject_p_vals, reject_mmd_vals)
    
    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary["MMDAgg test reject"] = False
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, l = kernel_bandwidths_l_list[j]
        for i in range(number_bandwidths):
            index = "Single test " + str(j + 1) + "." + str(i + 1)
            idx = number_bandwidths * j + i
            reject_dictionary[index] = {}
            reject_dictionary[index]["Reject"] = reject_p_vals[idx]
            reject_dictionary[index]["Kernel " + kernel] = True
            reject_dictionary[index]["Bandwidth"] = bandwidths[i]
            reject_dictionary[index]["MMD"] = mmd_vals[idx]
            reject_dictionary[index]["MMD quantile"] = quantiles[idx]
            reject_dictionary[index]["p-value"] = p_vals[i]
            reject_dictionary[index]["p-value threshold"] = thresholds[idx]
            # Aggregated test rejects if one single test rejects
            reject_dictionary["MMDAgg test reject"] = any((
                reject_dictionary["MMDAgg test reject"], 
                reject_p_vals[idx]
            ))

    if return_dictionary:
        return int(reject_dictionary["MMDAgg test reject"]), reject_dictionary
    else:
        return int(reject_dictionary["MMDAgg test reject"])

def compute_pairwise_matrix(X, Y, l):
    """
    Compute the pairwise distance matrix between all the points in X and Y,
    in L1 norm or L2 norm.

    inputs: X: (m,d) array of samples
            Y: (m,d) array of samples
            l: "l1" or "l2" or "l2sq"
    output: (2m,2m) pairwise distance matrix
    """
    Z = np.concatenate((X, Y))
    if l == "l1":
        return scipy.spatial.distance.cdist(Z, Z, 'cityblock')
    elif l == "l2":
        return scipy.spatial.distance.cdist(Z, Z, 'euclidean')
    else:
        raise ValueError("Third input should either be 'l1' or 'l2'.")

        
def kernel_matrix(pairwise_matrix, l, kernel_type, bandwidth):
    """
    Compute kernel matrix for a given kernel_type and bandwidth. 

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel_type: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel_type must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2":
        return  np.exp(-d ** 2)
    elif kernel_type == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif (kernel_type == "matern_0.5_l1" and l == "l1") or (kernel_type == "matern_0.5_l2" and l == "l2") or (kernel_type == "laplace" and l == "l1"):
        return  np.exp(-d)
    elif (kernel_type == "matern_1.5_l1" and l == "l1") or (kernel_type == "matern_1.5_l2" and l == "l2"):
        return (1 + np.sqrt(3) * d) * np.exp(- np.sqrt(3) * d)
    elif (kernel_type == "matern_2.5_l1" and l == "l1") or (kernel_type == "matern_2.5_l2" and l == "l2"):
        return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(- np.sqrt(5) * d)
    elif (kernel_type == "matern_3.5_l1" and l == "l1") or (kernel_type == "matern_3.5_l2" and l == "l2"):
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * np.sqrt(7) / 3 / 5 * d ** 3) * np.exp(- np.sqrt(7) * d)
    elif (kernel_type == "matern_4.5_l1" and l == "l1") or (kernel_type == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * np.exp(- 3 * d)
    else:
        raise ValueError(
            'The values of l and kernel_type are not valid.'
        )

        
def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of our paper.
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array([1 / N,] * N)
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights
