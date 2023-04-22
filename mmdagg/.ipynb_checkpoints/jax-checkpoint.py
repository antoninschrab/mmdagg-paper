import sys # why do we need?

# check if we add bandwidths parameter

import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from jax.flatten_util import ravel_pytree
from functools import partial


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 9, 10, 11))
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
    bandwidths=None,
):
    """
    Two-Sample MMDAgg test.
    
    Given data from one distribution and data from another distribution,
    return 0 if the test fails to reject the null 
    (i.e. data comes from the same distribution), 
    or return 1 if the test rejects the null 
    (i.e. data comes from different distributions).
    
    Fixing the two sample sizes and the dimension, the first time the function is
    run it is getting compiled. After that, the function can fastly be evaluated on 
    any data with same sample sizes and dimension.
    
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
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>> from mmdagg.jax import mmdagg, human_readable_dict

    # generate data for two-sample test
    >>> key = random.PRNGKey(0)
    >>> key, subkey = random.split(key)
    >>> subkeys = random.split(subkey, num=2)
    >>> X = random.uniform(subkeys[0], shape=(500, 10))
    >>> Y = random.uniform(subkeys[1], shape=(500, 10)) + 1

    # run MMDAgg test
    >>> output = mmdagg(X, Y)
    >>> output
    Array(1, dtype=int32)
    >>> output.item()
    1
    >>> output, dictionary = mmdagg(X, Y, return_dictionary=True)
    >>> output
    Array(1, dtype=int32)
    >>> human_readable_dict(dictionary)
    >>> dictionary
    {'MMDAgg test reject': True,
     'Single test 1': {'Bandwidth': 1.0,
      'MMD': 5.788900671177544e-05,
      'MMD quantile': 0.0009193826699629426,
      'Kernel IMQ': True,
      'Reject': False,
      'p-value': 0.41079461574554443,
      'p-value threshold': 0.01699146442115307},
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
    if type(bandwidths) == jnp.ndarray:
        assert bandwidths.ndim == 1
        number_bandwidths = len(bandwidths)
        bandwidths_l1 = bandwidths
        bandwidths_l2 = bandwidths_l1
    else:
        def compute_bandwidths(distances, number_bandwidths):
            # lambda_min / 2 * C^r for r = 0, ..., number_bandwidths -1
            # where C is such that lambda_max * 2 = lambda_min / 2 * C^(number_bandwidths - 1)
            distances = distances + (distances == 0) * jnp.median(distances)
            dd = jnp.sort(distances_l1)
            lambda_min = jax.lax.cond(
                jnp.min(distances) < 10 ** (-1), 
                lambda : jnp.maximum(dd[(jnp.floor(len(dd) * 0.05).astype(int))], 10 ** (-1)), 
                lambda : jnp.min(distances),
            )
            lambda_min = lambda_min / 2
            lambda_max = jnp.maximum(jnp.max(distances), 3 * 10 ** (-1))
            lambda_max = lambda_max * 2
            power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
            bandwidths = jnp.array([power ** i * lambda_min for i in range(number_bandwidths)])
            return bandwidths
        # bandwidths L1 for laplace, matern_0.5_l1, matern_1.5_l1, matern_2.5_l1, matern_3.5_l1, matern_4.5_l1
        if kernel in (
            "laplace", 
            "matern_0.5_l1", 
            "matern_1.5_l1", 
            "matern_2.5_l1", 
            "matern_3.5_l1", 
            "matern_4.5_l1", 
            "all_matern_l1", 
            "all_matern_l1_l2", 
            "all", 
            "laplace_gaussian", 
            "gaussian_laplace", 
        ):
            distances_l1 = jax_distances(X, Y, "l1", max_samples=500)
            bandwidths_l1 = compute_bandwidths(distances_l1, number_bandwidths)
        # bandwidths L2 for gaussian, imq, matern_0.5_l2, matern_1.5_l2, matern_2.5_l2, matern_3.5_l2, matern_4.5_l2
        if kernel in (
            "gaussian", 
            "imq", 
            "matern_0.5_l2", 
            "matern_1.5_l2", 
            "matern_2.5_l2", 
            "matern_3.5_l2", 
            "matern_4.5_l2", 
            "all_matern_l2", 
            "all_matern_l1_l2", 
            "all", 
            "laplace_gaussian", 
            "gaussian_laplace", 
        ):
            distances_l2 = jax_distances(X, Y, "l2", max_samples=500)
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
    if approx_type == "wild bootstrap":
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        R = random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(B1 + B2 + 1, n))  # (B1+B2+1, n) Rademacher
        R = R.at[B1].set(jnp.ones(n))
        R = R.transpose()
        R = jnp.concatenate((R, -R))  # (2n, B1+B2+1) 
    elif approx_type == "permutations":
        key = random.PRNGKey(seed)
        key, subkey = random.split(key)
        # (B1+B2+1, m+n): rows of permuted indices
        idx = random.permutation(subkey, jnp.array([[i for i in range(m + n)]] * (B1 + B2 + 1)), axis=1, independent=True)   
        #11
        v11 = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))  # (m+n, )
        V11i = jnp.tile(v11, (B1 + B2 + 1, 1))  # (B1+B2+1, m+n)
        V11 = jnp.take_along_axis(V11i, idx, axis=1)  # (B1+B2+1, m+n): permute the entries of the rows
        V11 = V11.at[B1].set(v11) # (B1+1)th entry is the original MMD (no permutation)
        V11 = V11.transpose()  # (m+n, B1+B2+1)
        #10
        v10 = jnp.concatenate((jnp.ones(m), jnp.zeros(n)))
        V10i = jnp.tile(v10, (B1 + B2 + 1, 1))
        V10 = jnp.take_along_axis(V10i, idx, axis=1)
        V10 = V10.at[B1].set(v10)
        V10 = V10.transpose() 
        #01
        v01 = jnp.concatenate((jnp.zeros(m), -jnp.ones(n)))
        V01i = jnp.tile(v01, (B1 + B2 + 1, 1))
        V01 = jnp.take_along_axis(V01i, idx, axis=1)
        V01 = V01.at[B1].set(v01)
        V01 = V01.transpose() 
    else:
        raise ValueError("Approximation type not defined.")
        
    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    N = number_bandwidths * len(kernel_bandwidths_l_list)
    M = jnp.zeros((N, B1 + B2 + 1))
    last_l_pairwise_matrix_computed = ""
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, l = kernel_bandwidths_l_list[j]
        # since kernel_bandwidths_l_list is ordered "l1" first, "l2" second
        # compute pairwise matrices the minimum amount of time
        # store only one pairwise matrix at once
        if l != last_l_pairwise_matrix_computed:
            Z = jnp.concatenate((X, Y))
            pairwise_matrix = jax_distances(Z, Z, l, matrix=True)
            last_l_pairwise_matrix_computed = l
        for i in range(number_bandwidths):
            bandwidth = bandwidths[i]
            K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
            if approx_type == "wild bootstrap": 
                # set diagonal elements of all four submatrices to zero
                paired_indices = jnp.diag_indices(n)
                K = K.at[paired_indices].set(0)
                K = K.at[paired_indices[0] + n, paired_indices[1]].set(0)
                K = K.at[paired_indices[0], paired_indices[1] + n].set(0)
                K = K.at[paired_indices[0] + n, paired_indices[1] + n].set(0)
                # compute MMD bootstrapped values
                M = M.at[number_bandwidths * j + i].set(jnp.sum(R * (K @ R), 0) / (n * (n - 1)))
            elif approx_type == "permutations": 
                # set diagonal elements to zero
                K = K.at[jnp.diag_indices(K.shape[0])].set(0)
                # compute MMD permuted values
                M = M.at[number_bandwidths * j + i].set(
                    jnp.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                    + jnp.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                    + jnp.sum(V11 * (K @ V11), 0) / (m * n)
                )  
            else:
                raise ValueError("Approximation type not defined.")           
    MMD_original = M[:, B1]
    M1_sorted = jnp.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)

    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = jnp.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0.
    u_max = jnp.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for j in range(len(kernel_bandwidths_l_list)):
            for i in range(number_bandwidths):
                quantiles = quantiles.at[number_bandwidths * j + i].set(
                    M1_sorted[
                        number_bandwidths * j + i, 
                        (jnp.ceil((B1 + 1) * (1 - u * weights[i]))).astype(int) - 1
                    ]
                )
        P_u = jnp.sum(jnp.max(M2 - quantiles, 0) > 0) / B2
        u_min, u_max = jax.lax.cond(P_u <= alpha, lambda: (u, u_max), lambda: (u_min, u))
    u = u_min
    for j in range(len(kernel_bandwidths_l_list)):
        for i in range(number_bandwidths):
            quantiles = quantiles.at[number_bandwidths * j + i].set(
                M1_sorted[
                    number_bandwidths * j + i, 
                    (jnp.ceil((B1 + 1) * (1 - u * weights[i]))).astype(int) - 1
                ]
            )
        
    # Step 3: output test result
    p_vals = jnp.mean((M1_sorted - MMD_original.reshape(-1, 1) >= 0), -1)
    all_weights = jnp.zeros(p_vals.shape)
    for j in range(len(kernel_bandwidths_l_list)):
         for i in range(number_bandwidths):
            all_weights = all_weights.at[number_bandwidths * j + i].set(weights[i])
    thresholds = u * all_weights
    # reject if p_val <= threshold
    reject_p_vals = p_vals <= thresholds

    mmd_vals = MMD_original
    quantiles = quantiles.reshape(-1)
    # reject if mmd_val > quantile
    reject_mmd_vals = mmd_vals > quantiles
    
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
            reject_dictionary["MMDAgg test reject"] = jnp.any(ravel_pytree(
                (reject_dictionary["MMDAgg test reject"],
                reject_p_vals[idx])
            )[0])

    if return_dictionary:
        return (reject_dictionary["MMDAgg test reject"]).astype(int), reject_dictionary
    else:
        return (reject_dictionary["MMDAgg test reject"]).astype(int)

        
def kernel_matrix(pairwise_matrix, l, kernel, bandwidth):
    """
    Compute kernel matrix for a given kernel and bandwidth. 

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel == "gaussian" and l == "l2":
        return  jnp.exp(-d ** 2)
    elif kernel == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif (kernel == "matern_0.5_l1" and l == "l1") or (kernel == "matern_0.5_l2" and l == "l2") or (kernel == "laplace" and l == "l1"):
        return  jnp.exp(-d)
    elif (kernel == "matern_1.5_l1" and l == "l1") or (kernel == "matern_1.5_l2" and l == "l2"):
        return (1 + jnp.sqrt(3) * d) * jnp.exp(- jnp.sqrt(3) * d)
    elif (kernel == "matern_2.5_l1" and l == "l1") or (kernel == "matern_2.5_l2" and l == "l2"):
        return (1 + jnp.sqrt(5) * d + 5 / 3 * d ** 2) * jnp.exp(- jnp.sqrt(5) * d)
    elif (kernel == "matern_3.5_l1" and l == "l1") or (kernel == "matern_3.5_l2" and l == "l2"):
        return (1 + jnp.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * jnp.sqrt(7) / 3 / 5 * d ** 3) * jnp.exp(- jnp.sqrt(7) * d)
    elif (kernel == "matern_4.5_l1" and l == "l1") or (kernel == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * jnp.exp(- 3 * d)
    else:
        raise ValueError(
            'The values of "l" and "kernel" are not valid.'
        )

        
def create_weights(N, weights_type):
    """
    Create weights as defined in Section 5.1 of our paper.
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = jnp.array([1 / N,] * N)
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = jnp.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = jnp.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = jnp.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = jnp.array(
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


def jax_distances(X, Y, l, max_samples=None, matrix=False):
    if l == "l1":
        def dist(x, y):
            z = x - y
            return jnp.sum(jnp.abs(z))
    elif l == "l2":
        def dist(x, y):
            z = x - y
            return jnp.sqrt(jnp.sum(jnp.square(z)))
    else:
        raise ValueError("Value of 'l' must be either 'l1' or 'l2'.")
    vmapped_dist = vmap(dist, in_axes=(0, None))
    pairwise_dist = vmap(vmapped_dist, in_axes=(None, 0))
    output = pairwise_dist(X[:max_samples], Y[:max_samples])
    if matrix:
        return output
    else:
        return output[jnp.triu_indices(output.shape[0])]

    
def human_readable_dict(dictionary):
    """
    Transform all jax arrays of one element into scalars.
    """
    meta_keys = dictionary.keys()
    for meta_key in meta_keys:
        if isinstance(dictionary[meta_key], jnp.ndarray):
            dictionary[meta_key] = dictionary[meta_key].item()
        elif isinstance(dictionary[meta_key], dict):
            for key in dictionary[meta_key].keys():
                if isinstance(dictionary[meta_key][key], jnp.ndarray):
                    dictionary[meta_key][key] = dictionary[meta_key][key].item()