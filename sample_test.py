import numpy as np
from tests import mmdagg, mmd_median_test, mmd_split_test
from ost import ost
from sampling import f_theta_sampler


def sample_and_test_uniform(
    function_type, seed, kernel_type, approx_type, m, n, d, p, s, 
    perturbation_multiplier, alpha, l_minus, l_plus, B1, B2, B3, bandwidth_multipliers
):  
    """
    Sample from uniform and perturbed uniform density and run two-sample test.
    inputs: function_type: "uniform", "increasing", "decreasing", "centred", "ost", 
                           "median", "split" or "split (doubled sample sizes)"
            seed: integer random seed
            kernel_type: "gaussian" or "laplace": 
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            m: non-negative integer (sample size for uniform distribution)
            n: non-negative integer (sample size for perturbed uniform distribution)
            d: non-negative integer (dimension of samples)
            p: non-negative integer (number of permutations)
            s: positive number (smoothness parameter of Sobolev ball (Eq. (1))
            perturbation_multiplier: perturbation_multiplier: positive number (c_d in Eq. (17)) 
            alpha: real number in (0,1) (level of the test)
            l_minus: integer (for collection of bandwidths Eq. (16) in our paper)
            l_plus: integer (for collection of bandwidths Eq. (16) in our paper)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
            bandwidth_multipliers: array such that mmd_split_test function (used for "split" 
                                   and "split (doubled sample sizes)") selects 'optimal' bandwidth from
                                   collection_bandwidths = [c*bandwidth_median for c in bandwidth_multipliers]
    output: result of test (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    if function_type == "split (doubled sample sizes)":
        m = 2 * m
        n = 2 * n
    rs = np.random.RandomState(seed)
    if p == 0:
        X = rs.uniform(0, 1, (m, d)) 
        Y = rs.uniform(0, 1, (n, d))         
    else:
        h = 1/p
        X = f_theta_sampler(seed, seed, m, p, s, perturbation_multiplier, d)
        Y = rs.uniform(0, 1, (n, d))
    if function_type == "median":
        return mmd_median_test(
            seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multiplier=1
        )
    elif function_type in ["split", "split (doubled sample sizes)"]:
        return mmd_split_test(
            seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multipliers
        )
    elif function_type == "ost":
        return ost(seed, X, Y, alpha, kernel_type, l_minus, l_plus)
    elif function_type in ["uniform", "increasing", "decreasing", "centred"]:
        return mmdagg(
            seed, X, Y, alpha, kernel_type, approx_type, 
            function_type, l_minus, l_plus, B1, B2, B3
        )
    else:
        raise ValueError(
            'Undefined function_type: function_type should be "median", "split",' 
            '"split (doubled sample sizes)", "ost", "uniform", "increasing", '
            '"decreasing" or "centred".'
        )  


def sample_and_test_mnist(
    P, Q, function_type, seed, kernel_type, approx_type, m, n, 
    alpha, l_minus, l_plus, B1, B2, B3, bandwidth_multipliers
):  
    """
    Sample from dataset P and dataset Q and run two-sample test.
    inputs: P: dataset of shape (number_points, dimension) from which to sample
            Q: dataset of shape (number_points, dimension) from which to sample
            function_type: "uniform", "increasing", "decreasing", "centred", "ost", 
                           "median", "split" or "split (doubled sample sizes)"
            seed: integer random seed
            kernel_type: "gaussian" or "laplace":
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            m: non-negative integer (sample size for uniform distribution)
            n: non-negative integer (sample size for perturbed uniform distribution)
            alpha: real number in (0,1) (level of the test)
            l_minus: integer (for collection of bandwidths Eq. (16) in our paper)
            l_plus: integer (for collection of bandwidths Eq. (16) in our paper)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
            bandwidth_multipliers: array such that mmd_split_test function (used for "split" 
                                   and "split (doubled sample sizes)") selects 'optimal' bandwidth from
                                   collection_bandwidths = [c*bandwidth for c in bandwidth_multipliers]
    output: result of test (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    rs = np.random.RandomState(seed)
    if function_type == "split (doubled sample sizes)":
        m = 2 * m
        n = 2 * n 
    idx_X = rs.randint(len(P), size=m)
    X = P[idx_X, :]
    idx_Y = rs.randint(len(Q), size=n)
    Y = Q[idx_Y, :]
    if function_type == "median":
        return mmd_median_test(
            seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multiplier=1
        )
    elif function_type in ["split", "split (doubled sample sizes)"]:
        return mmd_split_test(
            seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multipliers
        )
    elif function_type == "ost":
        return ost(seed, X, Y, alpha, kernel_type, l_minus, l_plus)
    elif function_type in ["uniform", "increasing", "decreasing", "centred"]:
        return mmdagg(
            seed, X, Y, alpha, kernel_type, approx_type, 
            function_type, l_minus, l_plus, B1, B2, B3
        )
    else:
        raise ValueError(
            'Undefined function_type: function_type should be "median", "split",' 
            '"split (doubled sample sizes)", "ost", "uniform", "increasing", '
            '"decreasing" or "centred".'
        )  
