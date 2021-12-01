"""
This code was extracted from https://github.com/MPI-IS/tests-wo-splitting under The MIT License.
Jonas M. Kübler, Wittawat Jitkrittum, Bernhard Schölkopf, Krikamol Muandet
Learning Kernel Tests Without Data Splitting
Neural Information Processing Systems 2020
https://papers.nips.cc/paper/2020/file/44f683a84163b3523afe57c2e008bc8c-Paper.pdf

We use their test as a comparison in our experiments.
We modified the PTKGauss class to use the exact same kernel
we use for our test. We have also added a PTKLaplace class 
and have defined the ost function which we use in our experiments.
"""

import torch
from scipy.stats import norm
from scipy.stats import chi as chi_stats
from cvxopt import matrix, solvers
from median import compute_median_bandwidth_subset
import numpy as np


class PTKGauss:
    """
    Pytorch implementation of the isotropic Gaussian kernel.
    Parameterization is the same as in the density of the standard normal
    distribution. sigma2 is analogous to the variance.
    
    Modifications: keep only case X.dim() = 2
                   change parameter to bandwidth = sqrt(2*sigma2)
    """

    def __init__(self, bandwidth):
        """
        bandwidth: a number 
        """
        bandwidth = torch.tensor(bandwidth)
        assert (bandwidth > 0).any(), 'bandwidth must be > 0. Was %s' % str(bandwidth)
        self.bandwidth = bandwidth

    def eval_lin(self, X, Y):
        """
        Evaluate only the relevant entries for the linear time mmd
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor
        Return
        ------
        K : a n/2 list of entries.
        """
        bandwidth = torch.sqrt(self.bandwidth ** 2)
        assert X.dim() == 2
        assert X.size() == Y.size()
        assert bandwidth.size()[0] == X.size()[1]
        D2 = torch.sum(((X - Y).div(bandwidth)) ** 2, dim=1).view(1, -1)
        K = torch.exp(-D2)
        # We have rewritten this to be similar to PTKLaplace.
        # This way of computing D2 is equivalent to the following:
        # sumx2 = torch.sum(X ** 2, dim=1).view(1, -1)
        # sumy2 = torch.sum(Y ** 2, dim=1).view(1, -1)
        # D2 = sumx2 - 2 * torch.sum(X * Y, dim=1).view(1, -1) + sumy2
        # K = torch.exp(-D2.div(bandwidth**2))
        return K
    

# added
class PTKLaplace:
    """
    Pytorch implementation of the isotropic Laplace kernel.
    """

    def __init__(self, bandwidth):
        """
        bandwidth: a number 
        """
        bandwidth = torch.tensor(bandwidth)
        assert (bandwidth > 0).any(), 'bandwidth must be > 0. Was %s' % str(bandwidth)
        self.bandwidth = bandwidth

    def eval_lin(self, X, Y):
        """
        Evaluate only the relevant entries for the linear time mmd
        ----------
        X : n1 x d Torch Tensor
        Y : n2 x d Torch Tensor
        Return
        ------
        K : a n/2 list of entries.
        """
        bandwidth = torch.sqrt(self.bandwidth ** 2)
        assert X.dim() == 2
        assert X.size() == Y.size()
        assert bandwidth.size()[0] == X.size()[1]
        D1 = torch.sum(torch.abs((X - Y).div(bandwidth)), dim=1).view(1, -1)
        K = torch.exp(-D1)
        return K


class LinearMMD:
    """
    To compute linear time MMD estimates and the covariance matrix of the asymptotic distribution of the linear time
    MMD for d different kernels.
    """

    def __init__(self, kernels):
        """
        :param kernels: list of kernels, which will be considered
        :returns
        mmd: linear time mmd estimates for all the kernels. Scaled with sqrt(n)
        Sigma: covariance matrix of the asymptotic normal distribution of linear mmd estimates
        """
        self.kernels = kernels

        # number of kernels considered
        self.d = len(kernels)

    def estimate(self, x_sample, y_sample):
        """
        Computes the linear time estimates of the MMD, for all kernels that should be considered. Further
        it computes the asymptotic covariance matrix of the linear time MMD for the kernels.
        The samplesize is taken into account on the side of the MMD, i.e., we estimate sqrt(n) MMD^2
        :param x_sample: data from P
        :param y_sample: data from Q
        :return:
        """
        if not isinstance(x_sample, torch.Tensor):
            # convert data to torch tensors
            x_sample = torch.tensor(x_sample)
            y_sample = torch.tensor(y_sample)
        assert list(x_sample.size())[0] == list(y_sample.size())[0], 'datasets must have same samplesize'

        # determine length of the sample
        size = list(x_sample.size())[0]
        # for linear time mmd assume that the number of samples is 2n. Truncate last data point if uneven
        size = size - size % 2
        n = int(size / 2)
        # define the
        x1, x2 = x_sample[:n], x_sample[n:size]
        y1, y2 = y_sample[:n], y_sample[n:size]

        # tensor of all functions h defined for the kernels
        h = torch.zeros(self.d, n)

        # compute values of h on the data
        for u in range(self.d):
            gram_xx = self.kernels[u].eval_lin(X=x1, Y=x2).squeeze() # added .squeeze()
            gram_xy = self.kernels[u].eval_lin(X=x1, Y=y2).squeeze() # added .squeeze()
            gram_yx = self.kernels[u].eval_lin(X=y1, Y=x2).squeeze() # added .squeeze()
            gram_yy = self.kernels[u].eval_lin(X=y1, Y=y2).squeeze() # added .squeeze()

            h[u] = gram_xx - gram_xy - gram_yx + gram_yy

        mmd = torch.sum(h, dim=1) / n
        Sigma = 1 / n * h.matmul(h.transpose(0,1)) - mmd.view(-1,1).matmul(mmd.view(1,-1))

        # We consider sqrt(n) * mmd. Therefore we will keep Sigma on a scale independent of n
        mmd = np.sqrt(n) * mmd

        return np.array(mmd), np.array(Sigma)


def truncation(beta_star, tau, Sigma, accuracy=1e-6):
    """
    Compute
    :param beta_star: optimal projection of tau
    :param tau: vector of test statistics
    :param Sigma: Covariance matrix of test statistics
    :param accuracy: threshold to determine whether an entry is zero
    :return: Lower threshold of conditional distribution of beta_star^T tau
    """
    # dimension of data
    d = len(tau)
    # determine non-zero entries of betastar
    non_zero = [1 if beta_i > accuracy else 0 for beta_i in beta_star]
    # define the arguments of the maximization of V^-
    arguments = [(tau[i] * (beta_star @ Sigma @ beta_star) - (np.eye(1, d, i) @ Sigma @ beta_star) * (beta_star @ tau))
                 / (np.sqrt(Sigma[i][i]) * np.sqrt(beta_star @ Sigma @ beta_star) - np.eye(1, d, i) @ Sigma @ beta_star)
                 for i in range(len(tau)) if non_zero[i] == 0]
    # catch cases for which we have 0/0 and hence nan. We dont consider these
    arguments = np.array([argument if argument > -10e6 else -10e6 for argument in arguments])
    if len(arguments) == 0:
        return -10e6
    v_minus = np.max(arguments)
    return v_minus


def truncated_gaussian(var, v_minus, level):
    """
    Computes the (1-level) threshold of  a truncated normal (the original normal is assumed to be centered)
    :param var: variance of the original normal
    :param v_minus: lower truncation
    :param level: desired level
    :return:
    """
    # normalize everything
    lower = v_minus / np.sqrt(var)
    # compute normalization of the truncated section
    renormalize = 1 - norm.cdf(lower)
    if renormalize == 0:
        # force a reject
        return np.sqrt(var) * 10000
    assert renormalize > 0, "renormalize is not positive"

    threshold = np.sqrt(var) * norm.ppf(renormalize * (1 - level) + norm.cdf(lower))
    return threshold


def optimization(tau, Sigma, selection='continuous'):
    """
    optimizes the signal to noise ratio. If tau has at least one positive entry, we fix the nominator to some constant
    by setting beta^T tau = 1 and then optimize the denominator.
    If tau has only negative entries, the signal to noise ratio is given by the optimum of the discrete optimization
    :param tau: Signal
    :param Sigma: noise
    :param selection: discrete (select from base tests) / continuous (OST in canoncical form)
    :return: optimal vector beta_star
    """

    if np.max(tau) < 1e-6:
        # If all entries are negative, then for the continuous case we also select the best of the base tests
        selection = 'discrete'

    # determine dimensionality
    d = len(tau)
    if selection == 'continuous':
        tau = np.ndarray.tolist(tau)
        Sigma = np.ndarray.tolist(Sigma)

        # define quadratic program in cvxopt
        P = matrix(Sigma)
        q = matrix(np.zeros(d))
        G = matrix(np.diag([-1.] * d))
        h = matrix(np.zeros(d))
        A = matrix(np.array([tau]))
        b = matrix([1.])

        initialization = matrix([1.] * d)
        solvers.options['reltol'] = 1e-40
        solvers.options['abstol'] = 1e-10
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 10000
        sol = solvers.qp(P, q, G, h, A, b, initvals=initialization)

        beta_star = np.array(sol['x']).flatten()
        # normalize betastar
        beta_star = beta_star / np.linalg.norm(beta_star, ord=1)
        return beta_star
    else:
        arguments = tau / np.sqrt(np.diag(Sigma))
        # in case of division by zero, we do not consider it since it implies also that the nominator is zero
        arguments = np.array([argument if argument > -10e6 else -10e6 for argument in arguments])
        j = int(np.argmax(arguments))
        beta_star = [0] * d
        beta_star[j] = 1
        return np.array(beta_star)
    

def ost_test(tau, Sigma, alpha=0.05, selection='discrete', max_condition=1e-6, accuracy=1e-6, constraints='Sigma',
             pval=False):
    """
    Runs the full test suggested in our paper.
    :param tau: observed statistic
    :param Sigma: covariance matrix
    :param alpha: level of test
    :param selection: continuous/discrete (discrete is not extensively tested)
    :param max_condition: at which condition number the covariance matrix is truncated.
    :param accuracy: threshold to determine whether an entry is zero
    :param constraints: if 'Sigma'  we work with the constraints (Sigma beta) >=0. If 'positive' we work with beta >= 0
    :param pval: if true, returns the conditional p value instead of the test result
    :return: 1 (reject), 0 (no reject)
    """
    assert constraints == 'Sigma' or constraints == 'positive', 'Constraints are not implemented'
    # if the selection is discrete we dont want any transformations
    if selection == 'discrete':
        constraints = 'positive'

    # check if there are entries with 0 variance
    zeros = [i for i in range(len(tau)) if Sigma[i][i] < 1e-15] # changed from 1e-10
    tau = np.delete(tau, zeros)
    Sigma = np.delete(Sigma, zeros, 0)
    Sigma = np.delete(Sigma, zeros, 1)

    if constraints == 'Sigma':
        # compute pseudoinverse to also handle singular covariances (see Appendix)
        r_cond = max_condition  # parameter which precision to use
        Sigma_inv = np.linalg.pinv(Sigma, rcond=r_cond, hermitian=True)

        # use Remark 1 to convert the problem
        tau = Sigma_inv @ tau
        Sigma = Sigma_inv

    # Apply Theorem 1 in the canonical form with beta>=0 constraints
    beta_star = optimization(tau=tau, Sigma=Sigma, selection=selection)

    # determine active set
    non_zero = [1 if beta_i > accuracy else 0 for beta_i in beta_star]

    projector = np.diag(non_zero)
    effective_sigma = projector @ Sigma @ projector

    # Use the rank of effective Sigma to determine how many degrees of freedom the covariance has after conditioning
    # for non-singular original covariance, this is the same number as the number of active dimensions |mathcal{U}|,
    # however, for singular cases using the rank is the right way to go.
    tol = max_condition * np.max(np.linalg.eigvalsh(Sigma))
    r = np.linalg.matrix_rank(effective_sigma, tol=tol, hermitian=True)
    # go back to notation used in the paper
    l = r
    if l > 1:
        test_statistic = beta_star @ tau / np.sqrt(beta_star @ Sigma @ beta_star)
        threshold = chi_stats.ppf(q=1 - alpha, df=l)
    else:
        vminus = truncation(beta_star=beta_star, tau=tau, Sigma=Sigma, accuracy=accuracy)
        threshold = truncated_gaussian(var=beta_star @ Sigma @ beta_star, v_minus=vminus, level=alpha)
        test_statistic = beta_star @ tau
    if not pval:
        if test_statistic > threshold:
            # reject
            return 1
        else:
            # cannot reject
            return 0
    if pval:
        if l > 1:
            test_statistic = beta_star @ tau / np.sqrt(beta_star @ Sigma @ beta_star)
            pvalue = 1 - chi_stats.cdf(x=test_statistic, df=l)
        else:
            test_statistic = beta_star @ tau / np.sqrt(beta_star @ Sigma @ beta_star)
            vminus = truncation(beta_star=beta_star, tau=tau, Sigma=Sigma, accuracy=accuracy) / \
                     np.sqrt(beta_star @ Sigma @ beta_star)
            pvalue = 1 - (norm.cdf(x=test_statistic) - norm.cdf(x=vminus)) / (1 - norm.cdf(x=vminus))
        return pvalue


# Run ost_test in our setting
def ost(seed, X, Y, alpha, kernel_type, l_minus, l_plus):
    assert X.shape == Y.shape
    assert kernel_type in ["gaussian", "laplace"]
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
    bandwidths = [median_bandwidth * (2 ** factor) for factor in range(l_minus, l_plus+1)]
    if kernel_type == "gaussian":
        kernels = [PTKGauss(bandwidths[u]) for u in range(len(bandwidths))]
    if kernel_type == "laplace":
        kernels = [PTKLaplace(bandwidths[u]) for u in range(len(bandwidths))]
    mmd = LinearMMD(kernels)
    tau, Sigma = mmd.estimate(X, Y)
    return ost_test(
        tau=tau, 
        Sigma=Sigma, 
        alpha=alpha, 
        selection="continuous",
        max_condition=1e-5,  # changed from 1e-6
        constraints='Sigma',
    )
