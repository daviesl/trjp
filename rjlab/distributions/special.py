import numpy as np
from scipy.stats import *
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import itertools
import scipy
from scipy.special import logsumexp
from rjlab.utils.linalgtools import *
from rjlab.distributions import Distribution
import dirichlet as dlt

np.set_printoptions(linewidth=200)


class SimplexTransform(object):
    @classmethod
    def forward(self, value):
        log_value = np.log(value)
        shift = np.sum(log_value, -1, keepdims=True) / value.shape[-1]
        return log_value[..., :-1] - shift

    @classmethod
    def backward(self, value):
        value = np.concatenate([value, -np.sum(value, -1, keepdims=True)], axis=-1)
        exp_value_max = np.exp(value - np.max(value, -1, keepdims=True))
        return exp_value_max / np.sum(exp_value_max, -1, keepdims=True)

    @classmethod
    def log_jac_det(self, value):
        N = value.shape[-1] + 1
        sum_value = np.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = np.concatenate(
            [value_sum_expanded, np.zeros(sum_value.shape)], -1
        )
        logsumexp_value_expanded = logsumexp(value_sum_expanded, -1, keepdims=True)
        res = np.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return np.sum(res, -1)


class SimplexSymmetricDirichletDistribution(Distribution):
    def __init__(self, alpha=alpha):
        """
        Allows the dimension to be set dynamically
        Dimension is always N-1 because we don't store the last entry.
        """
        self.alpha = alpha
        self.dim = 1
        # self.simplex = SimplexTransform()

    @classmethod
    def transformDirichletToSimplex(cls, X):
        fwd = SimplexTransform.forward(X)
        return fwd, -SimplexTransform.log_jac_det(fwd)

    @classmethod
    def transformSimplexToDirichlet(cls, X):
        ld = SimplexTransform.log_jac_det(X)
        return SimplexTransform.backward(X), ld

    def draw(self, size=1):
        if isinstance(size, tuple):
            if len(size) == 2:
                n = size[0]
                dim = size[1] + 1
            else:
                n = size[0]
                dim = 2
        else:
            dim = 2
            n = size
        samples = dirichlet(alpha=np.full(dim, self.alpha)).rvs(n)
        # transform to simplex
        lsamp, ld = self.transformDirichletToSimplex(samples)
        return lsamp

    def eval(self, X):
        return np.exp(self.logeval(X))

    def logeval(self, X):
        if X.ndim == 2:
            n = X.shape[0]
            dim = X.shape[1] + 1
        else:
            n = X.shape[0]
            dim = 2
        simplexX, ld = self.transformSimplexToDirichlet(X)
        dirichlet_logeval = dirichlet(alpha=np.full(dim, self.alpha)).logpdf(simplexX.T)
        return dirichlet_logeval + ld


class SymmetricDirichletDistribution(Distribution):
    def __init__(self, alpha=alpha):
        """
        Allows the dimension to be set dynamically
        """
        self.alpha = alpha
        self.dim = 1
        # self.dim = alpha.shape[0]

    def draw(self, size=1):
        if isinstance(size, tuple):
            if len(size) == 2:
                n = size[0]
                dim = size[1]
            else:
                n = size[0]
                dim = 1
        else:
            dim = 1
            n = size
        return dirichlet(alpha=np.full(dim, self.alpha)).rvs(n)

    def eval(self, X):
        return np.exp(self.logeval(X))

    def logeval(self, X):
        if X.ndim == 2:
            n = X.shape[0]
            dim = X.shape[1]
        else:
            n = X.shape[0]
            dim = 1
        return dirichlet(alpha=np.full(dim, self.alpha)).logpdf(X.T)

    @classmethod  # probably doesn't need to be a classmethod
    def _estimatemoments(self, theta, mk=None):
        """
        Special moment estimation for dirichlet using mle
        """
        if theta.shape[1] == 1:
            # degenerate case where value is always 1 and var=0. Use var=1 to avoid volume issues.
            # in WSSMC weight calculation, delta mean will always be zero in exp term
            return 1, np.array([[1]])
        # print("estimating dirichlet moments for \n",theta)
        a = dlt.mle(theta)
        print("Estimated dirichlet parameters", a)
        m, s = dlt.meanprecision(a)
        # cov = (k_ij * m_i - m_i * m_j) / (s + 1)
        cov = (np.diag(m) - np.outer(m, m)) / (s + 1)
        return m, cov


class BoundedPoissonDistribution(Distribution):
    def __init__(self, lam, kmin, kmax):
        self.kmin = kmin
        self.kmax = kmax
        super(BoundedPoissonDistribution, self).__init__(poisson(lam))

    def draw(self, size=1):
        # rejection sample
        samples = np.zeros(size)
        draw_idx = np.full(size, True)
        while draw_idx.sum() > 0:
            samples[draw_idx] = self.dist.rvs(size=draw_idx.sum())
            draw_idx = np.logical_or(samples > self.kmax, samples < self.kmin)
        return samples
