import numpy as np
from scipy.stats import *
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import itertools
import scipy
from scipy.special import logsumexp
from rjlab.utils.linalgtools import *

np.set_printoptions(linewidth=200)


class Distribution(object):
    # def __init__(self,distribution):
    def __init__(self, distribution, **kwargs):
        """
        Wrapper object for a scipy single variable distribution
        """
        self.dist = distribution
        if kwargs == None:
            self.kwargs = {}
        else:
            self.kwargs = kwargs
        self.dim = 1

    def draw(self, size=1):
        return self.dist.rvs(**self.kwargs, size=size)

    def eval(self, theta):
        if isinstance(self.dist.dist, rv_continuous):
            return self.dist.pdf(theta, **(self.kwargs))
        else:
            return self.dist.pmf(theta, **(self.kwargs))

    def logeval(self, theta):
        if isinstance(self.dist.dist, rv_continuous):
            return self.dist.logpdf(theta, **(self.kwargs))
        else:
            return self.dist.logpmf(theta, **(self.kwargs))

    def _estimatemoments(self, theta, mk=None):
        """
        Default, should be overridden.

        estimateMoments(theta) assumes all cols in theta are
        distributed according to the definition in this class.

        """
        mean = np.mean(theta, axis=0)
        cov = np.cov(theta.T)
        return mean, cov


class MVGaussianDistribution(Distribution):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        self.dim = mean.shape[0]

    def draw(self, size=1):
        return multivariate_normal.rvs(mean=self.mean, cov=self.cov, size=size)


class UniformDistribution(Distribution):
    def __init__(self, l=0, u=1):
        super(UniformDistribution, self).__init__(uniform(l, u - l))


class DirichletDistribution(Distribution):
    def __init__(self, alpha):
        super(DiricheltDistribution, self).__init__(dirichlet(alpha))


class NormalDistribution(Distribution):
    def __init__(self, mu=0, sigma=1):
        super(NormalDistribution, self).__init__(norm(mu, sigma))


class HalfNormalDistribution(Distribution):
    def __init__(self, sigma=1):
        super(HalfNormalDistribution, self).__init__(halfnorm(scale=sigma))


class LogHalfNormalDistribution(Distribution):
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.dim = 1
        # super(HalfNormalDistribution, self).__init__(halfnorm(scale=sigma))

    def draw(self, size=1):
        return np.log(halfnorm(scale=self.sigma).rvs(size))

    def eval(self, theta):
        return np.exp(self.logeval(theta))

    def logeval(self, theta):
        return (
            halfnorm(scale=self.sigma).logpdf(np.exp(theta)) + theta
        )  # log(d/dx exp(x)) = x, log(d/dx log(x)) = log(1/x)


class InvGammaDistribution(Distribution):
    def __init__(self, a=1, b=1):
        super(InvGammaDistribution, self).__init__(invgamma(a=a, scale=b))


class ImproperDistribution(Distribution):
    def __init__(self):
        # super(ImproperDistribution, self).__init__()
        self.dim = 1

    def draw(self, size=1):
        # hack it
        # raise RuntimeError("Cannot draw from improper distribution")
        return norm(0, 1).rvs(size)

    def eval(self, theta):
        # size=theta.shape[0]
        return np.ones(theta.shape)

    def logeval(self, theta):
        # size=theta.shape[0]
        return np.zeros(theta.shape)
