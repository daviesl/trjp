import numpy as np
from scipy.stats import *
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import itertools
import scipy
from scipy.special import logsumexp
from rjlab.utils.linalgtools import *
from rjlab.distributions import *

np.set_printoptions(linewidth=200)


# Each global parameter has
#  a prior distribution
class RandomVariable(object):
    def __init__(self, prior_distribution=Distribution(norm(0, 1))):
        self.priordist = prior_distribution
        # I don't think there is anything else... the prior completely defines the RV.

    def dim(self, model_key=None):
        return self.priordist.dim

    def setModel(self, m):
        self.pmodel = m

    def getModel(self):
        return self.pmodel

    def getModelIdentifier(self):
        """
        Overridden by child classes in transdimensional cases
        Always return a list of the identifying columns. [name]
        """
        return None

    def getRange(self):
        raise "getRange() not implemented"

    def draw(self, size=1):
        """
        Draw RV from the prior.
        """
        return self.priordist.draw(size)

    def eval_log_prior(self, theta):
        """
        Requires that theta is correct dimension for this RV
        """
        # return np.log(self.priordist.eval(theta))
        return self.priordist.logeval(theta)

    def _estimatemoments(self, theta, mk=None):
        # raise "Not implemented"
        return self.priordist._estimatemoments(theta, mk)


class ImproperRV(RandomVariable):
    def __init__(self):
        super(ImproperRV, self).__init__(prior_distribution=ImproperDistribution())


class NormalRV(RandomVariable):
    def __init__(self, mu=0, sigma=1):
        super(NormalRV, self).__init__(prior_distribution=NormalDistribution(mu, sigma))


class UniformRV(RandomVariable):
    def __init__(self, l=0, u=1):
        super(UniformRV, self).__init__(prior_distribution=UniformDistribution(l, u))


class HalfNormalRV(RandomVariable):
    def __init__(self, sigma=1):
        super(HalfNormalRV, self).__init__(
            prior_distribution=HalfNormalDistribution(sigma)
        )


class LogHalfNormalRV(RandomVariable):
    def __init__(self, sigma=1):
        super(LogHalfNormalRV, self).__init__(
            prior_distribution=LogHalfNormalDistribution(sigma)
        )


class InvGammaRV(RandomVariable):
    def __init__(self, a=1, b=1):
        super(InvGammaRV, self).__init__(prior_distribution=InvGammaDistribution(a, b))


class SortedMVUniformRV(UniformRV):
    def resize_by_dim(self, size):
        # behavior is dependent on dimensions (size).
        # the uniform dist has dim = 1 typically, but this draw may be called by a transd block
        # Therefore we need to divide the num columns in size by dim and use this as size.
        if isinstance(size, tuple) and len(size) == 2:
            return (size[0], size[1] / self.dim())
        else:
            return size

    def draw(self, size):
        """
        This RV should always have a dim > 1 otherwise it will throw an error when sorting.
        """
        v = super(SortedMVUniformRV, self).draw(size=size)
        # sort the columns to mimic layer depths
        # print("v before sort {}".format(v))

        sortv = np.sort(
            v, axis=1
        )  # axis=1 is sorta redundant bc np.sort defaults to sorting last axis
        # print("v after sort {}".format(sortv))
        # sys.exit(0)
        return sortv

    def eval_log_prior(self, theta):
        # return np.log(self.priordist.eval(theta))
        l = super(SortedMVUniformRV, self).eval_log_prior(theta)
        # print("sortedmvunif eval log prior",l)
        # print("sortedmvunif theta input",theta)
        return l


# class BoundedPoissonRV(RandomVariable):
#    def __init__(self,lam,imin,imax):
#        self.lam = lam
#        self.imin = imin
#        self.imax = imax
#        super(BoundedPoissonRV, self).__init__(prior_distribution=BoundedPoissonDistribution(lam,imin,imax))
#    def getRange(self):
#        return list(range(self.imin,self.imax+1))


class UniformIntegerRV(RandomVariable):
    def __init__(self, imin, imax):
        self.imin = imin
        self.imax = imax  # inclusive
        super(UniformIntegerRV, self).__init__(
            prior_distribution=Distribution(randint(low=imin, high=imax + 1))
        )

    def getRange(self):
        return list(range(self.imin, self.imax + 1))


# class DirichletRV(RandomVariable):
#    def __init__(self):
#        # TODO implement Dirichlet dist
#        super(DirichletRV, self).__init__(prior_distribution=UniformDistribution())


class DirichletRV(RandomVariable):
    def __init__(self, alpha):
        """
        alpha must be a 1D array, number of entries determines dimension of Dirichlet Distribution
        """
        super(DirichletRV, self).__init__(
            prior_distribution=DirichletDistribution(alpha)
        )


class SymmetricDirichletRV(RandomVariable):
    def __init__(self, alpha=1.0):
        super(SymmetricDirichletRV, self).__init__(
            prior_distribution=SymmetricDirichletDistribution(alpha)
        )


class SimplexSymmetricDirichletRV(RandomVariable):
    def __init__(self, alpha=1.0):
        super(SimplexSymmetricDirichletRV, self).__init__(
            prior_distribution=SimplexSymmetricDirichletDistribution(alpha)
        )
