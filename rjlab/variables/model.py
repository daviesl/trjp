import numpy as np
from scipy.stats import *
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import itertools
import scipy
from scipy.special import logsumexp
from rjlab.utils.linalgtools import *
from rjlab.distributions import *
from rjlab.variables.base import *
from rjlab.variables.block import *
from rjlab.proposals.base import *

np.set_printoptions(linewidth=200)


class ParametricModelSpace(RandomVariableBlock):
    """
    # Example for generating model
    mymodel = ParametricModelSpace(
        random_variables={
            'ConductiveLayers':TransDimensionalBlock({
                'Conductivity':UniformRV(),
                'Thickness':DirichletRV()}
            ),'ChargeableLayers':TransDimensionalBlock({
                'Chargeability':UniformRV(),
                'FrequencyDependence':UniformRV(),
                'TimeConstant':UniformRV(),
                'Thickness':DirichletRV()
            }),
            'Geometry1':UniformRV(),
            'Geometry2':UniformRV()},
        proposal=[
            EigDecComponentwiseNormalProposal(['ConductiveLayers','ChargeableLayers']),
            BirthDeathProposal(['ConductiveLayers',IndependentProposal('Conductivity',BetaDistribution(alpha,beta)])
        ])
    prop_theta = mymodel.propose(theta)
    """

    def __init__(self, random_variables, proposal, rv_transforms={}):
        super(ParametricModelSpace, self).__init__(random_variables)
        assert isinstance(proposal, Proposal)
        self.proposal = proposal
        self.proposal.setModel(self)
        self.em_method = "full"  # 'full' or 'block' for estimation of moments
        self.rv_transforms = rv_transforms  # by default, and empty dict. 
        # set the model for each random variable so that they can call self.pmodel.getModelIdentifier()
        for rvn in self.rv_names:
            self.rv[rvn].setModel(self)

    def sanitise(self, theta):
        return theta

    def sampleFromPrior(self, N):
        # sample from prior
        theta = np.zeros((N, self.dim()))
        cur_dim = 0
        for k in self.rv_names:
            thisdim = self.rv[k].dim()
            theta[:, cur_dim : cur_dim + thisdim] = (
                self.rv[k].draw(N).reshape(N, thisdim)
            )
            cur_dim += thisdim
        return theta

    def assertDimension(self, theta):
        assert (
            theta.shape[1] == self.dim()
        ), "Theta n cols {} is not equal to dimension of ParametricModelSpace {}".format(
            theta.shape[1], self.dim()
        )

    def dim(self, model_key=None):
        if model_key is None:
            return super(ParametricModelSpace, self).dim()
        else:
            ldim = 0
            for k in self.rv_names:
                ldim += self.rv[k].dim(model_key)
            return int(ldim)

    @staticmethod
    def plotJoints(theta, prop_theta, propname):
        nlayers, nlidx = np.unique(theta[:, 0].astype(int), return_inverse=True)
        for k in nlayers:
            ncols = int(k * 2 + 1)
            nrows = ncols
            thetaidx = theta[:, 0] == k
            propidx = prop_theta[:, 0] == k
            fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
            fig.suptitle(propname)
            colidx = (
                list(range(1, k + 1))
                + list(range(self.max_layers + 1, self.max_layers + 1 + k))
                + [self.max_layers * 2 + 1]
            )
            paramtheta = theta[thetaidx, :][:, colidx]
            paramprop = prop_theta[propidx, :][:, colidx]
            for i in range(2 * k + 1):
                for j in range(2 * k + 1):
                    if k == 0:
                        thisaxs = axs
                    else:
                        thisaxs = axs[i, j]
                    if i == j:
                        n, bins, patches = thisaxs.hist(
                            paramtheta[:, i], color="blue", density=True, bins=20
                        )
                        thisaxs.hist(
                            paramprop[:, i],
                            density=True,
                            color="red",
                            rwidth=0.5,
                            bins=20,
                        )
                        if k > 0:
                            for p in range(i, 2 * k + 1):
                                thisaxs.get_shared_y_axes().remove(axs[i, p])
                    elif i > j:
                        thisaxs.scatter(
                            paramtheta[:, j], paramtheta[:, i], color="blue", s=0.2
                        )
                        thisaxs.scatter(
                            paramprop[:, j], paramprop[:, i], color="red", s=0.2
                        )
                        thisaxs.get_shared_y_axes().remove(axs[j, j])
                    else:
                        fig.delaxes(thisaxs)
                    if i < 2 * k and k != 0:
                        thisaxs.xaxis.set_ticks_position("none")
                    if j > 0:
                        thisaxs.yaxis.set_ticks_position("none")

            plt.show()

    def getModelKeysFromSpec(self):
        ids = self.getModelIdentifier()
        # for each rv name, get the range
        if ids is not None:
            rvrangelist = []
            for i in ids:
                if i is not None:
                    rv = self.retrieveRV(i)
                    if rv is not None:
                        rvrangelist.append(rv.getRange())
                    else:
                        raise "Key error, rv name {} not found in model".format(i)
            # get all permutations
            keylist = []
            for p in itertools.product(*rvrangelist):
                keylist.append(tuple(p))
            return keylist
        else:
            return ()  # empty identifier

    def getModelKeys(self, theta=None):
        if theta is None:
            return self.getModelKeysFromSpec()
        ids = self.getModelKeyColumns()
        tags = theta[:, ids]
        unique_rows, tuple_indices = np.unique(tags, return_inverse=True, axis=0)
        # convert each unique row to immutable tuple for use as a dict key.
        # return list of these tuples and a numpy array of indices to this list
        return list(map(tuple, unique_rows))

    def getModelKeyColumns(self):
        ids = self.getModelIdentifier()
        rv_index_dict = (
            self.generateRVIndices()
        )  # DO NOT PASS MODEL KEY: generateRVIndices() calls this getModelKeyColumns() method to obtain keys.
        indices = []
        for k in ids:
            if k is not None:
                indices += rv_index_dict[k]
        return indices

    def enumerateModels(self, theta):
        """
        Traverses the rv structure and returns identifying columns from theta.
        If a rv is static it will return None and this should not be included in the list
        of keys. Most rvs are static but TransDimensionalBlocks are not and should return
        the column of nlayers
        """
        indices = self.getModelKeyColumns()
        # use to index theta
        tags = theta[:, indices]
        unique_rows, tuple_indices = np.unique(tags, return_inverse=True, axis=0)
        # convert each unique row to immutable tuple for use as a dict key.
        # return list of these tuples and a numpy array of indices to this list
        keys = list(map(tuple, unique_rows))
        model_enumeration = {}
        for i, k in enumerate(keys):
            model_enumeration[k] = np.where(tuple_indices == i)[0]
        return model_enumeration, tuple_indices

    def hasConditionalApproximation(self, mk):
        """
        Override this method to return true when a MVN approximation of the target is available
        """
        return False

    def getConditionalApproximation(self, mk):
        pass

    def useBarycenterCombination(self):
        return False

    def calibrateProposalsWeighted(self, theta, weights, N, t):
        # calibrate the proposals
        self.proposal.generateAndMapRVIndices(theta)
        m_indices, rev = self.enumerateModels(theta)
        self.proposal.calibrateweighted(theta, weights, m_indices, N, t)

    def calibrateProposalsMMMPD(self, mmmpd, N, t):
        # calibrate the proposals
        self.proposal.generateAndMapRVIndices(
            theta=None, model_keys=mmmpd.getModelKeys()
        )
        self.proposal.calibratemmmpd(mmmpd, N, t)

    def calibrateProposalsUnweighted(self, theta, N, t):
        # calibrate the proposals
        self.proposal.generateAndMapRVIndices(theta)
        N2 = theta.shape[0]
        m_indices, rev = self.enumerateModels(theta)
        self.proposal.calibrateweighted(theta, np.full(N2, 1.0 / N2), m_indices, N, t)

    def deconcatAllParameters(
        self, param_mat, theta, model_key, dest_model_key=None, transform=False
    ):
        self.proposal.generateAndMapRVIndices(theta, model_keys=self.getModelKeys())
        if transform:
            raise Exception("deconcatAllParameters() not implemented with transforms")
        else:
            return self.proposal.deconcatParameters(
                param_mat, theta, model_key, dest_model_key
            )

    def concatAllParameters(self, theta, mk, transform=False):
        self.proposal.generateAndMapRVIndices(theta, model_keys=self.getModelKeys())
        if transform:
            # use self.rv_transforms dict to transform each parameter
            params, splitidx = self.proposal.explodeParameters(theta, mk)
            for rv_name, tf in self.rv_transforms.items():
                params[rv_name] = tf(params[rv_name])
            Ttheta = self.proposal.applyProposedParameters(params, theta, model_key=mk)
            return self.proposal.concatParameters(Ttheta, mk)
        else:
            return self.proposal.concatParameters(theta, mk)

    def _estimatemoments(self, theta, mk):
        """
        ParametricModelSpace::_estimatemoments() follows that for randomvariable block
        For the input theta, returns moments for rows matching model mk.
        returns mean for each rv concatenated and covariance estimate for each rv as a block-diagonal cov matrix
        """
        if self.em_method == "block":
            rv_dim_dict = self.generateRVIndices(
                flatten_tree=False
            )  # do not pass model key, leave that to TransDimensionalBlock et al
            ncols = int(self.dim(mk))
            mean = np.zeros(ncols)
            cov = np.eye(ncols)  # by default, identity.
            curdim = 0
            # for key,rvdim in rv_dim_dict.items():
            for key in self.rv_names:
                rvdim = self.rv[key].dim(mk)
                print(
                    "ParametricModelSpace: Estimating moments for ",
                    key,
                    " dim ",
                    rvdim,
                    " idx ",
                    rv_dim_dict[key],
                )
                (
                    mean[curdim : curdim + rvdim],
                    cov[curdim : curdim + rvdim, curdim : curdim + rvdim],
                ) = self.rv[key]._estimatemoments(theta[:, rv_dim_dict[key]], mk)
                curdim += rvdim
            return mean, cov
        elif self.em_method == "full":
            # Assume all rows are for model mk
            theta_k = self.concatAllParameters(theta, mk, transform=True)
            # print("theta_k",theta_k)
            mean = np.mean(theta_k, axis=0)
            cov = np.cov(theta_k.T)
            return mean, cov
        else:
            raise Exception(
                "Unsupported moment estimation method: {}".format(self.em_method)
            )

    def propose(self, theta, N):
        """
        theta is numpy array.
        returns proposed_theta same dimensions as theta
        TODO should return function pointers to acceptance ratio functions for once llh is computed.
        """
        self.assertDimension(theta)
        # call the top level proposal.
        self.proposal.generateAndMapRVIndices(theta)
        # subset theta to only columns being proposed
        return self.proposal.draw(theta, N)

    def compute_prior(self, theta):
        # traverse RVs and compute prior for each.
        # returns log of the prior evaluated at theta
        n = theta.shape[0]
        rv_dim_dict = self.generateRVIndices(flatten_tree=False)
        prop_prior = np.zeros(n)
        for key in self.rv_names:
            prop_prior += (
                self.rv[key].eval_log_prior(theta[:, rv_dim_dict[key]]).reshape(n)
            )
        return prop_prior

    def compute_llh(self, theta):
        return 1

    def setStartingDistribution(self, starting_dist):
        # TODO assert is correct distribution
        self.starting_dist = starting_dist

    def evalStartingDistribution(self, theta):
        return self.starting_dist.compute_prior(theta)

    def compute_lar(
        self,
        cur_theta,
        prop_theta,
        prop_lpqratio,
        prop_llh,
        llh,
        cur_prior,
        prop_prior,
        temperature,
    ):
        """
        Computes the log acceptance ratio.
        The technical difficulty is the computation of this ratio after the log likelihood has been computed.
        In the language of probability, the MH acceptance ratio requires computation of expensive densities.
        These densities are the likelihood, the prior, and the proposal.
        For RWMH the proposal is often symmetric and cancels.

        prop_lpqratio : the log of the prior and proposal ratio. Often this is log(1)=0.
        prop_llh : the proposed log likelihood
        llh : the current log likelihood
        temperature: a value between 0 and 1 representing the geometric annealing temperature.
        """
        #
        cur_start_dist = self.evalStartingDistribution(cur_theta)
        prop_start_dist = self.evalStartingDistribution(prop_theta)
        lar = (
            temperature * (prop_llh - llh + prop_prior - cur_prior)
            + (1 - temperature) * (prop_start_dist - cur_start_dist)
            + prop_lpqratio
        )
        lar[np.isnan(lar)] = np.NINF
        lar[np.isneginf(lar)] = np.NINF
        lar[np.isposinf(lar)] = 0
        lar[lar > 0] = 0
        return lar
