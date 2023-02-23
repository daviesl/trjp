import numpy as np
from scipy.stats import *
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import itertools
import scipy
from scipy.special import logsumexp
from rjlab.utils.linalgtools import *
from rjlab.distributions import *
from rjlab.variables import *

np.set_printoptions(linewidth=200)


class Proposal(object):
    """
    A proposal is one component of a transition kernel. It is not the acceptance
    ratio, it is merely the density used to propose mutations.

    Can be a single proposal, a single layer of proposals, or mutliple layers.
    Each proposal transition density takes as input a vector of random variables.
    TODO figure out a way to make hierarchical.

    FIXME this scheme is broken.
          it needs to use the ParametricModelSpace definition and links to random variables
          somehow pass in the dimension of each rv and the size of the proposal.
          somehow index the current theta/proposed theta with the proposed values. THIS IS THE CRUX

    Usage:
        ps = Proposal(Componentwise(MVN(),birth(),death())
        prop_theta = ps.draw(theta,size=N)

        ps2 = Proposal(Componentwise(EigDecComponentwiseNormal(),birth(),death())
        prop_theta = ps2.draw(theta,size=N)
    """

    idnamedict = {}
    idpropdict = {}

    def __init__(self, proposals=[]):
        self.ps = []
        self.rv_names = []
        self.rv_indices = {}
        # print(proposals)
        for arg in proposals:
            # assert arg is a proposal strategy
            if isinstance(arg, Proposal):
                self.ps.append(arg)
            elif isinstance(arg, str):
                self.rv_names.append(arg)
            else:
                print(
                    "Unrecognised type. Should be rv name or proposal.", arg, type(arg)
                )
                assert False
        # print("INIT PROPOSAL RV NAMES",self.rv_names, "proposals",proposals)
        self.splitby_val = None
        self.setID()
        self.ar = []  # 0.44 # target default
        self.t_ar = []
        self.n_ar = []
        self.exclude_concat = []

    def make_copy(self):
        # some variables are references, some need to be duplicated
        to_be_copied = {"rv_names", "rv_indices", "ar", "t_ar", "n_ar"}
        c = copy(self)
        c.__dict__ = {
            attr: copy(self.__dict__[attr])
            if attr in to_be_copied
            else self.__dict__[attr]
            for attr in self.__dict__
        }
        # recursive copy the subproposals
        c.ps = [p.make_copy() for p in self.ps]
        c.setID()
        return c

    def printName(self):
        # TODO use model key
        # if self.splitby_val is not None:
        #    return self.__class__.__name__ + str(self.splitby_val)
        # else:
        return self.__class__.__name__

    def setModel(self, m):
        # TODO assert m is a parameteric model
        # from rjlab.variables.model import ParametricModelSpace
        # assert(isinstance(m,ParametricModelSpace))
        self.pmodel = m
        for prop in self.ps:
            prop.setModel(m)

    def getModel(self):
        return self.pmodel

    @classmethod
    def setAcceptanceRates(cls, prop_id, log_acceptance_ratio, t):
        unique_pids, pid_indices = np.unique(prop_id, return_inverse=True)
        unique_pids = unique_pids.astype(int)
        # if 0 in unique_pids:
        #    print("prop ids are:\n{}".format(prop_id.astype(int)))
        for i, pid in enumerate(unique_pids):
            # print ("setting {} in idpropdict where keys={}, unique_pids={}".format(int(pid),cls.idpropdict.keys(),unique_pids))
            new_ar = np.exp(logsumexp(log_acceptance_ratio[pid_indices == i])) / np.sum(
                pid_indices == i
            )
            if not np.isfinite(new_ar):
                print(
                    "Non-finite acceptance rate for prop",
                    pid,
                    "\n",
                    log_acceptance_ratio[pid_indices == i],
                    "\n",
                    np.sum(pid_indices == i),
                )
                new_ar = 0
            cls.idpropdict[int(pid)].setAR(new_ar, t, np.sum(pid_indices == i))

    def setAR(self, cur_ar, t, n):
        self.ar.append(cur_ar)
        self.t_ar.append(t)
        self.n_ar.append(n)

    def getLastAR(self):
        if len(self.ar) > 0:
            return self.ar[-1]
        else:
            return 1.0

    def getAvgARN(self, l):
        """
        Return the inner product of the number of particles proposed and the avg acceptance rate for that proposal, over l time-steps.
        """
        if l > len(self.ar):
            return 0
        return np.dot(np.clip(self.ar[-l:], None, 1), np.clip(self.n_ar[-l:], None, 1))

    @classmethod
    def clearIDs(cls):
        cls.idnamedict = {}
        cls.idpropdict = {}

    def setID(self):
        self.idnamedict[id(self)] = self.__class__.__name__
        self.idpropdict[id(self)] = self

    def setModelIdentifier(self, i):
        self.setID()
        self.model_identifier = (
            i  # a proposal may need to know what model it is assigne to.
        )

    def getModelIdentifier(self):
        return (
            self.model_identifier
        )  # a proposal may need to know what model it is assigned to.

    def dim(self, model_key):
        ldim = 0
        # print("rv_indices",self.rv_indices)
        for name, param_range in self.rv_indices[model_key].items():
            if name not in self.exclude_concat:
                ldim += len(param_range)
        return int(ldim)

    def getIndicesForModelKey(self, theta, model_key):
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        return model_key_indices[model_key]

    def explodeParameters(self, theta, model_key):
        self.setID()
        param_dict = {}
        # this needs to get rows of theta that satisfy model_key
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        rows = model_key_indices[model_key].shape[0]
        split_indices = model_key_indices[model_key]
        # assumes self.rv_indices has been populated.
        # print("rv_indices",self.rv_indices)
        for name, param_range in self.rv_indices[model_key].items():
            param_dict[name] = theta[split_indices, :][:, param_range]
        return param_dict, split_indices

    def getVariable(self, theta, vname):
        """
        unintelligent method to return column(s) for variable
        does not trim columns
        """
        columns = self.getModel().generateRVIndices()  # todo move to constructor
        return theta[:, columns[vname]]

    def setVariable(self, theta, vname, values):
        """
        unintelligent method to set column(s) for variable
        does not trim columns
        """
        # values = np.array(values)
        # if len(values.shape)==0:
        if not isinstance(values, np.ndarray):
            values = np.full(theta.shape[0], values)
        columns = self.getModel().generateRVIndices()  # todo move to constructor
        # print(columns,vname,values.shape)
        theta[:, columns[vname]] = values.reshape(
            (values.shape[0], len(columns[vname]))
        )
        return theta

    def concatParameters(self, theta, model_key, return_indices=False):
        self.setID()
        # useful for homogeneous proposals
        cur_dim = 0
        concat_indices = []
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        try:
            rows = model_key_indices[model_key].shape[0]
        except KeyError:
            print("Model key not in ensemble")
            print(model_key, model_key_indices, theta)
            raise Exception("Model key {} not in ensemble".format(model_key))
            sys.exit(0)
        split_indices = model_key_indices[model_key]
        param_mat = np.zeros((rows, self.dim(model_key)))
        for name, param_range in self.rv_indices[model_key].items():
            if name not in self.exclude_concat:
                thisdim = len(param_range)
                param_mat[:, cur_dim : cur_dim + thisdim] = theta[
                    np.ix_(split_indices, param_range)
                ]
                concat_indices += param_range
                cur_dim += thisdim
        # print("concat k",model_key,"param_mat ",param_mat.shape)
        if return_indices:
            return param_mat, concat_indices
        else:
            return param_mat

    def deconcatParameters(self, param_mat, theta, model_key, dest_model_key=None):
        """
        the reverse of concatParameters()
        """
        if dest_model_key is None:
            dest_model_key = model_key
        # TODO assert param_mat has correct dimensions
        cur_dim = 0
        # get indices of split by
        p_theta = theta.copy()
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(p_theta)
        rows = model_key_indices[model_key].shape[0]
        split_indices = model_key_indices[model_key]
        for name, param_range in self.rv_indices[dest_model_key].items():
            if name not in self.exclude_concat:
                thisdim = len(param_range)
                p_theta[np.ix_(split_indices, param_range)] = param_mat[
                    :, cur_dim : cur_dim + thisdim
                ]
                cur_dim += thisdim
        return p_theta

    def applyProposedParametersAllModels(self, proposed_dict, theta, model_map=None):
        """
        the reverse of explodeParameters(). Run at end of draw()
        """
        prop_theta = theta.copy()
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        # print(model_map)
        for model_key, proposed in proposed_dict.items():
            rows = model_key_indices[model_key].shape[0]
            split_indices = model_key_indices[model_key]
            # assumes self.rv_indices has been populated.
            if model_map is not None:
                idx_model_key = model_map[model_key]
            else:
                idx_model_key = model_key
            for name, param_range in self.rv_indices[idx_model_key].items():
                prop_theta[np.ix_(split_indices, param_range)] = proposed[name]
        return prop_theta

    def applyProposedParameters(self, proposed, theta, model_key):
        """
        the reverse of explodeParameters(). Run at end of draw()
        """
        prop_theta = theta.copy()
        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(theta)
        rows = model_key_indices[model_key].shape[0]
        split_indices = model_key_indices[model_key]
        # assumes self.rv_indices has been populated.
        for name, param_range in self.rv_indices[model_key].items():
            prop_theta[np.ix_(split_indices, param_range)] = proposed[name]
        return prop_theta

    def initRVIndices(self):
        # FIXME currently does nothing
        # self.rv_indices = {}
        for k, i in self.rv_indices.items():
            pass
        for prop in self.ps:
            prop.initRVIndices()

    def generateAndMapRVIndices(self, theta, model_keys=None):
        if model_keys is None:
            model_keys = self.getModel().getModelKeys(theta)
        self.initRVIndices()
        for model_key in model_keys:
            proposal_columns = self.getModel().generateRVIndices(model_key=model_key)
            # print("rv names",self.rv_names)
            # print("prop cols",proposal_columns)
            self.mapRVIndices(proposal_columns, model_key=model_key)
            # print("rv_indices after map",self.rv_indices)

    def mapRVIndices(self, rv_index_dict, model_key):
        """
        generates and index of random vars used in a particular model
        indexed by model_key.
        Recursively goes through sub-proposals and maps items
        from the input rv_index_dict to each model.
        Only collects indices used in that model.
        """
        if model_key not in self.rv_indices:
            self.rv_indices[model_key] = {}  # init for this model
        self.mapRVIndices_internal(rv_index_dict, model_key=model_key)
        # now search child proposals
        for prop in self.ps:
            prop.mapRVIndices(rv_index_dict, model_key)

    def mapRVIndices_internal(self, rv_index_dict, model_key):
        # should only be called by generateAndMapRVIndices() above
        for key, value in rv_index_dict.items():
            # if value is a list of indices, set it in the internal indices
            if isinstance(value, list):
                # print("mapping ",value,"to key",key)
                if key in self.rv_names:
                    self.rv_indices[model_key][key] = value
            elif isinstance(value, dict):
                # NOTE the behaviour here is that if key is a proposal rv-name, we don't search sub-proposal rv-names.
                if key in self.rv_names:
                    self.rv_indices[model_key][key] = collect_rv_dict_indices(value)
                else:
                    self.mapRVIndices_internal(value, model_key)
            else:
                raise "Unsupported index type"

    @staticmethod
    def resample_idx(weights, n=None):
        if n == None:
            n = weights.shape[0]
        indices = np.zeros(n, dtype=np.int32)
        # C = [0.] + [sum(weights[:i+1]) for i in range(n)]
        weights = weights / np.sum(weights)
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
            indices[i] = j
        return indices

    def calibrateweighted(self, theta, weights, m_indices_dict, size, t):
        """ """
        cal = getattr(self, "calibrate", None)
        if callable(cal):
            # resample to a reasonable size
            # Need to resample per model.
            resampled_theta_list = []
            for i, m_indices in m_indices_dict.items():
                print("Theta[{}].shape = {}".format(i, m_indices.shape[0]))
                idx = self.resample_idx(
                    weights[m_indices], n=min(2000, m_indices.shape[0])
                )
                resampled_theta_list.append(theta[m_indices][idx])
            cal(np.vstack(resampled_theta_list), size, t)
        else:
            for prop in self.ps:
                prop.calibrateweighted(theta, weights, m_indices_dict, size, t)

    def calibratemmmpd(self, mmmpd, size, t):
        """
        Calls calibratemmmpd() for each sub-proposal
        The MMMPD is the MultiModelMPD object containing a MixturePosteriorDensity for each model target.
        """
        cal = getattr(self, "calibrate", None)
        if callable(cal):
            # resample to a reasonable size
            rs, rs_w = mmmpd.getParticleDensityForTemperature(
                t, resample=True, resample_max_size=2000
            )
            cal(rs, size, t)
        else:
            for prop in self.ps:
                prop.calibratemmmpd(mmmpd, size, t)

    # def calibrate(self,theta,size,t):
    #    """
    #    By default, call calibration on all child proposals
    #    """
    #    for prop in self.ps:
    #        prop.calibrate(theta,size,t)
    def draw(self, theta, size=1):
        """
        Must be overridden. Just calls all proposals and returns the expected value.
        TODO Size should be set from theta.shape[0]
        The difficulty here is to obtain the correct columns of theta before passing to this method.

        It is a requirement that theta only contains the columns for this proposal.
        For transdimensional proposals all columns for the transdimensional parameters are passed in.

        rv_indices should be rv_columns
        with get_named_variables(theta,self.rv_indices) as named_vars:
            depths = named_vars['conductivity_depth']

        """
        assert False  # always throw an error.
        # prop_theta = np.zeros_like(theta)
        # for p in self.ps:
        #    prop_theta += (1./len(self.ps)) * p.draw(theta,size)
        # return prop_theta


def collect_rv_dict_indices(rv_index_dict):
    rv_indices = []
    for key, value in rv_index_dict.items():
        if isinstance(value, list):
            rv_indices.append(value)
        elif isinstance(value, dict):
            rv_indices.append(collect_rv_dict_indices(value))
    return rv_indices


class RepeatKernel(Proposal):
    def __init__(self, kernel, nrepeats=1):
        super(RepeatKernel, self).__init__([kernel])
        self.kernel = kernel
        self.nrepeats = nrepeats

    def calibrate(self, theta, size, t):
        self.kernel.calibrate(theta, size, t)
        self.t = t

    def draw(self, theta, size=1):
        N = theta.shape[0]
        if N == 0:
            return self.kernel.draw(theta, size)
        cur_theta = theta.copy()
        llh = self.pmodel.compute_llh(cur_theta)  # inefficient but we'll deal later.
        for i in range(self.nrepeats - 1):
            prop_theta, prop_lpqratio, prop_id = self.kernel.draw(cur_theta, size)
            prop_theta = self.pmodel.sanitise(prop_theta)
            cur_prior = self.pmodel.compute_prior(theta)
            prop_prior = self.pmodel.compute_prior(prop_theta)
            valid_theta = np.logical_and(
                np.isfinite(prop_prior), np.isfinite(prop_lpqratio)
            )
            prop_llh = np.full(N, np.NINF)
            if valid_theta.sum() > 0:
                prop_llh[valid_theta] = self.pmodel.compute_llh(
                    prop_theta[valid_theta, :]
                )
            log_acceptance_ratio = self.pmodel.compute_lar(
                cur_theta,
                prop_theta,
                prop_lpqratio,
                prop_llh,
                llh,
                cur_prior,
                prop_prior,
                self.t,
            )
            Proposal.setAcceptanceRates(prop_id, log_acceptance_ratio, self.t)
            log_u = np.log(uniform.rvs(0, 1, size=N))
            reject_indices = log_acceptance_ratio < log_u
            prop_theta[reject_indices] = cur_theta[reject_indices]
            cur_theta = prop_theta
            prop_llh[reject_indices] = llh[reject_indices]
            llh = prop_llh
        return self.kernel.draw(cur_theta, size)


class UniformChoiceProposal(Proposal):
    def draw(self, theta, size=1):
        """
        This is the top-level proposal that will choose between K sub-proposals.
        It will typically accept a full theta as the input argument.
        It will then divide the N particles into approx N/K subpopulations and feed into each proposal
        """
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_lpqratio = np.zeros(n)
        choice = np.random.randint(len(self.ps), size=n)
        # print("choice ",choice)
        ids = np.zeros(n)
        # print("proposals",self.ps)
        for i in range(len(self.ps)):
            # print("choice {} idx {}".format(i,np.where(choice==i)))
            (
                prop_theta[choice == i],
                prop_lpqratio[choice == i],
                ids[choice == i],
            ) = self.ps[i].draw(
                theta[choice == i, ...], np.sum(choice == i)
            )  # TODO remove size
        return prop_theta, prop_lpqratio, ids


class SystematicChoiceProposal(Proposal):
    def __init__(self, proposals=[]):
        super(SystematicChoiceProposal, self).__init__(proposals)
        self.counter = 0

    def draw(self, theta, size=1):
        """
        This is the top-level proposal that will choose between K sub-proposals.
        It will typically accept a full theta as the input argument.
        It will then divide the N particles into approx N/K subpopulations and feed into each proposal
        """
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_lpqratio = np.zeros(n)
        choice = np.full(n, self.counter)
        self.counter = (self.counter + 1) % len(self.ps)
        # print("choice ",choice)
        ids = np.zeros(n)
        # print("proposals",self.ps)
        for i in range(len(self.ps)):
            # print("choice {} idx {}".format(i,np.where(choice==i)))
            (
                prop_theta[choice == i],
                prop_lpqratio[choice == i],
                ids[choice == i],
            ) = self.ps[i].draw(
                theta[choice == i, ...], np.sum(choice == i)
            )  # TODO remove size
        return prop_theta, prop_lpqratio, ids


# TODO make a new block split proposal but make it split by the indices of enumerateModels.
class ModelEnumerateProposal(Proposal):
    """
    The purpose of this class is to propagate the sub-proposal type
    independently for each model. In the calibrate() method using the
    nblocks column to differentiate rows we run the calibration of
    the sub-proposal on each model. Then in the draw() method we
    do the same - run draw() of the subproposal associated with each model
    """

    def __init__(self, subproposal):
        # just use self.ps[0] for the sub proposal storage
        # splitby_name is the name of the column to split by e.g. nlayers or nblocks
        assert isinstance(subproposal, Proposal)
        super(ModelEnumerateProposal, self).__init__([subproposal])
        self.blocksplitps = {}  # list of k-model subproposals

    def enumerateModels(self, theta):
        m_indices_dict, rev = self.pmodel.enumerateModels(theta)
        enumerated_theta_dict = {}
        for k, idx in m_indices_dict.items():
            enumerated_theta_dict[k] = theta[idx]
        return enumerated_theta_dict, m_indices_dict

    # def calibrateweighted(self,theta,weights,m_indices_dict,size,t):
    #    cal = getattr(self, "calibrate", None)
    #    if callable(cal):
    #        # resample to a reasonable size
    #        # Need to resample per model.
    #        resampled_theta_list = []
    #        for i,m_indices in m_indices_dict.items():
    #            print("Theta[{}].shape = {}".format(i,m_indices.shape[0]))
    #            idx = self.resample_idx(weights[m_indices],n=min(2000,m_indices.shape[0]))
    #            resampled_theta_list.append(theta[m_indices][idx])
    #        cal(np.vstack(resampled_theta_list),size,t)
    #    else:
    #        for prop in self.ps:
    #            prop.calibrateweighted(theta,weights,m_indices_dict,size,t)
    def old_calibrate(self, theta, size, t):
        # here we do the same as Proposal.calibrate() but split by block as per draw() below.
        enumerated_theta_list, m_indices_dict = self.enumerateModels(theta)
        # first calibration will copy the sub-proposal
        for i, enumerated_theta in enumerated_theta_list.items():
            print(
                "Count {} particles for model {}".format(enumerated_theta.shape[0], i)
            )  # n,k
            if i not in self.blocksplitps:
                # FIXME don't use the below for loop. Use self.ps[0]
                for prop in self.ps:
                    # print("Copying prop ",prop.__class__.__name__,id(prop),vars(prop))
                    # prop2 = deepcopy(prop)
                    prop2 = prop.make_copy()
                    # prop2.setSplitBy(int(split_by_values[i]))
                    print("Model identifier for proposal set to {}".format(i))
                    prop2.setModelIdentifier(i)
                    self.blocksplitps[i] = prop2
            self.blocksplitps[i].calibrate(enumerated_theta, size, t)

    def calibratemmmpd(self, mmmpd, size, t):
        mklist = self.pmodel.getModelKeys()
        for mk in mklist:
            # theta,theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=2000)
            # print("Count {} particles for model {}".format(theta.shape[0],mk)) # n,k
            if mk not in self.blocksplitps:
                # FIXME don't use the below for loop. Use self.ps[0]
                for prop in self.ps:
                    prop2 = prop.make_copy()
                    # print("Model identifier for proposal set to {}".format(mk))
                    prop2.setModelIdentifier(mk)
                    self.blocksplitps[mk] = prop2
            self.blocksplitps[mk].calibratemmmpd(mmmpd, size, t)

    def draw(self, theta, size=1):
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_lpqratio = np.zeros(n)
        enumerated_theta_dict, mid = self.enumerateModels(theta)
        ids = np.zeros(n)
        for i, enumerated_theta in enumerated_theta_dict.items():
            mi = mid[i]
            prop_theta[mi], prop_lpqratio[mi], ids[mi] = self.blocksplitps[i].draw(
                enumerated_theta, mi.shape[0]
            )  # TODO remove size
        return prop_theta, prop_lpqratio, ids
