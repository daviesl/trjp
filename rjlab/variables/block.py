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

np.set_printoptions(linewidth=200)


class RandomVariableBlock(object):
    """
    Block of random variables. Defines one block. A layered model (for example) defines each layer as a block.
    """

    def __init__(self, random_variables={}):
        assert isinstance(random_variables, dict)
        self.rv = {}
        self.rv_names = []  # the order in here is immutable
        self.ldim = 0
        self.max_blocks = 1
        self.min_blocks = 1
        for key, value in random_variables.items():
            # TODO should just use assert perhaps
            # TODO FIXME check that key does not already exist in dict
            if isinstance(value, RandomVariable):
                self.rv_names.append(key)
                self.rv[key] = value
                self.ldim += self.rv[key].dim()
            elif isinstance(value, RandomVariableBlock):
                self.rv_names.append(key)
                self.rv[key] = value  # different behavior?
                self.ldim += self.rv[key].dim()
            else:
                raise ("Not an instance of RandomVariable or RandomVariableBlock")

    def getModelIdentifier(self):
        ids = []
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids += i
        if len(ids) == 0:
            return None
        else:
            return ids

    def setModel(self, m):
        self.pmodel = m

    def getModel(self):
        return self.pmodel

    def dim(self, mk=None):
        return self.ldim

    def islike(self, r):
        """
        Returns true if column names and indices match input r RandomVariableBlock
        """
        assert isinstance(r, RandomVariableBlock)
        selfidx = self.generateRVIndices()
        ridx = r.generateRVIndices()
        return ridx == selfidx

    def generateRVIndices(self, cur_dim=0, model_key=None, flatten_tree=True):
        """
        Called by ParametricModelSpace and all children random variable blocks
        Returns a dict of rv names as keys, with values as either a list of indices
        or another dict
        or possibly a slice object.
        The idea is to use the values to index parameters theta

        Returns a dictionary of possibly more dictionaries.
        The geometry of the indices should be unchanged for the duration of the sampling.
        Auxiliary variables for transdimensional moves are fixed in place.

        flatten_tree when True will generate a dict of the end nodes.
                     when False will generate a dict of one level of the tree
                     This has the result that when True all entries will be
                     the names of instances of RandomVariable, whereas
                     when False the entries will be a mix of RandomVariable
                     and RandomVariableBlock name instances.
        """
        rv_dim_dict = {}
        # use rv_names to preserve order
        # for key,value in self.rv.items():
        for key in self.rv_names:
            if isinstance(self.rv[key], RandomVariable):
                next_dim = cur_dim + self.rv[key].dim()
                rv_dim_dict[key] = list(range(cur_dim, next_dim))
                # use a slice object
                # rv_dim_dict[key]=slice(cur_dim,next_dim)
                cur_dim = next_dim
            elif isinstance(self.rv[key], RandomVariableBlock):
                # rv_dim_dict[key]=self.rv[key].generateRVIndices(cur_dim)
                temp_rv_dim_dict = self.rv[key].generateRVIndices(
                    cur_dim=cur_dim, model_key=model_key, flatten_tree=flatten_tree
                )
                if flatten_tree:
                    # the 'key' is discarded. It is a group name only.
                    # append to dictionary
                    rv_dim_dict.update(
                        temp_rv_dim_dict
                    ) 
                else:
                    rv_dim_dict[key] = []
                    for k, a in temp_rv_dim_dict.items():
                        rv_dim_dict[key] += a
                cur_dim += self.rv[key].dim()
        return rv_dim_dict

    def retrieveRV(self, rv_name):
        """
        returns rv object corresponding to rv_name.
        if rv object is not found, return none
        """
        if rv_name in self.rv:
            return self.rv[rv_name]
        else:
            for key in self.rv_names:
                if isinstance(self.rv[key], RandomVariableBlock):
                    rv = self.rv[key].retrieveRV(rv_name)
                    if rv is not None:
                        return rv
            return None

    def resetDimension(self):
        """
        Call resetDimension() when one of the random variables has changed dimension.
        """
        self.dim = 0
        for key, value in self.rv.items():
            self.dim += self.rv[key].dim()

    def draw(self, size=1):
        d = np.zeros((size, self.dim()))
        curdim = 0
        for key in self.rv_names:
            d[:, curdim : curdim + self.rv[key].dim()] = (
                self.rv[key].draw(size).reshape(size, self.rv[key].dim())
            )
            curdim += self.rv[key].dim()
        return d

    def eval_log_prior(self, theta):
        """
        Requires that theta is correct dimension for this RV
        """
        size = theta.shape[0]
        log_prior = np.zeros(size)
        rv_dim_dict = self.generateRVIndices(flatten_tree=False)
        # print("Block eval log prior, rv_dim_dict",rv_dim_dict)
        for key in self.rv_names:
            log_prior += (
                self.rv[key].eval_log_prior(theta[:, rv_dim_dict[key]]).reshape(size)
            )
        return log_prior

    def _estimatemoments(self, theta, mk):
        """
        Requires that theta is correct dimension for this RV
        returns mean for each rv concatenated and covariance estimate for each rv as a block-diagonal cov matrix
        """
        rv_dim_dict = self.generateRVIndices(model_key=mk, flatten_tree=False)
        ncols = int(self.dim(mk))
        mean = np.zeros(ncols)
        cov = np.eye(ncols)  # by default, identity.
        curdim = 0
        for key in self.rv_names:
            rvdim = self.rv[key].dim(mk)
            print(
                "RVB dim for rv",
                key,
                " is ",
                rvdim,
                " mk=",
                mk,
                " idx ",
                rv_dim_dict[key],
            )
            (
                mean[curdim : curdim + rvdim],
                cov[curdim : curdim + rvdim, curdim : curdim + rvdim],
            ) = self.rv[key]._estimatemoments(theta[:, rv_dim_dict[key]], mk)
            curdim += rvdim
        return mean, cov


class ConditionalVariableBlock(RandomVariableBlock):
    def __init__(self, random_variables, rv_conditions, indicator_rv, indicator_name):
        assert isinstance(random_variables, dict)
        assert isinstance(indicator_rv, UniformIntegerRV)
        all_rvs = random_variables.copy()
        self.rv_conditions = rv_conditions
        for rvn, rv in random_variables.items():
            assert rvn in self.rv_conditions
            assert isinstance(
                self.rv_conditions[rvn], int
            )  # for now, all conditions are >=(int)
        all_rvs[indicator_name] = indicator_rv
        super(ConditionalVariableBlock, self).__init__(all_rvs)
        self.indicator_name = indicator_name

    def getModelIdentifier(self):
        ids = []
        # append the indicator identifier
        ids.append(self.indicator_name)
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)
        if len(ids) == 0:
            return None
        else:
            return ids

    def get_inclusion_from_key(self, model_key, condition):
        ids = self.getModel().getModelIdentifier()
        for i, rvn in enumerate(ids):
            if rvn == self.indicator_name:
                return int(model_key[i] >= condition)
        assert False  # failsafe

    def generateRVIndices(self, cur_dim=0, model_key=None, flatten_tree=True):
        """
        Called by ParametricModelSpace and all children random variable blocks
        Returns a dict of rv names as keys, with values as either a list of indices
        or another dict
        or possibly a slice object.
        The idea is to use the values to index parameters theta

        Returns a dictionary of possibly more dictionaries. 
        The geometry of the indices should be unchanged for the duration of the sampling.
        Auxiliary variables for transdimensional moves are fixed in place.
        """
        rv_dim_dict = {}
        # if a model key is provided, query each random variable for the slice amount
        # use rv_names to preserve order
        for key in self.rv_names:
            if key == self.indicator_name:
                mb = 1
            else:
                if model_key is not None:
                    # determine if we're slicing
                    mb = self.get_inclusion_from_key(model_key, self.rv_conditions[key])
                else:
                    mb = 1
            if mb > 0:
                range_dim = int(self.rv[key].dim())
            else:
                range_dim = 0
            if range_dim > 0:
                rv_dim_dict[key] = list(range(cur_dim, cur_dim + range_dim))
            cur_dim += int(self.rv[key].dim())
        # if model_key is not None:
        #    print("CVB ",model_key,"generate rv indices",rv_dim_dict)
        return rv_dim_dict

    def dim(self, model_key=None):
        """
        If model_key is not provided, return full matrix dim including nblocks dim
        otherwise, return summed dim of sliced rvs, not including nblocks dim.

        This method is used by both the traversing for rvs for a full array of parameters
        and by the estimatemoments methods (_estimatemoments(), estimateMoments())
        The latter only wants moments of continuous variables.

        This method does not call child rv.dim() methods because
        this class assumes all rvs are the dimension of the slice
        """
        if model_key is None:
            return super(ConditionalVariableBlock, self).dim()
        else:
            # get slices for each RV in block
            # do not include the nblocks rv dim
            ldim = 0
            nb_dim = self.get_inclusion_from_key(model_key)
            for key in self.rv_names:
                if key == self.indicator_name:
                    continue
                nb_dim_m = self.rv[key].dim() * nb_dim
                ldim += nb_dim_m
            return ldim

    def draw(self, size=1):
        d = np.zeros((size, self.dim()))
        curdim = 0
        # draw indicator first, then use that to draw remaining
        ind_dim = self.rv[self.indicator_name].dim()
        indicator = self.rv[self.indicator_name].draw((size, ind_dim)).flatten()
        # get unique indicator values
        unique_indicator = np.unique(indicator).astype(int)

        # curdim+=thisdim
        for key in self.rv_names:
            rv_dim = self.rv[key].dim()
            if key == self.indicator_name:
                d[:, curdim : curdim + rv_dim] = indicator.reshape(
                    size, rv_dim
                )  
                curdim += rv_dim
                continue
            for indicator_value in unique_indicator:
                ind_idx = indicator == indicator_value
                ind_size = np.sum(ind_idx)
                if ind_size == 0:
                    continue
                ind_condition_met = indicator_value >= self.rv_conditions[key]
                if ind_condition_met:
                    d[ind_idx, curdim : curdim + rv_dim] = (
                        self.rv[key].draw((ind_size,)).reshape((ind_size, rv_dim))
                    )
                else:
                    d[ind_idx, curdim : curdim + rv_dim] = 0
            curdim += rv_dim
        return d

    def eval_log_prior(self, theta):
        """
        Requires that theta is correct dimension for this RV
        """
        size = theta.shape[0]
        log_prior = np.zeros(size)
        rv_dim_dict = self.generateRVIndices(flatten_tree=False)
        for key in self.rv_names:
            if key == self.indicator_name:
                # model indicator, always evaluate
                log_prior += (
                    self.rv[key]
                    .eval_log_prior(theta[:, rv_dim_dict[key]])
                    .reshape(size)
                )
            else:
                log_prior += (
                    theta[:, rv_dim_dict[self.indicator_name]]
                    >= self.rv_conditions[key]
                ).reshape(size) * self.rv[key].eval_log_prior(
                    theta[:, rv_dim_dict[key]]
                ).sum(
                    axis=1
                ).reshape(
                    size
                )
        return log_prior

    def _estimatemoments(self, theta, mk):
        """
        ConditionalVariableBlock::_estimatemoments()
        excludes indicator
        Requires that theta is correct dimension for this RV
        returns mean for each rv concatenated and covariance estimate for each rv as a block-diagonal cov matrix
        """
        rv_dim_dict = self.generateRVIndices(model_key=mk, flatten_tree=False)
        ncols = int(self.dim(mk))
        mean = np.zeros(ncols)
        cov = np.eye(ncols)  # by default, identity.
        curdim = 0
        for key in self.rv_names:
            if key == self.indicator_name:
                continue
            rvdim = self.rv[key].dim(mk)
            print(
                "CVB dim for rv",
                key,
                " is ",
                rvdim,
                " mk=",
                mk,
                " idx ",
                rv_dim_dict[key],
            )
            (
                mean[curdim : curdim + rvdim],
                cov[curdim : curdim + rvdim, curdim : curdim + rvdim],
            ) = self.rv[key]._estimatemoments(theta[:, rv_dim_dict[key]], mk)
            curdim += rvdim
        return mean, cov



class TransDimensionalBlock(RandomVariableBlock):
    def __init__(
        self,
        random_variables={},
        nblocks_name="nblocks",
        nblocks_modifiers={},
        minimum_blocks=1,
        maximum_blocks=1,
        nblocks_position="first",
        nblocks_dist="uniform",
    ):
        """
        nblocks_modifiers is a dict (indexed by rv name) of lambda functions that take the nblocks variable as input and return an integer.
                          This return value modifies the minimum and maximum number of blocks for that variable.
        """
        assert isinstance(random_variables, dict)
        self.rv = {}
        self.rv_names = []  # the order in here is immutable
        self.rv_dim = (
            {}
        )  # used because nblocks has dim 1 but the other RVs have dim maxblocks. TODO make dim() return sum of this.
        self.ldim = 0
        self.max_blocks = maximum_blocks
        self.min_blocks = minimum_blocks
        # NLayers rv
        self.nblocks_name = nblocks_name
        self.nblocks_modifiers = nblocks_modifiers  # needs to have an entry for each RV. If none exists, add the default identity modifier.
        assert nblocks_position in ["first", "last"]
        self.nblocks_position = nblocks_position
        if nblocks_position == "first":
            self.rv_names.append(nblocks_name)
            self.ldim += 1
        if nblocks_dist == "uniform":
            self.rv[nblocks_name] = UniformIntegerRV(self.min_blocks, self.max_blocks)
        elif nblocks_dist == "poisson":
            self.rv[nblocks_name] = BoundedPoissonRV(
                1, self.min_blocks, self.max_blocks
            )
        else:
            raise Exception(
                "Unsupported distribution for {} in TransDimensionalBlock".format(
                    nblocks_name
                )
            )
        self.rv_dim[nblocks_name] = 1
        for key, value in random_variables.items():
            # set modifier if none exists
            if key not in self.nblocks_modifiers.keys():
                self.nblocks_modifiers[key] = lambda n: n
            if isinstance(value, RandomVariable):
                self.rv_names.append(key)
                self.rv[key] = value
                self.rv_dim[key] = self.rv[key].dim() * self.nblocks_modifiers[key](
                    self.max_blocks
                )
                self.ldim += self.rv_dim[key]
            elif isinstance(value, RandomVariableBlock):
                raise (
                    "RandomVariableBlocks are not supported in TransDimensionalBlocks"
                )
            else:
                raise ("Not an instance of RandomVariable")
        if nblocks_position == "last":
            self.rv_names.append(nblocks_name)
            self.ldim += 1

    def appendBlock(self, theta, block_vars):
        pass

    def getModelIdentifier(self):
        ids = []
        # append the nblocks identifier
        ids.append(self.nblocks_name)
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(i)
        if len(ids) == 0:
            return None
        else:
            return ids

    def generateRVIndices(self, cur_dim=0, model_key=None, flatten_tree=True):
        """
        Called by ParametricModelSpace and all children random variable blocks
        Returns a dict of rv names as keys, with values as either a list of indices
        or another dict
        or possibly a slice object.
        The idea is to use the values to index parameters theta

        Returns a dictionary of possibly more dictionaries. 
        The geometry of the indices should be unchanged for the duration of the sampling.
        Auxiliary variables for transdimensional moves are fixed in place.
        """
        rv_dim_dict = {}
        # if a model key is provided, query each random variable for the slice amount
        # add the nblocks parameter
        # use rv_names to preserve order
        for key in self.rv_names:
            if isinstance(self.rv[key], RandomVariable):
                if key == self.nblocks_name:
                    mb = 1
                    max_mb = 1
                else:
                    if model_key is not None:
                        # determine if we're slicing
                        mb = self.nblocks_modifiers[key](
                            self.get_model_slice_from_key(model_key)
                        )
                        max_mb = self.nblocks_modifiers[key](self.max_blocks)
                    else:
                        mb = self.nblocks_modifiers[key](self.max_blocks)
                        max_mb = self.nblocks_modifiers[key](self.max_blocks)
                range_dim = int(self.rv[key].dim() * mb)
                next_dim = int(cur_dim + self.rv[key].dim() * max_mb)
                rv_dim_dict[key] = list(range(cur_dim, cur_dim + range_dim))
                cur_dim = next_dim
            elif isinstance(self.rv[key], RandomVariableBlock):
                raise ("RandomVariableBlock not supported in TransDimensionalBlock")
        return rv_dim_dict

    def get_model_slice_from_key(self, model_key):
        # determines which part of the model key affects this rv block
        ids = self.getModel().getModelIdentifier()
        # always slice for now
        for i, rvn in enumerate(ids):
            if rvn == self.nblocks_name:
                # specific to TransdimensionalBlock
                return model_key[i]
        assert False  # failsafe

    def dim(self, model_key=None):
        """
        If model_key is not provided, return full matrix dim including nblocks dim
        otherwise, return summed dim of sliced rvs, not including nblocks dim.

        This method is used by both the traversing for rvs for a full array of parameters
        and by the estimatemoments methods (_estimatemoments(), estimateMoments())
        The latter only wants moments of continuous variables.

        This method does not call child rv.dim() methods because
        this class assumes all rvs are the dimension of the slice
        """
        if model_key is None:
            return super(TransDimensionalBlock, self).dim()
        else:
            # get slices for each RV in block
            # do not include the nblocks rv dim
            ldim = 0
            nb_dim = self.get_model_slice_from_key(model_key)
            for key in self.rv_names:
                if key == self.nblocks_name:
                    continue
                nb_dim_m = self.nblocks_modifiers[key](nb_dim)
                ldim += nb_dim_m
            return int(ldim)

    def draw(self, size=1):
        """
        Unable to reuse code from RandomVariableBlock because the RV.dim() needs to be scaled by max_blocks
        """
        d = np.zeros((size, self.dim()))
        curdim = 0
        # draw nblocks first, then use that to draw remaining
        thisdim = self.rv_dim[self.nblocks_name]
        nblocks = (
            self.rv[self.nblocks_name].draw((size, thisdim)).reshape(size, thisdim)
        )
        if self.nblocks_position == "first":
            d[:, curdim : curdim + thisdim] = nblocks
            curdim += thisdim
        elif self.nblocks_position == "last":
            d[:, -thisdim:] = nblocks

        # get unique nblocks values
        nblocks = nblocks.flatten()
        unique_nblocks = np.unique(nblocks).astype(int)

        for key in self.rv_names:
            if key == self.nblocks_name:
                continue
            for nb_dim in unique_nblocks:
                nb_idx = nblocks == nb_dim
                nb_dim_m = self.nblocks_modifiers[key](nb_dim)
                if nb_dim_m == 0:
                    continue
                nb_size = np.sum(nb_idx)
                thisdim = self.rv_dim[key]
                d[nb_idx, curdim : curdim + nb_dim_m] = (
                    self.rv[key].draw((nb_size, nb_dim_m)).reshape(nb_size, nb_dim_m)
                )
            curdim += thisdim
        return d

    def eval_log_prior(self, theta):
        """
        Requires that theta is correct dimension for this RV
        """
        size = theta.shape[0]
        log_prior = np.zeros(size)
        # TODO eval priors for all model types, generating indices for each.
        rv_dim_dict = self.generateRVIndices(flatten_tree=False)

        # do key blocks first, then get nblocks from it.
        log_prior += (
            self.rv[self.nblocks_name]
            .eval_log_prior(theta[:, rv_dim_dict[self.nblocks_name]])
            .reshape(size)
        )

        # get unique blocks, then eval each separately. This will help for priors that are a dirichlet dist and any other multivariate dist.
        nblocks = theta[:, rv_dim_dict[self.nblocks_name]].flatten()
        unique_nblocks = np.unique(nblocks).astype(int)

        for key in self.rv_names:
            if key == self.nblocks_name:
                # model indicator, always evaluate
                pass
            else:
                # The below assumes that zero'd entries are not active. This is a mistake, because some distributions could have mass at zero.
                for nb_dim in unique_nblocks:
                    nb_match = nblocks == nb_dim
                    nb_dim_m = self.nblocks_modifiers[key](nb_dim)
                    if nb_dim_m == 0:
                        continue
                    nb_size = np.sum(nb_match)
                    nb_idx = np.where(nb_match)[0]
                    trimmed_idx = rv_dim_dict[key][:nb_dim_m]
                    log_prior[nb_idx] += sum_along_axis(
                        self.rv[key].eval_log_prior(theta[np.ix_(nb_idx, trimmed_idx)]),
                        axis=1,
                    ).reshape(nb_size)
        return log_prior

    def _estimatemoments(self, theta, mk):
        """
        TransDimensionalBlock::_estimatemoments()
        excludes nblocks rv
        Requires that theta is correct dimension for this RV
        returns mean for each rv concatenated and covariance estimate for each rv as a block-diagonal cov matrix
        """
        rv_dim_dict = self.generateRVIndices(model_key=mk, flatten_tree=False)
        ncols = int(self.dim(mk))
        mean = np.zeros(ncols)
        cov = np.eye(ncols)  # by default, identity.
        curdim = 0
        for key in self.rv_names:
            if key == self.nblocks_name:
                continue
            rvdim = len(rv_dim_dict[key])
            print(
                "TVB dim for rv",
                key,
                " is ",
                rvdim,
                " mk=",
                mk,
                " idx ",
                rv_dim_dict[key],
            )
            (
                mean[curdim : curdim + rvdim],
                cov[curdim : curdim + rvdim, curdim : curdim + rvdim],
            ) = self.rv[key]._estimatemoments(theta[:, rv_dim_dict[key]], mk)
            curdim += rvdim
        return mean, cov
