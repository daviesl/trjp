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




# The current ParametricModelSpace class should be changed to ParametricModelSpaceNestedSpace or something.
# Then we define how model transitions map using explicit bijections
# A couple of thoughts for this:
# 1) A Bijection class which takes as arguments two models and maps between them
# 2) ??
# The first point implies we need a ParametricModelSpace class which is essentially an instance defined by the model space
#   with particular parameters enabled/disabled
# Currently, parameters being enabled/disabled are implicity set by the model key. We should rectify this.
# Need to define
# 1) How model keys are derived from number of element columns
# 2) Are the number-of-element columns simply a model sub-identifier?
# 3) What is the simplest way to upgrade from model-id-columns currently implemented to 
# The end goal is to more easily retrieve a bijection in a proposal and apply it as an operation.
#
# A model should really be defined by the configuration of the priors.
# A model space should be defined by the prior distribution of models
# A bijection can be any mapping from a model+auxilliary variables to another model
# A model space defines the columns used in the sampler, the model operates on a view of these columns.
# Later down the road, the saturated space will use unused "aux" columns in the model space with their own priors. However, these priors are not evaluated in the acceptance ratio.

# The above will probably get rid of the TransdimensionalBlock. This is a big change. But necessary.
# We'll be switching from the "smooth" way of specifying parameters to atomic bijections.
# We'll also be needing to change how these are interpreted in the compute_llh() method. 
# Currently we just take the nblocks variable and use that to map to the likelihood variables.

# Also, note that we don't pass around model objects in particles. Each particle is a row in a big matrix, and the dimensions of this matrix are pre-set by the model space.
# So model classes essentially define the transformation from a row in the above matrix to parameters/random variables 
# This means that we should be able to get a model class and call a member function that gives us the random variables for that class... somehow.
# another issue is do we want to get random variables by name (sometimes we do) and do we just want to be agnostic?
# Ideally, we use the bijections to do all the name juggling. We'd have to specify one bijection for each rjmcmc move. The number of them being dependent on the models. We should make a factory for the layered model to work this out for us.

from abc import ABC, abstractmethod
# So what would it look like?
# TODO make Bijection extend the nflows Transformation
class Bijection(object): 
    @abstractmethod
    def forward(self,x,u):
        """
        Vectorised method to transform y=h(x,u)
        Returns transformed x matrix and logdet vector
        """
        pass

    @abstractmethod
    def inverse(self,y):
        pass

class NaiveChangePointBijection(Bijection):
    def __init__(self,model1,auxvars,model2):
        """
        model1 is a ParametricModel
        model2 is a ParametricModel
        auxvars is a dict {"name":RandomVariable,"name2":RandomVariable,...}
        """
        # In this function I need to define a way to map model1 to model2 and map auxvars to model2
        # TODO assert number of parameters in model1 + auxvars = model2
        # One thing I did not think of is the subset of parameters in each model to which the bijection maps.
        # in the rjmcmc layerered model proposal, we identify the background param, the transd value param and transd depth param.
        # How does this translate to the naive change point bijection?
        # I need to specify the parameters in the model1 and model2. 
        # I can't specify the explicit mapping because the aux vars determine the choice of bijection implicitly.
        # 
        # How do I want this class to be used?
        # in the RJ Proposal, I want to 
        # iterate mk for all models in the space
        #     X <- get particles for mk
        #     draw mk_primes for each row in X
        #     for each mk_prime in mk_primes
        #         X_mk_prime <- get particles from X 
        #         Y_mk_prime, logdet_mk_prime <- biject transform X_mk_prime
        #     
        # Another thing I need to ask is whereabout are the particles converted from the raw theta to model parameters?
        # DO I cookie-cutter in the get particles for mk method? X = mk.getstuff(allX)? Do I remove zeros? How do I handle it? How does it get passed to the bijection?
        self.model1 = model1
        self.model2 = model2
        self.auxvars = auxvars
        # in this naive transformation, the auxvars determine the mapping. Note that the logdet is not zero! it is -log(k+1) for forward (birth)
        # for the inverse transform, an auxiliary index needs to be generated. Don't know if better to do in-method or outside.
              
    def forward(self,x,u):
        # TODO assert dimensions of x and u are model1 and auxvars respectively
        # u will contain n pairs of uniform variables conforming to the auxvars spec. Usually U(0,1). 
        # 
        pass
    def inverse(self,y):
        # TODO assert dimesions of y match model2
        #k_max = self.model2.

        #delete_at = np.random.randint(,size=n)
        pass

class ParametricModel(object):
    # don't know if this should extend RandomVariableBlock.
    # Objects of this class are a view to a ParametricModelSpace
    # Specifically, the number and kind of parameters are fixed.
    # How does it work with a TransDimensionalBlock???
    # Should I ditch TDB and explicitly specify all parameters?
    # It all depends on how well I can make a view of a TDB.
    # {'rvname':number_of_cols,'rvname2':num_cols_2,...}?
    def __init__(self,pmspace):
        # TODO assert pmspace is ParametricModelSpace
        self.pmspace = pmspace
        self.rv_view = {}
    def addRV(self,rvname,ncols=1):
        self.rv_view[rvname]=ncols
    def getRVCols(self,rvname):
        if rvname not in self.rv_view:
            return 0
        

