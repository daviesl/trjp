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
from rjlab.proposals.base import *
from rjlab.utils.progressbar import progress

np.set_printoptions(linewidth=200)


def VERBOSITY_HIGH():
    # return True # set when debugging
    return False


def VERBOSITY_ERROR():
    return True


PLOTMARGINALS = False
TESTBIAS = False


def setTESTBIAS(b):
    global TESTBIAS
    TESTBIAS = b


def getTESTBIAS():
    global TESTBIAS
    return TESTBIAS


def setPLOTMARGINALS(b):
    global PLOTMARGINALS
    PLOTMARGINALS = b


class SMCQuantities(object):
    def __init__(self, m_indices, llh, log_prior, theta, pmodel, mk):
        self.pmodel = pmodel  # reference
        self.mk = mk
        self.indices = m_indices
        self.N = m_indices.shape[0]
        # self.log_w_norm = -np.log(self.N * np.ones(self.N)) # log(1/N)
        # self.log_w = np.zeros(self.N)
        self.theta = theta.copy()
        self.log_w = np.full(self.N, -np.log(self.N))
        self.log_w_norm = self.log_w.copy()  # log(1/N)
        self.log_lh = llh.copy()  # np.zeros(self.N)
        self.log_prior = log_prior.copy()  # np.zeros(self.N)
        self.log_Zt = 0  # running total estimate of marginal likelihood
        self.log_ESS = np.log(self.N)  # running total estimate of marginal likelihood
        self.log_CESS = np.log(self.N)  # running total estimate of marginal likelihood

    def updateAfterMutate(self, m_indices, llh, log_prior, theta):
        self.indices = m_indices
        self.N = m_indices.shape[0]
        self.theta = theta.copy()
        self.log_lh = llh.copy()
        self.log_prior = log_prior.copy()
        self.log_w = np.full(self.N, -np.log(self.N))
        self.log_w_norm = self.log_w.copy()  # log(1/N)
        self.log_ESS = np.log(self.N)  # running total estimate of marginal likelihood
        self.log_CESS = np.log(self.N)  # running total estimate of marginal likelihood

    def updateAcceptanceRate(self, a):
        self.ar = a

    def getAcceptanceRate(self):
        """
        for now get a summary acceptance rate
        """
        return self.ar

    def makePPPD(self, gamma_t):
        return PowerPosteriorParticleDensity(
            self.pmodel,
            self.mk,
            self.log_lh,
            self.log_prior,
            self.theta,
            gamma_t,
            self.log_w,
        )

    def __str__(self):
        return "log_Zt: {}\nindices: {}\nN: {}\nlog_w_norm: {}\nlog_w: {}\nlog_ESS: {}".format(
            self.log_Zt, self.indices, self.N, self.log_w_norm, self.log_w, self.log_ESS
        )

    def __repr__(self):
        return self.__str__()
        # return self.__dict__


class WSSMCQuantities(SMCQuantities):
    def makePPPD(self, gamma_t, log_D_t=None):
        if log_D_t is None:
            log_D_t = {mk: 0 for mk in self.pmodel.getModelKeys()}
        return WeightedPowerPosteriorParticleDensity(
            self.pmodel,
            self.mk,
            self.log_lh,
            self.log_prior,
            self.theta,
            gamma_t,
            self.log_w,
            log_D_t,
        )


class StratifiedSMC(object):
    def __init__(self, T, parametric_model, starting_distribution="prior"):
        """
        ESS-Adaptive Static Sequential Monte Carlo Sampler

        T                     : number of intermediate distributions
        parametric_model      : a ParametricModelSpace object defining the space of parameters and models
        starting_distribution : string to state whether sampler starts from prior,
                                or a RandomVariableBlock object that matches the parametric_model
        """
        self.pmodel = parametric_model
        self.T = T
        self.nmutations = 0
        if starting_distribution == "prior":
            self.starting_dist = self.pmodel
        else:
            assert self.pmodel.islike(starting_distribution)
            self.starting_dist = starting_distribution
        # self.start_from_prior = (starting_distribution=='prior')
        assert isinstance(self.pmodel, ParametricModelSpace)
        self.pmodel.setStartingDistribution(self.starting_dist)
        # self.compute_llh = lh_function
        # init future things
        self.init_more()

    def init_more(self):
        pass

    def run(self, N=100, ess_threshold=0.5):
        """
        Sample from prior
        For t=1,...,T
            Importance Sample
            Resample
            Mutate

        Particles (theta) are represented as 2D matrix (N-particles,Theta-dim)
        Uses adaptive resampling at ESS < threshold * N
        """
        self.rbar_list = []
        # init memory for particles
        theta = np.zeros((N, self.pmodel.dim()))
        llh = np.zeros(N)  # log likelihood
        log_prior = np.zeros(N)  # log prior
        # theta[:] = self.pmodel.sampleFromPrior(N)
        theta[:] = self.starting_dist.draw(N)
        # compute initial likelihood
        llh[:] = self.pmodel.compute_llh(theta)
        log_prior[:] = self.pmodel.compute_prior(theta)
        # init the SMC quantities: weights and Z
        models_grouped_by_indices = self.enumerateModels(theta)
        if VERBOSITY_HIGH():
            print("Model identifiers", models_grouped_by_indices.keys())
        smc_quantities = {}
        for k, m_indices in models_grouped_by_indices.items():
            smc_quantities[k] = SMCQuantities(
                m_indices,
                llh[m_indices],
                log_prior[m_indices],
                theta[m_indices],
                self.pmodel,
                k,
            )
        self.appendToMixtureTargetDensity(smc_quantities, llh, log_prior, theta, 0.0)
        # iterate over sequence of distributions
        if True:
            t = 0.0

            def ESSThreshold(new_t, last_t, ess_threshold, llh, smcq, pmodel):
                log_ESS_collected = []
                for k, smcq in smc_quantities.items():
                    # importance sample
                    # new_gamma = new_t - last_t
                    # llh_gamma_delta = llh[smcq.indices] * new_gamma
                    # log_target_ratio = (new_t-last_t) * (llh[smcq.indices] + log_prior[smcq.indices]) + ((1-new_t)-(1-last_t)) * pmodel.evalStartingDistribution(smcq.theta[smcq.indices])
                    log_target_ratio = (new_t - last_t) * (
                        llh[smcq.indices] + log_prior[smcq.indices]
                    ) + ((1 - new_t) - (1 - last_t)) * pmodel.evalStartingDistribution(
                        theta[smcq.indices]
                    )
                    log_w = smcq.log_w_norm + log_target_ratio
                    log_w_sum = logsumexp(log_w)
                    log_w_norm = log_w - log_w_sum
                    log_ESS = -logsumexp(2 * log_w_norm)
                    log_ESS_collected.append(log_ESS)
                # compute total ESS.
                log_ESS_total = logsumexp(np.array(log_ESS_collected))
                # print("in situ ess total",np.exp(log_ESS_total))
                return np.exp(log_ESS_total) - ess_threshold

            while t < self.T:
                last_t = t
                if (
                    ESSThreshold(
                        1.0,
                        last_t / self.T,
                        N * ess_threshold,
                        llh,
                        smc_quantities,
                        self.pmodel,
                    )
                    < 0
                ):
                    next_t, rres = scipy.optimize.bisect(
                        ESSThreshold,
                        t / self.T,
                        1.0,
                        args=(
                            last_t / self.T,
                            N * ess_threshold,
                            llh,
                            smc_quantities,
                            self.pmodel,
                        ),
                        full_output=True,
                        rtol=1e-10,
                    )
                    # print("Next t",next_t,"rres",rres)
                    t = next_t * self.T
                else:
                    t = self.T
                # apply to smc_quantities
                log_ESS_collected = []
                for k, smcq in smc_quantities.items():
                    # importance sample
                    new_gamma = t / self.T - last_t / self.T
                    # llh_gamma_delta = llh[smcq.indices] * new_gamma
                    log_target_ratio = new_gamma * (
                        llh[smcq.indices] + log_prior[smcq.indices]
                    ) - new_gamma * self.pmodel.evalStartingDistribution(
                        theta[smcq.indices]
                    )
                    smcq.log_w[:] = smcq.log_w_norm + log_target_ratio
                    log_w_sum = logsumexp(smcq.log_w)
                    smcq.log_Zt += log_w_sum
                    if VERBOSITY_HIGH():
                        print("logZt[{},{}]={}".format(k, t / self.T, smcq.log_Zt))
                    smcq.log_w_norm[:] = smcq.log_w - log_w_sum
                    smcq.log_ESS = -logsumexp(2 * smcq.log_w_norm)
                    log_ESS_collected.append(smcq.log_ESS)
                log_ESS_total = logsumexp(np.array(log_ESS_collected))

                # CALIBRATE HERE
                self.appendToMixtureTargetDensity(
                    smc_quantities, llh, log_prior, theta, float(t) / self.T
                )
                self.pmodel.calibrateProposalsMMMPD(
                    self.getMixtureTargetDensity(), N, float(t) / self.T
                )
                self.getMixtureTargetDensity()

                if VERBOSITY_HIGH():
                    print(
                        "Resampling at t = {}, Total ESS = {}...".format(
                            t, np.exp(log_ESS_total)
                        )
                    )
                # resample within groupings
                for k, smcq in smc_quantities.items():
                    smcq.resample_indices = self.resample(
                        np.exp(smcq.log_w_norm), smcq.N
                    )
                    resample_indices_global = smcq.indices[smcq.resample_indices]
                    theta[smcq.indices] = theta[resample_indices_global]
                    llh[smcq.indices] = llh[resample_indices_global]
                    log_prior[smcq.indices] = log_prior[resample_indices_global]
                if VERBOSITY_HIGH():
                    print(
                        "Calibrating at t = {}, Total ESS = {}...".format(
                            t, np.exp(log_ESS_total)
                        )
                    )
                # mutate
                theta[:], llh[:], log_prior[:], accepted = self.mutate(
                    theta, llh, log_prior, N, t, smc_quantities
                )
                # update the smc quantities
                models_grouped_by_indices = self.enumerateModels(theta)
                for k, m_indices in models_grouped_by_indices.items():
                    if k not in smc_quantities:
                        smc_quantities[k] = SMCQuantities(
                            m_indices,
                            llh[m_indices],
                            log_prior[m_indices],
                            theta[m_indices],
                            self.pmodel,
                            k,
                        )
                    else:
                        smc_quantities[k].updateAfterMutate(
                            m_indices,
                            llh[m_indices],
                            log_prior[m_indices],
                            theta[m_indices],
                        )
                # Set this posterior to use the final particles after mutation
                # print("Appending again")
                # self.appendToMixtureTargetDensity(smc_quantities, llh, log_prior, theta, float(t)/self.T)
        return (
            theta,
            smc_quantities,
            llh,
            log_prior,
            self.getMixtureTargetDensity(),
            self.rbar_list,
        )

    def computeMixtureTargetDensityOld(
        self, smc_quantities, llh, log_prior, theta, gamma_t, resample=True
    ):
        """
        Deprecated
        """
        n = calibrate_theta.shape[0]
        if resample:
            calibrate_theta = np.zeros_like(theta)
            for k, smcq in smc_quantities.items():
                resample_indices = self.resample(np.exp(smcq.log_w_norm), smcq.N)
                resample_indices_global = smcq.indices[resample_indices]
                calibrate_theta[smcq.indices] = theta[resample_indices_global]
            return calibrate_theta, np.full(n, 1.0 / n)
        else:
            return theta, np.full(n, 1.0 / n)

    def computeMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t, resample=True
    ):
        """
        Deprecated
        """
        # init previous_model_targets each time
        previous_model_targets = MultiModelMPD()
        for k, smcq in smc_quantities.items():
            # previous_model_targets.addComponent(k,PowerPosteriorParticleDensity(smcq,llh,log_prior,theta,gamma_t))
            previous_model_targets.addComponent(k, smcq.makePPPD(gamma_t))
        return previous_model_targets.getParticleDensityForTemperature(
            gamma_t, resample
        )

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        # init previous_model_targets each time
        self.previous_model_targets = MultiModelMPD()
        for k, smcq in smc_quantities.items():
            self.previous_model_targets.addComponent(k, smcq.makePPPD(gamma_t))

    def getMixtureTargetDensity(self):
        return self.previous_model_targets

    def enumerateModels(self, theta):
        """
        Associate each model with a key. Typically in a single rjmcmc scheme this would be a tuple of the number of layers (n,)
        """
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        return model_key_dict

    def reverseEnumerateModels(self, theta):
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        return reverse_key_ref

    def getModelKeyArray(self, theta):
        """
        return numpy array of model keys of each row of theta
        """
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        mk_keys = model_key_dict.keys()
        ncols = len(list(mk_keys)[0])
        keyarray = np.zeros((theta.shape[0], ncols))
        for mk, idx in model_key_dict.items():
            keyarray[idx] = np.array(list(mk))
        return keyarray

    def resample(self, weights, n):
        # n = weights.shape[0]
        # print("weights sum = ",weights.sum(),n)
        indices = np.zeros(n, dtype=np.int32)
        # indices = np.zeros_like(weights,dtype=np.int32)
        # C = [0.] + [sum(weights[:i+1]) for i in range(n)]
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
            indices[i] = j
        return indices

    def mutate(self, theta, llh, log_prior, N, t, smc_quantities):
        global TESTBIAS
        global PLOTMARGINALS
        # switch to say whether to use total acceptance rate or min acceptance rate
        use_arn_total = False
        """
        Uses Rt method (Drovandi & Pettitt)
        R_t = ceil(log(0.01) / log(1 - ar))
        where 0<=ar<=1 is the acceptance rate 
        TODO pass in smc_quantities so that the ESS for each model density can be used to mix the independent proposals with the priors
        this will probably need to be passed into self.pmodel.proposal.setSMCQuantities(smc_quantities) or something similar
        """
        # pilot run with acceptance rate
        ar = 0.44
        new_theta = np.zeros_like(theta)
        new_llh = np.zeros_like(llh)
        new_log_prior = np.zeros_like(log_prior)
        new_theta[:] = theta
        new_llh[:] = llh
        new_log_prior[:] = log_prior
        ar_total = 1.0
        arn_total = 0
        arn_repeats = 3
        min_ar = 1.0
        for i in range(arn_repeats):
            new_theta[:], new_llh[:], new_log_prior[:], accepted = self.single_mutation(
                new_theta, new_llh, N, t
            )
        for pid, prop in Proposal.idpropdict.items():
            if VERBOSITY_HIGH():
                print(
                    "After trial mutation prop pid {}\tprop {}\tar {}".format(
                        pid,
                        prop.printName(),
                        prop.getAvgARN(arn_repeats) / (N * arn_repeats),
                    )
                )
            this_ar = prop.getAvgARN(arn_repeats) / (N * arn_repeats)
            arn_total += this_ar
            min_ar = min(min_ar, prop.getLastAR())
        if use_arn_total:
            ar_total = arn_total
        else:
            ar_total = min_ar
        # TODO compute minimum acceptance ratio given each proposal type
        if VERBOSITY_HIGH():
            print("Acceptance rate = {}".format(ar_total))
        # R_t = int(np.ceil(np.log(0.001) / np.log(1 - ar_total)))
        R_t = np.ceil(np.log(0.001) / np.log(1 - ar_total))
        if np.isfinite(R_t):
            R_t = int(R_t)
        else:
            R_t = 50
        if TESTBIAS:
            R_t = max(100, R_t)
        elif True:
            R_t = min(50, R_t)
        if VERBOSITY_HIGH():
            print("mutating {} times".format(R_t))
        total_accepted = np.zeros(N) + accepted
        # include pilot run and do remaining runs
        for r in range(R_t):
            new_theta[:], new_llh[:], new_log_prior[:], accepted = self.single_mutation(
                new_theta, new_llh, N, t
            )
            total_accepted += accepted
        return (
            new_theta,
            new_llh,
            new_log_prior,
            total_accepted,
        )  # what are we doing with the accepted totals? Also include proposals?

    def single_mutation(self, theta, llh, N, t):
        """
        By default use RWMH kernel. Proposals are set in __init__ as proposal_strategy.
        """
        global PLOTMARGINALS
        kblocks, rev = self.pmodel.enumerateModels(theta)
        # print("kblocks",kblocks)
        for mk, mkidx in kblocks.items():
            if VERBOSITY_HIGH():
                print(
                    "Single mutation count {} particles for model {}".format(
                        mkidx.shape[0], mk
                    )
                )  # n,k
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros_like(llh)
        prop_llh = np.full(llh.shape, np.NINF)
        cur_prior = np.zeros_like(llh)
        prop_prior = np.zeros_like(llh)
        prop_id = np.zeros_like(llh)
        prop_lpqratio = np.zeros_like(llh)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        # propose
        # Proposal.clearIDs()
        # if float(t)/self.T >= 0.9 :
        #   PLOTMARGINALS = True
        prop_theta[:], prop_lpqratio[:], prop_id[:] = self.pmodel.propose(
            theta, N
        )  # TODO FIXME self.pmodel.propose(theta,N)
        cur_prior[:] = self.pmodel.compute_prior(theta)
        prop_prior[:] = self.pmodel.compute_prior(prop_theta)
        # print(prop_prior)
        ninfprioridx = np.where(~np.isfinite(cur_prior))
        # print("ninf cur_prior indices",ninfprioridx)
        # print("ninf curprior theta",theta[ninfprioridx])
        # print("prop_prior",prop_prior)
        # print("cur_prior",cur_prior)
        # sys.exit(0)
        # TODO prior check!
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        # print("update llh indices",np.isfinite(prop_prior))
        valid_theta = np.logical_and(
            np.isfinite(prop_prior), np.isfinite(prop_lpqratio)
        )
        # prop_llh[np.isfinite(prop_prior)] = self.pmodel.compute_llh(prop_theta[np.isfinite(prop_prior),:])
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        # print("prop llh",prop_llh)
        # sys.exit(0)

        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta,
            prop_theta,
            prop_lpqratio,
            prop_llh,
            llh,
            cur_prior,
            prop_prior,
            float(t) / self.T,
        )  # TODO implement

        # store acceptance ratios
        # self.rbar_list.append(rbar(log_acceptance_ratio,float(t)/self.T,self.reverseEnumerateModels(theta),self.reverseEnumerateModels(prop_theta)))
        self.rbar_list.append(
            rbar(
                log_acceptance_ratio,
                float(t) / self.T,
                self.getModelKeyArray(theta),
                self.getModelKeyArray(prop_theta),
            )
        )

        Proposal.setAcceptanceRates(prop_id, log_acceptance_ratio, float(t) / self.T)
        for pid, prop in Proposal.idpropdict.items():
            if VERBOSITY_HIGH():
                print(
                    "for pid {}\tprop {}\tar {}".format(
                        pid, prop.printName(), prop.getLastAR()
                    )
                )
            # pididx = prop_id==pid
            # print("prop_llh,prop_prior,prop_lpqratio\n",np.column_stack([prop_llh[pididx],prop_prior[pididx],prop_lpqratio[pididx]]))
            # if t > 9000:
            #    ParametricModelSpace.plotJoints(theta[pid==prop_id,:],prop_theta[pid==prop_id,:],prop.printName())
            #    #self.pmodel.plotJoints(theta[pid==prop_id,:],prop_theta[pid==prop_id,:],prop.printName())
            # if self.nmutations > 80:
            #    self.pmodel.plotJoints(theta[pid==prop_id,:],prop_theta[pid==prop_id,:],prop.printName())

        # Print acceptance ratios
        # print(Proposal.idnamedict)
        # ar_dict_name = {} # name : ar
        # ar_dict = {} # pid : ar
        # unique_pids, pid_indices = np.unique(prop_id, return_inverse=True)
        # for i,pid in enumerate(unique_pids):
        #    k = Proposal.idnamedict[int(pid)]
        #    if k not in ar_dict_name:
        #        ar_dict_name[k] = []
        #    ar_dict[int(pid)] =np.exp(logsumexp(log_acceptance_ratio[pid_indices==i]))/np.sum(pid_indices==i)
        #    ar_dict_name[k].append(ar_dict[int(pid)])
        #    #print("lar {} {}\n{}".format(k,logsumexp(log_acceptance_ratio[pid_indices==i]),log_acceptance_ratio[pid_indices==i]))
        # for propname,propar in ar_dict_name.items():
        #    print("{}\t: {}\n".format(propname,propar))
        # sys.exit(0)
        # accept/reject
        log_u = np.log(uniform.rvs(0, 1, size=N))
        # reject_indices = (prop_llh - llh < log_u) | ~np.isfinite(prop_llh)
        reject_indices = log_acceptance_ratio < log_u
        prop_theta[reject_indices] = theta[reject_indices]
        prop_llh[reject_indices] = llh[reject_indices]
        prop_prior[reject_indices] = cur_prior[reject_indices]
        # a boolean array of accepted proposals
        accepted = np.ones(N)
        accepted[reject_indices] = 0
        # return prop_theta,prop_llh,accepted
        # print("log acceptance ratio",log_acceptance_ratio)
        self.nmutations += 1
        return prop_theta, prop_llh, prop_prior, np.exp(log_acceptance_ratio)


class PowerPosteriorParticleDensity(object):
    # def __init__(self,smcq,llh,log_prior,theta,gamma_t):
    #    self.smcq = deepcopy(smcq) # could just need copy.
    #    self.theta = theta[smcq.indices].copy() # theta.copy()
    #    self.llh = llh[smcq.indices].copy() # llh.copy()
    #    self.log_prior = log_prior[smcq.indices].copy() #log_prior.copy()
    #    self.gamma_t = gamma_t
    def __init__(self, pmodel, mk, llh, log_prior, theta, gamma_t, original_log_w):
        if mk is not None:
            mk_list = pmodel.getModelKeys(theta)
            assert mk in mk_list
            assert len(mk_list) == 1
            # Do we assign mk to self?
        self.theta = theta.copy()  # theta.copy()
        self.llh = llh.copy()  # llh.copy()
        self.log_prior = log_prior.copy()  # log_prior.copy()
        self.gamma_t = gamma_t
        self.pmodel = pmodel
        self.original_log_w = original_log_w.copy()

    def size(self):
        return self.llh.shape[0]

    def dim(self):
        """need to ensure theta dimensions == 2. Could come unstuck in particle impoverishment scenarios."""
        return self.theta.shape[1]

    def log_Zt_Zt1(self, gamma_t_1):
        delta_gamma = self.gamma_t - gamma_t_1
        lZtZt1 = logsumexp(
            delta_gamma * (self.llh + self.log_prior)
            - delta_gamma * self.pmodel.evalStartingDistribution(self.theta)
        ) - np.log(self.size())
        # print("log_Zt_Zt1 for gamma_t (",gamma_t_1,self.gamma_t,") is ",lZtZt1, ", deltagamma=",delta_gamma)
        return lZtZt1
        # print("log_Zt_Zt1 for gamma_t (",gamma_t_1,self.gamma_t,") is ",logsumexp(self.original_log_w))
        # print("log_Zt_Zt1({})={}".format(gamma_t_1,logsumexp(self.original_log_w)))
        # return logsumexp(self.original_log_w)

    def log_Zt_Zt1_k(self, k, gamma_t_1):
        """
        TODO deprecate
        """
        log_Zt_Zt1_k = 0
        mkdict, rev = self.pmodel.enumerateModels(self.theta)
        delta_gamma = self.gamma_t - gamma_t_1
        if k in mkdict.keys():
            # log_proportion = np.log(mkdict[k].shape[0]) - np.log(self.theta.shape[0])
            # log_Zt_Zt1_k = logsumexp(self.original_log_w[mkdict[k]]) - log_proportion
            weights = delta_gamma * (
                self.llh[mkdict[k]] + self.log_prior[mkdict[k]]
            ) - delta_gamma * self.pmodel.evalStartingDistribution(
                self.theta[mkdict[k]]
            )
            log_Zt_Zt1_k = logsumexp(weights) - np.log(mkdict[k].shape[0])
        # else:
        #    print("FAILED key ",k," not in mkdict",list(mkdict.keys()))
        #    sys.exit(0)
        # print("log_Zt_Zt1_k for k",k,"gamma_t",gamma_t_1,self.gamma_t,"is",log_Zt_Zt1_k)
        return log_Zt_Zt1_k


class WeightedPowerPosteriorParticleDensity(PowerPosteriorParticleDensity):
    def __init__(
        self, pmodel, mk, llh, log_prior, theta, gamma_t, original_log_w, log_D_t
    ):
        self.log_D_t = log_D_t  # TODO assert has all model keys in pmodel
        super(WeightedPowerPosteriorParticleDensity, self).__init__(
            pmodel, mk, llh, log_prior, theta, gamma_t, original_log_w
        )

    def getLogDt(self, theta=None):
        if theta is None:
            theta = self.theta
        mk_indices, rev = self.pmodel.enumerateModels(theta)
        log_D_t_vec = np.zeros(theta.shape[0])
        for mk, mkidx in mk_indices.items():
            log_D_t_vec[mkidx] = self.log_D_t[mk]
        return log_D_t_vec


class MixtureParticleDensity(object):
    def __init__(self, mk_id=None):
        self.mixture_components = {}
        self.target_weights = {}
        self.target_log_unnorm_weights = {}
        self.mk_id = mk_id

    def getComponentKeys(self):
        return self.mixture_components.keys()

    def log_Zt(self, gamma_t=1):
        """
        Returns estimate of log_Zt at returned last_gamma_t
        input arg gamma_t is default 1, used as cutoff for density used in estimation.
        """
        last_gamma_t = 0
        n_components = len(self.mixture_components)
        log_Zt_Zt1 = np.zeros(
            n_components
        )  # don't need +1 because we pushed the gamma_0 state
        # print("sorted(self.mixture_components.items())",sorted(self.mixture_components.items()))
        for i, (k, c) in enumerate(sorted(self.mixture_components.items())):
            if c.gamma_t > gamma_t:
                continue
            log_Zt_Zt1[i] = c.log_Zt_Zt1(last_gamma_t)
            last_gamma_t = c.gamma_t
        log_Zt = np.cumsum(log_Zt_Zt1)
        return log_Zt, last_gamma_t

    def log_Zt_k(self, mk, gamma_t=1):
        """
        TODO Deprecate, I don't think this works.
        Returns estimate of log_Zt at returned last_gamma_t
        input arg gamma_t is default 1, used as cutoff for density used in estimation.
        """
        last_gamma_t = 0
        n_components = len(self.mixture_components)
        log_Zt_Zt1_k = np.zeros(
            n_components
        )  # don't need +1 because we pushed the gamma_0 state
        for i, (k, c) in enumerate(sorted(self.mixture_components.items())):
            if c.gamma_t > gamma_t:
                break
            log_Zt_Zt1_k[i] = c.log_Zt_Zt1_k(mk, last_gamma_t)
            last_gamma_t = c.gamma_t
        log_Zt_k = np.cumsum(log_Zt_Zt1_k)
        return log_Zt_k, last_gamma_t

    def nComponentsTo(self, gamma_t):
        n = 0
        for gt in self.mixture_components.keys():
            if gt < gamma_t:
                n += 1
        return n

    def addComponent(self, pppd):
        # TODO either move the pmodel ref from pppd to this class or assert it matches all others in the list
        self.mixture_components[
            pppd.gamma_t
        ] = pppd  # replaces any pppd at this temperature
        self._recomputeWeights(pppd.gamma_t)

    def deleteComponentAt(self, gamma_t):
        assert gamma_t in self.mixture_components.keys()
        # TODO either move the pmodel ref from pppd to this class or assert it matches all others in the list
        del self.mixture_components[gamma_t]  # replaces any pppd at this temperature
        # recompute at last weight
        if len(self.mixture_components.keys()) > 0:
            gamma_list = sorted(self.mixture_components.keys())
            self._recomputeWeights(gamma_list[-1])

    def _recomputeWeights(self, gamma_t):
        assert gamma_t in self.mixture_components.keys()
        # print(self.mixture_components.keys())
        assert 0 in self.mixture_components.keys()
        # get gamma index
        gamma_list = sorted(self.mixture_components.keys())
        # print("gamma list",gamma_list)
        nth_component = gamma_list.index(gamma_t)
        # print("nth component",nth_component)
        n_components = nth_component + 1
        # re-compute weights
        log_w = (
            []
        )  # we use a list instead of a 2D array because each target will have a different number of particles
        w = []
        log_w = []
        log_Zt, last_gamma_t = self.log_Zt()
        if VERBOSITY_HIGH():
            print("_recomputeWeights log_Z[{}]={}".format(gamma_t, log_Zt))
        extradensity = {}
        # for i,(k,c) in enumerate(sorted(self.mixture_components.items())):
        # for i,k in enumerate(gamma_list):
        for i in range(n_components):
            k = gamma_list[i]
            c = self.mixture_components[k]
            extradensity[i] = np.zeros((c.size(), n_components))
            # print("extra density shape",extradensity[i].shape)
            # for j,(k2,c2) in enumerate(sorted(self.mixture_components.items())):
            # for j,k2 in enumerate(gamma_list):
            for j in range(n_components):
                k2 = gamma_list[j]
                c2 = self.mixture_components[k2]
                extradensity[i][:, j] = (
                    c2.gamma_t * (c.llh + c.log_prior)
                    + (1 - c2.gamma_t) * c.pmodel.evalStartingDistribution(c.theta)
                    - log_Zt[j]
                )
        # for i,(k,c) in enumerate(sorted(self.mixture_components.items())):
        # for i,k in enumerate(gamma_list):
        for i in range(n_components):
            k = gamma_list[i]
            c = self.mixture_components[k]
            log_w_t = (
                gamma_t * (c.llh + c.log_prior)
                + (1 - gamma_t) * c.pmodel.evalStartingDistribution(c.theta)
                - (-np.log(n_components) + logsumexp(extradensity[i], axis=1))
            )  # I hope I'm getting the number of particles right.

            # print("Recomputed weights")
            # print(i,log_w_t)
            # print(i,"firstbit",gamma_t * (c.llh + c.log_prior) + (1-gamma_t) * c.pmodel.evalStartingDistribution(c.theta))
            # print(i,"secondbit",- (-np.log(n_components) + logsumexp(extradensity[i],axis=1)))
            # print(i,"gamma_t",gamma_t,"\nllh",c.llh,"\nlogprior",c.log_prior,"\nevalStartingDist",c.pmodel.evalStartingDistribution(c.theta),"\nextradensity.sum",logsumexp(extradensity[i],axis=1),"\nlog_ncomponents",-np.log(n_components))
            # plt.plot(log_w_t)
            # plt.title("log_w_t")
            # plt.show()
            # log_w_t = gamma_t * c.llh + c.log_prior - (-np.log(c.size()*(i+1)) + logsumexp(np.log(c.size())+extradensity[i][:,:i+1],axis=1)) # I hope I'm getting the number of particles right.
            # log_w_t = c.gamma_t * c.llh + c.log_prior - (-np.log(n_components) + logsumexp(extradensity[i],axis=1)) # I hope I'm getting the number of particles right.
            # log_w_t = pppd.gamma_t * c.llh + c.log_prior - (-np.log(n_components) + logsumexp(extradensity[i],axis=1)) # I hope I'm getting the number of particles right.
            # w.append(np.exp(log_w_t-logsumexp(log_w_t)))
            log_w.append(log_w_t)
            # log_w_sum[i]=logsumexp(log_w_t)
        # normalise the weights
        # denom = logsumexp(log_w_sum)
        denom = logsumexp(np.concatenate(log_w))
        for i, lw in enumerate(log_w):
            # w.append(np.exp(lw))
            w.append(np.exp(lw - denom))
        self.target_weights[gamma_t] = w
        self.target_log_unnorm_weights[gamma_t] = log_w
        # print("Model",self.mk_id,"at gamma_t",gamma_t," Non recycled logZ",log_Zt," recycled logZ",logsumexp(np.concatenate(log_w))-np.log(np.concatenate(log_w).shape[0]))

    def getOriginalParticleDensityForTemperature(self, gamma_t, normalise_weights=True):
        """returns weighted particle density"""
        c = self.mixture_components[gamma_t]
        if normalise_weights:
            return c.theta, c.original_log_w - logsumexp(c.original_log_w)
        else:
            return c.theta, c.original_log_w

    def getParticleDensityForTemperature(
        self, gamma_t, normalise_weights=True, return_logpdfs=False
    ):
        """returns weighted particle density"""
        self._recomputeWeights(gamma_t)
        size = 0
        firstkey = list(self.mixture_components.keys())[0]
        dim = self.mixture_components[firstkey].dim()
        if normalise_weights:
            target_weights = self.target_weights
        else:
            target_weights = self.target_log_unnorm_weights
        # for i,(k,c) in enumerate(sorted(self.mixture_components.items())):
        #    #print("size ",c.size()," for temp ",c.gamma_t)
        #    #size += c.size()
        # print(target_weights.keys(),gamma_t)
        for i, c in enumerate(target_weights[gamma_t]):
            size += c.shape[0]
        target = np.zeros((size, dim))
        target_w = np.zeros(size)
        if return_logpdfs:
            target_llh = np.zeros(size)
            target_log_prior = np.zeros(size)
        offset = 0
        for i, (k, c) in enumerate(sorted(self.mixture_components.items())):
            if i >= len(target_weights[gamma_t]):
                break
            # print("Getting weights for target ",i,k)
            target[offset : offset + c.size(), :] = c.theta
            target_w[offset : offset + c.size()] = target_weights[gamma_t][i]
            if return_logpdfs:
                target_llh[offset : offset + c.size()] = c.llh
                target_log_prior[offset : offset + c.size()] = c.log_prior
            offset += c.size()
        if return_logpdfs:
            return target, target_w, target_llh, target_log_prior
        else:
            return target, target_w


class WeightStabilisedMixtureParticleDensity(MixtureParticleDensity):
    """
    A MixtureParticleDensity class that is aware of the joint target of models and parameters
    and will compute stabilising weights D_{t,k} for each model k in K at each target pi_t.

    This class assumes that any PPPD in the mixture_components dict is a WeightedPowerPosteriorParticleDensity
    """

    def _recomputeWeights(self, gamma_t):
        assert gamma_t in self.mixture_components.keys()
        assert 0 in self.mixture_components.keys()
        if VERBOSITY_HIGH():
            print(
                "WeightStabilisedMixtureParticleDensity._recomputeWeights({})".format(
                    gamma_t
                )
            )
        # TODO assert that gamma_t is greater than all existing gamma_t in mixture_components.keys()
        # first check if there is more than one model. If there isn't, call the parent method.
        pmodel = self.mixture_components[0].pmodel
        mk_list = pmodel.getModelKeys()
        if len(mk_list) == 1:
            # case where we are doing single model SMC and don't use weight-stabilising correction.
            return super(
                WeightStabilisedMixtureParticleDensity, self
            )._recomputeWeights(gamma_t)

        # get gamma index
        gamma_list = sorted(self.mixture_components.keys())

        # If we are here, there is more than one model in mk_list.
        # indexed by gamma_t, then by model index mk
        # self.log_D_t = {temp:{mk:0 for mk in mk_list} for temp in gamma_list}
        # stratify by model, then compute weights for each?..... wait,
        # all I need to do is index by model on c.theta, then set D accordingly

        nth_component = gamma_list.index(gamma_t)
        n_components = nth_component + 1
        # re-compute weights
        log_w = (
            []
        )  # we use a list instead of a 2D array because each target will have a different number of particles
        w = []
        log_w = []
        log_Zt, last_gamma_t = self.log_Zt()
        if VERBOSITY_HIGH():
            print("WSOPR._recomputeWeights log_Z[{}]={}".format(gamma_t, log_Zt))
        extradensity = {}
        for i in range(n_components):
            k = gamma_list[i]
            c = self.mixture_components[k]
            extradensity[i] = np.zeros((c.size(), n_components))
            # set weights given theta
            for j in range(n_components):
                k2 = gamma_list[j]
                c2 = self.mixture_components[k2]
                log_D_t = c2.getLogDt(c.theta)
                extradensity[i][:, j] = (
                    log_D_t
                    + c2.gamma_t * (c.llh + c.log_prior)
                    + (1 - c2.gamma_t) * c.pmodel.evalStartingDistribution(c.theta)
                    - log_Zt[j]
                )
        for i in range(n_components):
            k = gamma_list[i]
            c = self.mixture_components[k]
            # set weights given theta
            log_D_t = c.getLogDt()
            log_w_t = (
                log_D_t
                + gamma_t * (c.llh + c.log_prior)
                + (1 - gamma_t) * c.pmodel.evalStartingDistribution(c.theta)
                - (-np.log(n_components) + logsumexp(extradensity[i], axis=1))
            )  # I hope I'm getting the number of particles right.
            log_w.append(log_w_t)
        # normalise the weights
        denom = logsumexp(np.concatenate(log_w))
        for i, lw in enumerate(log_w):
            w.append(np.exp(lw - denom))
        self.target_weights[gamma_t] = w
        self.target_log_unnorm_weights[gamma_t] = log_w


class MultiModelMPD(object):
    def __init__(self):
        self.densities = {}

    def getTemperatures(self):
        dk = list(self.densities.keys())
        return list(self.densities[dk[0]].getComponentKeys())

    def addComponent(self, k, pppd):
        # TODO assert(k is a model key)
        if k not in self.densities:
            self.densities[k] = MixtureParticleDensity(mk_id=k)
        self.densities[k].addComponent(pppd)

    def getModelKeys(self):
        """
        Just return keys of densities
        """
        return self.densities.keys()

    def resample(self, weights, n):
        # n = weights.shape[0]
        # print("weights sum = ",weights.sum(),n)
        indices = np.zeros(n, dtype=np.int32)
        # indices = np.zeros_like(weights,dtype=np.int32)
        # C = [0.] + [sum(weights[:i+1]) for i in range(n)]
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
            indices[i] = j
        return indices

    def getlogZForModelAndTemperature_old(self, k, gamma_t):
        target, targetw = self.densities[k].getParticleDensityForTemperature(
            gamma_t, normalise_weights=False
        )
        # print("weights sum",logsumexp(targetw),"n",targetw.shape,"gamma_t",gamma_t)
        return logsumexp(targetw) - np.log(
            targetw.shape[0]
        )  # shape of targetw should be N*T, T is the number of components in the power posterior.
        # return logsumexp(targetw) - np.log(self.densities[k].nComponentsTo(gamma_t)*targetw.shape[0])

    def getlogZForModelAndTemperature(self, k, gamma_t):
        logZt, last_gamma_t = self.densities[k].log_Zt(gamma_t)
        # print("LogZt",logZt,"gamma_t",gamma_t)
        return logZt[self.densities[k].nComponentsTo(gamma_t)]

    def getParticleDensityForModelAndTemperature(
        self, k, gamma_t, resample=False, resample_max_size=2000
    ):
        """
        TODO make more Pythonic as an iterator
        """
        if k not in self.densities:
            raise "Unknown model key {}".format(k)
        target_list = []
        target_w_list = []

        target, targetw = self.densities[k].getParticleDensityForTemperature(
            gamma_t, normalise_weights=False
        )
        targetw_norm = np.exp(targetw - logsumexp(targetw))
        if resample:
            # print("weights norm sum",targetw_norm.sum())
            resample_size = min(resample_max_size, targetw_norm.shape[0] * 2)
            target_list.append(target[self.resample(targetw_norm, resample_size)])
        else:
            target_list.append(target)
        target_w_list.append(targetw_norm)

        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack

    def getOriginalParticleDensityForTemperature(
        self, gamma_t, resample=False, resample_max_size=2000
    ):
        target_list = []
        target_w_list = []
        n = 0
        for k, density in self.densities.items():
            target, targetw = density.getOriginalParticleDensityForTemperature(
                gamma_t, normalise_weights=True
            )
            target_list.append(target)
            target_w_list.append(targetw)
            n += target.shape[0]
            # print("target weights for k ",k, " and gamma_t ",gamma_t, " are ",targetw[:20])
        if resample:
            for i, (t, tw) in enumerate(zip(target_list, target_w_list)):
                targetw_norm = np.exp(tw - logsumexp(tw))
                # print("temp ",gamma_t)
                # print("weights norm sum",targetw_norm.sum())
                # print("orig pd size",targetw_norm.shape[0])
                resample_size = int(t.shape[0] * 1.0 / n * resample_max_size)
                target_list[i] = t[self.resample(targetw_norm, resample_size)]
                target_w_list[i] = tw[self.resample(targetw_norm, resample_size)]
        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if VERBOSITY_HIGH():
            print("whole orig pd size", n)
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack

    def getParticleDensityForTemperature_broken(
        self, gamma_t, resample=False, resample_max_size=2000
    ):
        """
        TODO this implementation is broken.
        """
        target_list = []
        target_w_list = []
        for k, density in self.densities.items():
            target, targetw = density.getParticleDensityForTemperature(
                gamma_t
            )  # normalises weights
            if resample:
                resample_size = min(resample_max_size, targetw.shape[0] * 2)
                target_list.append(target[self.resample(targetw, resample_size)])
            else:
                target_list.append(target)
            target_w_list.append(targetw)
            # print("target weights for k ",k, " and gamma_t ",gamma_t, " are ",targetw[:20])
        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        n = target_stack.shape[0]
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack


# Wrapper obect for a single model MPD that behaves as a MultiModelMPD
class SingleModelMPD(MultiModelMPD):
    def __init__(self, pmodel):
        self.pmodel = pmodel
        self.density = MixtureParticleDensity()

    def getTemperatures(self):
        return list(self.density.getComponentKeys())

    def addComponent(self, pppd):
        # TODO assert(k is a model key)
        self.density.addComponent(pppd)

    def deleteComponentAt(self, gamma_t):
        self.density.deleteComponentAt(gamma_t)

    def getModelKeys(self):
        """
        Just return keys of densities
        """
        return self.pmodel.getModelKeys()

    def getlogZForModelAndTemperature(self, k, gamma_t):
        target, targetw = self.density.getParticleDensityForTemperature(
            gamma_t, normalise_weights=False
        )
        # get model k from the target
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(target)
        if k not in model_key_dict.keys():
            raise "Model {} not found in recycled target density.".format(k)
        mk_targetw = targetw[model_key_dict[k]]
        mk_logZOPR = logsumexp(mk_targetw) - np.log(mk_targetw.shape[0])
        total_logZOPR = logsumexp(targetw) - np.log(targetw.shape[0])
        total_logZOPR_list = []
        for mk, idx in model_key_dict.items():
            total_logZOPR_list.append(
                logsumexp(targetw[idx]) - np.log(targetw[idx].shape[0])
            )
        total_logZOPR_2 = logsumexp(np.array(total_logZOPR_list))
        # print("total logZ comparison: ",total_logZOPR, total_logZOPR_2)
        prob_mk_lZOPR = np.exp(mk_logZOPR - total_logZOPR_2)
        # print("weights sum",logsumexp(mk_targetw),"n",mk_targetw.shape,"mk_logZ_OPR",mk_logZOPR,"prob mk lzopr",prob_mk_lZOPR,"gamma_t",gamma_t)
        return mk_logZOPR

    def getSMCLogZForModelAndTemperature(self, k, gamma_t):
        """
        Danger, I don't think this works. TODO deprecate.
        """
        # print("getSMCLogZForModelAndTemperature k gamma_t",k,gamma_t)
        logZtk, last_gamma_t = self.density.log_Zt_k(k, gamma_t)
        # print("logZtk for k=",k," is ",logZtk)
        return logZtk[self.density.nComponentsTo(gamma_t)]

    def getSMCLogZForTemperature(self, gamma_t):
        if VERBOSITY_HIGH():
            print("getSMCLogZTemperature gamma_t", gamma_t)
        logZt, last_gamma_t = self.density.log_Zt(gamma_t)
        if VERBOSITY_HIGH():
            print("logZt is ", logZt)
        return logZt[self.density.nComponentsTo(gamma_t)]

    def getParticleDensityForModelAndTemperature(
        self, k, gamma_t, resample=False, resample_max_size=2000
    ):
        """
        TODO make more Pythonic as an iterator
        """
        target_list = []
        target_w_list = []

        target, targetw = self.density.getParticleDensityForTemperature(
            gamma_t, normalise_weights=False
        )
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(target)
        if k not in model_key_dict.keys():
            raise BaseException(
                "Model {} not found in recycled target density.".format(k)
            )
        mk_target = target[model_key_dict[k]]
        mk_targetw = targetw[model_key_dict[k]]
        targetw_norm = np.exp(mk_targetw - logsumexp(mk_targetw))
        if resample:
            # print("weights norm",targetw_norm)
            resample_size = min(resample_max_size, targetw_norm.shape[0] * 2)
            resample_idx = self.resample(targetw_norm, resample_size)
            target_list.append(mk_target[resample_idx])
            target_w_list.append(targetw_norm[resample_idx])
        else:
            # print("not resampling weights norm",targetw_norm)
            target_list.append(mk_target)
            target_w_list.append(targetw_norm)

        target_stack = np.vstack(target_list)
        target_w_stack = np.concatenate(target_w_list)
        # plt.plot(target_w_stack)
        # plt.show()
        n = target_stack.shape[0]
        if resample:
            return target_stack, np.full(n, 1.0 / n)
        else:
            return target_stack, target_w_stack

    def getParticleDensityForTemperature(
        self, gamma_t, resample=False, resample_max_size=2000, return_logpdfs=False
    ):
        if return_logpdfs:
            (
                target,
                targetw,
                targetllh,
                targetlogprior,
            ) = self.density.getParticleDensityForTemperature(
                gamma_t, normalise_weights=False, return_logpdfs=True
            )
            # plt.plot(targetw)
            # plt.show()
            if resample:
                resample_size = min(resample_max_size, targetw.shape[0] * 2)
                targetw_norm = np.exp(targetw - logsumexp(targetw))
                resample_indices = self.resample(targetw_norm, resample_size)
                return (
                    target[resample_indices],
                    np.full(resample_size, 1.0 / resample_size),
                    targetllh[resample_indices],
                    targetlogprior[resample_indices],
                )
            else:
                # plt.plot(targetw)
                # plt.show()
                return target, targetw, targetllh, targetlogprior
        else:
            target, targetw = self.density.getParticleDensityForTemperature(
                gamma_t, normalise_weights=False, return_logpdfs=False
            )
            # plt.plot(targetw)
            # plt.show()
            if resample:
                resample_size = min(resample_max_size, targetw.shape[0] * 2)
                targetw_norm = np.exp(targetw - logsumexp(targetw))
                resample_indices = self.resample(targetw_norm, resample_size)
                return target[resample_indices], np.full(
                    resample_size, 1.0 / resample_size
                )
            else:
                # plt.plot(targetw)
                # plt.show()
                return target, targetw

    def getOriginalParticleDensityForTemperature(
        self, gamma_t, resample=False, resample_max_size=2000
    ):
        target_list = []
        target_w_list = []
        target, targetw = self.density.getOriginalParticleDensityForTemperature(
            gamma_t, normalise_weights=True
        )
        if resample:
            resample_size = min(resample_max_size, targetw.shape[0] * 2)
            targetw_norm = np.exp(targetw - logsumexp(targetw))
            resample_indices = self.resample(targetw_norm, resample_size)
            return target[resample_indices], np.full(resample_size, 1.0 / resample_size)
        else:
            return target, targetw


class WeightedSingleModelMPD(SingleModelMPD):
    def __init__(self, pmodel):
        self.pmodel = pmodel
        self.density = WeightStabilisedMixtureParticleDensity()


# Online particle recycling RJSMC
class StratifiedSMCOPR(StratifiedSMC):
    def init_more(self):
        # come up with a structure for previous targets
        self.previous_model_targets = MultiModelMPD()

    def computeMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t, resample=True
    ):
        """
        Deprecated
        """
        """
        This function needs to return the mixture density object.

        Currently it takes as input the llh, log_prior, theta, gamma_t, and smc_quantities of the current particle set (at step t).
        It outputs a set of particles and associated weights.
        If resample==False, the output weights are the computed weights.
        If resample==True, the output weights are the 1/N where N is the number of particles returned.

        The need for change here is that we are constantly "appending" new PowerPosteriorParticleDensity objects to each MixtureParticleDensity
        Why is this a problem? Um... maybe it isn't a problem.

        What do we need the MixtureParticleDensity objects to do?
        MPD objects need to be input to proposals for calibration. 
            Currently we give proposals weighted particles. 
            An annoyance here is that we are just appending particles for all models to the one ensemble.
        Instead, it would be good to keep models separate.
        """
        for k, smcq in smc_quantities.items():
            # self.previous_model_targets.addComponent(k,PowerPosteriorParticleDensity(smcq,llh,log_prior,theta,gamma_t))
            self.previous_model_targets.addComponent(k, smcq.makePPPD(gamma_t))
        return self.previous_model_targets.getParticleDensityForTemperature(
            gamma_t, resample
        )

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        for k, smcq in smc_quantities.items():
            # self.previous_model_targets.addComponent(k,PowerPosteriorParticleDensity(smcq,llh,log_prior,theta,gamma_t))
            self.previous_model_targets.addComponent(k, smcq.makePPPD(gamma_t))

    def getMixtureTargetDensity(self):
        return self.previous_model_targets


class rbar(object):
    def __init__(self, lar, power, pre_model_ids, post_model_ids):
        self.lar = lar
        self.power = power
        self.pre_model_ids = pre_model_ids
        self.post_model_ids = post_model_ids


class SMC1(object):
    def __init__(
        self,
        parametric_model,
        starting_distribution="prior",
        temperature_sequence=None,
        n_mutations=None,
        store_ar=False,
    ):
        """
        ESS-Adaptive Static Sequential Monte Carlo Sampler

        T                     : number of intermediate distributions
        parametric_model      : a ParametricModelSpace object defining the space of parameters and models
        starting_distribution : string to state whether sampler starts from prior,
                                or a RandomVariableBlock object that matches the parametric_model
        """
        self.pmodel = parametric_model
        self.tempseq = temperature_sequence
        self.store_ar = store_ar
        self.tempseq_done = [
            0.0,
        ]
        self.essseq_done = {0: 0}
        self.mkseq_done = {}
        if self.tempseq is not None:
            assert self.tempseq[0] == 0
            assert self.tempseq[-1] == 1
            assert np.sum(np.array(self.tempseq) > 1) == 0
            assert np.sum(np.array(self.tempseq) < 0) == 0
            assert (
                np.sum((np.array(self.tempseq)[1:] - np.array(self.tempseq)[:1]) < 0)
                == 0
            )
        self.nmutations = 0
        if starting_distribution == "prior":
            self.starting_dist = self.pmodel
        else:
            assert self.pmodel.islike(starting_distribution)
            self.starting_dist = starting_distribution
        if n_mutations is not None:
            assert isinstance(n_mutations, int)
        self.n_mutations = n_mutations
        # self.start_from_prior = (starting_distribution=='prior')
        assert isinstance(self.pmodel, ParametricModelSpace)
        self.pmodel.setStartingDistribution(self.starting_dist)
        # self.compute_llh = lh_function
        # init future things
        self.previous_model_targets = SingleModelMPD(self.pmodel)
        self.init_more()

    def init_more(self):
        pass

    def computeIncrementalWeights(
        self,
        new_t,
        last_t,
        llh,
        log_prior,
        pmodel,
        theta,
        log_w_norm_last_t,
        store_D=False,
    ):
        log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
            (1 - new_t) - (1 - last_t)
        ) * pmodel.evalStartingDistribution(theta)
        return log_target_ratio

    def preImportanceSampleHook(
        self, last_t, llh, log_prior, smcq, pmodel, next_t=None
    ):
        pass

    def ESSThreshold(
        self, new_t, last_t, ess_threshold, llh, log_prior, smcq, pmodel, N
    ):
        # importance sample
        # log_target_ratio = (new_t-last_t) * (llh + log_prior) + ((1-new_t)-(1-last_t)) * pmodel.evalStartingDistribution(smcq.theta)
        log_target_ratio = self.computeIncrementalWeights(
            new_t,
            last_t,
            llh,
            log_prior,
            pmodel,
            smcq.theta,
            smcq.log_w_norm,
            store_D=False,
        )
        log_w = smcq.log_w_norm + log_target_ratio
        log_w_sum = logsumexp(log_w)
        log_w_norm = log_w - log_w_sum
        log_ESS = -logsumexp(2 * log_w_norm)
        if VERBOSITY_HIGH():
            print("in situ ess", np.exp(log_ESS))
        return np.exp(log_ESS) - ess_threshold(new_t, N)

    def makeSMCQuantities(self, N, llh, log_prior, theta, pmodel, mk):
        return SMCQuantities(np.arange(N), llh, log_prior, theta, pmodel, mk)

    def getEmpiricalModelProbs(self, theta):
        kblocks, rev = self.pmodel.enumerateModels(theta)
        Ninv = 1.0 / float(theta.shape[0])
        mkprobs = {mk: len(idx) * Ninv for mk, idx in kblocks.items()}
        # account for zero prob models
        mklist = self.pmodel.getModelKeys()
        for mk in mklist:
            if mk not in mkprobs.keys():
                mkprobs[mk] = 0
        return mkprobs

    def run(self, N=100, ess_threshold=0.5):
        """
        Sample from prior
        For t=0,...,1
            Importance Sample
            Resample
            Mutate

        Particles (theta) are represented as 2D matrix (N-particles,Theta-dim)
        Uses adaptive resampling at ESS < threshold * N
        """
        if isinstance(ess_threshold, float):
            assert ess_threshold < 1 and ess_threshold > 0

            def ess_th_fn(t, N):
                return ess_threshold * N

        else:
            assert callable(ess_threshold)
            ess_th_fn = ess_threshold

        def dummyzero(a, b):
            return 0

        self.rbar_list = []
        # init memory for particles
        theta = np.zeros((N, self.pmodel.dim()))
        llh = np.zeros(N)  # log likelihood
        log_prior = np.zeros(N)  # log prior
        # theta[:] = self.pmodel.sampleFromPrior(N)
        theta[:] = self.starting_dist.draw(N)
        # print("start theta",theta[:10])
        # print(theta[:10,:])
        # compute initial likelihood
        llh[:] = self.pmodel.compute_llh(theta)
        log_prior[:] = self.pmodel.compute_prior(theta)
        theta, llh, log_prior = self.orderByModel(theta, llh, log_prior)
        # print("orderedtheta",theta[-10:])
        # init the SMC quantities: weights and Z
        models_grouped_by_indices = self.enumerateModels(theta)
        if VERBOSITY_HIGH():
            print("Model identifiers", models_grouped_by_indices.keys())
        # smc_quantities = SMCQuantities(np.arange(N), llh, log_prior,theta,self.pmodel,None)
        smc_quantities = self.makeSMCQuantities(
            N, llh, log_prior, theta, self.pmodel, None
        )
        self.setInitialDensity(smc_quantities, llh, log_prior, theta)
        # self.appendToMixtureTargetDensity(smc_quantities, llh, log_prior, theta, 0.)
        # print("kblocks",kblocks)
        # for mk,mkidx in kblocks.items():
        #    print("Single mutation count {} particles for model {}".format(mkidx.shape[0],mk)) # n,k
        # iterate over sequence of distributions
        t = 0.0
        self.essseq_done[0] = N
        self.mkseq_done[0] = self.getEmpiricalModelProbs(theta)
        progress(t, 1, status="Initialising SMC sampler")
        while t < 1.0:
            last_t = t
            if self.tempseq is None:
                self.preImportanceSampleHook(
                    last_t, llh, log_prior, smc_quantities, self.pmodel, next_t=None
                )
                # last_t = t
                # safety for early stages
                # do a safe threshold check
                if (
                    self.ESSThreshold(
                        t,
                        last_t,
                        ess_th_fn,
                        llh,
                        log_prior,
                        smc_quantities,
                        self.pmodel,
                        N,
                    )
                    > 0
                ):
                    max_t = max(1e-20, min(1.0, 2 ** np.ceil(np.log2(t))))
                    if VERBOSITY_HIGH():
                        print("Starting bisection search with max_t={}".format(max_t))
                    while (
                        max_t < 1
                        and self.ESSThreshold(
                            max_t,
                            last_t,
                            ess_th_fn,
                            llh,
                            log_prior,
                            smc_quantities,
                            self.pmodel,
                            N,
                        )
                        > 0
                    ):
                        max_t = 2 ** np.ceil(np.log2(max_t) + 1)
                        if VERBOSITY_HIGH():
                            print("Increasing to max_t={}".format(max_t))
                    if (
                        self.ESSThreshold(
                            max_t,
                            last_t,
                            ess_th_fn,
                            llh,
                            log_prior,
                            smc_quantities,
                            self.pmodel,
                            N,
                        )
                        < 0
                    ):
                        next_t, rres = scipy.optimize.bisect(
                            self.ESSThreshold,
                            t,
                            max_t,
                            args=(
                                last_t,
                                ess_th_fn,
                                llh,
                                log_prior,
                                smc_quantities,
                                self.pmodel,
                                N,
                            ),
                            full_output=True,
                            rtol=1e-6,
                        )
                        # print("Next t",next_t,"rres",rres)
                        t = next_t
                    else:
                        t = 1.0
            else:
                # use the temperature sequence provided.
                # last_t = t
                # assumes last element of self.tempseq is 1
                t = self.tempseq[self.tempseq.index(t) + 1]
                self.preImportanceSampleHook(
                    last_t, llh, log_prior, smc_quantities, self.pmodel, next_t=t
                )
            if True:
                self.tempseq_done.append(t)
                # self.essseq_done[t] = self.ESSThreshold(t,last_t,dummyzero,llh,log_prior,smc_quantities,self.pmodel,N)
                # importance sample
                # new_gamma = t - last_t
                # log_target_ratio = new_gamma * (llh + log_prior) - new_gamma * self.pmodel.evalStartingDistribution(theta)
                # llh_gamma_delta = llh[smc_quantities.indices] * new_gamma
                log_target_ratio = self.computeIncrementalWeights(
                    t,
                    last_t,
                    llh,
                    log_prior,
                    self.pmodel,
                    theta,
                    smc_quantities.log_w_norm,
                    store_D=True,
                )
                smc_quantities.log_w[:] = smc_quantities.log_w_norm + log_target_ratio
                log_w_sum = logsumexp(smc_quantities.log_w)
                smc_quantities.log_Zt += log_w_sum
                smc_quantities.log_w_norm[:] = smc_quantities.log_w - log_w_sum
                smc_quantities.log_ESS = -logsumexp(2 * smc_quantities.log_w_norm)
                if VERBOSITY_HIGH():
                    print(
                        "Calibrating at t = {}, Total ESS = {}...".format(
                            t, np.exp(smc_quantities.log_ESS)
                        )
                    )
                progress(
                    t,
                    1,
                    status="Calibrating at inverse temperature = {}, ESS = {}".format(
                        t, np.exp(smc_quantities.log_ESS)
                    ),
                )
                self.essseq_done[t] = np.exp(smc_quantities.log_ESS)

                # CALIBRATE HERE
                self.appendToMixtureTargetDensity(
                    smc_quantities, llh, log_prior, theta, float(t)
                )
                self.pmodel.calibrateProposalsMMMPD(
                    self.getMixtureTargetDensity(), N, float(t)
                )
                # self.getMixtureTargetDensity()

                if VERBOSITY_HIGH():
                    print(
                        "Resampling at t = {}, Total ESS = {}...".format(
                            t, np.exp(smc_quantities.log_ESS)
                        )
                    )
                progress(
                    t,
                    1,
                    status="Resampling at inverse temperature = {}, ESS = {}".format(
                        t, np.exp(smc_quantities.log_ESS)
                    ),
                )

                # resample
                smc_quantities.resample_indices = self.resample(
                    np.exp(smc_quantities.log_w_norm), N
                )
                resample_indices_global = smc_quantities.indices[
                    smc_quantities.resample_indices
                ]
                theta[smc_quantities.indices] = theta[resample_indices_global]
                llh[smc_quantities.indices] = llh[resample_indices_global]
                log_prior[smc_quantities.indices] = log_prior[resample_indices_global]

                # set empirical model probs
                self.mkseq_done[t] = self.getEmpiricalModelProbs(theta)

                if VERBOSITY_HIGH():
                    print(
                        "Mutating at t = {}, Total ESS = {}...".format(
                            t, np.exp(smc_quantities.log_ESS)
                        )
                    )
                progress(
                    t,
                    1,
                    status="MCMC mutation at inverse temperature = {}, ESS = {}".format(
                        t, np.exp(smc_quantities.log_ESS)
                    ),
                )
                # mutate
                theta[:], llh[:], log_prior[:], accepted = self.mutate(
                    theta, llh, log_prior, N, t, smc_quantities
                )
                # TODO may need to order accepted too
                theta, llh, log_prior = self.orderByModel(theta, llh, log_prior)
                # update the smc quantities
                smc_quantities.updateAfterMutate(np.arange(N), llh, log_prior, theta)
                # Set this posterior to use the final particles after mutation
                # print("Appending again")
                # self.appendToMixtureTargetDensity(smc_quantities, llh, log_prior, theta, float(t))
        progress(t, 1, status="Finished")
        return (
            theta,
            smc_quantities,
            llh,
            log_prior,
            self.getMixtureTargetDensity(),
            self.rbar_list,
        )

    def orderByModel(self, theta, llh, log_prior):
        # order by model
        ordered_theta_list = []
        ordered_llh_list = []
        ordered_log_prior_list = []
        models_grouped_by_indices = self.enumerateModels(theta)
        for k, m_indices in models_grouped_by_indices.items():
            ordered_theta_list.append(theta[m_indices])
            ordered_llh_list.append(llh[m_indices])
            ordered_log_prior_list.append(log_prior[m_indices])
        return (
            np.vstack(ordered_theta_list),
            np.hstack(ordered_llh_list),
            np.hstack(ordered_log_prior_list),
        )

    def computeMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t, resample=True
    ):
        """
        Deprecated
        """
        # init previous_model_targets each time
        previous_model_targets = SingleModelMPD(self.pmodel)
        previous_model_targets.addComponent(self.initialdensity)
        previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t))
        return previous_model_targets.getParticleDensityForTemperature(
            gamma_t, resample
        )

    def setInitialDensity(self, smc_quantities, llh, log_prior, theta):
        self.initialdensity = smc_quantities.makePPPD(0)
        self.previous_model_targets.addComponent(self.initialdensity)

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        # init previous_model_targets each time
        # print("append to target\n",theta)
        self.previous_model_targets = SingleModelMPD(self.pmodel)
        self.previous_model_targets.addComponent(self.initialdensity)
        self.previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t))

    def getMixtureTargetDensity(self):
        return self.previous_model_targets

    def enumerateModels(self, theta):
        """
        Associate each model with a key. Typically in a single rjmcmc scheme this would be a tuple of the number of layers (n,)
        """
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        return model_key_dict

    def reverseEnumerateModels(self, theta):
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        return reverse_key_ref

    def getModelKeyArray(self, theta):
        """
        return numpy array of model keys of each row of theta
        """
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        mk_keys = model_key_dict.keys()
        ncols = len(list(mk_keys)[0])
        keyarray = np.zeros((theta.shape[0], ncols))
        for mk, idx in model_key_dict.items():
            keyarray[idx] = np.array(list(mk))
        return keyarray

    def resample(self, weights, n):
        # n = weights.shape[0]
        # print("weights sum = ",weights.sum(),n)
        indices = np.zeros(n, dtype=np.int32)
        # indices = np.zeros_like(weights,dtype=np.int32)
        # C = [0.] + [sum(weights[:i+1]) for i in range(n)]
        C = np.cumsum(weights) * n
        u0 = np.random.uniform(0, 1)
        j = 0
        for i in range(n):
            u = u0 + i
            while u > C[j]:
                j += 1
                if j >= n:
                    j = 0
            indices[i] = j
        return indices

    def mutate(self, theta, llh, log_prior, N, t, smc_quantities):
        global TESTBIAS
        global PLOTMARGINALS
        # switch to say whether to use total acceptance rate or min acceptance rate
        use_arn_total = False
        """
        Uses Rt method (Drovandi & Pettitt)
        R_t = ceil(log(0.01) / log(1 - ar))
        where 0<=ar<=1 is the acceptance rate 
        TODO pass in smc_quantities so that the ESS for each model density can be used to mix the independent proposals with the priors
        this will probably need to be passed into self.pmodel.proposal.setSMCQuantities(smc_quantities) or something similar
        """
        # pilot run with acceptance rate
        ar = 0.44
        new_theta = np.zeros_like(theta)
        new_llh = np.zeros_like(llh)
        new_log_prior = np.zeros_like(log_prior)
        new_theta[:] = theta
        new_llh[:] = llh
        new_log_prior[:] = log_prior
        ar_total = 1.0
        arn_total = 0
        arn_repeats = 3
        min_ar = 1.0
        if self.n_mutations is not None:
            R_t = self.n_mutations
            total_accepted = np.zeros(N)
        else:
            for i in range(arn_repeats):
                (
                    new_theta[:],
                    new_llh[:],
                    new_log_prior[:],
                    accepted,
                ) = self.single_mutation(new_theta, new_llh, N, t)
            for pid, prop in Proposal.idpropdict.items():
                # print("After trial mutation prop pid {}\tprop {}\tar {}".format(pid,prop.printName(),prop.getAvgARN(arn_repeats) / (N*arn_repeats)))
                # print("After trial mutation prop pid {}\tprop {}\tar {}".format(pid,prop.printName(),prop.getAvgARN(arn_repeats)/arn_repeats))
                this_ar = prop.getAvgARN(arn_repeats) / (N * arn_repeats)
                arn_total += this_ar
                min_ar = min(min_ar, prop.getLastAR())
            if use_arn_total:
                ar_total = arn_total
            else:
                ar_total = min_ar
            # TODO compute minimum acceptance ratio given each proposal type
            if VERBOSITY_HIGH():
                print("Acceptance rate = {}".format(ar_total))
            # R_t = int(np.ceil(np.log(0.001) / np.log(1 - ar_total)))
            R_t = np.ceil(np.log(0.001) / np.log(1 - ar_total))
            if np.isfinite(R_t):
                R_t = int(R_t)
            else:
                R_t = 50
            if TESTBIAS:
                R_t = max(100, R_t)
            elif True:
                R_t = R_t  # min(500,R_t)
            total_accepted = np.zeros(N) + accepted
        if VERBOSITY_HIGH():
            print("mutating {} times".format(R_t))
        # include pilot run and do remaining runs
        for r in range(R_t):
            new_theta[:], new_llh[:], new_log_prior[:], accepted = self.single_mutation(
                new_theta, new_llh, N, t
            )
            total_accepted += accepted
        return (
            new_theta,
            new_llh,
            new_log_prior,
            total_accepted,
        )  # what are we doing with the accepted totals? Also include proposals?

    def single_mutation(self, theta, llh, N, t, blocksize=1000):
        """
        By default use RWMH kernel. Proposals are set in __init__ as proposal_strategy.
        """
        global PLOTMARGINALS
        # kblocks = np.unique(theta[:,0])
        kblocks, rev = self.pmodel.enumerateModels(theta)
        # print("kblocks",kblocks)
        for mk, mkidx in kblocks.items():
            if VERBOSITY_HIGH():
                print(
                    "Single mutation count {} particles for model {}".format(
                        mkidx.shape[0], mk
                    )
                )  # n,k
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros_like(llh)
        prop_llh = np.full(llh.shape, np.NINF)
        cur_prior = np.zeros_like(llh)
        prop_prior = np.zeros_like(llh)
        prop_id = np.zeros_like(llh)
        prop_lpqratio = np.zeros_like(llh)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        # propose
        # Proposal.clearIDs()
        # if float(t)/self.T >= 0.9 :
        #   PLOTMARGINALS = True
        nblocks = int(np.ceil((1.0 * N) / blocksize))
        blocks = [
            np.arange(i * blocksize, min(N, (i + 1) * blocksize))
            for i in range(nblocks)
        ]
        # TODO reuse this computation from constructor for mmmpd
        for bidx in blocks:
            prop_theta[bidx], prop_lpqratio[bidx], prop_id[bidx] = self.pmodel.propose(
                theta[bidx], bidx.shape[0]
            )  # TODO FIXME self.pmodel.propose(theta,N)
        # prop_theta[:], prop_lpqratio[:], prop_id[:] = self.pmodel.propose(theta,N) # TODO FIXME self.pmodel.propose(theta,N)
        cur_prior[:] = self.pmodel.compute_prior(theta)
        prop_prior[:] = self.pmodel.compute_prior(prop_theta)
        # print(prop_prior)
        ninfprioridx = np.where(~np.isfinite(cur_prior))
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(
            np.isfinite(prop_prior), np.isfinite(prop_lpqratio)
        )
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta,
            prop_theta,
            prop_lpqratio,
            prop_llh,
            llh,
            cur_prior,
            prop_prior,
            float(t),
        )  # TODO implement

        # store acceptance ratios
        if self.store_ar:
            # self.rbar_list.append(rbar(log_acceptance_ratio,float(t),self.reverseEnumerateModels(theta),self.reverseEnumerateModels(prop_theta)))
            self.rbar_list.append(
                rbar(
                    log_acceptance_ratio,
                    float(t),
                    self.getModelKeyArray(theta),
                    self.getModelKeyArray(prop_theta),
                )
            )

        Proposal.setAcceptanceRates(prop_id, log_acceptance_ratio, float(t))
        for pid, prop in Proposal.idpropdict.items():
            if VERBOSITY_HIGH():
                print(
                    "for pid {}\tprop {}\tar {}".format(
                        pid, prop.printName(), prop.getLastAR()
                    )
                )

        # accept/reject
        log_u = np.log(uniform.rvs(0, 1, size=N))
        # reject_indices = (prop_llh - llh < log_u) | ~np.isfinite(prop_llh)
        reject_indices = log_acceptance_ratio < log_u
        prop_theta[reject_indices] = theta[reject_indices]
        prop_llh[reject_indices] = llh[reject_indices]
        prop_prior[reject_indices] = cur_prior[reject_indices]
        # a boolean array of accepted proposals
        accepted = np.ones(N)
        accepted[reject_indices] = 0
        # return prop_theta,prop_llh,accepted
        # print("log acceptance ratio",log_acceptance_ratio)
        self.nmutations += 1
        return prop_theta, prop_llh, prop_prior, np.exp(log_acceptance_ratio)


# Online particle recycling RJSMC
class SMC1OPR(SMC1):
    def init_more(self):
        # come up with a structure for previous targets
        # self.previous_model_targets = SingleModelMPD(self.pmodel)
        pass

    def computeMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t, resample=True
    ):
        """
        Deprecated
        """
        """
        This function needs to return the mixture density object.

        Currently it takes as input the llh, log_prior, theta, gamma_t, and smc_quantities of the current particle set (at step t).
        It outputs a set of particles and associated weights.
        If resample==False, the output weights are the computed weights.
        If resample==True, the output weights are the 1/N where N is the number of particles returned.

        The need for change here is that we are constantly "appending" new PowerPosteriorParticleDensity objects to each MixtureParticleDensity
        Why is this a problem? Um... maybe it isn't a problem.

        What do we need the MixtureParticleDensity objects to do?
        MPD objects need to be input to proposals for calibration. 
            Currently we give proposals weighted particles. 
            An annoyance here is that we are just appending particles for all models to the one ensemble.
        Instead, it would be good to keep models separate.
        """
        self.previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t))
        return self.previous_model_targets.getParticleDensityForTemperature(
            gamma_t, resample
        )

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        self.previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t))

    def getMixtureTargetDensity(self):
        return self.previous_model_targets


def weightedMoments(theta_k, theta_k_w):
    mu_pi_t = np.average(theta_k, weights=theta_k_w, axis=0)
    cov_pi_t = np.cov(theta_k.T, aweights=theta_k_w)
    if np.any(~np.isfinite(cov_pi_t)):
        if VERBOSITY_ERROR():
            print("non-finite cov_pi_t ", cov_pi_t)
        cov_pi_t = cov_s
        sys.exit(0)
    covdim = theta_k.shape[1]  # cov_pi_t.shape[0]
    # print("covdim",covdim)
    return mu_pi_t, cov_pi_t


# def estimateFinalTargetMoments(new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s):
def estimateFinalTargetMoments(new_t, cov_pi_t, mu_pi_t, cov_s, cov_s_inv, mu_s):
    # covdim = cov_pi_t.shape[0]
    covdim = getdimof(mu_pi_t)
    if covdim < 2:
        cov_pi_t_inv = 1.0 / cov_pi_t
        # print("cov_pi_t_inv",cov_pi_t_inv)
        # print("cov_s_inv",cov_s_inv)
        # cov_pi = new_t * 1./(cov_pi_t_inv - (1 - new_t) * cov_s_inv)
        cov_pi = new_t * (
            cov_pi_t
            + 1.0
            / (1 + cov_pi_t * (new_t - 1) * cov_s_inv)
            * (cov_pi_t * (new_t - 1) * cov_s_inv * cov_pi_t)
        )
        cov_pi = np.abs(cov_pi)  # force positive definite
        mu_pi = (
            1.0
            / new_t
            * cov_pi
            * (cov_pi_t_inv * mu_pi_t - (1 - new_t) * cov_s_inv * mu_s)
        )
    else:
        # cov_pi_t_inv = np.linalg.inv(cov_pi_t)
        # print("pre cov pi t")
        cov_pi_t_inv = safe_inv(cov_pi_t)
        # cov_pi = new_t * np.linalg.inv(cov_pi_t_inv - (1 - new_t) * cov_s_inv)
        # print("pre cov pi")
        # cov_pi = new_t * safe_inv(cov_pi_t_inv - (1 - new_t) * cov_s_inv)
        # safer way
        cov_pi = new_t * (
            cov_pi_t
            + safe_inv(np.eye(covdim) + cov_pi_t @ ((new_t - 1) * cov_s_inv))
            @ (cov_pi_t @ ((new_t - 1) * cov_s_inv) @ cov_pi_t)
        )
        cov_pi = make_pos_def(cov_pi)
        mu_pi = (
            1.0
            / new_t
            * cov_pi
            @ (cov_pi_t_inv @ mu_pi_t - (1 - new_t) * cov_s_inv @ mu_s)
        )
    return mu_pi, cov_pi


# def combineCondApprox(mu_pi_approx,cov_pi_approx,pmodel,new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s):
def combineCondApprox(
    mu_pi_approx,
    cov_pi_approx,
    pmodel,
    new_t,
    cov_pi_t,
    mu_pi_t,
    cov_s,
    cov_s_inv,
    mu_s,
    last_t,
):
    """
    uses last_t as the threshold for the barycenter approx cutoff.
    """
    if pmodel.useBarycenterCombination():
        # weight = np.clip(1.01*new_t - 0.01,0,1)
        a = 0.01  # 0.01
        b = 0.03  # 0.05
        # weight = np.clip((new_t - a)/(b-a),0,1)
        weight = np.clip((last_t - a) / (b - a), 0, 1)
        if VERBOSITY_HIGH():
            print("weight for combine approx", weight, new_t)
        if weight > 0:
            # mu_pi_est, cov_pi_est = estimateFinalTargetMoments(new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s)
            mu_pi_est, cov_pi_est = estimateFinalTargetMoments(
                new_t, cov_pi_t, mu_pi_t, cov_s, cov_s_inv, mu_s
            )
            from rjlab.utils.barycenter import bures_barycenter

            mu_pi, cov_pi = bures_barycenter(
                mu_pi_est, cov_pi_est, mu_pi_approx, cov_pi_approx, weight
            )
            return mu_pi, cov_pi
    return mu_pi_approx, cov_pi_approx


def getdimof(a):
    return np.array(a).flatten().shape[0]


def computeLogDtk(mk, new_t, cov_pi, mu_pi, cov_s, cov_s_inv, mu_s):
    PRINTSTUFF = False
    if new_t == 0:
        return 0
    if new_t == 1:
        return 0
    dim = getdimof(mu_pi)  # theta_k.shape[1]
    if dim < 2:
        if PRINTSTUFF:
            # print("cov_pi_t_inv",cov_pi_t_inv)
            print("cov_s_inv", cov_s_inv)
            print("cov_pi", cov_pi)
            print("mu_pi", mu_pi)
        # print("pi cov",cov_pi,"mu",mu_pi)
        # ok, ordeal halfway there.  Some trivial quantities
        cov_mix_k = (1 - new_t) * cov_pi + new_t * cov_s
        delta_mu_k = mu_pi - mu_s
        logdet_cov_mix_k = np.log(np.abs(cov_mix_k))  # we know it's positive
        logdet_cov_s = np.log(cov_s)  # TODO move to init_more()
        logdet_cov_pi = np.log(np.abs(cov_pi))
        inv_cov_mix_k = 1.0 / cov_mix_k
        # Now we compute D_t
        if PRINTSTUFF:
            print("logdet_cov_s", logdet_cov_s)
            print("logdet_cov_pi", logdet_cov_pi)
            print("logdet_cov_mix_k", logdet_cov_mix_k)
            print("inv_cov_mix_k", inv_cov_mix_k)
        log_D_t_k = -0.5 * (
            (1 - new_t) * logdet_cov_pi
            + new_t * logdet_cov_s
            - logdet_cov_mix_k
            - (new_t * (1 - new_t)) * (delta_mu_k.T * inv_cov_mix_k * delta_mu_k)
        )
    else:
        if PRINTSTUFF:
            # print("cov_pi_t_inv",cov_pi_t_inv)
            print("cov_s_inv", cov_s_inv)
            print("cov_pi", cov_pi)
            print("mu_pi", mu_pi)
        # ok, ordeal halfway there.  Some trivial quantities
        cov_mix_k = (1 - new_t) * cov_pi + new_t * cov_s
        delta_mu_k = mu_pi - mu_s
        sign, logdet_cov_mix_k = np.linalg.slogdet(cov_mix_k)
        sign, logdet_cov_s = np.linalg.slogdet(cov_s)  # TODO move to init_more()
        sign, logdet_cov_pi = np.linalg.slogdet(cov_pi)
        # sign, logdet_cov_pi = safe_logdet(cov_pi)
        # inv_cov_mix_k = np.linalg.inv(cov_mix_k)
        inv_cov_mix_k = safe_inv(cov_mix_k)
        # Now we compute D_t
        if PRINTSTUFF:
            # print("logdet_cov_s",logdet_cov_s)
            print("logdet_cov_pi", logdet_cov_pi)
            print("logdet_cov_mix_k", logdet_cov_mix_k)
            print("inv_cov_mix_k", inv_cov_mix_k)
        log_D_t_k = -0.5 * (
            (1 - new_t) * logdet_cov_pi
            + new_t * logdet_cov_s
            - logdet_cov_mix_k
            - (new_t * (1 - new_t)) * (delta_mu_k.T @ inv_cov_mix_k @ delta_mu_k)
        )
    log_D_t_k = log_D_t_k.flatten()
    if VERBOSITY_HIGH():
        print("log_D_t_k for mk={} at t={} is {}".format(mk, new_t, log_D_t_k))
    return log_D_t_k


# def computeLogDtkFromParticles(pmodel,mk,theta_k,theta_k_w,new_t,cov_s,cov_s_inv,mu_s):
#    if new_t == 0:
#        return 0
#    if new_t == 1:
#        return 0
#    # check if the model has a conditional approximation
#    if pmodel.hasConditionalApproximation(mk):
#        mu_pi_approx, cov_pi_approx = pmodel.getConditionalApproximation(mk)
#        mu_pi, cov_pi = combineCondApprox(mu_pi_approx,cov_pi_approx,pmodel,new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s)
#    else:
#        mu_pi, cov_pi = estimateFinalTargetMoments(new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s)
#    return computeLogDtk(mk,new_t,cov_pi,mu_pi,cov_s,cov_s_inv,mu_s)
#
# def getFinalTargetMoments(pmodel,mk,theta_k,theta_k_w,new_t,cov_s,cov_s_inv,mu_s):
#    # check if the model has a conditional approximation
#    if pmodel.hasConditionalApproximation(mk):
#        mu_pi_approx, cov_pi_approx = pmodel.getConditionalApproximation(mk)
#        mu_pi, cov_pi = combineCondApprox(mu_pi_approx,cov_pi_approx,pmodel,new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s)
#    else:
#        mu_pi, cov_pi = estimateFinalTargetMoments(new_t,theta_k,theta_k_w,cov_s,cov_s_inv,mu_s)
#    return mu_pu, cov_pi


class WSSMC1(SMC1):
    """
    Weight Stabilising SMC1 where each model is reweighted online at each iteration to avoid torpid sampling over the temperature sequence.
    """

    def makeSMCQuantities(self, N, llh, log_prior, theta, pmodel, mk):
        """
        Overrides parent method to just instantiate an object of WSSMCQuantities
        """
        return WSSMCQuantities(np.arange(N), llh, log_prior, theta, pmodel, mk)

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        # init previous_model_targets each time
        # print("append to target\n",theta)
        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
        self.previous_model_targets.addComponent(self.initialdensity)
        self.previous_model_targets.addComponent(
            smc_quantities.makePPPD(gamma_t, self.log_D_t[gamma_t])
        )

    def init_more2(self):
        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
        pass

    def init_more(self):
        """
        Creates a dict of D_{t,k} values for each model. D_{0,k}=1 for all k.
        """
        self.log_D_t = {}
        self.log_D_t[0] = {}
        self.log_Z_t = {}  # used in ZWSSMC1
        self.log_Z_t[0] = {}
        for mk in self.pmodel.getModelKeys():
            self.log_D_t[0][mk] = 0
            self.log_Z_t[0][mk] = 0
        # sample prior and get empirical moments
        theta = self.starting_dist.draw(2000)  # TODO get N from input
        self.mu_s = {}
        self.cov_s = {}
        self.cov_s_inv = {}
        models_grouped_by_indices = self.enumerateModels(theta)
        # for mk in self.pmodel.getModelKeys():
        for mk, m_indices in models_grouped_by_indices.items():
            if (
                False
            ):  # TODO delete this block, it's now decided in ParametricModelSpace
                # estimate moments empirically all at once, which can be unstable
                # theta_k = self.pmodel.concatAllParameters(theta[m_indices],mk)
                theta_k = self.pmodel.concatAllParameters(
                    theta[m_indices], mk, transform=True
                )
                # print("theta_k",theta_k)
                self.mu_s[mk] = np.mean(theta_k, axis=0)
                self.cov_s[mk] = np.cov(theta_k.T)
            else:
                # block diagonal moment estimation, marginal to each rv
                # print("theta shape init_more()",theta.shape,mk)
                self.mu_s[mk], self.cov_s[mk] = self.pmodel._estimatemoments(
                    theta[m_indices], mk
                )
            if VERBOSITY_HIGH():
                print("cov_s", self.cov_s[mk], "shape", self.cov_s[mk].shape)
            if len(self.cov_s[mk].shape) < 2:
                self.cov_s_inv[mk] = 1.0 / self.cov_s[mk]
            else:
                # self.cov_s_inv[mk] = np.linalg.inv(self.cov_s[mk])
                self.cov_s_inv[mk] = safe_inv(self.cov_s[mk])
        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)

        # experimental - precompute final conditonal target moments once per iteration
        self.mu_pi_estimate = {}
        self.cov_pi_estimate = {}
        # self.init_more2()

    def computeIncrementalWeights(
        self,
        new_t,
        last_t,
        llh,
        log_prior,
        pmodel,
        theta,
        log_w_norm_last_t,
        store_D=False,
    ):
        """
        stratify by model
        compute each model D_t,k weight correction
        Also store it.
        """
        mk_list = self.pmodel.getModelKeys()
        if store_D:
            self.log_D_t[new_t] = {}  # TODO skip if new_t==0
        if len(mk_list) == 1:
            # case where we are doing single model SMC and don't use weight-stabilising correction.
            if store_D:
                self.log_D_t[new_t][mk_list[0]] = 0  # always 0.
            log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
                (1 - new_t) - (1 - last_t)
            ) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        elif new_t == 0:
            for mk in mk_list:
                self.log_D_t[new_t][mk] = 0
                # todo set current pi_t to s?
            log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
                (1 - new_t) - (1 - last_t)
            ) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        elif new_t == last_t:
            log_target_ratio = 0  # (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        else:
            # compute weight stabilising correction
            models_grouped_by_indices = self.enumerateModels(theta)
            # for mk in mk_list:
            mu_pi_t = {}
            cov_pi_t = {}
            mu_pi = {}
            cov_pi = {}
            # we need to compute the current target uncorrected first for the covariances we will compute later
            log_target_ratio_uncorrected = (new_t - last_t) * (llh + log_prior) + (
                last_t - new_t
            ) * pmodel.evalStartingDistribution(theta)
            log_w_uncorrected = log_w_norm_last_t + log_target_ratio_uncorrected
            # log_w_unc_norm = log_w_uncorrected - logsumexp(log_w_uncorrected)

            if store_D:
                for mk in mk_list:
                    self.log_D_t[new_t][mk] = 0

            log_target_ratio = np.zeros(theta.shape[0])
            for mk, m_indices in models_grouped_by_indices.items():
                # commpute using stored final target moments.
                log_D_t_k = computeLogDtk(
                    mk,
                    new_t,
                    self.cov_pi_estimate[mk],
                    self.mu_pi_estimate[mk],
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )
                log_D_t1_k_star = computeLogDtk(
                    mk,
                    last_t,
                    self.cov_pi_estimate[mk],
                    self.mu_pi_estimate[mk],
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )

                if VERBOSITY_HIGH():
                    print("Uncorrected log_D_t[{}][{}]={}".format(new_t, mk, log_D_t_k))
                log_D_t_k = log_D_t_k + (
                    ((1 - new_t) / (1 - last_t)) ** (1000 / new_t)
                ) * (self.log_D_t[last_t][mk] - log_D_t1_k_star)
                # log_D_t_k = log_D_t_k + np.sqrt((1-new_t)/(1-last_t))*np.exp(-10./(new_t)*(new_t-last_t))*(self.log_D_t[last_t][mk] - log_D_t1_k_star) # old one
                # log_D_t_k = log_D_t_k + np.sqrt(1-new_t)*np.exp(-1./(new_t)*(new_t-last_t))*(self.log_D_t[last_t][mk] - log_D_t1_k_star)
                if VERBOSITY_HIGH():
                    print("Corrected log_D_t[{}][{}]={}".format(new_t, mk, log_D_t_k))
                log_target_ratio[m_indices] = (
                    log_D_t_k
                    - self.log_D_t[last_t][mk]
                    + (new_t - last_t) * (llh[m_indices] + log_prior[m_indices])
                    + (last_t - new_t)
                    * pmodel.evalStartingDistribution(theta[m_indices])
                )
                if store_D:
                    self.log_D_t[new_t][
                        mk
                    ] = log_D_t_k  # TODO now test, and report these weights!
            return log_target_ratio

    def preImportanceSampleHook(
        self, last_t, llh, log_prior, smcq, pmodel, next_t=None
    ):
        """
        In this hook we pre-compute the final conditional target moments given current particles
        """
        self.precomputeConditionalTargetMoments(
            last_t, llh, log_prior, pmodel, smcq.theta, smcq.log_w_norm, next_t
        )

    def precomputeConditionalTargetMoments(
        self, last_t, llh, log_prior, pmodel, theta, log_w_norm_last_t, next_t
    ):
        """
        We hypothesise that there is an "optimal" temperature at which to estimate the final target moments from the current set of particles.
        We don't know what it is, so we choose an ESS threshold and bisect till we find t that gives that threshold.
        The ESS is not for all particles, but rather for each model. A conditional ESS maybe.

        """
        mk_list = pmodel.getModelKeys()
        models_grouped_by_indices = self.enumerateModels(theta)
        # log_target_ratio_uncorrected = (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
        # log_w_uncorrected = log_w_norm_last_t + log_target_ratio_uncorrected
        # TODO stratify by model, do bisection for individual model ESS, use a minimum ESS for each model and a ratio of last ESS.

        # for mk, m_indices in models_grouped_by_indices.items():
        for mk in mk_list:
            if mk not in models_grouped_by_indices.keys():
                if VERBOSITY_HIGH():
                    print("Particle Impoverishment Encountered in model ", mk)
                self.mu_pi_estimate[mk] = self.mu_s[mk]
                self.cov_pi_estimate[mk] = self.cov_s[mk]
                continue

            m_indices = models_grouped_by_indices[mk]

            def quickWeightUpdateNorm(
                new_t, last_t, log_w_last, llh_mk, log_prior_mk, pmodel, theta_mk
            ):
                log_target_ratio_uncorrected = (new_t - last_t) * (
                    llh_mk + log_prior_mk
                ) + (last_t - new_t) * pmodel.evalStartingDistribution(theta_mk)
                log_w_uncorrected = log_w_last + log_target_ratio_uncorrected
                return log_w_uncorrected - logsumexp(log_w_uncorrected)

            def quickESSThreshold(
                new_t,
                last_t,
                log_w_last,
                llh_mk,
                log_prior_mk,
                pmodel,
                theta_mk,
                minESS,
                ratioESS=0.5,
            ):
                """
                given input weights, first compute a threshold = max(min(lastESS,minESS),ratioESS*lastESS)
                then update the weights using a naive annealing sequence
                """
                lastESS = np.exp(-logsumexp(2 * log_w_last))
                threshold = max(min(lastESS, minESS), ratioESS * lastESS)
                new_log_w_norm = quickWeightUpdateNorm(
                    new_t, last_t, log_w_last, llh_mk, log_prior_mk, pmodel, theta_mk
                )
                val = np.exp(-logsumexp(2 * new_log_w_norm))
                # print("ESS at t={} is {}".format(new_t,val))
                # print("threshold is ",threshold)
                return val - threshold

            # use a bisection optimiser to find a new t for this model to compute moments.
            log_w_last_mk_norm = log_w_norm_last_t[m_indices] - logsumexp(
                log_w_norm_last_t[m_indices]
            )

            if next_t is None:
                # print("last log_w\n",log_w_last_mk_norm,logsumexp(log_w_last_mk_norm))
                # First check ESS at 1
                if VERBOSITY_HIGH():
                    print(
                        "f(a)=",
                        quickESSThreshold(
                            last_t,
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                    )
                    print(
                        "f(b)=",
                        quickESSThreshold(
                            1,
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                    )
                if (
                    quickESSThreshold(
                        last_t,
                        last_t,
                        log_w_last_mk_norm,
                        llh[m_indices],
                        log_prior[m_indices],
                        self.pmodel,
                        theta[m_indices],
                        20,
                        0.5,
                    )
                    < 0
                ):
                    # numerical precision error - we've run out of usable particles to predict final conditional moments.
                    new_t = last_t
                elif (
                    quickESSThreshold(
                        1,
                        last_t,
                        log_w_last_mk_norm,
                        llh[m_indices],
                        log_prior[m_indices],
                        self.pmodel,
                        theta[m_indices],
                        20,
                        0.5,
                    )
                    > 0
                ):
                    new_t = 1.0
                else:
                    # TODO move minESS to config, or set to some multiple of the model dimension.
                    new_t, rres = scipy.optimize.bisect(
                        quickESSThreshold,
                        last_t,
                        1.0,
                        args=(
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                        full_output=True,
                        rtol=1e-6,
                    )
            else:
                new_t = next_t

            new_log_w_mk_norm = quickWeightUpdateNorm(
                new_t,
                last_t,
                log_w_last_mk_norm,
                llh[m_indices],
                log_prior[m_indices],
                self.pmodel,
                theta[m_indices],
            )
            if VERBOSITY_HIGH():
                print(
                    "Temperature at which to pre-compute conditional target for mk={} is t={}, ESS={}, lastESS={}".format(
                        mk,
                        new_t,
                        np.exp(-logsumexp(2 * new_log_w_mk_norm)),
                        np.exp(-logsumexp(2 * log_w_last_mk_norm)),
                    )
                )
            if (
                False
            ):  # TODO remove this block, we resample instead and estimate moments as either full or block in ParametricModelSpace._estimatemoments()
                # Estimate moments empirically, can be unstable
                theta_mk = self.pmodel.concatAllParameters(
                    theta[m_indices], mk, transform=True
                )
                mu_pi_t, cov_pi_t = weightedMoments(theta_mk, np.exp(new_log_w_mk_norm))
                # self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = estimateFinalTargetMoments(new_t,theta_mk,np.exp(new_log_w_mk_norm),self.cov_s[mk],self.cov_s_inv[mk],self.mu_s[mk])
                dim = theta_mk.shape[1]
            else:
                # resample theta from weights
                resampled_idx = self.resample(
                    np.exp(new_log_w_mk_norm), max(1000, new_log_w_mk_norm.shape[0])
                )
                mu_pi_t, cov_pi_t = self.pmodel._estimatemoments(
                    theta[m_indices][resampled_idx], mk
                )
                # print("pi_t moments")
                # print(mu_pi_t)
                # print(cov_pi_t)
                (
                    self.mu_pi_estimate[mk],
                    self.cov_pi_estimate[mk],
                ) = estimateFinalTargetMoments(
                    new_t,
                    cov_pi_t,
                    mu_pi_t,
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )
                # print("estimated final pi[{}] moments".format(mk))
                # print(self.mu_pi_estimate[mk])
                # print(self.cov_pi_estimate[mk])
                # dim = cov_pi_t.shape[0] # assumes matrix, need to test this works for dim=1! Numpy is such a pain
                dim = getdimof(mu_pi_t)

            # Barycenter stuff or VI stuff
            if pmodel.hasConditionalApproximation(mk):
                mu_pi_approx, cov_pi_approx = pmodel.getConditionalApproximation(mk)
                # self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = combineCondApprox(mu_pi_approx,cov_pi_approx,pmodel,new_t,theta_mk,np.exp(new_log_w_mk_norm),self.cov_s[mk],self.cov_s_inv[mk],self.mu_s[mk]) # TODO just pass the estimates computed above
                self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = combineCondApprox(
                    mu_pi_approx,
                    cov_pi_approx,
                    pmodel,
                    new_t,
                    cov_pi_t,
                    mu_pi_t,
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                    last_t,
                )  # TODO just pass the estimates computed above

            # print("Final conditional target moments mk={} mu={} cov={}".format(mk,self.mu_pi_estimate[mk],self.cov_pi_estimate[mk]))
            if dim > 1:
                if VERBOSITY_HIGH():
                    print("Is cov PSD?", is_pos_def(self.cov_pi_estimate[mk]))
                if not is_pos_def(self.cov_pi_estimate[mk]):
                    if VERBOSITY_ERROR():
                        print("Cov not pos def for mk", mk, self.cov_pi_estimate[mk])
                    sys.exit(0)
            else:
                if VERBOSITY_HIGH():
                    print("Is cov PSD?", self.cov_pi_estimate[mk] > 0)
                if not self.cov_pi_estimate[mk] > 0:
                    if VERBOSITY_ERROR():
                        print("1D Cov not pos def for mk", mk, self.cov_pi_estimate[mk])
                    sys.exit(0)

    def single_mutation(self, theta, llh, N, t, blocksize=1000):
        """
        By default use RWMH kernel. Proposals are set in __init__ as proposal_strategy.
        """
        global PLOTMARGINALS
        # kblocks = np.unique(theta[:,0])
        kblocks, rev = self.pmodel.enumerateModels(theta)
        # print("kblocks",kblocks)
        for mk, mkidx in kblocks.items():
            if VERBOSITY_HIGH():
                print(
                    "Single mutation count {} particles for model {}".format(
                        mkidx.shape[0], mk
                    )
                )  # n,k
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros_like(llh)
        prop_llh = np.full(llh.shape, np.NINF)
        cur_prior = np.zeros_like(llh)
        prop_prior = np.zeros_like(llh)
        prop_id = np.zeros_like(llh)
        prop_lpqratio = np.zeros_like(llh)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        # propose
        # Proposal.clearIDs()
        # if float(t)/self.T >= 0.9 :
        #   PLOTMARGINALS = True
        nblocks = int(np.ceil((1.0 * N) / blocksize))
        blocks = [
            np.arange(i * blocksize, min(N, (i + 1) * blocksize))
            for i in range(nblocks)
        ]
        # TODO reuse this computation from constructor for mmmpd
        for bidx in blocks:
            prop_theta[bidx], prop_lpqratio[bidx], prop_id[bidx] = self.pmodel.propose(
                theta[bidx], bidx.shape[0]
            )  # TODO FIXME self.pmodel.propose(theta,N)
        # prop_theta[:], prop_lpqratio[:], prop_id[:] = self.pmodel.propose(theta,N) # TODO FIXME self.pmodel.propose(theta,N)
        cur_prior[:] = self.pmodel.compute_prior(theta)
        prop_prior[:] = self.pmodel.compute_prior(prop_theta)
        # print(prop_prior)
        ninfprioridx = np.where(~np.isfinite(cur_prior))
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(
            np.isfinite(prop_prior), np.isfinite(prop_lpqratio)
        )
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        # here we weight the likelihoods (and thus targets) by D_{t,k}
        prop_lpqratio_D = prop_lpqratio.copy()
        for mk, mkidx in kblocks.items():
            prop_lpqratio_D[mkidx] -= self.log_D_t[t][mk]
        kblocks_prop, rev = self.pmodel.enumerateModels(prop_theta)
        for mk, mkidx in kblocks_prop.items():
            prop_lpqratio_D[mkidx] += self.log_D_t[t][mk]

        # log_acceptance_ratio[:] = self.pmodel.compute_lar(theta,prop_theta,prop_lpqratio,prop_llh,llh,cur_prior,prop_prior,float(t)) # TODO implement
        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta,
            prop_theta,
            prop_lpqratio_D,
            prop_llh,
            llh,
            cur_prior,
            prop_prior,
            float(t),
        )  # TODO implement

        # store acceptance ratios
        if self.store_ar:
            # self.rbar_list.append(rbar(log_acceptance_ratio,float(t),self.reverseEnumerateModels(theta),self.reverseEnumerateModels(prop_theta)))
            self.rbar_list.append(
                rbar(
                    log_acceptance_ratio,
                    float(t),
                    self.getModelKeyArray(theta),
                    self.getModelKeyArray(prop_theta),
                )
            )

        Proposal.setAcceptanceRates(prop_id, log_acceptance_ratio, float(t))
        for pid, prop in Proposal.idpropdict.items():
            if VERBOSITY_HIGH():
                print(
                    "for pid {}\tprop {}\tar {}".format(
                        pid, prop.printName(), prop.getLastAR()
                    )
                )

        # accept/reject
        log_u = np.log(uniform.rvs(0, 1, size=N))
        # reject_indices = (prop_llh - llh < log_u) | ~np.isfinite(prop_llh)
        reject_indices = log_acceptance_ratio < log_u
        prop_theta[reject_indices] = theta[reject_indices]
        prop_llh[reject_indices] = llh[reject_indices]
        prop_prior[reject_indices] = cur_prior[reject_indices]
        # a boolean array of accepted proposals
        accepted = np.ones(N)
        accepted[reject_indices] = 0
        # return prop_theta,prop_llh,accepted
        # print("log acceptance ratio",log_acceptance_ratio)
        self.nmutations += 1
        return prop_theta, prop_llh, prop_prior, np.exp(log_acceptance_ratio)


class WSSMC1OPR(WSSMC1):
    def init_more2(self):
        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
        pass

    def makeSMCQuantities(self, N, llh, log_prior, theta, pmodel, mk):
        """
        Overrides parent method to just instantiate an object of WSSMCQuantities
        """
        return WSSMCQuantities(np.arange(N), llh, log_prior, theta, pmodel, mk)

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        self.previous_model_targets.addComponent(
            smc_quantities.makePPPD(gamma_t, self.log_D_t[gamma_t])
        )

    def getMixtureTargetDensity(self):
        return self.previous_model_targets

    def preImportanceSampleHook(
        self, last_t, llh, log_prior, smcq, pmodel, next_t=None
    ):
        """
        In this hook we pre-compute the final conditional target moments given the OPR density of particles
        """
        mpd = self.previous_model_targets  # ref
        (
            pp_theta_t,
            pp_theta_t_w,
            pp_llh,
            pp_log_prior,
        ) = mpd.getParticleDensityForTemperature(
            last_t, resample=False, return_logpdfs=True
        )
        self.precomputeConditionalTargetMoments(
            last_t, pp_llh, pp_log_prior, pmodel, pp_theta_t, pp_theta_t_w, next_t
        )


class ZWSSMC1(WSSMC1):
    def computeIncrementalWeights(
        self,
        new_t,
        last_t,
        llh,
        log_prior,
        pmodel,
        theta,
        log_w_norm_last_t,
        store_D=False,
    ):
        """
        ZWSSMC::computeIncrementalWeights()
        stratify by model
        compute each model D_t,k weight correction
        Also store it.
        In this method, we approximate D_{t,k} = Z_{t,k}^{1-gamma_t}.
        """
        mk_list = self.pmodel.getModelKeys()
        if store_D:
            self.log_D_t[new_t] = {}  # TODO skip if new_t==0
            self.log_Z_t[new_t] = {}  # TODO skip if new_t==0
        if len(mk_list) == 1:
            # case where we are doing single model SMC and don't use weight-stabilising correction.
            if store_D:
                self.log_D_t[new_t][mk_list[0]] = 0  # always 0.
                self.log_Z_t[new_t][mk_list[0]] = 0  #
            log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
                (1 - new_t) - (1 - last_t)
            ) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        elif new_t == 0:
            for mk in mk_list:
                self.log_D_t[new_t][mk] = 0
                self.log_Z_t[new_t][mk] = 0
                # todo set current pi_t to s?
            log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
                (1 - new_t) - (1 - last_t)
            ) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        elif new_t == last_t:
            log_target_ratio = 0  # (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        else:
            # compute weight stabilising correction
            models_grouped_by_indices = self.enumerateModels(theta)
            # for mk in mk_list:
            mu_pi_t = {}
            cov_pi_t = {}
            mu_pi = {}
            cov_pi = {}
            # we need to compute the current target uncorrected first for the covariances we will compute later
            # log_target_ratio_uncorrected = (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
            # log_w_uncorrected = log_w_norm_last_t + log_target_ratio_uncorrected
            # log_w_unc_norm = log_w_uncorrected - logsumexp(log_w_uncorrected)

            if store_D:
                for mk in mk_list:
                    self.log_D_t[new_t][mk] = np.NINF
                    self.log_Z_t[new_t][mk] = np.NINF

            N_particles = log_w_norm_last_t.shape[0]

            log_target_ratio = np.zeros(theta.shape[0])
            for mk, m_indices in models_grouped_by_indices.items():
                # importance sample ahead and compute evidence ratio
                log_w_increment_mk = (new_t - last_t) * (
                    llh[m_indices] + log_prior[m_indices]
                ) + (last_t - new_t) * pmodel.evalStartingDistribution(theta[m_indices])
                mk_particles = log_w_increment_mk.shape[0]
                if VERBOSITY_HIGH():
                    print("logsumexp(last weights) = ", logsumexp(log_w_norm_last_t))
                log_w_norm_last_t_mk = (
                    log_w_norm_last_t[m_indices]
                    + np.log(N_particles)
                    - np.log(mk_particles)
                )  # set to log(1/M), assumes D_t_k not in weights
                if VERBOSITY_HIGH():
                    print(
                        "logsumexp(last weights for model {}) = ".format(mk),
                        logsumexp(log_w_norm_last_t_mk),
                    )
                log_w_unnorm_mk = log_w_norm_last_t_mk + log_w_increment_mk
                log_Z_t_k_Z_t1_k = logsumexp(log_w_unnorm_mk)
                log_Z_t_k = log_Z_t_k_Z_t1_k + self.log_Z_t[last_t][mk]
                if False:
                    # Naive, does not predict Z_T_k
                    # W_t_k = Z_t_k * Z_T_k ^ {-gamma_t} # given perfect Z_T_k
                    #       = W_t1_k * Z_t_k/Z_t1_k
                    # log_D_t_k = (1.-new_t) * ((1./(1-last_t)) * self.log_D_t[last_t][mk] + log_Z_t_k_Z_t1_k)
                    log_D_t_k = log_Z_t_k - new_t * log_Z_t_k
                elif False:
                    # Linearly predicts Z_T_k in log space
                    approx_log_Z_T_k_Z_t_k = (
                        (1 - new_t) / (new_t - last_t)
                    ) * log_Z_t_k_Z_t1_k
                    # log_D_t_k = log_Z_t_k * (1.-new_t) - (new_t) * approx_log_Z_T_k_Z_t_k
                    log_D_t_k = (
                        log_Z_t_k - new_t * log_Z_t_k - new_t * approx_log_Z_T_k_Z_t_k
                    )
                elif False:
                    # hack
                    log_D_t_k = log_Z_t_k - (new_t**new_t) * log_Z_t_k
                elif False:
                    # hack number 2: beta = 1-gamma_t(1-gamma_t)
                    log_D_t_k = log_Z_t_k - (1 - new_t * (1 - new_t)) * log_Z_t_k
                elif False:
                    # BIC approx
                    k = pmodel.dim(mk)
                    if VERBOSITY_HIGH():
                        print("model dim({})={}".format(mk, k))
                    log_Z_T_k_BIC = -2 * np.max(llh[m_indices]) + k * np.log(
                        1000
                    )  # TODO get this in a sec. lower values prolly lead to greater torpidity!
                    log_D_t_k = log_Z_t_k + new_t * log_Z_T_k_BIC
                elif False:
                    # importance sample ahead to final target
                    log_w_increment_mk_T = (1 - new_t) * (
                        llh[m_indices] + log_prior[m_indices]
                    ) - (1 - new_t) * pmodel.evalStartingDistribution(theta[m_indices])
                    # compute ESS
                    log_w_unnorm_mk_T = log_w_unnorm_mk + log_w_increment_mk_T
                    log_Z_T_k_Z_t_k = logsumexp(log_w_unnorm_mk_T)
                    log_w_norm_mk_T = log_w_unnorm_mk_T - log_Z_T_k_Z_t_k
                    alpha = np.exp(
                        -logsumexp(2 * log_w_norm_mk_T) - np.log(mk_particles)
                    )
                    if VERBOSITY_HIGH():
                        print("ESS_tT[{}] alpha = {}".format(mk, alpha))
                    # W_t_k = Z_t_k * Z_T_k ^ {-gamma_t} # given perfect Z_T_k
                    #       = Z_t_k ^ {1-gamma_t) * (Z_T_k/Z_t_k)^{-gamma_t)
                    # log_D_t_k = (1.-new_t) * ((1./(1-last_t)) * self.log_D_t[last_t][mk] + log_Z_t_k_Z_t1_k) - (new_t-last_t) * log_Z_T_k_Z_t_k
                    log_D_t_k = log_Z_t_k * (1.0 - new_t) - (new_t) * log_Z_T_k_Z_t_k
                elif False:
                    # Rao Blackwellisation attempt
                    # importance sample ahead to final target
                    log_w_increment_mk_T = (1 - new_t) * (
                        llh[m_indices] + log_prior[m_indices]
                    ) - (1 - new_t) * pmodel.evalStartingDistribution(theta[m_indices])
                    # compute ESS
                    log_w_unnorm_mk_T = (-new_t) * log_w_unnorm_mk + (
                        -new_t
                    ) * log_w_increment_mk_T
                    log_Z_T_k_Z_t_k_neg_gamma_t = (-new_t) * logsumexp(
                        log_w_unnorm_mk
                    ) + logsumexp(-new_t * log_w_increment_mk_T)
                    log_w_norm_mk_T = log_w_unnorm_mk_T - log_Z_T_k_Z_t_k_neg_gamma_t
                    alpha = np.exp(
                        -logsumexp(2 * log_w_norm_mk_T) - np.log(mk_particles)
                    )
                    if VERBOSITY_HIGH():
                        print("RB ESS_tT[{}] alpha = {}".format(mk, alpha))
                    # W_t_k = Z_t_k * Z_T_k ^ {-gamma_t} # given perfect Z_T_k
                    #       = Z_t_k ^ {1-gamma_t) * (Z_T_k/Z_t_k)^{-gamma_t)
                    # log_D_t_k = (1.-new_t) * ((1./(1-last_t)) * self.log_D_t[last_t][mk] + log_Z_t_k_Z_t1_k) - (new_t-last_t) * log_Z_T_k_Z_t_k
                    log_D_t_k = log_Z_t_k * (1.0 - new_t) + log_Z_T_k_Z_t_k_neg_gamma_t
                elif False:
                    # Raw, no rao-blackwellisation
                    # importance sample ahead to final target
                    log_w_increment_mk_T = (1 - last_t) * (
                        llh[m_indices] + log_prior[m_indices]
                    ) - (1 - last_t) * pmodel.evalStartingDistribution(theta[m_indices])
                    # compute ESS
                    log_w_unnorm_mk_T = log_w_norm_last_t_mk + log_w_increment_mk_T
                    log_Z_T_k_Z_t1_k = logsumexp(log_w_unnorm_mk_T)
                    log_w_norm_mk_T = log_w_unnorm_mk_T - log_Z_T_k_Z_t1_k
                    alpha = np.exp(logsumexp(2 * log_w_norm_mk_T))
                    if VERBOSITY_HIGH():
                        print("Raw ESS_tT[{}] alpha = {}".format(mk, alpha))
                        print("log_Z_T_k_Z_t1_k[{}]".format(mk), log_Z_T_k_Z_t1_k)
                    log_D_t_k = (
                        log_Z_t_k
                        - new_t * self.log_Z_t[last_t][mk]
                        - new_t * log_Z_T_k_Z_t1_k
                    )
                    log_D_t_k = -log_D_t_k
                elif True:
                    # Using a tight lower bound for Z_T/Z_t_1, we proceed
                    log_LB_Z_T_k_Z_t1_k = (
                        log_Z_t_k_Z_t1_k * (1 - last_t) / (new_t - last_t)
                    )
                    if VERBOSITY_HIGH():
                        print(
                            "Lower bound log Z_T_k_Z_t1_k [{}] = {}".format(
                                mk, log_LB_Z_T_k_Z_t1_k
                            )
                        )
                    log_D_t_k = log_Z_t_k - new_t * (
                        self.log_Z_t[last_t][mk] + log_LB_Z_T_k_Z_t1_k
                    )
                    log_D_t_k = -log_D_t_k
                elif False:
                    # pi(x)pi(mode)^{1-gamma_t} method from Tawn using the lower bound to Z_T,k/Z_t-1,k
                    # Using a tight lower bound for Z_T/Z_t_1, we proceed
                    # Best so far. I can see why this is wrong.
                    log_LB_Z_T_k_Z_t1_k = (
                        log_Z_t_k_Z_t1_k * (1 - last_t) / (new_t - last_t)
                    )
                    if VERBOSITY_HIGH():
                        print(
                            "Lower bound log Z_T_k_Z_t1_k [{}] = {}".format(
                                mk, log_LB_Z_T_k_Z_t1_k
                            )
                        )
                    log_D_t_k = (-new_t) * (
                        self.log_Z_t[last_t][mk] + log_LB_Z_T_k_Z_t1_k
                    )
                elif False:
                    # pi(x)pi(mode)^{1-gamma_t} method from Tawn using the lower bound to Z_T,k/Z_t-1,k
                    # Using a tight lower bound for Z_T/Z_t_1, we proceed
                    # trying something.... still no. Do the maths properly. Work out the form of the weight W_t_k.
                    log_LB_Z_T_k_Z_t1_k = (
                        log_Z_t_k_Z_t1_k * (1 - last_t) / (new_t - last_t)
                    )
                    if VERBOSITY_HIGH():
                        print(
                            "Lower bound log Z_T_k_Z_t1_k [{}] = {}".format(
                                mk, log_LB_Z_T_k_Z_t1_k
                            )
                        )
                    log_D_t_k = (
                        (-new_t)
                        * (1 - new_t)
                        * (self.log_Z_t[last_t][mk] + log_LB_Z_T_k_Z_t1_k)
                    )
                else:
                    # Rao Blackwellisation attempt 2 - reverse
                    # ALL THE BADNESS 10,000
                    # importance sample ahead to final target
                    log_w_increment_mk_t1_T = (
                        llh[m_indices] + log_prior[m_indices]
                    ) - pmodel.evalStartingDistribution(theta[m_indices])
                    # compute ESS
                    log_Z_T_k_Z_t1_k_neg_gamma_t = (
                        new_t * (1 - last_t) * logsumexp(-log_w_increment_mk_t1_T)
                    )
                    if VERBOSITY_HIGH():
                        print(
                            "log_Z_T_k_Z_t1_k_neg_gamma_t[{}]".format(mk),
                            log_Z_T_k_Z_t1_k_neg_gamma_t,
                        )
                    log_D_t_k = (
                        log_Z_t_k
                        - new_t * self.log_Z_t[last_t][mk]
                        + log_Z_T_k_Z_t1_k_neg_gamma_t
                    )

                log_D_t1_k_star = self.log_D_t[last_t][
                    mk
                ]  # delete this whole term log_D_t1_k_star

                # commpute using stored final target moments.
                # log_D_t_k = computeLogDtk(mk,new_t,self.cov_pi_estimate[mk],self.mu_pi_estimate[mk],self.cov_s[mk],self.cov_s_inv[mk],self.mu_s[mk])
                # log_D_t1_k_star = computeLogDtk(mk,last_t,self.cov_pi_estimate[mk],self.mu_pi_estimate[mk],self.cov_s[mk],self.cov_s_inv[mk],self.mu_s[mk])

                if VERBOSITY_HIGH():
                    print(
                        "ZWSSMC1 Uncorrected log_D_t[{}][{}]={}".format(
                            new_t, mk, log_D_t_k
                        )
                    )
                log_D_t_k = log_D_t_k + (
                    ((1 - new_t) / (1 - last_t)) ** (1000 / new_t)
                ) * (self.log_D_t[last_t][mk] - log_D_t1_k_star)
                # log_D_t_k = log_D_t_k + np.sqrt((1-new_t)/(1-last_t))*np.exp(-10./(new_t)*(new_t-last_t))*(self.log_D_t[last_t][mk] - log_D_t1_k_star) # old one
                # log_D_t_k = log_D_t_k + np.sqrt(1-new_t)*np.exp(-1./(new_t)*(new_t-last_t))*(self.log_D_t[last_t][mk] - log_D_t1_k_star)
                if VERBOSITY_HIGH():
                    print(
                        "ZWSSMC1 Corrected log_D_t[{}][{}]={}".format(
                            new_t, mk, log_D_t_k
                        )
                    )
                log_target_ratio[m_indices] = (
                    log_D_t_k
                    - self.log_D_t[last_t][mk]
                    + (new_t - last_t) * (llh[m_indices] + log_prior[m_indices])
                    + (last_t - new_t)
                    * pmodel.evalStartingDistribution(theta[m_indices])
                )
                if store_D:
                    self.log_D_t[new_t][
                        mk
                    ] = log_D_t_k  # TODO now test, and report these weights!
                    self.log_Z_t[new_t][
                        mk
                    ] = log_Z_t_k  # TODO now test, and report these weights!
            return log_target_ratio

    def preImportanceSampleHook(
        self, last_t, llh, log_prior, smcq, pmodel, next_t=None
    ):
        """
        In this hook we pre-compute the final conditional target moments given current particles
        """
        pass
        # self.precomputeConditionalTargetMoments(last_t,llh,log_prior,pmodel,smcq.theta,smcq.log_w_norm,next_t)


# A simplified estimation of final target moments where the pi_t is standard gaussian.
# TODO list for implementing Transport WSSMC
# 1) New TWSSMC class which has the same look-ahead as WSSMC for each model, but then picks a temp average to all. Probably just use standard ESS look ahead for all models?
# 2) New estimateFinalTargetMoments() function that assumes transports have been done on prior
# 3) Pushforward current distribution for each model.
# 4) I can't think of anything else. Maybe the computeIncrementalWeights() function? Also, the preImportanceSampleHook()?

from rjlab.transforms.base import *
import nflows
from nflows.distributions.uniform import *
from nflows.transforms import *


class TWSSMC1(WSSMC1):
    """
    Transport Weight Stabilising SMC1 where each model is reweighted online at each iteration to avoid torpid sampling over the temperature sequence.
    """

    # def makeSMCQuantities(self,N,llh,log_prior,theta,pmodel,mk):
    #    """
    #    Overrides parent method to just instantiate an object of WSSMCQuantities
    #    """
    #    return WSSMCQuantities(np.arange(N), llh, log_prior,theta,pmodel,mk)
    # def appendToMixtureTargetDensity(self,smc_quantities, llh, log_prior, theta, gamma_t):
    #    # init previous_model_targets each time
    #    #print("append to target\n",theta)
    #    self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
    #    self.previous_model_targets.addComponent(self.initialdensity)
    #    self.previous_model_targets.addComponent(smc_quantities.makePPPD(gamma_t,self.log_D_t[gamma_t]))
    # def init_more2(self):
    #    self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
    #    pass
    def init_more(self):
        """
        Creates a dict of D_{t,k} values for each model. D_{0,k}=1 for all k.
        """
        # rest of the owl
        self.log_D_t = {}
        self.log_D_t[0] = {}
        for mk in self.pmodel.getModelKeys():
            self.log_D_t[0][mk] = 0
        # assert(self.tempseq is not None, "Temperature sequence must be provided for Transport WSSMC")
        # in an incarnation of this algo, mu_s and cov_s represent pussforward IS dist mean and diagonalised cov.
        # Similarly for mu_pi and cov_pi.
        self.mu_s = {}
        self.cov_s = {}
        self.cov_s_inv = {}
        self.mu_pi_estimate = {}
        self.cov_pi_estimate = {}

        # set up transports dict
        self.transports = {}

        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
        # self.init_more2()

    def getInitialDistributionMoments(self, mk, ndraws=10000):
        # we need to estimate mu_s and cov_s at every SMC step.
        # sample prior and get empirical moments
        theta = self.starting_dist.draw(10000)  # TODO get N from input
        models_grouped_by_indices = self.enumerateModels(theta)
        # for mk, m_indices in models_grouped_by_indices.items():
        m_indices = models_grouped_by_indices[mk]
        if True:  # mad hack
            theta_mk_cc = self.pmodel.concatAllParameters(theta[m_indices], mk)
            T_theta_mk_cc, ld = self.transports[mk]._transform.inverse(
                (torch.tensor(theta_mk_cc, dtype=torch.float32))
            )
            # T_theta_mk_cc, ld = self.transports[mk]._transform.forward((torch.tensor(theta_mk_cc,dtype=torch.float32)))
            T_theta_mk = self.pmodel.deconcatAllParameters(
                T_theta_mk_cc.detach().numpy(), theta[m_indices], mk
            )
            self.mu_s[mk], self.cov_s[mk] = self.pmodel._estimatemoments(T_theta_mk, mk)
            # print("cov_s",self.cov_s[mk],"shape",self.cov_s[mk].shape)
            if len(self.cov_s[mk].shape) < 2:
                self.cov_s_inv[mk] = 1.0 / self.cov_s[mk]
            else:
                self.cov_s_inv[mk] = safe_inv(self.cov_s[mk])

    # TODO delete this method and just use parent
    def computeIncrementalWeights(
        self,
        new_t,
        last_t,
        llh,
        log_prior,
        pmodel,
        theta,
        log_w_norm_last_t,
        store_D=False,
    ):
        """
        stratify by model
        compute each model D_t,k weight correction
        Also store it.

        This version uses a transport to pushforward the particles at the current target. May slow down Bisect.
        Actually this version does not run the transport at all. So we should delete this method.
        """
        mk_list = self.pmodel.getModelKeys()
        if store_D:
            self.log_D_t[new_t] = {}  # TODO skip if new_t==0
        if len(mk_list) == 1:
            # case where we are doing single model SMC and don't use weight-stabilising correction.
            self.log_D_t[new_t][mk_list[0]] = 0  # always 0.
            log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
                (1 - new_t) - (1 - last_t)
            ) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        elif new_t == 0:
            for mk in mk_list:
                self.log_D_t[new_t][mk] = 0
                # todo set current pi_t to s?
            log_target_ratio = (new_t - last_t) * (llh + log_prior) + (
                (1 - new_t) - (1 - last_t)
            ) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        elif new_t == last_t:
            log_target_ratio = 0  # (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
            return log_target_ratio
        else:
            # compute weight stabilising correction
            models_grouped_by_indices = self.enumerateModels(theta)
            # for mk in mk_list:
            mu_pi_t = {}
            cov_pi_t = {}
            mu_pi = {}
            cov_pi = {}
            # we need to compute the current target uncorrected first for the covariances we will compute later
            # log_target_ratio_uncorrected = (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
            # log_w_uncorrected = log_w_norm_last_t + log_target_ratio_uncorrected
            # log_w_unc_norm = log_w_uncorrected - logsumexp(log_w_uncorrected)

            if store_D:
                for mk in mk_list:
                    self.log_D_t[new_t][mk] = 0

            log_target_ratio = np.zeros(theta.shape[0])
            for mk, m_indices in models_grouped_by_indices.items():
                # commpute using stored final target moments.
                log_D_t_k = computeLogDtk(
                    mk,
                    new_t,
                    self.cov_pi_estimate[mk],
                    self.mu_pi_estimate[mk],
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )
                log_D_t1_k_star = computeLogDtk(
                    mk,
                    last_t,
                    self.cov_pi_estimate[mk],
                    self.mu_pi_estimate[mk],
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )

                if VERBOSITY_HIGH():
                    print("Uncorrected log_D_t[{}][{}]={}".format(new_t, mk, log_D_t_k))
                log_D_t_k = log_D_t_k + (
                    ((1 - new_t) / (1 - last_t)) ** (1000 / new_t)
                ) * (self.log_D_t[last_t][mk] - log_D_t1_k_star)
                # log_D_t_k = log_D_t_k + np.sqrt((1-new_t)/(1-last_t))*np.exp(-10./(new_t)*(new_t-last_t))*(self.log_D_t[last_t][mk] - log_D_t1_k_star) # old one
                # log_D_t_k = log_D_t_k + np.sqrt(1-new_t)*np.exp(-1./(new_t)*(new_t-last_t))*(self.log_D_t[last_t][mk] - log_D_t1_k_star)
                if VERBOSITY_HIGH():
                    print("Corrected log_D_t[{}][{}]={}".format(new_t, mk, log_D_t_k))
                log_target_ratio[m_indices] = (
                    log_D_t_k
                    - self.log_D_t[last_t][mk]
                    + (new_t - last_t) * (llh[m_indices] + log_prior[m_indices])
                    + (last_t - new_t)
                    * pmodel.evalStartingDistribution(theta[m_indices])
                )
                if store_D:
                    self.log_D_t[new_t][
                        mk
                    ] = log_D_t_k  # TODO now test, and report these weights!
            return log_target_ratio

    def precomputeConditionalTargetMoments(
        self, last_t, llh, log_prior, pmodel, theta, log_w_norm_last_t, next_t
    ):
        """
        We hypothesise that there is an "optimal" temperature at which to estimate the final target moments from the current set of particles.
        We don't know what it is, so we choose an ESS threshold and bisect till we find t that gives that threshold.
        The ESS is not for all particles, but rather for each model. A conditional ESS maybe.

        """
        mk_list = pmodel.getModelKeys()
        models_grouped_by_indices = self.enumerateModels(theta)
        # log_target_ratio_uncorrected = (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
        # log_w_uncorrected = log_w_norm_last_t + log_target_ratio_uncorrected
        # TODO stratify by model, do bisection for individual model ESS, use a minimum ESS for each model and a ratio of last ESS.

        # for mk, m_indices in models_grouped_by_indices.items():
        for mk in mk_list:
            if mk not in models_grouped_by_indices.keys():
                if VERBOSITY_HIGH():
                    print("Particle Impoverishment Encountered in model ", mk)
                self.mu_pi_estimate[mk] = self.mu_s[mk]
                self.cov_pi_estimate[mk] = self.cov_s[mk]
                # raise Exception("Particle impoverishment encountered in model {}".format(mk))
                continue

            m_indices = models_grouped_by_indices[mk]

            def quickWeightUpdateNorm(
                new_t, last_t, log_w_last, llh_mk, log_prior_mk, pmodel, theta_mk
            ):
                log_target_ratio_uncorrected = (new_t - last_t) * (
                    llh_mk + log_prior_mk
                ) + (last_t - new_t) * pmodel.evalStartingDistribution(theta_mk)
                log_w_uncorrected = log_w_last + log_target_ratio_uncorrected
                return log_w_uncorrected - logsumexp(log_w_uncorrected)

            def quickESSThreshold(
                new_t,
                last_t,
                log_w_last,
                llh_mk,
                log_prior_mk,
                pmodel,
                theta_mk,
                minESS,
                ratioESS=0.5,
            ):
                """
                given input weights, first compute a threshold = max(min(lastESS,minESS),ratioESS*lastESS)
                then update the weights using a naive annealing sequence
                """
                lastESS = np.exp(-logsumexp(2 * log_w_last))
                threshold = max(min(lastESS, minESS), ratioESS * lastESS)
                new_log_w_norm = quickWeightUpdateNorm(
                    new_t, last_t, log_w_last, llh_mk, log_prior_mk, pmodel, theta_mk
                )
                val = np.exp(-logsumexp(2 * new_log_w_norm))
                # print("ESS at t={} is {}".format(new_t,val))
                # print("threshold is ",threshold)
                return val - threshold

            # use a bisection optimiser to find a new t for this model to compute moments.
            log_w_last_mk_norm = log_w_norm_last_t[m_indices] - logsumexp(
                log_w_norm_last_t[m_indices]
            )

            if next_t is None:
                # print("last log_w\n",log_w_last_mk_norm,logsumexp(log_w_last_mk_norm))
                # First check ESS at 1
                if VERBOSITY_HIGH():
                    print(
                        "f(a)=",
                        quickESSThreshold(
                            last_t,
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                    )
                    print(
                        "f(b)=",
                        quickESSThreshold(
                            1,
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                    )
                if (
                    quickESSThreshold(
                        last_t,
                        last_t,
                        log_w_last_mk_norm,
                        llh[m_indices],
                        log_prior[m_indices],
                        self.pmodel,
                        theta[m_indices],
                        20,
                        0.5,
                    )
                    < 0
                ):
                    # numerical precision error - we've run out of usable particles to predict final conditional moments.
                    new_t = last_t
                elif (
                    quickESSThreshold(
                        1,
                        last_t,
                        log_w_last_mk_norm,
                        llh[m_indices],
                        log_prior[m_indices],
                        self.pmodel,
                        theta[m_indices],
                        20,
                        0.5,
                    )
                    > 0
                ):
                    new_t = 1.0
                else:
                    # TODO move minESS to config, or set to some multiple of the model dimension.
                    new_t, rres = scipy.optimize.bisect(
                        quickESSThreshold,
                        last_t,
                        1.0,
                        args=(
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                        full_output=True,
                        rtol=1e-6,
                    )
            else:
                new_t = next_t

            new_log_w_mk_norm = quickWeightUpdateNorm(
                new_t,
                last_t,
                log_w_last_mk_norm,
                llh[m_indices],
                log_prior[m_indices],
                self.pmodel,
                theta[m_indices],
            )
            if VERBOSITY_HIGH():
                print(
                    "Temperature at which to pre-compute conditional target for mk={} is t={}, ESS={}, lastESS={}".format(
                        mk,
                        new_t,
                        np.exp(-logsumexp(2 * new_log_w_mk_norm)),
                        np.exp(-logsumexp(2 * log_w_last_mk_norm)),
                    )
                )
            if True:
                # Train the transport flow. This is where this method differs from the parent method.
                self.transports[mk] = self.makeTransport(
                    mk, theta[m_indices], np.exp(new_log_w_mk_norm)
                )
                self.getInitialDistributionMoments(
                    mk
                )  # need to get pushforward initial IS distribution moments
                if False:
                    theta_mk_cc = self.pmodel.concatAllParameters(theta[m_indices], mk)
                    T_theta_mk_cc, mk_ld = self.transports[mk]._transform.forward(
                        torch.tensor(theta_mk_cc, dtype=torch.float32)
                    )
                    T_theta_mk = self.pmodel.deconcatAllParameters(
                        T_theta_mk_cc.detach().numpy(), theta[m_indices], mk
                    )
                    # resample theta from weights
                    resampled_idx = self.resample(
                        np.exp(new_log_w_mk_norm), max(1000, new_log_w_mk_norm.shape[0])
                    )
                    mu_pi_t, cov_pi_t = self.pmodel._estimatemoments(
                        T_theta_mk[resampled_idx], mk
                    )  # at current temp, should be standard normal
                    if VERBOSITY_HIGH():
                        print(
                            "Trained flow for pi_t mk {},\nmu={},\ncov={}".format(
                                mk, mu_pi_t, cov_pi_t
                            )
                        )
                else:
                    mu_pi_t = np.zeros_like(self.mu_s[mk])
                    cov_pi_t = np.eye(self.cov_s[mk].shape[0])
                # print("pi_t moments")
                # print(mu_pi_t)
                # print(cov_pi_t)
                (
                    self.mu_pi_estimate[mk],
                    self.cov_pi_estimate[mk],
                ) = estimateFinalTargetMoments(
                    new_t,
                    cov_pi_t,
                    mu_pi_t,
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )
                # print("estimated final pi[{}] moments".format(mk))
                # print(self.mu_pi_estimate[mk])
                # print(self.cov_pi_estimate[mk])
                dim = cov_pi_t.shape[
                    0
                ]  # assumes matrix, need to test this works for dim=1! Numpy is such a pain
            # Barycenter stuff or VI stuff
            if pmodel.hasConditionalApproximation(mk):
                mu_pi_approx, cov_pi_approx = pmodel.getConditionalApproximation(mk)
                # self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = combineCondApprox(mu_pi_approx,cov_pi_approx,pmodel,new_t,theta_mk,np.exp(new_log_w_mk_norm),self.cov_s[mk],self.cov_s_inv[mk],self.mu_s[mk]) # TODO just pass the estimates computed above
                self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = combineCondApprox(
                    mu_pi_approx,
                    cov_pi_approx,
                    pmodel,
                    new_t,
                    cov_pi_t,
                    mu_pi_t,
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                    last_t,
                )  # TODO just pass the estimates computed above

            # print("Final conditional target moments mk={} mu={} cov={}".format(mk,self.mu_pi_estimate[mk],self.cov_pi_estimate[mk]))
            if dim > 1:
                if VERBOSITY_HIGH():
                    print("Is cov PSD?", is_pos_def(self.cov_pi_estimate[mk]))
                if not is_pos_def(self.cov_pi_estimate[mk]):
                    if VERBOSITY_ERROR():
                        print("Cov not pos def for mk", mk, self.cov_pi_estimate[mk])
                    sys.exit(0)
            else:
                if VERBOSITY_HIGH():
                    print("Is cov PSD?", self.cov_pi_estimate[mk] > 0)
                if not self.cov_pi_estimate[mk] > 0:
                    if VERBOSITY_ERROR():
                        print("1D Cov not pos def for mk", mk, self.cov_pi_estimate[mk])
                    sys.exit(0)

    def makeTransport(self, mk, mk_theta, mk_theta_w):
        ls = nflows.transforms.Sigmoid()
        X = self.pmodel.concatAllParameters(mk_theta, mk)
        beta_dim = X.shape[1]
        bnorm = StandardNormal((beta_dim,))
        if ~np.any(np.isfinite(np.std(X, axis=0))):
            if VERBOSITY_ERROR():
                print(X)
                print("X is singular", X)
            sys.exit(0)
        # if self.affine:
        if False:
            # affine flow
            fn = nflows.transforms.InverseTransform(
                NaiveGaussianTransform(torch.Tensor(X), torch.Tensor(mk_theta_w))
            )
            return Flow(fn, bnorm)
        else:
            weights = torch.full([beta_dim], 1.0 / beta_dim)
            fn = FixedNorm(torch.Tensor(X), weights=torch.Tensor(mk_theta_w))
            return RationalQuadraticFlow2.factory(
                X, bnorm, ls, fn, input_weights=mk_theta_w
            )


class TWSSMC1OPR(TWSSMC1, WSSMC1OPR):
    pass


class PTWSSMC1(WSSMC1):
    """
    Prior Transport Weight Stabilising SMC1 where each model is reweighted online at each iteration to avoid torpid sampling over the temperature sequence.
    Transports priors only.
    """

    def makeSMCQuantities(self, N, llh, log_prior, theta, pmodel, mk):
        """
        Overrides parent method to just instantiate an object of WSSMCQuantities
        """
        return WSSMCQuantities(np.arange(N), llh, log_prior, theta, pmodel, mk)

    def appendToMixtureTargetDensity(
        self, smc_quantities, llh, log_prior, theta, gamma_t
    ):
        # init previous_model_targets each time
        # print("append to target\n",theta)
        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
        self.previous_model_targets.addComponent(self.initialdensity)
        self.previous_model_targets.addComponent(
            smc_quantities.makePPPD(gamma_t, self.log_D_t[gamma_t])
        )

    def init_more2(self):
        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)
        pass

    def init_more(self):
        """
        Creates a dict of D_{t,k} values for each model. D_{0,k}=1 for all k.
        """
        # rest of the owl
        self.log_D_t = {}
        self.log_D_t[0] = {}
        for mk in self.pmodel.getModelKeys():
            self.log_D_t[0][mk] = 0
        # assert(self.tempseq is not None, "Temperature sequence must be provided for Transport WSSMC")
        # in an incarnation of this algo, mu_s and cov_s represent pussforward IS dist mean and diagonalised cov.
        # Similarly for mu_pi and cov_pi.
        self.mu_s = {}
        self.cov_s = {}
        self.cov_s_inv = {}
        self.mu_pi_estimate = {}
        self.cov_pi_estimate = {}

        # set up transports dict
        self.transports = {}

        self.previous_model_targets = WeightedSingleModelMPD(self.pmodel)

        if VERBOSITY_HIGH():
            print("PTWSSMC training flows on starting dist")
        # train flow on initial dist
        self.getInitialDistributionMoments()
        # self.init_more2()
        self.pmodel.runFinalTargetMoments(self.transports)

    def getInitialDistributionMoments(self, ndraws=10000):
        # we need to estimate mu_s and cov_s at every SMC step.
        # sample prior and get empirical moments
        theta = self.starting_dist.draw(ndraws)  # TODO get N from input
        models_grouped_by_indices = self.enumerateModels(theta)
        for mk, m_indices in models_grouped_by_indices.items():
            theta_mk_cc = self.pmodel.concatAllParameters(theta[m_indices], mk)
            self.transports[mk] = self.makeTransport(mk, theta[m_indices], None)
            self.mu_s[mk] = np.zeros(theta_mk_cc.shape[1])
            self.cov_s[mk] = np.eye(theta_mk_cc.shape[1])
            self.cov_s_inv[mk] = self.cov_s[mk]
            # print("cov_s",self.cov_s[mk],"shape",self.cov_s[mk].shape)
            # if len(self.cov_s[mk].shape) < 2:
            #    self.cov_s_inv[mk] = 1./self.cov_s[mk]
            # else:
            #    self.cov_s_inv[mk] = safe_inv(self.cov_s[mk])

    def precomputeConditionalTargetMoments(
        self, last_t, llh, log_prior, pmodel, theta, log_w_norm_last_t, next_t
    ):
        """
        We hypothesise that there is an "optimal" temperature at which to estimate the final target moments from the current set of particles.
        We don't know what it is, so we choose an ESS threshold and bisect till we find t that gives that threshold.
        The ESS is not for all particles, but rather for each model. A conditional ESS maybe.

        """
        mk_list = pmodel.getModelKeys()
        models_grouped_by_indices = self.enumerateModels(theta)
        # log_target_ratio_uncorrected = (new_t-last_t) * (llh + log_prior) + (last_t-new_t) * pmodel.evalStartingDistribution(theta)
        # log_w_uncorrected = log_w_norm_last_t + log_target_ratio_uncorrected
        # TODO stratify by model, do bisection for individual model ESS, use a minimum ESS for each model and a ratio of last ESS.

        # for mk, m_indices in models_grouped_by_indices.items():
        for mk in mk_list:
            if mk not in models_grouped_by_indices.keys():
                if VERBOSITY_HIGH():
                    print("Particle Impoverishment Encountered in model ", mk)
                self.mu_pi_estimate[mk] = self.mu_s[mk]
                self.cov_pi_estimate[mk] = self.cov_s[mk]
                # raise Exception("Particle impoverishment encountered in model {}".format(mk))
                continue

            m_indices = models_grouped_by_indices[mk]

            def quickWeightUpdateNorm(
                new_t, last_t, log_w_last, llh_mk, log_prior_mk, pmodel, theta_mk
            ):
                log_target_ratio_uncorrected = (new_t - last_t) * (
                    llh_mk + log_prior_mk
                ) + (last_t - new_t) * pmodel.evalStartingDistribution(theta_mk)
                log_w_uncorrected = log_w_last + log_target_ratio_uncorrected
                return log_w_uncorrected - logsumexp(log_w_uncorrected)

            def quickESSThreshold(
                new_t,
                last_t,
                log_w_last,
                llh_mk,
                log_prior_mk,
                pmodel,
                theta_mk,
                minESS,
                ratioESS=0.5,
            ):
                """
                given input weights, first compute a threshold = max(min(lastESS,minESS),ratioESS*lastESS)
                then update the weights using a naive annealing sequence
                """
                lastESS = np.exp(-logsumexp(2 * log_w_last))
                threshold = max(min(lastESS, minESS), ratioESS * lastESS)
                new_log_w_norm = quickWeightUpdateNorm(
                    new_t, last_t, log_w_last, llh_mk, log_prior_mk, pmodel, theta_mk
                )
                val = np.exp(-logsumexp(2 * new_log_w_norm))
                # print("ESS at t={} is {}".format(new_t,val))
                # print("threshold is ",threshold)
                return val - threshold

            # use a bisection optimiser to find a new t for this model to compute moments.
            log_w_last_mk_norm = log_w_norm_last_t[m_indices] - logsumexp(
                log_w_norm_last_t[m_indices]
            )

            if next_t is None:
                # print("last log_w\n",log_w_last_mk_norm,logsumexp(log_w_last_mk_norm))
                # First check ESS at 1
                if VERBOSITY_HIGH():
                    print(
                        "f(a)=",
                        quickESSThreshold(
                            last_t,
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                    )
                    print(
                        "f(b)=",
                        quickESSThreshold(
                            1,
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                    )
                if (
                    quickESSThreshold(
                        last_t,
                        last_t,
                        log_w_last_mk_norm,
                        llh[m_indices],
                        log_prior[m_indices],
                        self.pmodel,
                        theta[m_indices],
                        20,
                        0.5,
                    )
                    < 0
                ):
                    # numerical precision error - we've run out of usable particles to predict final conditional moments.
                    new_t = last_t
                elif (
                    quickESSThreshold(
                        1,
                        last_t,
                        log_w_last_mk_norm,
                        llh[m_indices],
                        log_prior[m_indices],
                        self.pmodel,
                        theta[m_indices],
                        20,
                        0.5,
                    )
                    > 0
                ):
                    new_t = 1.0
                else:
                    # TODO move minESS to config, or set to some multiple of the model dimension.
                    new_t, rres = scipy.optimize.bisect(
                        quickESSThreshold,
                        last_t,
                        1.0,
                        args=(
                            last_t,
                            log_w_last_mk_norm,
                            llh[m_indices],
                            log_prior[m_indices],
                            self.pmodel,
                            theta[m_indices],
                            20,
                            0.5,
                        ),
                        full_output=True,
                        rtol=1e-6,
                    )
            else:
                new_t = next_t

            new_log_w_mk_norm = quickWeightUpdateNorm(
                new_t,
                last_t,
                log_w_last_mk_norm,
                llh[m_indices],
                log_prior[m_indices],
                self.pmodel,
                theta[m_indices],
            )
            if VERBOSITY_HIGH():
                print(
                    "Temperature at which to pre-compute conditional target for mk={} is t={}, ESS={}, lastESS={}".format(
                        mk,
                        new_t,
                        np.exp(-logsumexp(2 * new_log_w_mk_norm)),
                        np.exp(-logsumexp(2 * log_w_last_mk_norm)),
                    )
                )
            if True:
                theta_mk_cc = self.pmodel.concatAllParameters(theta[m_indices], mk)
                T_theta_mk_cc, mk_ld = self.transports[mk]._transform.forward(
                    torch.tensor(theta_mk_cc, dtype=torch.float32)
                )
                T_theta_mk = self.pmodel.deconcatAllParameters(
                    T_theta_mk_cc.detach().numpy(), theta[m_indices], mk
                )
                # resample theta from weights
                resampled_idx = self.resample(
                    np.exp(new_log_w_mk_norm), max(1000, new_log_w_mk_norm.shape[0])
                )
                mu_pi_t, cov_pi_t = self.pmodel._estimatemoments(
                    T_theta_mk[resampled_idx], mk
                )  # at current temp, should be standard normal
                # print("pi_t moments")
                # print(mu_pi_t)
                # print(cov_pi_t)
                (
                    self.mu_pi_estimate[mk],
                    self.cov_pi_estimate[mk],
                ) = estimateFinalTargetMoments(
                    new_t,
                    cov_pi_t,
                    mu_pi_t,
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                )
                # print("estimated final pi[{}] moments".format(mk))
                # print(self.mu_pi_estimate[mk])
                # print(self.cov_pi_estimate[mk])
                dim = cov_pi_t.shape[
                    0
                ]  # assumes matrix, need to test this works for dim=1! Numpy is such a pain
            # Barycenter stuff or VI stuff
            if pmodel.hasConditionalApproximation(mk):
                mu_pi_approx, cov_pi_approx = pmodel.getConditionalApproximation(mk)
                # self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = combineCondApprox(mu_pi_approx,cov_pi_approx,pmodel,new_t,theta_mk,np.exp(new_log_w_mk_norm),self.cov_s[mk],self.cov_s_inv[mk],self.mu_s[mk]) # TODO just pass the estimates computed above
                self.mu_pi_estimate[mk], self.cov_pi_estimate[mk] = combineCondApprox(
                    mu_pi_approx,
                    cov_pi_approx,
                    pmodel,
                    new_t,
                    cov_pi_t,
                    mu_pi_t,
                    self.cov_s[mk],
                    self.cov_s_inv[mk],
                    self.mu_s[mk],
                    last_t,
                )  # TODO just pass the estimates computed above

            # print("Final conditional target moments mk={} mu={} cov={}".format(mk,self.mu_pi_estimate[mk],self.cov_pi_estimate[mk]))
            if dim > 1:
                if VERBOSITY_HIGH():
                    print("Is cov PSD?", is_pos_def(self.cov_pi_estimate[mk]))
                if not is_pos_def(self.cov_pi_estimate[mk]):
                    if VERBOSITY_ERROR():
                        print("Cov not pos def for mk", mk, self.cov_pi_estimate[mk])
                    sys.exit(0)
            else:
                if VERBOSITY_HIGH():
                    print("Is cov PSD?", self.cov_pi_estimate[mk] > 0)
                if not self.cov_pi_estimate[mk] > 0:
                    if VERBOSITY_ERROR():
                        print("1D Cov not pos def for mk", mk, self.cov_pi_estimate[mk])
                    sys.exit(0)

    def makeTransport(self, mk, mk_theta, mk_theta_w=None):
        ls = nflows.transforms.Sigmoid()
        X = self.pmodel.concatAllParameters(mk_theta, mk)
        beta_dim = X.shape[1]
        bnorm = StandardNormal((beta_dim,))
        if ~np.any(np.isfinite(np.std(X, axis=0))):
            if VERBOSITY_ERROR():
                print(X)
                print("X is singular", X)
            sys.exit(0)
        # if self.affine:
        if False:
            # affine flow
            fn = nflows.transforms.InverseTransform(
                NaiveGaussianTransform(torch.Tensor(X), torch.Tensor(mk_theta_w))
            )
            return Flow(fn, bnorm)
        else:
            weights = torch.full([beta_dim], 1.0 / beta_dim)
            if mk_theta_w is not None:
                fn = FixedNorm(torch.Tensor(X), weights=torch.Tensor(mk_theta_w))
                return RationalQuadraticFlow2.factory(
                    X, bnorm, ls, fn, input_weights=mk_theta_w
                )
            else:
                fn = FixedNorm(torch.Tensor(X))
                return RationalQuadraticFlow2.factory(X, bnorm, ls, fn)
