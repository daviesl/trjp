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
from rjlab.samplers.smc import SingleModelMPD, PowerPosteriorParticleDensity
from rjlab.utils.progressbar import progress

np.set_printoptions(linewidth=200)


def VERBOSITY_HIGH():
    # return True # set when debugging
    return False


def VERBOSITY_ERROR():
    return True


class RJMCMC(object):
    def __init__(self, parametric_model, calibrate_draws=None):
        """
        Standard RJMCMC Sampler
        """
        self.pmodel = parametric_model
        self.nmutations = 0
        self.use_mmmpd = True
        from rjlab.variables.model import ParametricModelSpace

        assert isinstance(self.pmodel, ParametricModelSpace)
        # init future things
        self.init_more()
        if calibrate_draws is None:
            N = 1000
            self.pmodel.setStartingDistribution(self.pmodel)
            calibrate_draws = self.drawFromInitialDistribution(N)
        else:
            N = calibrate_draws.shape[0]

        if not self.use_mmmpd:
            self.pmodel.calibrateProposalsUnweighted(calibrate_draws, N, 1)
        else:
            self.pmodel.setStartingDistribution(self.pmodel)

            # create the MMMPD
            mmmpd = SingleModelMPD(self.pmodel)

            # add prior draws to avoid an exception
            init_draws = self.pmodel.draw(N)
            init_llh = self.pmodel.compute_llh(init_draws)
            init_log_prior = self.pmodel.compute_prior(init_draws)
            init_pppd = PowerPosteriorParticleDensity(
                self.pmodel,
                None,
                init_llh,
                init_log_prior,
                init_draws,
                0.0,
                np.log(np.ones_like(init_llh) * 1.0 / init_draws.shape[0]),
            )
            mmmpd.addComponent(init_pppd)

            # now add the target for calibration
            llh = self.pmodel.compute_llh(calibrate_draws)
            log_prior = self.pmodel.compute_prior(calibrate_draws)
            pppd = PowerPosteriorParticleDensity(
                self.pmodel,
                None,
                llh,
                log_prior,
                calibrate_draws,
                1.0,
                np.ones_like(llh) * 1.0 / N,
            )
            mmmpd.addComponent(pppd)
            self.pmodel.calibrateProposalsMMMPD(mmmpd, N, 1)

    def init_more(self):
        pass

    def drawFromInitialDistribution(self, M):
        # return  self.pmodel.sampleFromPrior(M)
        return self.pmodel.draw(M)

    def run(self, M=1000, start_theta=None):
        progress(0, M, status="Initialising RJMCMC")
        theta = np.zeros((M, self.pmodel.dim()))
        prop_theta = np.zeros((M, self.pmodel.dim()))
        llh = np.zeros(M)  # log likelihood
        log_prior = np.zeros(M)  # log prior
        if start_theta is not None:
            theta[0] = start_theta
        else:
            theta[0] = self.drawFromInitialDistribution(1)
        llh[0] = self.pmodel.compute_llh(theta[0][np.newaxis, :])
        log_prior[0] = self.pmodel.compute_prior(theta[0][np.newaxis, :])
        # print("starting llh, logprior",llh[0],log_prior[0])
        ar = np.zeros(M)
        progress(1, M, status="Running RJMCMC")
        for step in range(1, M):
            (
                theta[step],
                prop_theta[step],
                llh[step],
                log_prior[step],
                ar[step],
            ) = self.single_mutation(
                theta[step - 1][np.newaxis, :], np.array([llh[step - 1]]), 1
            )
            if VERBOSITY_HIGH():
                print(theta[step])
            progress(step, M, status="Running RJMCMC")
        return theta, prop_theta, llh, log_prior, ar

    def single_mutation(self, theta, llh, N):
        if False:
            assert N == llh.shape[0]
        # init the SMC quantities: weights and Z
        # kblocks = np.unique(theta[:,0])
        # print("kblocks",kblocks)
        # for i in kblocks:
        #    print("Single mutation count {} particles for model {}".format((theta[:,0]==i).sum(),i)) # n,k
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros(N)
        prop_llh = np.full(N, np.NINF)
        cur_prior = np.zeros(N)
        prop_prior = np.zeros(N)
        prop_id = np.zeros(N)
        prop_lpqratio = np.zeros(N)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        prop_theta[:], prop_lpqratio[:], prop_id[:] = self.pmodel.propose(
            theta, N
        )  # TODO FIXME self.pmodel.propose(theta,N)
        cur_prior[:] = self.pmodel.compute_prior(theta)
        prop_prior[:] = self.pmodel.compute_prior(prop_theta)
        ninfprioridx = np.where(~np.isfinite(cur_prior))
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(
            np.isfinite(prop_prior), np.isfinite(prop_lpqratio)
        )
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        # print("acceptance ratio",prop_lpqratio,prop_llh,llh,cur_prior,prop_prior) # TODO implement
        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta, prop_theta, prop_lpqratio, prop_llh, llh, cur_prior, prop_prior, 1
        )  # TODO implement

        # store acceptance ratios
        # self.rbar_list.append(rbar(log_acceptance_ratio,1,self.reverseEnumerateModels(theta),self.reverseEnumerateModels(prop_theta)))

        Proposal.setAcceptanceRates(
            prop_id, log_acceptance_ratio, 1
        )  # TODO for rjmcmc just log all accetance rates
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
        accepted_theta = prop_theta.copy()
        accepted_theta[reject_indices] = theta[reject_indices]
        prop_llh[reject_indices] = llh[reject_indices]
        prop_prior[reject_indices] = cur_prior[reject_indices]
        # a boolean array of accepted proposals
        accepted = np.ones(N)
        accepted[reject_indices] = 0
        # return prop_theta,prop_llh,accepted
        # print("log acceptance ratio",log_acceptance_ratio)
        self.nmutations += 1
        return (
            accepted_theta,
            prop_theta,
            prop_llh,
            prop_prior,
            np.exp(log_acceptance_ratio),
        )


class RJBridge(object):
    def __init__(self, parametric_model, target_draws):
        """
        target_draws should be an enumerated model list perhaps. TODO determine this.
        This class uses equation (16) from Bartolucci et al (2006)
        """
        self.pmodel = parametric_model
        self.nmutations = 0
        self.use_mmmpd = True
        assert isinstance(self.pmodel, ParametricModelSpace)
        # init future things
        self.init_more()
        self.target_draws = target_draws
        if not self.use_mmmpd:
            self.pmodel.calibrateProposalsUnweighted(
                calibrate_draws, calibrate_draws.shape[0], 1
            )
        else:
            N = target_draws.shape[0]
            self.pmodel.setStartingDistribution(self.pmodel)
            # create the MMMPD
            mmmpd = SingleModelMPD(self.pmodel)
            # add prior draws to avoid an exception
            init_draws = self.pmodel.draw(N)
            init_llh = self.pmodel.compute_llh(init_draws)
            init_log_prior = self.pmodel.compute_prior(init_draws)
            init_pppd = PowerPosteriorParticleDensity(
                self.pmodel,
                None,
                init_llh,
                init_log_prior,
                init_draws,
                0.0,
                np.log(np.ones_like(init_llh) * 1.0 / N),
            )
            mmmpd.addComponent(init_pppd)
            # add draws
            llh = self.pmodel.compute_llh(target_draws)
            log_prior = self.pmodel.compute_prior(target_draws)
            pppd = PowerPosteriorParticleDensity(
                self.pmodel,
                None,
                llh,
                log_prior,
                target_draws,
                1.0,
                np.log(np.ones_like(llh) * 1.0 / N),
            )
            mmmpd.addComponent(pppd)
            self.pmodel.calibrateProposalsMMMPD(mmmpd, N, 1)

    def init_more(self):
        pass

    def estimate_log_p_mk(self, target_draws, blocksize=None):
        theta = target_draws
        N = theta.shape[0]
        if blocksize is None:
            blocksize = N
        llh = np.zeros(N)  # log likelihood
        cur_prior = np.zeros(N)  # log prior
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros(N)
        prop_llh = np.full(N, np.NINF)
        prop_prior = np.zeros(N)
        prop_id = np.zeros(N)
        prop_lpqratio = np.zeros(N)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        # get indices for computation blocks
        nblocks = int(np.ceil((1.0 * N) / blocksize))
        blocks = [
            np.arange(i * blocksize, min(N, (i + 1) * blocksize))
            for i in range(nblocks)
        ]
        # print("blocks\n",blocks)
        # TODO reuse this computation from constructor for mmmpd
        progress(0, N, status="Initialising RJMCMC proposals")
        for bidx in blocks:
            if VERBOSITY_HIGH():
                print(
                    "proposing for block shape",
                    bidx.shape,
                    "index ",
                    bidx.min(),
                    " to ",
                    bidx.max(),
                )
            llh[bidx] = self.pmodel.compute_llh(theta[bidx])
            cur_prior[bidx] = self.pmodel.compute_prior(theta[bidx])
            prop_theta[bidx], prop_lpqratio[bidx], prop_id[bidx] = self.pmodel.propose(
                theta[bidx], N
            )
            prop_prior[bidx] = self.pmodel.compute_prior(prop_theta[bidx])
            progress(bidx.max(), N, status="Running RJMCMC proposals")
        ninfprioridx = np.where(~np.isfinite(cur_prior))  # is this used?
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(
            np.isfinite(prop_prior), np.isfinite(prop_lpqratio)
        )
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta, :])

        # print("acceptance ratio",prop_lpqratio,prop_llh,llh,cur_prior,prop_prior) # TODO implement
        log_acceptance_ratio[:] = self.pmodel.compute_lar(
            theta, prop_theta, prop_lpqratio, prop_llh, llh, cur_prior, prop_prior, 1
        )  # TODO implement
        if False:
            print("theta first 10:", theta[:10])
            print("prop theta first 10:", prop_theta[:10])
            print("Log AR first 10:", log_acceptance_ratio[:10])
            print("prior_first 10:", cur_prior[:10])
            print("prop prior_first 10:", prop_prior[:10])
            print("llh 10:", llh[:10])
            print("prop llh 10:", prop_llh[:10])
            print("Log AR last 10:", log_acceptance_ratio[-10:])
            print("prior_last 10:", cur_prior[-10:])
            print("prop prior_last 10:", prop_prior[-10:])
            print("llh last 10:", llh[-10:])
            print("prop llh last 10:", prop_llh[-10:])
            print(
                "non-fininte Log AR:",
                np.where(~np.isfinite(log_acceptance_ratio))[0],
                np.sum(~np.isfinite(log_acceptance_ratio)),
            )
        if False:
            import seaborn as sns
            import pandas as pd
            import matplotlib.pyplot as plt

            idxX8 = [0, 1, 3, 4, 6, 7, 9, 10]
            idxk8 = [2, 5, 8, 11]
            idxX4 = [0, 1, 3, 4]
            idxk4 = [2, 5, 6, 7]
            idxXFA = list(range(0, 15)) + list(range(16, 22))
            idxkFA = 15
            k0 = [1]  # [1,0,0,0] #[1,1,1,0]
            k1 = [2]  # [1,1,0,0] #[1,1,1,1]
            X = theta[:, idxXFA]
            k = theta[:, idxkFA]
            propX = prop_theta[:, idxXFA]
            propk = prop_theta[:, idxkFA]
            print("k==k0", k == k0)
            print("propk==k1", propk == k1)
            # k_up = np.logical_and(np.all(k==k0,axis=1),np.all(propk==k1,axis=1))
            # k_down = np.logical_and(np.all(k==k1,axis=1),np.all(propk==k0,axis=1))
            k_up = np.logical_and(k == k0, propk == k1)
            k_down = np.logical_and(k == k1, propk == k0)
            print("kup", k_up)
            labels = ["beta{}".format(i) for i in range(X.shape[1])]
            labels = labels + ["Proposed"]
            # didx = np.random.choice(X[ki].shape[0],size=1000,replace=False)
            # data = np.vstack([np.column_stack([propX[didx],np.ones(1000)+ki[didx]]),np.column_stack([X[didx],np.zeros(1000)+propki[didx]])])
            data_lower = np.vstack(
                [
                    np.column_stack([propX[k_down], np.ones((k_down).sum())]),
                    np.column_stack([X[k_up], np.zeros((k_up).sum())]),
                ]
            )
            # didx = np.random.choice(X[~ki].shape[0],size=1000,replace=False)
            data_upper = np.vstack(
                [
                    np.column_stack([propX[k_up], np.ones((k_up).sum())]),
                    np.column_stack([X[k_down], np.zeros((k_down).sum())]),
                ]
            )
            df = pd.DataFrame(data_lower, columns=labels)
            g = sns.PairGrid(df, hue=labels[-1], palette="Paired")
            g.map_lower(sns.scatterplot, s=1)
            g.map_diag(sns.distplot)
            g.add_legend()
            plt.show()
            # again for down
            df = pd.DataFrame(data_upper, columns=labels)
            g = sns.PairGrid(df, hue=labels[-1], palette="Paired")
            g.map_lower(sns.scatterplot, s=1)
            g.map_diag(sns.distplot)
            g.add_legend()
            plt.show()

        # Proposal.setAcceptanceRates(prop_id,log_acceptance_ratio,1) # TODO for rjmcmc just log all accetance rates
        # for pid,prop in Proposal.idpropdict.items():
        #    print("for pid {}\tprop {}\tar {}".format(pid,prop.printName(),prop.getLastAR()))

        # store acceptance ratios
        # self.rbar_list.append(rbar(log_acceptance_ratio,1,self.reverseEnumerateModels(theta),self.reverseEnumerateModels(prop_theta)))
        model_key_dict, reverse_key_ref = self.pmodel.enumerateModels(theta)
        prop_model_key_dict, prop_reverse_key_ref = self.pmodel.enumerateModels(
            prop_theta
        )
        # FIXME Wrong! Want the rbar style jump index
        # mk_ar = {mk:log_acceptance_ratio[idx] for mk,idx in model_key_dict.items()}
        # prop_mk_ar = {mk:log_acceptance_ratio[idx] for mk,idx in prop_model_key_dict.items()}
        # ar by pair (model key, prop model key). We need this for forward and reverse bayes factors.
        for mk, idx in model_key_dict.items():
            model_key_dict[mk].sort()
        for mk, idx in prop_model_key_dict.items():
            prop_model_key_dict[mk].sort()
        mkmklar = {
            (mk, prop_mk): log_acceptance_ratio[np.intersect1d(idx, prop_idx)]
            for mk, idx in model_key_dict.items()
            for prop_mk, prop_idx in prop_model_key_dict.items()
        }
        # compute the Bartolucci eqn (17) rao-blackwellised estimator for bayes factors
        mk_list = self.pmodel.getModelKeys()
        nmodels = len(mk_list)
        elar_mat = np.zeros((nmodels, nmodels))  # expected log acceptance ratio
        # elar_mat2 = np.zeros((nmodels,nmodels)) # expected log acceptance ratio
        for i, mk in enumerate(mk_list):
            for j, prop_mk in enumerate(mk_list):
                if mkmklar[(mk, prop_mk)].shape[0] > 0:
                    # elar_mat2[i,j] = logsumexp(mkmklar[(mk,prop_mk)]) - np.log(1000)
                    elar_mat[i, j] = logsumexp(mkmklar[(mk, prop_mk)]) - np.log(
                        mkmklar[(mk, prop_mk)].shape[0]
                    )
        # square matrix B_ij, with log(B_ij)=-log(B_ji), and log(B_ii)=0 down diagonal.
        # print(elar_mat,"\n",elar_mat2)
        if VERBOSITY_HIGH():
            print(elar_mat)
        log_B_star = elar_mat.T - elar_mat
        # log_B_star2 = elar_mat2.T - elar_mat2
        # print("log_B_star\n",log_B_star,"\nlog_B_star2\n",log_B_star2)
        if VERBOSITY_HIGH():
            print("log_B_star\n", log_B_star)
            print("-logsumexp(log_B_star,axis=0)", -logsumexp(log_B_star, axis=0))
        # now compute model probabilities from bayes factors
        # p(i) = B_i1 / (1 + sum_j (B_ji)) = 1 / (B_1i ( 1 + sum_{j=2..d} (B_ji)))
        if True:
            log_pmk = -logsumexp(log_B_star, axis=0)
            # so maybe the above, log_pmk, is the denominator.
            mk_lpmk = {mk: log_pmk[i] for i, mk in enumerate(mk_list)}
            if True:
                log_pmk_all = log_B_star - logsumexp(log_B_star, axis=0)
                log_pmk2 = logsumexp(log_pmk_all, axis=1) - np.log(nmodels)
                max_denom = np.argmax(log_pmk)
                log_pmk3 = log_pmk_all[:, max_denom]
                mk_lpmk2 = {mk: log_pmk2[i] for i, mk in enumerate(mk_list)}
                mk_lpmk3 = {mk: log_pmk3[i] for i, mk in enumerate(mk_list)}
                mk_pmk2 = {mk: np.exp(log_pmk2[i]) for i, mk in enumerate(mk_list)}
                mk_pmk3 = {mk: np.exp(log_pmk3[i]) for i, mk in enumerate(mk_list)}
                mk_pmk = {mk: np.exp(lp) for mk, lp in mk_lpmk.items()}
                if VERBOSITY_HIGH():
                    print("probs1 = ", mk_pmk)
                    print("probs1 sum = ", np.sum([p for mk, p in mk_pmk.items()]))
                p1sum = np.sum([p for mk, p in mk_pmk.items()])
                if VERBOSITY_HIGH():
                    print("probs1norm = ", {mk: p / p1sum for mk, p in mk_pmk.items()})
                    print("probs2 = ", mk_pmk2)
                    print("probs2 sum = ", np.sum([p for mk, p in mk_pmk2.items()]))
                    print("probs3 = ", mk_pmk3)
                    print("probs3 sum = ", np.sum([p for mk, p in mk_pmk3.items()]))
            # return mk_lpmk, mk_lpmk2, mk_lpmk3
            return mk_lpmk
        else:
            log_pmk = log_B_star - logsumexp(log_B_star, axis=0)
            if VERBOSITY_HIGH():
                print("log model probabilities for")
            # print(mk_list)
            if VERBOSITY_HIGH():
                print(log_pmk)
                print(logsumexp(log_pmk, axis=1) - np.log(nmodels))
            log_pmk = logsumexp(log_pmk, axis=1) - np.log(nmodels)
            return {mk: log_pmk[i] for i, mk in enumerate(mk_list)}
            # return prop_theta,prop_llh,prop_prior,np.exp(log_acceptance_ratio)
