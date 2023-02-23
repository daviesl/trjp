import sys

sys.path.append("../../")
from rjlab.distributions import *
from rjlab.variables import *
from rjlab.proposals.base import *
from rjlab.proposals.standard import *
from rjlab.transforms.base import *
import nflows as n
from nflows.distributions.uniform import *
from nflows.transforms import *
import shelve

PLOT_PROGRESS = False


class RJZGlobalRobustBlockVSProposalSaturated(Proposal):
    def __init__(
        self,
        blocksizes,
        blocknames,
        gammanames,
        betanames,
        within_model_proposal,
        is_naive=False,
    ):
        self.blocksizes = blocksizes
        self.blocknames = blocknames
        self.gammanames = gammanames
        self.betanames = betanames
        assert isinstance(within_model_proposal, Proposal)
        self.within_model_proposal = within_model_proposal
        self.rv_names = (
            gammanames + betanames
        )  # [self.k_name,self.t1_name,self.t2_name]
        self.is_naive = is_naive
        super(RJZGlobalRobustBlockVSProposalSaturated, self).__init__(
            self.rv_names + [self.within_model_proposal]
        )
        self.exclude_concat = gammanames
        # self.exclude_concat = [self.k_name]

    # def calibrateweighted(self,theta,weights,m_indices_dict,size,t):
    # def calibrate(self,theta,size,t):
    def calibratemmmpd(self, mmmpd, size, t):
        global PLOT_PROGRESS
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)
        mklist = self.pmodel.getModelKeys()  # get all keys
        cols = self.pmodel.generateRVIndices()
        orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(
            t, resample=True, resample_max_size=10000
        )
        orig_mkdict, rev = self.pmodel.enumerateModels(orig_theta)

        self.mk_logZhat = {}
        for mk in mklist:
            if mk in orig_mkdict.keys():
                self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0] * 1.0 / 10000)
            else:
                self.mk_logZhat[mk] = np.NINF

        if self.is_naive:
            # get dim
            theta, theta_w = mmmpd.getParticleDensityForTemperature(t, resample=False)
            rvidx = self.rv_indices[mklist[0]]
            X = np.column_stack([theta[:, rvidx[rvn]] for rvn in self.betanames])
            beta_dim = X.shape[1]
            self.flow = Flow(
                n.transforms.IdentityTransform(), StandardNormal((beta_dim,))
            )
        else:
            # train flow on all betas
            theta, theta_w = mmmpd.getParticleDensityForTemperature(t, resample=False)
            # print("theta_w",theta_w)
            rvidx = self.rv_indices[mklist[0]]
            X = np.column_stack([theta[:, rvidx[rvn]] for rvn in self.betanames])
            Y = np.column_stack([theta[:, rvidx[rvn]] for rvn in self.gammanames])
            U = norm(0, 10).rvs(X.shape)
            Ymask = Y.repeat(self.blocksizes, axis=1).astype(bool)
            X[~Ymask] = U[~Ymask]
            # print("xhsape yshape",X.shape,Y.shape)
            # reweight target to give equal weights to each model for training
            un, un_inv = np.unique(Y, return_inverse=True, axis=0)
            for i, yu in enumerate(un):
                # print("reweighting for model",yu)
                yidx = i == un_inv
                theta_w[yidx] = theta_w[yidx] - logsumexp(theta_w[yidx])
            theta_w = np.exp(theta_w - logsumexp(theta_w))
            # Y = np.dot(np.array([2, 4, 8]),Y.T).T.reshape((X.shape[0],1))
            # Y -= 0.5
            beta_dim = X.shape[1]
            gamma_dim = Y.shape[1]
            # print("calibrate X\n",X[-10:],"\nY\n",Y[-10:])
            ls = n.transforms.Sigmoid()
            if True:
                args = [nn.Linear(gamma_dim, 128), nn.ReLU()]
                for i in range(29):
                    args += [nn.Linear(128, 128), nn.ReLU()]
                args += [nn.Linear(128, 2 * beta_dim)]
                context_net = nn.Sequential(*args)

            bnorm = ConditionalDiagonalNormal(
                shape=(beta_dim,), context_encoder=context_net
            )
            # bnorm = ConditionalDiagonalNormal(shape=(beta_dim,),context_encoder=nn.Linear(Y.shape[1], X.shape[1]*2))
            # fn = FixedNorm(torch.Tensor(X),torch.Tensor(theta_w))

            bs = torch.Tensor(tuple(self.blocksizes)).type(torch.int)
            fn_param = MaskedFixedNorm(
                torch.Tensor(X),
                torch.Tensor(theta_w),
                Ymask,
                lambda y: y.repeat_interleave(bs, dim=1).type(torch.bool),
            )
            fn_aux = ConditionalMaskedTransform(
                FixedLinear(shift=0, scale=1.0 / 10),
                lambda y: ~y.repeat_interleave(bs, dim=1).type(torch.bool),
            )
            fn = CompositeTransform([fn_param, fn_aux])
            # self.flow = ConditionalRationalQuadraticFlow.factory(X,Y,base_dist=bnorm,boxing_transform=ls,initial_transform=fn,input_weights=theta_w)
            priordist = torch.distributions.normal.Normal(0, 10)
            self.flow = ConditionalMaskedRationalQuadraticFlow.factory(
                X,
                Y,
                ~Ymask,
                priordist,
                base_dist=bnorm,
                boxing_transform=ls,
                initial_transform=fn,
                input_weights=theta_w,
            )

        if PLOT_PROGRESS and t > 0.999:
            import seaborn as sns
            import pandas as pd
            import matplotlib.pyplot as plt

            Ylabel = np.dot(Y[:, 1:], np.array([1, 2]))  # .reshape((Y.shape[0],1))
            labels = ["beta{}".format(i) for i in range(beta_dim)]
            labels.append("Y")
            df = pd.DataFrame(np.column_stack([X, Ylabel]), columns=labels)
            g = sns.PairGrid(df, hue=labels[-1])
            g.map_lower(sns.scatterplot, s=1)
            g.map_diag(sns.distplot)
            plt.show()
            Z = np.zeros_like(X)
            for mk in mklist:
                yval = np.dot(list(mk)[1:], np.array([1, 2]))
                base, ld = self.transformToBase(theta[Ylabel == yval], mk)
                Z[Ylabel == yval, :] = np.column_stack(
                    [base[:, rvidx[rvn]] for rvn in self.betanames]
                )
            labels = ["z{}".format(i) for i in range(beta_dim)]
            labels.append("Y")
            df = pd.DataFrame(np.column_stack([Z, Ylabel]), columns=labels)
            g = sns.PairGrid(df, hue=labels[-1])
            g.map_lower(sns.scatterplot, s=1)
            g.map_diag(sns.distplot)
            plt.show()

        # set probabilities
        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        pp_mk_prob = np.exp(pp_mk_log_prob)

        def anneal_down(a, logp, thres):
            alogp = a * logp
            npa = np.exp(alogp - logsumexp(alogp))
            return np.max(thres - npa)

    def transformToBase(self, inputs, mk):
        # if self.getModelDim(mk)==0:
        #    return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        # Y = np.tile(list(mk),(X.shape[0],1))
        Y = np.tile(np.array(list(mk), dtype=np.float32), (X.shape[0], 1))
        # Y -= 0.5
        XX, logdet = self.flow._transform.forward(
            torch.tensor(X, dtype=torch.float32),
            context=torch.tensor(Y, dtype=torch.float32),
        )
        # do one volume preserving transform. Here just a permutation
        XXn = XX.detach().numpy()
        XXn = XXn[
            np.arange(len(XXn))[:, None], np.random.randn(*XXn.shape).argsort(axis=1)
        ]
        # print("transform to base",XX.shape,mk)
        return self.deconcatParameters(XXn, inputs, mk), logdet.detach().numpy()

    def transformFromBase(self, inputs, mk):
        # if self.getModelDim(mk)==0:
        #    return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        Y = np.tile(np.array(list(mk), dtype=np.float32), (X.shape[0], 1))
        # Y -= 0.5
        XX, logdet = self.flow._transform.inverse(
            torch.tensor(X, dtype=torch.float32),
            context=torch.tensor(Y, dtype=torch.float32),
        )
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )

    # def getModelDim(self,mk):
    #    return int(np.array([bs * list(mk)[i] for i,bs in enumerate(self.blocksizes)]).sum())
    #    #return int(np.array(list(mk)).sum()*self.blocksize) # + static extra params
    def toggleGamma(self, mk, tidx):
        mkl = list(mk)
        mkl[tidx] = 1 - mkl[tidx]
        return tuple(mkl)

    def toggleIDX(self, mk, new_mk):
        mkl = np.array(list(mk))
        new_mkl = np.array(list(new_mk))
        on = np.logical_and(new_mkl, np.logical_not(mkl))
        off = np.logical_and(mkl, np.logical_not(new_mkl))
        return np.where(on)[0], np.where(off)[0]

    def draw(self, theta, size=1):
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1  # std dev of auxiliary u dist

        prop_theta = theta.copy()

        # This method does the following
        # For each model
        #   select one gamma to toggle
        #   For that gamma, if toggling off, evaluate aux var
        #   otherwise draw aux var, transform to beta for toggled gamma

        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)

        model_enumeration, rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]
            params, splitdict = self.explodeParameters(theta, mk)
            betas = np.column_stack(
                [params[xn] for xn in self.betanames]
            )  # should only return active blocks
            gammas = np.column_stack(
                [params[xn] for xn in self.gammanames]
            )  # should return all gammas. Rows should be identical.
            gamma_vec = gammas[0]

            Tmktheta, lpq1 = self.transformToBase(mk_theta, mk)

            # global proposal to all models
            this_log_prob = self.mk_logZhat[mk] - logsumexp(pp_mk_logZ)
            pidx = np.random.choice(
                np.arange(pp_mk_logZ.shape[0]), p=np.exp(pp_mk_log_prob), size=mk_n
            )  # gagnon
            lpq_mk[mk] = this_log_prob - pp_mk_log_prob[pidx]

            proposed[mk], mk_idx[mk] = self.explodeParameters(Tmktheta, mk)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # for each mk in pidx
            # separate theta further into model transitions
            for p_i in np.unique(pidx):
                tn = (pidx == p_i).sum()
                new_mk = pp_mk_keys[p_i]
                # print("new mk",new_mk)
                at_idx = pidx == p_i
                if new_mk == mk:
                    (
                        prop_mk_theta[at_idx],
                        lpq_mk[mk][at_idx],
                        mk_prop_ids[at_idx],
                    ) = self.within_model_proposal.draw(mk_theta[at_idx], tn)
                    continue
                # get column ids for toggled on and off blocks
                on_idx, off_idx = self.toggleIDX(mk, new_mk)
                # toggle ons
                for idx in on_idx:
                    Tmktheta[at_idx] = self.setVariable(
                        Tmktheta[at_idx], self.gammanames[idx], np.ones(tn)
                    )
                # for toggle off
                for idx in off_idx:
                    Tmktheta[at_idx] = self.setVariable(
                        Tmktheta[at_idx], self.gammanames[idx], np.zeros(tn)
                    )
                # transform back
                prop_mk_theta[at_idx], lpq2 = self.transformFromBase(
                    Tmktheta[at_idx], new_mk
                )
                lpq_mk[mk][at_idx] += lpq1[at_idx] + lpq2
                mk_prop_ids[at_idx] = np.full(tn, id(self))
            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids
        # sys.exit(0)
        return prop_theta, logpqratio, prop_ids


class RJZGlobalBlockVSProposalIndiv(Proposal):
    def __init__(
        self,
        blocksizes,
        blocknames,
        gammanames,
        betanames,
        within_model_proposal,
        affine=False,
        propose_all=False,
        use_opr_calib=False,
    ):
        self.affine = affine
        self.propose_all = (
            propose_all  # Even if a model has zero particles, fit a proposal.
        )
        self.blocksizes = blocksizes
        self.blocknames = blocknames
        self.gammanames = gammanames
        self.betanames = betanames
        assert isinstance(within_model_proposal, Proposal)
        self.within_model_proposal = within_model_proposal
        self.rv_names = (
            gammanames + betanames
        )  # [self.k_name,self.t1_name,self.t2_name]
        super(RJZGlobalBlockVSProposalIndiv, self).__init__(
            self.rv_names + [self.within_model_proposal]
        )
        self.exclude_concat = gammanames
        self.use_opr_calib = use_opr_calib

    def calibratemmmpd(self, mmmpd, size, t):
        global PLOT_PROGRESS
        self.within_model_proposal.calibratemmmpd(mmmpd, size, t)
        mklist = self.pmodel.getModelKeys()  # get all keys
        cols = self.pmodel.generateRVIndices()

        self.flows = {}
        self.mk_logZhat = {}

        # USE_OPR_MP = True # Does not use WS target for some reason. Investigate. Also needs to add non-zero prop probs back in.

        if self.use_opr_calib:
            full_theta, full_theta_w = mmmpd.getParticleDensityForTemperature(
                t, resample=False
            )
            full_mkdict, rev = self.pmodel.enumerateModels(full_theta)
            full_N = full_theta.shape[0]

            for mk in mklist:
                if mk in full_mkdict.keys():
                    self.mk_logZhat[mk] = logsumexp(
                        full_theta_w[full_mkdict[mk]]
                    ) - np.log(full_N)
                else:
                    self.mk_logZhat[mk] = np.NINF

        else:
            orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(
                t, resample=True, resample_max_size=10000
            )
            orig_mkdict, rev = self.pmodel.enumerateModels(orig_theta)

            for mk in mklist:
                if mk in orig_mkdict.keys():
                    self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0] * 1.0 / 10000)
                else:
                    self.mk_logZhat[mk] = np.NINF

        # compute model probs from Z
        # Need to make sure we can precisely set floor of 0.001 to prop probs.
        # if self.propose_all:
        #    self.mk_logZhat = {mk:np.log(max(np.exp(p),0.001)) for mk,p in self.mk_logZhat.items()} # very hacky but does give non-zero proposal probs

        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        pp_mk_prob = np.exp(pp_mk_log_prob)

        # print("model probabilities") #{mk:np.exp(lZ) for mk,lZ in self.mk_logZhat.items()})
        # for mk,lZ in self.mk_logZhat.items():
        #    print(mk,2000*np.exp(lZ))
        # for i,mk in enumerate(pp_mk_keys):
        #    print(mk,pp_mk_prob[i])

        orig_theta, orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(
            t, resample=True, resample_max_size=10000
        )
        orig_mkdict, rev = self.pmodel.enumerateModels(orig_theta)

        # print("Calibrating TRJP for model {} with Z_hat={}".format(mk,np.exp(self.mk_logZhat[mk])))
        for mk in mklist:
            if self.use_opr_calib:
                # mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=10000)
                mk_theta, mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(
                    mk, t, resample=False
                )
                self.flows[mk] = self.makeFlow(
                    mk,
                    mk_theta,
                    np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w))),
                )
            else:
                if mk in orig_mkdict.keys():
                    mk_theta = orig_theta[orig_mkdict[mk]]
                    mk_theta_w = orig_theta_w[orig_mkdict[mk]]
                    # model_key_indices, m_tuple_indices = self.getModel().enumerateModels(mk_theta)
                    # print("NOT USING OPR")
                    self.flows[mk] = self.makeFlow(
                        mk,
                        mk_theta,
                        np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w))),
                    )
                else:
                    self.flows[mk] = self.makeDummyFlow(mk)

    def makeDummyFlow(self, mk):
        dim = self.getModelDim(mk)
        bnorm = StandardNormal((dim,))
        fn = n.transforms.IdentityTransform()
        return Flow(fn, bnorm)

    def makeFlow(self, mk, mk_theta, mk_theta_w):
        ls = n.transforms.Sigmoid()
        X = self.concatParameters(mk_theta, mk)
        beta_dim = X.shape[1]
        bnorm = StandardNormal((beta_dim,))
        if ~np.any(np.isfinite(np.std(X, axis=0))):
            print(X)
            print("X is singular", X)
            sys.exit(0)
        if self.affine:
            fn = n.transforms.InverseTransform(
                NaiveGaussianTransform(torch.Tensor(X), torch.Tensor(mk_theta_w))
            )
            return Flow(fn, bnorm)
        else:
            weights = torch.full([beta_dim], 1.0 / beta_dim)
            fn = FixedNorm(torch.Tensor(X))  # ,weights=weights)
            return RationalQuadraticFlow2.factory(
                X, bnorm, ls, fn
            )  # ,input_weights=weights.detach().numpy())

    def transformToBase(self, inputs, mk):
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.forward(
            torch.tensor(X, dtype=torch.float32)
        )
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )

    def transformFromBase(self, inputs, mk):
        if self.getModelDim(mk) == 0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs, mk)
        XX, logdet = self.flows[mk]._transform.inverse(
            torch.tensor(X, dtype=torch.float32)
        )
        return (
            self.deconcatParameters(XX.detach().numpy(), inputs, mk),
            logdet.detach().numpy(),
        )

    def getModelDim(self, mk):
        return int(
            np.array([bs * list(mk)[i] for i, bs in enumerate(self.blocksizes)]).sum()
        )
        # return int(np.array(list(mk)).sum()*self.blocksize) # + static extra params

    def toggleGamma(self, mk, tidx):
        mkl = list(mk)
        mkl[tidx] = 1 - mkl[tidx]
        return tuple(mkl)

    def toggleIDX(self, mk, new_mk):
        mkl = np.array(list(mk))
        new_mkl = np.array(list(new_mk))
        on = np.logical_and(new_mkl, np.logical_not(mkl))
        off = np.logical_and(mkl, np.logical_not(new_mkl))
        return np.where(on)[0], np.where(off)[0]

    def draw(self, theta, size=1):
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1  # std dev of auxiliary u dist

        prop_theta = theta.copy()

        # This method does the following
        # For each model
        #   select one gamma to toggle
        #   For that gamma, if toggling off, evaluate aux var
        #   otherwise draw aux var, transform to beta for toggled gamma

        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i] = pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)

        model_enumeration, rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]
            params, splitdict = self.explodeParameters(theta, mk)
            betas = np.column_stack(
                [params[xn] for xn in self.betanames]
            )  # should only return active blocks
            gammas = np.column_stack(
                [params[xn] for xn in self.gammanames]
            )  # should return all gammas. Rows should be identical.
            gamma_vec = gammas[0]

            Tmktheta, lpq1 = self.transformToBase(mk_theta, mk)

            # global proposal to all models
            this_log_prob = self.mk_logZhat[mk] - logsumexp(pp_mk_logZ)
            pidx = np.random.choice(
                np.arange(pp_mk_logZ.shape[0]), p=np.exp(pp_mk_log_prob), size=mk_n
            )  # gagnon
            lpq_mk[mk] = this_log_prob - pp_mk_log_prob[pidx]

            proposed[mk], mk_idx[mk] = self.explodeParameters(Tmktheta, mk)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # for each mk in pidx
            # separate theta further into model transitions
            for p_i in np.unique(pidx):
                tn = (pidx == p_i).sum()
                new_mk = pp_mk_keys[p_i]
                at_idx = pidx == p_i
                if new_mk == mk:
                    (
                        prop_mk_theta[at_idx],
                        lpq_mk[mk][at_idx],
                        mk_prop_ids[at_idx],
                    ) = self.within_model_proposal.draw(mk_theta[at_idx], tn)
                    continue
                # get column ids for toggled on and off blocks
                on_idx, off_idx = self.toggleIDX(mk, new_mk)
                # toggle ons
                for idx in on_idx:
                    for i in range(self.blocksizes[idx]):
                        cbs = int(np.array(self.blocksizes[:idx]).sum())
                        u = norm(0, sigma_u).rvs(tn)
                        log_u = norm(0, sigma_u).logpdf(u)  # for naive, mean is u
                        lpq_mk[mk][at_idx] -= log_u
                        Tmktheta[at_idx] = self.setVariable(
                            Tmktheta[at_idx], self.betanames[cbs + i], u
                        )  # for naive, update with random walk using old u
                    Tmktheta[at_idx] = self.setVariable(
                        Tmktheta[at_idx], self.gammanames[idx], np.ones(tn)
                    )
                # for toggle off
                for idx in off_idx:
                    for i in range(self.blocksizes[idx]):
                        cbs = int(np.array(self.blocksizes[:idx]).sum())
                        u = self.getVariable(
                            Tmktheta[at_idx], self.betanames[cbs + i]
                        ).flatten()
                        log_u = norm(0, sigma_u).logpdf(u)  # for naive, mean is u
                        lpq_mk[mk][at_idx] += log_u
                    Tmktheta[at_idx] = self.setVariable(
                        Tmktheta[at_idx], self.gammanames[idx], np.zeros(tn)
                    )
                # transform back
                prop_mk_theta[at_idx], lpq2 = self.transformFromBase(
                    Tmktheta[at_idx], new_mk
                )
                lpq_mk[mk][at_idx] += lpq1[at_idx] + lpq2
                mk_prop_ids[at_idx] = np.full(tn, id(self))
            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids
        return prop_theta, logpqratio, prop_ids
