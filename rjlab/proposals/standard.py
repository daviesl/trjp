import sys
from rjlab.distributions.base import *
from rjlab.proposals.base import *
from rjlab.samplers import *


class IndependentProposal(Proposal):
    def __init__(self, rv_name, proposal_distribution):
        """
        a single proposal distribution. Usually the prior. In independent copula proposal this is a parametric distribution fitted to the marginal.
        """
        assert isinstance(rv_name, str)
        super(IndependentProposal, self).__init__([rv_name])
        assert isinstance(proposal_distribution, Distribution)

        self.propdist = proposal_distribution

    def draw(self, theta, size=1):
        """
        Draw from prior
        The problem is I need to link the priors now.
        """
        # dim = len(self.rv_names)
        n = theta.shape[0]
        prop_lpqratio = np.zeros(n)
        return (
            self.propdist.draw(n),
            prop_lpqratio,
            np.full(n, id(self)),
        )  # TODO pq ratio is not 1. Compute it.


class NormalRandomWalkProposal(Proposal):
    def draw(self, theta, size=1):
        n = theta.shape[0]
        dim = len(self.rv_names)
        return (
            normal.rvs(theta, 1, dim),
            np.zeros(n),
            np.full(n, id(self)),
        )  # FIXME should be size, not dim


class MVNProposal(Proposal):
    """
    Does not have sub-proposals.
    """

    def __init__(self, proposals=[]):
        self.propscale = 1.0
        super(MVNProposal, self).__init__(proposals)

    def calibrate(self, theta, size, t):
        pmat = self.concatParameters(theta)
        m = pmat.shape[1]  # self.splitby_val # FIXME a hack
        self.propscale = 0.1 * (0.238**2) / m
        print("Proposal scale for (d={}) MVN is {}".format(m, self.propscale))

        n = pmat.shape[0]
        self.cov = np.cov(pmat.T)
        # seed with the identity weighted by nrows
        m = pmat.shape[1]
        if m > 1:
            self.cov += np.eye(m) * (max(1 - np.linalg.det(self.cov), 0))
        else:
            self.cov += max(1 - self.cov, 0)

    def draw(self, theta, size=1):
        pmat = self.concatParameters(theta)
        n = pmat.shape[0]
        m = pmat.shape[1]
        if m > 1:
            prop_pmat = pmat + multivariate_normal(
                np.zeros(m), self.cov * self.propscale
            ).rvs(n)
        else:
            prop_pmat = pmat + norm(0, self.cov * self.propscale).rvs(n)
        prop_theta = self.deconcatParameters(prop_pmat, theta)
        return prop_theta, np.zeros(n), np.full(n, id(self))


class EigDecComponentwiseNormalProposalTrial(Proposal):
    """
    Does not have sub-proposals.
    """

    def __init__(self, proposals=[]):
        self.propscale = 1.0
        super(EigDecComponentwiseNormalProposalTrial, self).__init__(proposals)
        self.dimension = 1
        self.propscale = 1

    # def setAR(self,ar,t):
    #    super(EigDecComponentwiseNormalProposal,self).setAR(ar,t)
    #    if len(self.ar)>10:
    #        # average the last 10
    #        avg_ar = np.mean(self.ar[-10:])
    #        self.propscale = np.exp(2.0 * self.dimension * (avg_ar - 0.44)) # not scaling by dim
    #        print("eig[{}].propscale = {} for ar {}".format(self.dimension,self.propscale, avg_ar))

    # def setAR(self,ar,t,n):
    #    super(EigDecComponentwiseNormalProposal,self).setAR(ar,t,n)
    #    # average the last 100
    #    if len(self.ar)<10:
    #        avg_ar = 0.44
    #    else:
    #        alen = min(len(self.ar)-1,100)
    #        #alen = len(self.ar)-1
    #        avg_ar = np.sum(np.array(self.ar[-alen:])*np.array(self.n_ar[-alen:])*np.array(self.t_ar[-alen:]))/np.sum(np.array(self.n_ar[-alen:])*np.array(self.t_ar[-alen:]))
    #    self.propscale = np.exp(2.0 * self.dimension * (min(avg_ar,0.7) - 0.44)) # not scaling by dim
    #    print("eig[{}].propscale = {} for ar {}".format(self.dimension,self.propscale, avg_ar))

    # def calibrate(self,theta,size,t):
    #    # compute eigenvalue decomposition here
    #    pmat = self.concatParameters(theta,self.getModelIdentifier())
    def calibratemmmpd(self, mmmpd, size, t):
        mk = self.getModelIdentifier()
        # print("Calibrating Component-wise MVN proposal for model {}".format(mk)) # TODO set as status
        theta, theta_w = mmmpd.getParticleDensityForModelAndTemperature(
            mk, t, resample=True, resample_max_size=500
        )

        pmat = self.concatParameters(theta, mk)

        m = pmat.shape[1]  # self.splitby_val # FIXME a hack
        self.dimension = m
        # FIXME the below gives extreme numbers.
        # self.propscale = np.exp(2.0 * m * (self.ar - 0.44)) # not scaling by dim
        # self.propscale = np.exp(- m)
        # self.propscale = (1.25-t)/m #(0.238**2)/ m

        # self.propscale = np.exp(-2*t)/m #(0.238**2)/ m

        # self.propscale = np.exp(2.0 * m * (min(self.getLastAR(),0.44) - 0.44)) # not scaling by dim
        # print("eig[{}].propscale = {} for ar {}".format(m,self.propscale, self.getLastAR()))

        # average the last 100
        # if len(self.ar)<10:
        #    avg_ar = 0.44
        # else:
        #    alen = min(len(self.ar)-1,10)
        #    #alen = len(self.ar)-1
        #    avg_ar = np.sum(np.array(self.ar[-alen:])*np.array(self.n_ar[-alen:]))/np.sum(self.n_ar[-alen:])
        #    if not np.isfinite(avg_ar):
        #        print("Non finite ar.",self.ar[-alen:],"\n",self.n_ar[-alen:])
        # self.propscale = np.exp(2.0 * self.dimension * (min(avg_ar,0.44) - 0.44)) # not scaling by dim
        # print("eig[{}].propscale = {} for ar {}".format(self.dimension,self.propscale, avg_ar))

        n = pmat.shape[0]
        cov = np.cov(pmat.T)
        if len(cov.shape) == 0:
            # store the valid indices
            self.valid_indices = np.arange(1)  # trivial.
            self.num_valid_indices = 1
            if not np.isfinite(cov):
                cov = 0.5
            self.eigvals = cov  # we just use variance to scale.
            ##print("cov for m=1 is {}".format(cov))
        elif cov.shape[0] == 0:
            self.valid_indices = np.arange(0)  # trivial.
            self.num_valid_indices = 0
            self.eigvals = 0
            return
        else:
            # print("Eig prop pmat shape = ",pmat.shape)
            # print("Eig[{}].pmat\n{}".format(m,pmat))
            ##print("Eig[{}].cov\n{}".format(self.getModelIdentifier(),cov))
            # seed with the identity weighted by nrows
            m = cov.shape[0]
            if not np.isfinite(cov).all():
                cov = np.eye(m) * 0.5
            # print("cov for m={} is {}".format(m,cov))
            while np.linalg.det(cov) < 0.001:
                cov += np.eye(m) * 0.001 * (max(1 - np.linalg.det(cov), 0))
                # print("Adjusted covariance due to low volume\n")
            # if m>1:
            #    cov += np.eye(m) * (max(1-np.linalg.det(cov),0))
            # else:
            #    cov += max(1-cov,0)
            if False:
                seed = np.eye(m)
                w = max(float(m - n), 0.0) / m
                # print("Eig[{}].w_seed={}".format(m,w))
                if ~np.isfinite(cov).any():
                    cov = seed
                else:
                    cov += w * seed
                if ~np.isfinite(cov).any():
                    # print("Inv values in cov for Eig[{}]\n{}\npmat\n{}".format(m,cov,pmat))
                    sys.exit(0)
            # print("cov for eig.d={} is\n{}".format(m,cov))
            try:
                self.eigvals, self.eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                print("Linalg error in eigenvalue decomposition")
                print(cov)
                sys.exit(0)
            # print("Eigvals ",self.eigvals)
            # store the valid indices
            self.valid_indices = np.arange(self.eigvals.shape[0])  # trivial.
            self.num_valid_indices = self.valid_indices.shape[0]
            # print("valid indices ", self.valid_indices)
            # Now scale the proposal length using samples from the target
            target_ar = 0.234  # 0.44
            toler = 0.0001
            maxiter = 20
            avg_ar = target_ar
            self.propscale = 0.1  # starting point

            ssize = 64
            ids = np.random.choice(pmat.shape[0], ssize)
            llh = self.pmodel.compute_llh(theta[ids])
            cur_prior = self.pmodel.compute_prior(theta[ids])

            # ssize=128

            def get_ar():
                # ids = np.random.choice(theta.shape[0],ssize)
                # llh = self.pmodel.compute_llh(theta[ids])
                # cur_prior = self.pmodel.compute_prior(theta[ids])
                prop_theta, prop_lpqratio, prop_id = self.draw(theta[ids])
                prop_prior = self.pmodel.compute_prior(prop_theta)
                prop_llh = self.pmodel.compute_llh(prop_theta)
                log_ar = self.pmodel.compute_lar(
                    theta[ids],
                    prop_theta,
                    np.zeros(ssize),
                    prop_llh,
                    llh,
                    cur_prior,
                    prop_prior,
                    t,
                )  # TODO implement
                ar = np.exp(logsumexp(log_ar) - np.log(log_ar.shape[0]))
                return ar

            def get_propscale(ar):
                return np.exp(
                    2.0 * self.dimension * (ar - target_ar)
                )  # not scaling by dim

            lb = 0.0001
            ub = 2.0

            next_ps = 0.5 * (lb + ub)
            ps = ub

            while maxiter > 0:
                if abs(ps - next_ps) < toler:
                    break
                ps = next_ps
                self.propscale = ps
                ar = get_ar()
                if ar < target_ar:
                    ub = ps
                else:
                    lb = ps
                next_ps = 0.5 * (lb + ub)
                maxiter -= 1
                # print("Adapting eig[{}].propscale = {} for ar {}".format(self.dimension,self.propscale,ar))

    def draw(self, theta, size=1):
        pmat = self.concatParameters(theta, self.getModelIdentifier())
        n = pmat.shape[0]
        m = pmat.shape[1]
        if self.num_valid_indices == 0:
            return theta, np.zeros(n), np.full(n, id(self))
        # if len(self.eigvals.shape)==0:
        if m == 1:
            noise = norm(0, np.sqrt(self.eigvals) * self.propscale).rvs(n)
            prop_pmat = pmat + noise.reshape(pmat.shape)
        else:
            ii = np.random.randint(self.num_valid_indices, size=n)
            i = self.valid_indices[ii]
            pmat_r = np.einsum("ij,kj->ki", self.eigvecs.T, pmat)
            pmat_r[(np.arange(n), i)] += norm(
                0, np.sqrt(np.abs(self.eigvals[i])) * self.propscale
            ).rvs(n)
            prop_pmat = np.einsum("ij,kj->ki", self.eigvecs, pmat_r)
        prop_theta = self.deconcatParameters(
            prop_pmat, theta, self.getModelIdentifier()
        )
        return prop_theta, np.zeros(n), np.full(n, id(self))


class EigDecComponentwiseNormalProposal(Proposal):
    """
    Does not have sub-proposals.
    """

    def __init__(self, proposals=[]):
        self.propscale = 1.0
        super(EigDecComponentwiseNormalProposal, self).__init__(proposals)
        self.dimension = 1
        self.propscale = 1

    # def setAR(self,ar,t):
    #    super(EigDecComponentwiseNormalProposal,self).setAR(ar,t)
    #    if len(self.ar)>10:
    #        # average the last 10
    #        avg_ar = np.mean(self.ar[-10:])
    #        self.propscale = np.exp(2.0 * self.dimension * (avg_ar - 0.44)) # not scaling by dim
    #        print("eig[{}].propscale = {} for ar {}".format(self.dimension,self.propscale, avg_ar))

    def setAR(self, ar, t, n):
        super(EigDecComponentwiseNormalProposal, self).setAR(ar, t, n)
        # average the last 100
        if len(self.ar) < 10:
            avg_ar = 0.44
        else:
            alen = min(len(self.ar) - 1, 100)
            # alen = len(self.ar)-1
            avg_ar = np.sum(
                np.array(self.ar[-alen:])
                * np.array(self.n_ar[-alen:])
                * np.array(self.t_ar[-alen:])
            ) / np.sum(np.array(self.n_ar[-alen:]) * np.array(self.t_ar[-alen:]))
        self.propscale = np.exp(
            2.0 * self.dimension * (min(avg_ar, 0.7) - 0.44)
        )  # not scaling by dim
        print(
            "eig[{}].propscale = {} for ar {}".format(
                self.dimension, self.propscale, avg_ar
            )
        )

    def calibrate(self, theta, size, t):
        # compute eigenvalue decomposition here
        pmat = self.concatParameters(theta)
        m = pmat.shape[1]  # self.splitby_val # FIXME a hack
        self.dimension = m
        # FIXME the below gives extreme numbers.
        # self.propscale = np.exp(2.0 * m * (self.ar - 0.44)) # not scaling by dim
        # self.propscale = np.exp(- m)
        # self.propscale = (1.25-t)/m #(0.238**2)/ m

        # self.propscale = np.exp(-2*t)/m #(0.238**2)/ m

        # self.propscale = np.exp(2.0 * m * (min(self.getLastAR(),0.44) - 0.44)) # not scaling by dim
        # print("eig[{}].propscale = {} for ar {}".format(m,self.propscale, self.getLastAR()))

        # average the last 100
        # if len(self.ar)<10:
        #    avg_ar = 0.44
        # else:
        #    alen = min(len(self.ar)-1,10)
        #    #alen = len(self.ar)-1
        #    avg_ar = np.sum(np.array(self.ar[-alen:])*np.array(self.n_ar[-alen:]))/np.sum(self.n_ar[-alen:])
        #    if not np.isfinite(avg_ar):
        #        print("Non finite ar.",self.ar[-alen:],"\n",self.n_ar[-alen:])
        # self.propscale = np.exp(2.0 * self.dimension * (min(avg_ar,0.44) - 0.44)) # not scaling by dim
        # print("eig[{}].propscale = {} for ar {}".format(self.dimension,self.propscale, avg_ar))

        n = theta.shape[0]
        cov = np.cov(pmat.T)
        if len(cov.shape) == 0:
            # store the valid indices
            self.valid_indices = np.arange(1)  # trivial.
            self.num_valid_indices = 1
            if not np.isfinite(cov):
                cov = 0.5
            self.eigvals = cov  # we just use variance to scale.
            # print("cov for m=1 is {}".format(cov))
        else:
            # print("Eig prop pmat shape = ",pmat.shape)
            # print("Eig[{}].pmat\n{}".format(m,pmat))
            # print("Eig[].cov\n{}".format(cov))
            # seed with the identity weighted by nrows
            m = cov.shape[0]
            if not np.isfinite(cov).all():
                cov = np.eye(m) * 0.5
            # print("cov for m={} is {}".format(m,cov))
            while np.linalg.det(cov) < 0.001:
                cov += np.eye(m) * 0.001 * (max(1 - np.linalg.det(cov), 0))
                # print("Adjusted covariance due to low volume\n")
            # if m>1:
            #    cov += np.eye(m) * (max(1-np.linalg.det(cov),0))
            # else:
            #    cov += max(1-cov,0)
            if False:
                seed = np.eye(m)
                w = max(float(m - n), 0.0) / m
                # print("Eig[{}].w_seed={}".format(m,w))
                if ~np.isfinite(cov).any():
                    cov = seed
                else:
                    cov += w * seed
                if ~np.isfinite(cov).any():
                    # print("Inv values in cov for Eig[{}]\n{}\npmat\n{}".format(m,cov,pmat))
                    sys.exit(0)
            # print("cov for eig.d={} is\n{}".format(m,cov))
            try:
                self.eigvals, self.eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                print("Linalg error in eigenvalue decomposition")
                print(cov)
                sys.exit(0)
            # print("Eigvals ",self.eigvals)
            # store the valid indices
            self.valid_indices = np.arange(self.eigvals.shape[0])  # trivial.
            self.num_valid_indices = self.valid_indices.shape[0]
            # print("valid indices ", self.valid_indices)

    def draw(self, theta, size=1):
        pmat = self.concatParameters(theta)
        n = pmat.shape[0]
        m = pmat.shape[1]
        # if len(self.eigvals.shape)==0:
        if m == 1:
            prop_pmat = pmat + norm(0, np.sqrt(self.eigvals) * self.propscale).rvs(n)
        else:
            ii = np.random.randint(self.num_valid_indices, size=n)
            i = self.valid_indices[ii]
            pmat_r = np.einsum("ij,kj->ki", self.eigvecs.T, pmat)
            pmat_r[(np.arange(n), i)] += norm(
                0, np.sqrt(np.abs(self.eigvals[i])) * self.propscale
            ).rvs(n)
            prop_pmat = np.einsum("ij,kj->ki", self.eigvecs, pmat_r)
        prop_theta = self.deconcatParameters(prop_pmat, theta)
        return prop_theta, np.zeros(n), np.full(n, id(self))


class MixtureProposal(Proposal):
    def __init__(self, subproposals, weights):
        """
        subproposals are all instances of a proposal
        weights must sum to 1 and have the same number of elements as subproposal
        """
        assert isinstance(subproposals, list)
        assert isinstance(weights, list)
        assert len(weights) == len(subproposals)
        super(MixtureProposal, self).__init__(subproposals)
        wsum = 0.0
        for w in weights:
            assert isinstance(w, float)
            wsum += w
        assert wsum == 1
        self.weights = np.array(weights)
        self.cumsumweights = np.cumsum(self.weights)

    def draw(self, theta, size=1):
        n = theta.shape[0]
        prop_theta = np.zeros_like(theta)
        prop_pqratio = np.zeros(n)
        # print("proposals",self.ps)
        for i in range(len(self.ps)):
            pt, plq, ids = self.ps[i].draw(theta, n)  # TODO remove size
            prop_theta += self.weights[i] * pt
            prop_pqratio += self.weights[i] * np.exp(plq)
        return prop_theta, np.log(prop_pqratio), np.full(n, id(self))
