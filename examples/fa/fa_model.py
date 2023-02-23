import sys
import numpy as np
sys.path.append("../../")
from rjlab.samplers import *
from rjlab.variables import *
from rjlab.proposals.standard import *
from rjlab.transforms import *
import nflows as n
from nflows.flows.base import Flow
from nflows.distributions.uniform import *
from nflows.transforms import *
from nflows.transforms.nonlinearities import CauchyCDF
from nflows.distributions.normal import StandardNormal
import shelve

PLOT_PROGRESS = False

class FARWProposal(Proposal):
    def __init__(self,betaii_names,betaij_names,lambda_names):
        self.betaii_names = betaii_names
        self.betaij_names = betaij_names
        self.lambda_names = lambda_names
        super(FARWProposal, self).__init__(betaii_names+betaij_names+lambda_names)
    def transformToBase(self,inputs,mk):
        X = self.concatParameters(inputs,mk)
        XX,logdet = self.flow.forward(torch.tensor(X,dtype=torch.float32))
        return XX.detach().numpy(),logdet.detach().numpy()
    def transformFromBase(self,X,inputs,mk):
        XX,logdet = self.flow.inverse(torch.tensor(X,dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(),inputs,mk),logdet.detach().numpy()
    def calibratemmmpd(self,mmmpd,size,t):
        mk = self.getModelIdentifier()
        #mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True)
        #mk_theta_w = np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w)))
        mk_theta,mk_theta_w = mmmpd.getOriginalParticleDensityForTemperature(t,resample=True)
        mk_theta_w = np.exp(mk_theta_w - logsumexp(mk_theta_w))

        model_key_indices, m_tuple_indices = self.getModel().enumerateModels(mk_theta)

        X,concat_indices = self.concatParameters(mk_theta,mk,return_indices=True)
        #print(concat_indices)
        # transform the betaii variables and lambda variables using log first
        spec = {}
        dim = 0
        for rvn,rv_indices in self.pmodel.generateRVIndices(model_key=mk,flatten_tree=True).items():
            if len(rv_indices)>0:
                dim += len(rv_indices)
                #if rvn in self.betaii_names:
                #if rvn in self.lambda_names:
                if rvn in self.betaii_names or rvn in self.lambda_names:
                    for i in rv_indices:
                        j = concat_indices.index(i)
                        spec[j] = LogTransform()
                        #print("adding log transform for rv ",rvn, " indices ", rv_indices)
                #elif rvn in self.lambda_names:
                #elif rvn in self.betaii_names:
                #if rvn in self.betaii_names or rvn in self.lambda_names:
                #    for i in rv_indices:
                #        j = concat_indices.index(i)
                #        spec[j] = n.transforms.CompositeTransform([
                #            LogTransform(),
                #            CauchyCDF(),
                #            n.transforms.InverseTransform(n.transforms.Sigmoid(temperature=np.sqrt(8/np.pi)))
                #        ])
                #        print("adding log-cauchyCDF-invsigmoid transform for rv ",rvn, " indices ", rv_indices)
                #elif rvn in self.betaij_names:
                #    for i in rv_indices:
                #        j = concat_indices.index(i)
                #        spec[j] = n.transforms.CompositeTransform([
                #            CauchyCDF(),
                #            n.transforms.InverseTransform(n.transforms.Sigmoid(temperature=np.sqrt(8/np.pi)))
                #        ])
                #        print("adding log-cauchyCDF-invsigmoid transform for rv ",rvn, " indices ", rv_indices)
        self.flow = ColumnSpecificTransform(spec)
        # Calibrate to transformed target
        Tk_theta,ld1 = self.transformToBase(mk_theta,mk)
        d = Tk_theta.shape[1]
        N = Tk_theta.shape[0]
        self.cov = np.cov(Tk_theta.T)
        #self.propscale = 2.38/np.sqrt(d)
        self.propscale = 0.05 # starting point
        #self.propscale = 1 # replicate Leah's setup
        if False:
            # Now scale the proposal length using samples from the target
            target_ar = 0.234 #0.44
            toler = 0.0001
            maxiter = 20
            avg_ar = target_ar

            ssize=64
            ids = np.random.choice(N,size=ssize)
            llh = self.pmodel.compute_llh(mk_theta[ids])
            cur_prior = self.pmodel.compute_prior(mk_theta[ids])

            #ssize=128

            def get_ar():
                prop_theta, prop_lpqratio, prop_id = self.draw(mk_theta[ids])
                prop_prior = self.pmodel.compute_prior(prop_theta)
                prop_llh = self.pmodel.compute_llh(prop_theta)
                log_ar = self.pmodel.compute_lar(mk_theta[ids],prop_theta,prop_lpqratio,prop_llh,llh,cur_prior,prop_prior,t) # TODO implement
                ar = np.exp(logsumexp(log_ar) - np.log(log_ar.shape[0]))
                return ar

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
                #print("Adapting rw[{}].propscale = {} for ar {}".format(d,self.propscale,ar))
    def draw(self,theta,size=1):
        mk = self.getModelIdentifier()
        N = theta.shape[0]
        TkX,lpq1 = self.transformToBase(theta,mk)
        d = TkX.shape[1]
        propTkX = TkX + multivariate_normal(np.zeros(d),self.cov * self.propscale).rvs(N)
        proptheta,lpq2 = self.transformFromBase(propTkX,theta.copy(),mk)
        return proptheta, lpq1+lpq2, np.full(N,id(self))


class RJFlowGlobalFactorAnalysisProposal(Proposal):
    def __init__(self,indicator_name,betaii_names,betaij_names,lambda_names,within_model_proposal,flowtype='rq'):
        self.indicator_name = indicator_name
        self.betaii_names = betaii_names
        self.betaij_names = betaij_names
        self.lambda_names = lambda_names
        assert(isinstance(within_model_proposal,Proposal))
        self.within_model_proposal = within_model_proposal
        self.rv_names = betaii_names + betaij_names + lambda_names + [indicator_name]
        super(RJFlowGlobalFactorAnalysisProposal, self).__init__(self.rv_names + [self.within_model_proposal])
        self.exclude_concat = [indicator_name]
        self.flowtype=flowtype
        assert(self.flowtype in ['rq','af'])
    def calibratemmmpd(self,mmmpd,size,t):
        global PLOT_PROGRESS, mk_safety_threshold
        self.within_model_proposal.calibratemmmpd(mmmpd,size,t)
        mklist = self.pmodel.getModelKeys() # get all keys
        #theta,theta_w = mmmpd.getParticleDensityForTemperature(t,resample=True,resample_max_size=2000)
        #theta,theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=2000)
        # set up transforms
        # for this, get size of model via num of sum of gammas
        # index by model key
        # use StandardNormal((numgammas,)) as base
        #params = self.explodeParameters(theta,mk)
        #gammas = np.column_stack([int(params[xn])==1 for xn in self.gammanames]) # should return all gammas. Rows should be identical.
        cols = self.pmodel.generateRVIndices()
        #gammas = np.column_stack([theta[:,cols[i]] for i in self.gammanames])
        orig_theta,orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(t,resample=False)
        orig_theta_w = np.exp(orig_theta_w)
        orig_mkdict,rev = self.pmodel.enumerateModels(orig_theta)

        self.flows = {}
        self.mk_logZhat = {}
        for mk in mklist:
            #mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=10000)
            if False:
                # weighted, including prior
                mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=False)
                mk_theta_w = np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w)))
            else:
                # use original particles at temperature t.
                mk_theta = orig_theta[orig_mkdict[mk]]
                mk_theta_w = orig_theta_w[orig_mkdict[mk]]
                mk_theta_w = np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w)))
            #self.mk_logZhat[mk] = mmmpd.getlogZForModelAndTemperature(mk,t)
            #if mk in orig_mkdict.keys():
            #    self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0]*1./10000)
            #else:
            #    self.mk_logZhat[mk] = np.log(mk_safety_threshold)
            self.mk_logZhat[mk] = -np.log(len(mklist))

            model_key_indices, m_tuple_indices = self.getModel().enumerateModels(mk_theta)
            print("Calibrating model {} with Z_hat={}".format(mk,np.exp(self.mk_logZhat[mk])))

            X,concat_indices = self.concatParameters(mk_theta,mk,return_indices=True)
            # transform the betaii variables and lambda variables using log first
            spec = {}
            dim = 0
            for rvn,rv_indices in self.pmodel.generateRVIndices(model_key=mk,flatten_tree=True).items():
                if len(rv_indices)>0:
                    dim += len(rv_indices)
                    if rvn in self.betaii_names or rvn in self.lambda_names:
                        for i in rv_indices:
                            j = concat_indices.index(i)
                            spec[j] = n.transforms.CompositeTransform([
                                LogTransform(),
                                CauchyCDF(),
                                n.transforms.InverseTransform(n.transforms.Sigmoid(temperature=np.sqrt(8/np.pi)))
                            ])
                            #print("adding log-cauchyCDF-invsigmoid transform for rv ",rvn, " indices ", rv_indices)
                    elif rvn in self.betaij_names:
                        for i in rv_indices:
                            j = concat_indices.index(i)
                            spec[j] = n.transforms.CompositeTransform([
                                CauchyCDF(),
                                n.transforms.InverseTransform(n.transforms.Sigmoid(temperature=np.sqrt(8/np.pi)))
                            ])
                            #print("adding cauchyCDF-invsigmoid transform for rv ",rvn, " indices ", rv_indices)
            dim -= 1
            bmmt = n.transforms.IdentityTransform()

            ls = n.transforms.Sigmoid(temperature=np.sqrt(8/np.pi))
            bnorm = StandardNormal((dim,))
            if ~np.any(np.isfinite(np.std(X,axis=0))):
                print("X is singular",X)
                sys.exit(0)
            #fn = FixedNorm(torch.Tensor(X))
            spectransform = ColumnSpecificTransform(spec)
            X_,_ = spectransform.forward(torch.Tensor(X))
            if self.flowtype=='rq':
                fn = n.transforms.CompositeTransform([
                    ColumnSpecificTransform(spec),
                    FixedNorm(X_,torch.Tensor(mk_theta_w))
                    ])
                #fn = n.transforms.CompositeTransform([
                #    ColumnSpecificTransform(spec),
                #    n.transforms.InverseTransform(NaiveGaussianTransform(X_,torch.Tensor(mk_theta_w)))
                #    ])
                self.flows[mk] = RationalQuadraticFlowFAV.factory(X,bnorm,ls,fn,input_weights=mk_theta_w)
                #try:
                #    #self.flows[mk] = RationalTailQuadraticFlow.factory(X,bnorm,fn,input_weights=mk_theta_w)
                #    #self.flows[mk] = RationalQuadraticFlow.factory(X,bnorm,bmmt,ls,fn,input_weights=mk_theta_w)
                #    self.flows[mk] = RationalQuadraticFlowFAV.factory(X,bnorm,ls,fn,input_weights=mk_theta_w)
                #    #self.flows[mk] = MaskedAffineFlow.factory(X,bnorm,bmmt,ls,fn,input_weights=mk_theta_w)
                #except Exception as ex:
                #    print("ERROR: Training flow failed.")
                #    print("Inputs:\n",X)
                #    print(ex)
                #    sys.exit(0)
                #    #np.savetxt('bad_inputs.txt',X)
                #    #sys.exit(0)
                #    self.flows[mk] = fn
            elif self.flowtype=='af':
                tf = n.transforms.CompositeTransform([
                    ColumnSpecificTransform(spec),
                    n.transforms.InverseTransform(NaiveGaussianTransform(X_,torch.Tensor(mk_theta_w)))
                    ])
                self.flows[mk] = Flow(tf,bnorm)
            else:
                raise Exception("Unsupported flow type {}".format(self.flowtype))

            if PLOT_PROGRESS:
                #print(mk_theta_w)
                import seaborn as sns
                import pandas as pd
                ncols = X.shape[1]+1
                labels = ['B{}'.format(i) for i in range(ncols)]
                X_draws = fn.forward(self.flows[mk].sample(500))[0].detach().numpy()
                X = fn.forward(torch.tensor(X))[0].detach().numpy()
                Xtrain = np.column_stack([X[np.random.choice(X.shape[0],replace=False,size=500)],np.zeros(500)])
                Xtest = np.column_stack([X_draws,np.ones(500)])
                #Xtrain[:,0] = np.log(Xtrain[:,0])
                #Xtrain[:,6] = np.log(Xtrain[:,6])
                #Xtrain[:,-7:] = np.log(Xtrain[:,-7:])
                #Xtest[:,0] = np.log(Xtest[:,0])
                #Xtest[:,6] = np.log(Xtest[:,6])
                #Xtest[:,-7:] = np.log(Xtest[:,-7:])

                Xtrain = np.clip(Xtrain,-1e6,1e6)
                Xtest = np.clip(Xtest,-1e6,1e6)
                print("Xtrain.shape",Xtrain.shape,"Xtest.shape",Xtest.shape)
                dfXd = pd.DataFrame(np.vstack([Xtrain,Xtest]),columns=labels)
                dfXd[labels[-1]] = dfXd[labels[-1]].map('istest_{}'.format)
                g = sns.PairGrid(dfXd,hue=labels[-1])
                #g.map_lower(sns.scatterplot,size=0.1)
                g.map_lower(sns.scatterplot,s=0.2)
                #g.map_lower(sns.kdeplot)
                g.map_diag(sns.distplot)
                g.add_legend()
                plt.show()


        if False:
            # Ensure probabilities do not drop below 1%
            pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
            pp_mk_keys = []
            for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
                pp_mk_logZ[i]=pmk_logZ
                pp_mk_keys.append(pmk)
            pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
            pp_mk_prob = np.exp(pp_mk_log_prob)
            def anneal_down(a,logp,thres):
                alogp = a*logp
                npa = np.exp(alogp - logsumexp(alogp))
                return np.max(thres - npa)
            if np.any(pp_mk_prob < mk_safety_threshold):
                print("Enabling safety threshold for jump probabilities.",pp_mk_prob)
                a, rres = scipy.optimize.bisect(anneal_down,0.,1.,args=(pp_mk_log_prob,mk_safety_threshold),full_output=True,rtol=1e-10)
                pp_mk_log_prob *= a
                pp_mk_log_prob -= logsumexp(pp_mk_log_prob)
                for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
                    self.mk_logZhat[pmk] = pp_mk_log_prob[i]
                print("New safer jump probabilities.",[np.exp(p) for pmk,p in self.mk_logZhat.items()])

    def transformToBase(self,inputs,mk):
        if self.getModelDim(mk)==0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs,mk)
        XX,logdet = self.flows[mk]._transform.forward(torch.tensor(X,dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(),inputs,mk),logdet.detach().numpy()
    def transformFromBase(self,inputs,mk):
        if self.getModelDim(mk)==0:
            return inputs, np.zeros(inputs.shape[0])
        X = self.concatParameters(inputs,mk)
        XX,logdet = self.flows[mk]._transform.inverse(torch.tensor(X,dtype=torch.float32))
        return self.deconcatParameters(XX.detach().numpy(),inputs,mk),logdet.detach().numpy()

    def getModelDim(self,mk):
        dim = 0
        for rvn,rv_indices in self.pmodel.generateRVIndices(model_key=mk,flatten_tree=True).items():
            if len(rv_indices)>0:
                dim += len(rv_indices)
        return dim-1

        #return int(np.array([bs * list(mk)[i] for i,bs in enumerate(self.blocksizes)]).sum())
        #return int(np.array(list(mk)).sum()*self.blocksize) # + static extra params
    def toggleGamma(self,mk,tidx):
        mkl = list(mk)
        mkl[tidx] = 1-mkl[tidx]
        return tuple(mkl)
    def auxToCols(self,mk,new_mk):
        # return list of column names which are enabled from mk to new_mk
        mk_cols = self.pmodel.generateRVIndices(model_key=mk,flatten_tree=True)
        #print("mk_cols",mk,mk_cols)
        new_mk_cols = self.pmodel.generateRVIndices(model_key=new_mk,flatten_tree=True)
        #print("new_mk_cols",new_mk,new_mk_cols)
        return list(set(mk_cols.keys()) - set(new_mk_cols.keys()))
    def draw(self,theta,size=1):
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1 # std dev of auxiliary u dist

        prop_theta = theta.copy()

        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i]=pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        nmodels = pp_mk_logZ.shape[0]
        pp_mk_log_prob = np.zeros(nmodels) - np.log(nmodels-1) # hack for bartolucci estimator
        #print("Log probs ",pp_mk_log_prob,"probs",np.exp(pp_mk_log_prob))

        # for each model k
        #   transform to base
        #   draw k'
        #   if k==k', do within model proposal
        #   else if k'<k
        #       eval u for k-k'
        #       set transformed_theta(k-k')=0
        #       set k=k'
        #   else if k'>k
        #       draw u for k-k'
        #       eval u for k-k'
        #       set transformed_theta(k-k')=u
        #       set k=k'

        model_enumeration,rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]
            params,splitdict = self.explodeParameters(theta,mk)
            #betaiis = np.column_stack([params[xn] for xn in self.betaii_names]) # should only return active blocks
            #betaijs = np.column_stack([params[xn] for xn in self.betaij_names]) # should only return active blocks
            #lambdas = np.column_stack([params[xn] for xn in self.lambda_names]) # should only return active blocks
            k = params[self.indicator_name] # should return all gammas. Rows should be identical.

            Tmktheta,lpq1 = self.transformToBase(mk_theta,mk)

            # global proposal to all models
            this_log_prob = self.mk_logZhat[mk] - logsumexp(pp_mk_logZ)
            this_log_prob = 0 # hack
            this_mk_probs = np.exp(pp_mk_log_prob)
            this_p_i = pp_mk_keys.index(mk)
            this_mk_probs[this_p_i] = 0
            #pidx = np.random.choice(np.arange(pp_mk_logZ.shape[0]),p=np.exp(pp_mk_log_prob),size=mk_n) 
            pidx = np.random.choice(np.arange(pp_mk_logZ.shape[0]),p=this_mk_probs,size=mk_n)  # don't do within model move. Hack for bartolucci
            lpq_mk[mk] = np.zeros(mk_n) #this_log_prob - pp_mk_log_prob[pidx]

            proposed[mk],mk_idx[mk] = self.explodeParameters(Tmktheta,mk)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # for each mk in pidx
            # separate theta further into model transitions
            for p_i in np.unique(pidx):
                tn = (pidx==p_i).sum()
                new_mk = pp_mk_keys[p_i]
                at_idx = pidx==p_i
                if new_mk==mk:
                    prop_mk_theta[at_idx],lpq_mk[mk][at_idx],mk_prop_ids[at_idx] = self.within_model_proposal.draw(mk_theta[at_idx],tn)
                    continue
                # get column ids for toggled on and off blocks
                off_cols = self.auxToCols(mk,new_mk)
                on_cols = self.auxToCols(new_mk,mk)
                #print("off cols for ",mk,new_mk,off_cols)
                #print("on cols for ",mk,new_mk,on_cols)
                # toggle ons
                for c in on_cols:
                    u = norm(0,sigma_u).rvs(tn)
                    log_u = norm(0,sigma_u).logpdf(u) # for naive, mean is u
                    lpq_mk[mk][at_idx] -= log_u
                    Tmktheta[at_idx]=self.setVariable(Tmktheta[at_idx],c,u)
                # for toggle off
                for c in off_cols:
                    u = self.getVariable(Tmktheta[at_idx],c).flatten()
                    log_u = norm(0,sigma_u).logpdf(u) # for naive, mean is u
                    lpq_mk[mk][at_idx] += log_u
                Tmktheta[at_idx]=self.setVariable(Tmktheta[at_idx],self.indicator_name,np.full(tn,new_mk[0])) # hack for indicator value
                # transform back
                prop_mk_theta[at_idx],lpq2 = self.transformFromBase(Tmktheta[at_idx],new_mk)
                # zero toggle offs
                for c in off_cols:
                    prop_mk_theta[at_idx]=self.setVariable(prop_mk_theta[at_idx],c,0)
                lpq_mk[mk][at_idx] += lpq1[at_idx] + lpq2
                mk_prop_ids[at_idx] = np.full(tn,id(self))
            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids
        return prop_theta, logpqratio, prop_ids

class RJLWGlobalFactorAnalysisProposal(Proposal):
    def __init__(self,indicator_name,betaii_names,betaij_names,lambda_names,within_model_proposal):
        self.indicator_name = indicator_name
        self.betaii_names = betaii_names
        self.betaij_names = betaij_names
        self.lambda_names = lambda_names
        assert(isinstance(within_model_proposal,Proposal))
        self.within_model_proposal = within_model_proposal
        self.rv_names = betaii_names + betaij_names + lambda_names + [indicator_name]
        super(RJLWGlobalFactorAnalysisProposal, self).__init__(self.rv_names + [self.within_model_proposal])
        self.exclude_concat = [indicator_name]
    def calibratemmmpd(self,mmmpd,size,t):
        global PLOT_PROGRESS, mk_safety_threshold
        self.within_model_proposal.calibratemmmpd(mmmpd,size,t)
        mklist = self.pmodel.getModelKeys() # get all keys
        cols = self.pmodel.generateRVIndices()
        orig_theta,orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(t,resample=True,resample_max_size=10000)
        orig_mkdict,rev = self.pmodel.enumerateModels(orig_theta)

        self.flows = {}
        self.mk_logZhat = {}
        self.beta_col_names = {}
        self.beta_cov = {}
        self.beta_mean = {}
        self.lambda_a = {}
        self.lambda_scale = {}
        for mk in mklist:
            #mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=10000)
            mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=False)
            mk_theta_w = np.exp(np.log(mk_theta_w) - logsumexp(np.log(mk_theta_w)))
            #self.mk_logZhat[mk] = mmmpd.getlogZForModelAndTemperature(mk,t)
            if mk in orig_mkdict.keys():
                self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0]*1./10000)
            else:
                #self.mk_logZhat[mk] = np.log(mk_safety_threshold)
                self.mk_logZhat[mk] = -np.NINF

            model_key_indices, m_tuple_indices = self.getModel().enumerateModels(mk_theta)
            print("Calibrating model {} with Z_hat={}".format(mk,np.exp(self.mk_logZhat[mk])))

            #X,concat_indices = self.concatParameters(mk_theta,mk,return_indices=True)
            #X_beta = X[:,:-6]
            #X_lambda = X[:,-6:]

            self.beta_col_names[mk] = []
            for rvn,rv_indices in self.pmodel.generateRVIndices(model_key=mk,flatten_tree=True).items():
                if len(rv_indices)>0:
                    if rvn in self.betaii_names or rvn in self.betaij_names:
                        self.beta_col_names[mk].append(rvn)

            beta_all = np.zeros((mk_theta.shape[0],len(self.beta_col_names[mk])))
            for i,c in enumerate(self.beta_col_names[mk]):
                if c in self.betaii_names:
                    beta_all[:,i] = np.log(self.getVariable(mk_theta,c).flatten())
                else:
                    beta_all[:,i] = self.getVariable(mk_theta,c).flatten()
            self.beta_cov[mk] = 2*np.cov(beta_all.T,aweights = mk_theta_w)
            self.beta_mean[mk] = np.average(beta_all,axis=0,weights = mk_theta_w)

            #lambda_all = np.zeros((mk_theta.shape[0],len(self.lambda_names)))
            self.lambda_a[mk] = {}
            self.lambda_scale[mk] = {}
            for i,c in enumerate(self.lambda_names):
                #lambda_all[:,i] = self.getVariable(mk_theta,c).flatten()
                this_lambda = self.getVariable(mk_theta,c).flatten()
                # fit IG dist to lambdas
                #self.lambda_a[c], _loc_, self.lambda_scale[c] = invgamma.fit(this_lambda,floc=0)
                if False:
                    a,loc,scale = invgamma.fit(this_lambda)
                    # get mode
                    mode = scale/(a+1) + loc
                else:
                    h = gaussian_kde(this_lambda).pdf(this_lambda)
                    mode = this_lambda[np.argmax(h)]
                    print("mode for ",c," in mk ",mk," is ",mode)
                    if False:
                        plt.hist(this_lambda,bins=50,alpha=0.5,color='blue',density=True)
                        plt.hist(InvGammaDistribution(18,mode*18).draw(10000),bins=50,alpha=0.5,color='orange',density=True)
                        plt.show()
                self.lambda_a[mk][c] = 18
                self.lambda_scale[mk][c] = 18 * mode



    def draw(self,theta,size=1):
        logpqratio = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        prop_ids = np.zeros(theta.shape[0])
        sigma_u = 1 # std dev of auxiliary u dist

        prop_theta = theta.copy()

        pp_mk_logZ = np.zeros(len(self.mk_logZhat.keys()))
        pp_mk_keys = []
        for i, (pmk, pmk_logZ) in enumerate(self.mk_logZhat.items()):
            pp_mk_logZ[i]=pmk_logZ
            pp_mk_keys.append(pmk)
        pp_mk_log_prob = pp_mk_logZ - logsumexp(pp_mk_logZ)
        nmodels = pp_mk_logZ.shape[0]
        pp_mk_log_prob = np.zeros(nmodels) - np.log(nmodels-1) # hack for bartolucci estimator
        #print("Log probs ",pp_mk_log_prob,"probs",np.exp(pp_mk_log_prob))

        # for each model k
        #   transform to base
        #   draw k'
        #   if k==k', do within model proposal
        #   else if k'<k
        #       eval u for k-k'
        #       set transformed_theta(k-k')=0
        #       set k=k'
        #   else if k'>k
        #       draw u for k-k'
        #       eval u for k-k'
        #       set transformed_theta(k-k')=u
        #       set k=k'

        model_enumeration,rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            mk_n = mk_theta.shape[0]
            params,splitdict = self.explodeParameters(theta,mk)
            #betaiis = np.column_stack([params[xn] for xn in self.betaii_names]) # should only return active blocks
            #betaijs = np.column_stack([params[xn] for xn in self.betaij_names]) # should only return active blocks
            #lambdas = np.column_stack([params[xn] for xn in self.lambda_names]) # should only return active blocks
            k = params[self.indicator_name] # should return all gammas. Rows should be identical.

            # global proposal to all models
            this_log_prob = self.mk_logZhat[mk] - logsumexp(pp_mk_logZ)
            this_log_prob = 0 # hack
            this_mk_probs = np.exp(pp_mk_log_prob)
            this_p_i = pp_mk_keys.index(mk)
            this_mk_probs[this_p_i] = 0
            #pidx = np.random.choice(np.arange(pp_mk_logZ.shape[0]),p=np.exp(pp_mk_log_prob),size=mk_n) 
            pidx = np.random.choice(np.arange(pp_mk_logZ.shape[0]),p=this_mk_probs,size=mk_n)  # don't do within model move. Hack for bartolucci
            lpq_mk[mk] = np.zeros(mk_n) #this_log_prob - pp_mk_log_prob[pidx]

            #proposed[mk],mk_idx[mk] = self.explodeParameters(mk_theta,mk)

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            # for each mk in pidx
            # separate theta further into model transitions
            for p_i in np.unique(pidx):
                tn = (pidx==p_i).sum()
                new_mk = pp_mk_keys[p_i]
                at_idx = pidx==p_i
                if new_mk==mk:
                    prop_mk_theta[at_idx],lpq_mk[mk][at_idx],mk_prop_ids[at_idx] = self.within_model_proposal.draw(mk_theta[at_idx],tn)
                    continue

                # the Lopes & West RJ proposal is an independence proposal
                # evaluate old parameters
                old_betas = np.zeros((tn,len(self.beta_col_names[mk])))
                for vi,c in enumerate(self.beta_col_names[mk]):
                    if c in self.betaii_names:
                        old_betas[:,vi] = np.log(self.getVariable(mk_theta[at_idx],c).flatten())
                        lpq_mk[mk][at_idx] += -old_betas[:,vi]
                    else:
                        old_betas[:,vi] = self.getVariable(mk_theta[at_idx],c).flatten()
                lpq_mk[mk][at_idx] += multivariate_normal(self.beta_mean[mk],self.beta_cov[mk]).logpdf(old_betas)
                old_lambdas = np.zeros((tn,len(self.lambda_names)))
                for vi,c in enumerate(self.lambda_names):
                    old_lambdas[:,vi] = self.getVariable(mk_theta[at_idx],c).flatten()
                    lpq_mk[mk][at_idx] += InvGammaDistribution(self.lambda_a[mk][c],self.lambda_scale[mk][c]).logeval(old_lambdas[:,vi]) #.sum(axis=1)
                #lpq_mk[mk][at_idx] += InvGammaDistribution(1.1,0.05).logeval(old_lambdas).sum(axis=1)

                # We draw betas from a fitted MVN, and lambdas from IG(1.1,0.05) priors
                new_betas = multivariate_normal(self.beta_mean[new_mk],self.beta_cov[new_mk]).rvs(tn).reshape((tn,self.beta_mean[new_mk].shape[0]))
                lpq_mk[mk][at_idx] -= multivariate_normal(self.beta_mean[new_mk],self.beta_cov[new_mk]).logpdf(new_betas)
                for vi,c in enumerate(self.beta_col_names[new_mk]):
                    if c in self.betaii_names:
                        prop_mk_theta[at_idx]=self.setVariable(prop_mk_theta[at_idx],c,np.exp(new_betas[:,vi]))
                        lpq_mk[mk][at_idx] -= -new_betas[:,vi]
                    else:
                        prop_mk_theta[at_idx]=self.setVariable(prop_mk_theta[at_idx],c,new_betas[:,vi])
                #new_lambdas = InvGammaDistribution(1.1,0.05).draw(size=(tn,len(self.lambda_names))).reshape((tn,len(self.lambda_names)))
                #lpq_mk[mk][at_idx] -= InvGammaDistribution(1.1,0.05).logeval(new_lambdas).sum(axis=1)
                for vi,c in enumerate(self.lambda_names):
                    new_lambdas = InvGammaDistribution(self.lambda_a[new_mk][c],self.lambda_scale[new_mk][c]).draw(size=tn)
                    lpq_mk[mk][at_idx] -= InvGammaDistribution(self.lambda_a[new_mk][c],self.lambda_scale[new_mk][c]).logeval(new_lambdas)
                    prop_mk_theta[at_idx]=self.setVariable(prop_mk_theta[at_idx],c,new_lambdas)
                    #prop_mk_theta[at_idx]=self.setVariable(prop_mk_theta[at_idx],c,new_lambdas[:,vi])
                prop_mk_theta[at_idx]=self.setVariable(prop_mk_theta[at_idx],self.indicator_name,np.full(tn,new_mk[0]))

                mk_prop_ids[at_idx] = np.full(tn,id(self))

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids
        return prop_theta, logpqratio, prop_ids


class FactorAnalysisModel(ParametricModelSpace):
    def __init__(self,y_data,ft='rq',k_min=0,k_max=2):
        self.y_data = y_data
        random_variables = {}
        self.betaii_names = []
        self.betaij_names = []
        self.lambda_names = []
        blockrv = {}
        self.blockrvcond = {}
        for col in range(0,3):
            name = 'beta{}{}'.format(col,col)
            self.betaii_names.append(name)
            blockrv[name] = HalfNormalRV(1)
            self.blockrvcond[name] = col
            for row in range(col+1,6):
                name = 'beta{}{}'.format(row,col)
                self.betaij_names.append(name)
                blockrv[name] = NormalRV(0,1)
                self.blockrvcond[name] = col
        #print("blockrv",blockrv)
        #print("blockrvcond",self.blockrvcond)
        name = 'allbeta'
        random_variables[name]=ConditionalVariableBlock(blockrv,self.blockrvcond,UniformIntegerRV(k_min,k_max),'k')
        for i in range(6):
            name = 'lambda{}'.format(i)
            self.lambda_names.append(name)
            random_variables[name] = InvGammaRV(1.1,0.05)
        #print("Random variables:",random_variables)
        #componentwiseprop = EigDecComponentwiseNormalProposalTrial(proposals=self.betaii_names+self.betaij_names+self.lambda_names)
        #mep = ModelEnumerateProposal(subproposal=componentwiseprop)
        #mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=['allbeta']+self.lambda_names))
        #imep = ModelEnumerateProposal(subproposal=IndependenceFlowProposal(self.betaii_names,self.betaij_names,self.lambda_names))
        #transformedflowrwprop = MGRWFlowProposal(self.betaii_names,self.betaij_names,self.lambda_names)
        #imep = ModelEnumerateProposal(subproposal=transformedflowrwprop)
        transformedrwprop = FARWProposal(self.betaii_names,self.betaij_names,self.lambda_names)
        trwmep = ModelEnumerateProposal(subproposal=transformedrwprop)
        if k_min < k_max:
            if ft=='lw':
                rjp = RJLWGlobalFactorAnalysisProposal('k',self.betaii_names,self.betaij_names,self.lambda_names,trwmep)
            else:
                rjp = RJFlowGlobalFactorAnalysisProposal('k',self.betaii_names,self.betaij_names,self.lambda_names,trwmep,flowtype=ft)
            #proposal = SystematicChoiceProposal([RJFlowGlobalFactorAnalysisProposal('k',self.betaii_names,self.betaij_names,self.lambda_names,mep),mep])
            proposal = SystematicChoiceProposal([rjp,trwmep])
        else:
            #proposal = SystematicChoiceProposal([trwmep,imep])
            proposal = trwmep
            #proposal = imep
        super(FactorAnalysisModel, self).__init__(random_variables,proposal)

    def sanitise(self,inputs):
        # need to set betas to zero when not in block
        outputs = inputs.copy()
        mkdict,rev = self.enumerateModels(inputs)
        for mk,idx in mkdict.items(): 
            tn = idx.shape[0]
            #print(mk,tn)
            for nm in self.betaii_names+self.betaij_names:
                if mk[0]<self.blockrvcond[nm]:
                    #print("setting ",nm," to zero")
                    outputs[idx]=self.proposal.setVariable(outputs[idx],nm,np.zeros(tn))
        return outputs

    def compute_llh(self,theta):
        y_data = self.y_data
        cols = self.generateRVIndices()
        betas_stack = []
        n = theta.shape[0]
        W = np.zeros((n,6,6))
        L = np.zeros((n,6,6))
        # assumes order
        for i,bn in enumerate(self.betaii_names):
            if len(cols[bn]) > 0:
                W[:,i,i] = theta[:,cols[bn]].flatten()
        j=1
        i=0
        for bn in self.betaij_names:
            if len(cols[bn]) > 0:
                W[:,j,i] = theta[:,cols[bn]].flatten()
            j+=1
            if j==6:
                i+=1
                j=i+1
        for i,ln in enumerate(self.lambda_names):
            if len(cols[ln]) > 0:
                L[:,i,i] = theta[:,cols[ln]].flatten()
        cov = np.einsum('...ij,...jk',W,np.einsum('...ji',W))+L
        _,logdets = np.linalg.slogdet(cov)
        invs = np.linalg.inv(cov)
        return -0.5*(y_data.shape[0]*(logdets+6*np.log(2*np.pi))+np.einsum('i...j,i...j',np.einsum('...k,ijk',y_data,invs),y_data))

