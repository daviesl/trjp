import sys
from rjlab.samplers import *
from rjlab.proposals.standard import *
from rjlab.transforms import *
from rjlab.utils.kde import *
# NF
import nflows as n
from nflows.distributions.uniform import *
from nflows.transforms import *
import shelve

PLOT_PROGRESS = False
run_idx=0

# Things I need to do in this test script
# 1) Test a single model SMC. Known posterior. Make sure particle recycling delivers correct target.
# 2) Test a cross-dimensional move and make sure both targets are correct (pseudo marginal paper example 2009)

# For Flow-Augmented RJSMC
# 1) Make all distributions pytorch objects
# 2) Make the FASMC class extend RJSMC with run hooks
# 3) Make a NF class that builds the flow as a sequence of smaller flows that can also be evaluated in Adam
m1prob = 0.5
ep = [-2,1.5,-2]
dp = [1.0, 1.0, 1.5]
#ep = [-2,-2,-2]
#dp = [1.0, 1.0, 1.0]
#ep = [0.,0.,0.]
#dp = [1.,1.,1.]
#ep = [0.,0.,0.]
#dp = [2.,2.,2.]
# TODO FRIDAY OCT 15 2021: Make another RJ toy proposal that is even dumber: t1=t1 t2=u. Reverse is t1=t1, u=t2.


# TODO remove all nf
# then create a nf file with the nf stuff
# and a linear file with a linear proposal from Brooks ( this will be the hardest I guess)
# and a perfect file with the SAS transform with known parameters (subsitited for NF)
class RJToyModelProposalPerfect(Proposal):
    def __init__(self,model_id_name='k',t1_name='t1',t2_name='t2',within_model_proposal=None,proptype='affine'):
        self.proptype = proptype # todo assert in affine, nf, perfect
        self.k_name = model_id_name
        self.t1_name = t1_name
        self.t2_name = t2_name
        assert(isinstance(within_model_proposal,Proposal))
        self.within_model_proposal = within_model_proposal
        self.rv_names = [self.k_name,self.t1_name,self.t2_name]
        super(RJToyModelProposalPerfect, self).__init__(self.rv_names+[self.within_model_proposal])
        self.exclude_concat = [self.k_name]
    #def calibratemmmpd(self,mmmpd,size,t):
    #    rs,rs_w = mmmpd.getParticleDensityForTemperature(t,resample=True,resample_max_size=2000)
    #    self.calibrateweighted(rs,rs_w)
    #def calibrateweighted(self,theta,weights,m_indices_dict,size,t):
    #def calibrate(self,theta,size,t):
    #    global ep, dp, PLOT_PROGRESS
    def calibratemmmpd(self,mmmpd,size,t):
        """
        This version computes the exact model probabilities for the 1D 2D gaussian mixture example.
        """
        global ep, dp, m1prob, PLOT_PROGRESS
        self.within_model_proposal.calibratemmmpd(mmmpd,size,t)
        mks = self.pmodel.getModelKeys() # get all keys
        # set up transforms
        self.flows = {}
        self.mk_logZhat = {}
        #mks = self.getModel().getModelKeys(theta)
        orig_theta,orig_theta_w = mmmpd.getOriginalParticleDensityForTemperature(t,resample=True,resample_max_size=1000)
        orig_mkdict,rev = self.pmodel.enumerateModels(orig_theta)

        #mk1cov = 1./t
        #mk2cov = torch.tensor([[1.,.99],[.99,1.]]) * 1./t

        var_h = 25.
        var_h_inv = 1./var_h
        mk1sig = 1
        log_mk1sig = 0
        mk1cov_t = 1./((1.-var_h_inv)*t+var_h_inv)
        #log_mk1cov_t = -np.log((1.-var_h_inv)*t+var_h_inv)
        log_mk1cov_t = -np.log((1./mk1sig)*t+var_h_inv*(1-t))

        def make_mk2cov_t(t,var_h,mk2cov):
            var_h_inv = 1./var_h
            mk2L, mk2Q = torch.linalg.eigh(mk2cov)
            return torch.mm(mk2Q,torch.mm(torch.diag(1./(var_h_inv+t*((1./mk2L)-var_h_inv))),mk2Q.T))

        mk2cov = torch.tensor([[1.,.99],[.99,1.]]) 
        mk2cov_t = make_mk2cov_t(t,var_h,mk2cov)
        print("mk2cov_t",mk2cov_t)
        print("test t=1 mk2cov",make_mk2cov_t(0,var_h,mk2cov))

        # compute normalising constants.
        mk1sig_t = np.sqrt(mk1cov_t)
        log_mk1sig_t = 0.5 * log_mk1cov_t
        mk2detcov_t = torch.linalg.det(mk2cov_t)
        sign,log_mk2detcov_t = torch.linalg.slogdet(mk2cov_t)
        mk1sig_h_1t = np.sqrt(var_h) ** (1-t)
        log_mk1sig_h_1t = np.log(var_h) * (1-t) * 0.5
        mk2detcov_h_1t = (var_h * var_h) ** (1-t)
        log_mk2detcov_h_1t = (1-t) * 2 * np.log(var_h)
        mk2detcov = torch.linalg.det(mk2cov)
        sign,log_mk2detcov = torch.linalg.slogdet(mk2cov)
        mk1_inc = (m1prob ** t) * mk1sig_t / ((mk1sig ** t) * mk1sig_h_1t)# inverse normalising constant I think
        mk2_inc = np.sqrt(((1-m1prob) ** t) * mk2detcov_t / ((mk2detcov ** t) * mk2detcov_h_1t))
        log_mk1_inc = np.log(m1prob) * t + log_mk1sig_t - ((log_mk1sig * t) + log_mk1sig_h_1t) # inverse normalising constant I think
        log_mk2_inc = (np.log(1-m1prob) * t) + 0.5*(log_mk2detcov_t - ((log_mk2detcov * t) + log_mk2detcov_h_1t))
        print("t=",t,"log_mk2detcov_t",log_mk2detcov_t,"log_mk2detcov",log_mk2detcov,"log_mk2detcov_h_1t",log_mk2detcov_h_1t)
        print("t=",t,"mk1_inc=",np.exp(log_mk1_inc),"mk2_inc=",torch.exp(log_mk2_inc))

        #self.mk_logZhat[(0,)] = np.log(mk1_inc / (mk1_inc + mk2_inc))
        #self.mk_logZhat[(1,)] = np.log(mk2_inc / (mk1_inc + mk2_inc))
        self.mk_logZhat[(0,)] = log_mk1_inc - torch.log(np.exp(log_mk1_inc) + torch.exp(log_mk2_inc))
        self.mk_logZhat[(1,)] = log_mk2_inc - torch.log(np.exp(log_mk1_inc) + torch.exp(log_mk2_inc))
        # hack to show forward proposal
        self.mk_logZhat[(0,)] = np.NINF
        self.mk_logZhat[(1,)] = 1.

        if False:
            for mk in mks:
                mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=2000)
                #self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0]*1./1000)
                #self.mk_logZhat[mk] = 0 
                if self.getModelInt(mk)==0:
                    tf1  = n.transforms.InverseTransform(SinArcSinhTransform(ep[0],dp[0]))
                    #tf1  = n.transforms.InverseTransform(L1DTransform(np.sqrt(mk1cov_t)))
                    self.flows[mk] = Flow(tf1,StandardNormal((1,)))
                elif self.getModelInt(mk)==1:
                    tf2  = n.transforms.InverseTransform(n.transforms.CompositeTransform([LTransform(torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))),SAS2DTransform([ep[1],ep[2]],[dp[1],dp[2]])]))
                    #tf2  = n.transforms.InverseTransform(LTransform(torch.linalg.cholesky(mk2cov_t)))
                    self.flows[mk] = Flow(tf2,StandardNormal((2,)))
                X = self.extractModelConcatCols(self.concatParameters(mk_theta,mk),mk)
                if PLOT_PROGRESS:
                    tmp1 = self.flows[mk].sample(1000)
                    tmp1=tmp1.detach().numpy()
                    f,ax = plt.subplots(nrows=1,ncols=2)
                    if self.getModelInt(mk)==0:
                        ax[0].hist(X,bins=50,color='red')
                        ax[1].hist(tmp1,bins=50)
                        plt.show()
                    elif self.getModelInt(mk)==1:
                        ax[0].scatter(X[:,0],X[:,1],color='red')
                        ax[1].scatter(tmp1[:,0],tmp1[:,1])
                        plt.show()
            print("MK log Z",self.mk_logZhat)
            print("MK Z",[np.exp(z) for i,z in self.mk_logZhat.items()])
            return

        else:
            for mk in mks:
                mk_theta,mk_theta_w = mmmpd.getParticleDensityForModelAndTemperature(mk,t,resample=True,resample_max_size=2000)
                #mk_theta = orig_theta[orig_mkdict[mk]]
                #self.mk_logZhat[mk] = mmmpd.getlogZForModelAndTemperature(mk,t)
                #self.mk_logZhat[mk] = np.log(orig_mkdict[mk].shape[0]*1./1000)
                #self.mk_logZhat[mk] = mmmpd.getlogZForModelAndTemperature(mk,t)
    
                if self.getModelInt(mk)==0:
                    #b = Uniform(low=torch.Tensor([-100.]),high=torch.Tensor([100.]))
                    #ls = LinearSquish(torch.Tensor([-100.]),torch.Tensor([100.]))
                    ls = n.transforms.Sigmoid()
                    bnorm = StandardNormal((1,))
                    #tf1 = SinArcSinhTransform(ep[0],dp[0])
                    #self.flows[mk] = n.transforms.InverseTransform(tf1)
                    #tf = SinArcSinhTransform(-2,1.0)
                    #self.flows[mk] = tf
                elif self.getModelInt(mk)==1:
                    #b = Uniform(low=torch.Tensor([-100.,-100.]),high=torch.Tensor([100.,100.]))
                    #ls = LinearSquish(torch.Tensor([-100.,-100]),torch.Tensor([100.,100.]))
                    ls = n.transforms.Sigmoid()
                    bnorm = StandardNormal((2,))
                    #tf = n.transforms.CompositeTransform([LTransform(torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))),SAS2DTransform([1.5,-2],[1,1.5])])
                    #tf2 = n.transforms.CompositeTransform([LTransform(torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))),SAS2DTransform([ep[1],ep[2]],[dp[1],dp[2]])])
                    #tf2 = n.transforms.CompositeTransform([SAS2DTransform([ep[1],ep[2]],[dp[1],dp[2]])])
                    #self.flows[mk] = n.transforms.InverseTransform(tf2)
                    #self.flows[mk] = tf
                    # test it
                    #a,b=tf.inverse(torch.tensor([[1.,1.]]))
                    #print("test out",a,b)
                X = self.extractModelConcatCols(self.concatParameters(mk_theta,mk),mk)
                # Fit betamix to marginals
                #fn = n.transforms.IdentityTransform()
                #bmmt = BetaMixtureMarginalTransform(XX)
                bmmt = n.transforms.IdentityTransform()
                cbmmt = n.transforms.CompositeCDFTransform(n.transforms.Sigmoid(),bmmt)
                if self.proptype=='affine':
                    fn = n.transforms.InverseTransform(NaiveGaussianTransform(torch.Tensor(X)))
                    self.flows[mk] = Flow(fn,bnorm)
                elif self.proptype=='nf':
                    fn = FixedNorm(torch.Tensor(X))
                    #fn = n.transforms.InverseTransform(NaiveGaussianTransform(torch.Tensor(X)))
                    X_,_ = fn.forward(torch.Tensor(X))
                    XX,__ = ls.forward(X_)
                    if PLOT_PROGRESS:
                        #tmp1,tmp1ld = bmmt.forward(XX)
                        tmp1 = XX
                        tmp1=tmp1.detach().numpy()
                        if self.getModelInt(mk)==0:
                            plt.hist(tmp1,bins=50)
                            plt.show()
                        elif self.getModelInt(mk)==1:
                            plt.scatter(tmp1[:,0],tmp1[:,1])
                            plt.show()
                    self.flows[mk] = RationalQuadraticFlow.factory(X,bnorm,bmmt,ls,fn)
                else:
                    #perfect
                    if self.getModelInt(mk)==0:
                        tf  = n.transforms.InverseTransform(SinArcSinhTransform(ep[0],dp[0]))
                    else:
                        tf  = n.transforms.InverseTransform(n.transforms.CompositeTransform([LTransform(torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))),SAS2DTransform([ep[1],ep[2]],[dp[1],dp[2]])]))
                    self.flows[mk] = Flow(tf,bnorm)
                if PLOT_PROGRESS:
                    tmp1 = self.flows[mk].sample(1000)
                    tmp1=tmp1.detach().numpy()
                    f,ax = plt.subplots(nrows=1,ncols=2)
                    if self.getModelInt(mk)==0:
                        ax[0].hist(X,bins=50,color='red')
                        ax[1].hist(tmp1,bins=50)
                        plt.show()
                    elif self.getModelInt(mk)==1:
                        ax[0].scatter(X[:,0],X[:,1],color='red')
                        ax[1].scatter(tmp1[:,0],tmp1[:,1])
                        plt.show()
            print("MK Z",self.mk_logZhat)
            print("MK Z",[np.exp(z) for i,z in self.mk_logZhat.items()])

    def extractModelConcatCols(self,X,mk):
        if self.getModelInt(mk)==0:
            return X[:,0].reshape((X.shape[0],1))
        elif self.getModelInt(mk)==1:
            return X[:,:2]
    def returnModelConcatCols(self,XX,mk):
        if self.getModelInt(mk)==0:
            return np.column_stack([XX,np.zeros_like(XX)])
        elif self.getModelInt(mk)==1:
            return XX
    def transformToBase(self,inputs,mk):
        X = self.extractModelConcatCols(self.concatParameters(inputs,mk),mk)
        #print("mk",mk,"\ninputs",inputs,"\ntensor",torch.tensor(X,dtype=torch.float32))
        XX,logdet = self.flows[mk]._transform.forward(torch.tensor(X,dtype=torch.float32))
        return self.deconcatParameters(self.returnModelConcatCols(XX.detach().numpy(),mk),inputs,mk),logdet.detach().numpy()
    def transformFromBase(self,inputs,mk):
        X = self.extractModelConcatCols(self.concatParameters(inputs,mk),mk)
        #print("X_base",X)
        XX,logdet = self.flows[mk]._transform.inverse(torch.tensor(X,dtype=torch.float32))
        #print("XX",XX)
        return self.deconcatParameters(self.returnModelConcatCols(XX.detach().numpy(),mk),inputs,mk),logdet.detach().numpy()

    def getModelInt(self,mk):
        if isinstance(mk,list):
            return np.array([m[0] for m in mk])
        else:
            return mk[0]
    def draw(self,theta,size=1):
        global run_idx
        logpqratio = np.zeros(theta.shape[0])
        prop_theta = theta.copy()
        prop_ids = np.zeros(theta.shape[0])
        proposed = {}
        mk_idx = {}
        lpq_mk = {}
        #R = np.array([[-1,-1],[-1,1]])/np.sqrt(2)
        sqrt2 = np.sqrt(2)
        sigma_u = 1 # std dev of auxiliary u dist
        #print("start theta",theta)

        # instead of always proposing a jump, use the model probabilities
        # to determine whether it's a within model proposal or a jump

        #for mk in self.getModel().getModelKeys(theta):
        model_enumeration,rev = self.pmodel.enumerateModels(theta)
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            Tmktheta,lpq1 = self.transformToBase(mk_theta,mk)
            #print("checkpoint1 theta",theta,Tmktheta,lpq1)
            lpq_mk[mk] = lpq1
            proposed[mk],mk_idx[mk] = self.explodeParameters(Tmktheta,mk)
            t1 = proposed[mk][self.t1_name]#.flatten()
            t2 = proposed[mk][self.t2_name]#.flatten()
            k = proposed[mk][self.k_name]#.flatten()

            mk_n = t1.shape[0]

            mprobkeys = list(self.mk_logZhat.keys())

            totallogZ = logsumexp(np.array([z for mkz,z in self.mk_logZhat.items()]))
            mprobs = np.exp(np.array([self.mk_logZhat[mkz]-totallogZ for mkz in mprobkeys]))
            mprobsdict = {mk:mprobs[i] for i,mk in enumerate(mprobkeys)}

            mpropidx = np.random.choice(len(mprobkeys),size=mk_n,p=mprobs)
            mprop = [mprobkeys[i] for i in mpropidx]

            prop_mk_theta = mk_theta.copy()
            mk_prop_ids = np.zeros(mk_n)

            if self.getModelInt(mk)==0:
                jump_idx = self.getModelInt(mprop)==1
                within_idx = self.getModelInt(mprop)==0
                #print(mprop,self.getModelInt(mprop))
                #print("within idx",within_idx,"j",jump_idx)

                # within model move
                prop_mk_theta[within_idx],lpq_mk[mk][within_idx],mk_prop_ids[within_idx] = self.within_model_proposal.draw(mk_theta[within_idx],within_idx.sum())

                # switch to model 2.
                if jump_idx.sum()>0:
                    # draw the aux var. 
                    #u_unif = np.linspace(1e-5,1-1e-5,jump_idx.sum())[run_idx]
                    #u = norm(0,sigma_u).ppf(u_unif)
                    u = norm(0,sigma_u).rvs(jump_idx.sum())
                    #u = uniform(-100,200).rvs(t1.shape)
                    #prop_t1 = t1
                    #prop_t2 = u
                    log_u_eval = norm(0,sigma_u).logpdf(u.flatten())
                    #log_u_eval = uniform(-100,200).logpdf(u.flatten())
                    lpq_mk[mk][jump_idx] += -log_u_eval + np.log(mprobsdict[(0,)]) - np.log(mprobsdict[(1,)])
                    Tmktheta[jump_idx]=self.setVariable(Tmktheta[jump_idx],self.t2_name,u) # for naive, update with random walk using old u
                    Tmktheta[jump_idx]=self.setVariable(Tmktheta[jump_idx],self.k_name,1) # for naive, update with random walk using old u
    
                    #proposed[mk][self.t1_name][jump_idx] = prop_t1
                    #proposed[mk][self.t2_name][jump_idx] = prop_t2
                    #proposed[mk][self.k_name][jump_idx] = 2
                    # transform back
                    #prop_mk_theta[jump_idx],lpq2 = self.transformFromBase(Tmktheta[jump_idx],mprop[jump_idx][0])
                    prop_mk_theta[jump_idx],lpq2 = self.transformFromBase(Tmktheta[jump_idx],(1,))
                    lpq_mk[mk][jump_idx] += lpq2
                    mk_prop_ids[jump_idx] = np.full(jump_idx.sum(),id(self))

            elif self.getModelInt(mk)==1: 
                jump_idx = self.getModelInt(mprop)==0
                within_idx = self.getModelInt(mprop)==1

                # within model move
                prop_mk_theta[within_idx],lpq_mk[mk][within_idx],mk_prop_ids[within_idx] = self.within_model_proposal.draw(mk_theta[within_idx],within_idx.sum())

                if jump_idx.sum()>0:
                    u       = t2[jump_idx]
                    #prop_t1 = t1
                    log_u_eval = norm(0,sigma_u).logpdf(u.flatten())
                    #log_u_eval = uniform(-100,200).logpdf(u.flatten())
                    lpq_mk[mk][jump_idx] += log_u_eval - np.log(mprobsdict[(0,)]) + np.log(mprobsdict[(1,)])
                    #logpqratio[mk_idx[mk]] = log_u_eval + lpq_mk[mk]
                    #proposed[mk][self.t1_name] = prop_t1
                    #proposed[mk][self.t2_name] = 0
                    #proposed[mk][self.k_name] = 1
                    Tmktheta[jump_idx]=self.setVariable(Tmktheta[jump_idx],self.k_name,0) 
                    Tmktheta[jump_idx]=self.setVariable(Tmktheta[jump_idx],self.t2_name,0) 
    
                    # transform back
                    #prop_mk_theta[jump_idx],lpq2 = self.transformFromBase(Tmktheta[jump_idx],mprop[jump_idx][0])
                    prop_mk_theta[jump_idx],lpq2 = self.transformFromBase(Tmktheta[jump_idx],(0,))
                    lpq_mk[mk][jump_idx] += lpq2
                    mk_prop_ids[jump_idx] = np.full(jump_idx.sum(),id(self))

            prop_theta[mk_row_idx] = prop_mk_theta
            logpqratio[mk_row_idx] = lpq_mk[mk]
            prop_ids[mk_row_idx] = mk_prop_ids

        return prop_theta, logpqratio, prop_ids
                


class ToyModel(ParametricModelSpace):
    def __init__(self,proptype='affine'):
        random_variables = {'t1':ImproperRV(),'block2':TransDimensionalBlock({'t2':ImproperRV()},nblocks_name='k',minimum_blocks=0,maximum_blocks=1)}
        #random_variables_start = {'t1':NormalRV(0,5),'block2':TransDimensionalBlock({'t2':NormalRV(0,5)},nblocks_name='k',minimum_blocks=0,maximum_blocks=1)}
        wmp = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=['t1','t2']))
        #proposal = SystematicChoiceProposal([RJToyModelProposalPerfect('k','t1','t2',within_model_proposal=wmp),wmp])
        proposal = RJToyModelProposalPerfect('k','t1','t2',within_model_proposal=wmp,proptype=proptype)
        #self.starting_dist = ParametricModelSpace(random_variables_start,proposal)
        super(ToyModel, self).__init__(random_variables,proposal)
    
    def sas(self,x,i):
        # TODO compute jacobian for each i. Return it. Parameters are not dependent.
        def _sas(x, epsilon, delta):
            return np.sinh((np.arcsinh(x)+epsilon)/delta)
        epsilon = np.array([-2,2,-2,2,-2])
        delta = np.array([1,1,1,1,1])
        return _sas(x,epsilon[i],delta[i])

    def draw_perfect(self,M):
        global ep, dp
        cols = self.generateRVIndices()
        # draw it from prior first
        theta = np.zeros((M,self.dim()))
        # hack it
        theta[:] = self.sampleFromPrior(M)
        print(theta)
        # That sorts out model indicators for now.
        # FIXME for now we leave model indicators as drawn from prior
        # and just draw perfect draws for these models again.
        # This doesn't give us a complete joint posterior
        # because the model probability marginals are wrong, they're the prior.
        # For calibration, we don't care.

        k = theta[:,cols['k']].flatten()
        t1 = theta[:,cols['t1']].flatten()
        t2 = theta[:,cols['t2']].flatten()
        m1 = k==0
        m2 = k==1
        mk1cov = 1.
        mk2cov = torch.tensor([[1.,.99],[.99,1.]]) 
        if True:
            tf1  = n.transforms.InverseTransform(SinArcSinhTransform(ep[0],dp[0]))
            tf2  = n.transforms.InverseTransform(n.transforms.CompositeTransform([LTransform(torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))),SAS2DTransform([ep[1],ep[2]],[dp[1],dp[2]])]))
            #tf1  = n.transforms.InverseTransform(L1DTransform(np.sqrt(mk1cov)))
            #tf1  = n.transforms.IdentityTransform()
            #tf2  = n.transforms.InverseTransform(LTransform(torch.linalg.cholesky(mk2cov)))
            f1 = Flow(tf1,StandardNormal((1,)))
            f2 = Flow(tf2,StandardNormal((2,)))
            m1_v = f1.sample(t1[m1].shape[0]).detach().numpy()
            t1[m1]= m1_v[:,0]
            m2_v = f2.sample(t1[m2].shape[0]).detach().numpy()
            t1[m2]=m2_v[:,0]
            t2[m2]=m2_v[:,1]
            theta[:,cols['t1']] = t1.reshape(theta[:,cols['t1']].shape)
            theta[:,cols['t2']] = t2.reshape(theta[:,cols['t2']].shape)
        return theta

    def compute_llh(self,theta):
        global ep, dp, m1prob
        # we use the RJ target from Andrieu et al 2009
        # \pi(\theta,k) = 0.25 * N(\theta;0,1) * I(k==1) + 0.75 * N(\theta;[0,0],[[1,-0.9],[-0.9,1]]) * I(k==2)
        # k \in {1,2}
        # What kind of priors do I use here? Seems they are ill-formed. Perhaps really wide uniform priors, and hope they don't bias the posterior much.
        cols = self.generateRVIndices()
        k = theta[:,cols['k']].flatten()
        t1 = theta[:,cols['t1']].flatten()
        t2 = theta[:,cols['t2']].flatten()
        m1 = k==0
        m2 = k==1
        llh = np.zeros(k.shape[0])
        mk1cov = 1.
        mk2cov = torch.tensor([[1.,.99],[.99,1.]]) 
        if True:
            tf1  = n.transforms.InverseTransform(SinArcSinhTransform(ep[0],dp[0]))
            tf2  = n.transforms.InverseTransform(n.transforms.CompositeTransform([LTransform(torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))),SAS2DTransform([ep[1],ep[2]],[dp[1],dp[2]])]))
            #tf1  = n.transforms.InverseTransform(L1DTransform(np.sqrt(mk1cov)))
            #tf1  = n.transforms.IdentityTransform()
            #tf2  = n.transforms.InverseTransform(LTransform(torch.linalg.cholesky(mk2cov)))
            f1 = Flow(tf1,StandardNormal((1,)))
            f2 = Flow(tf2,StandardNormal((2,)))
            #print(torch.Tensor([t1[m1]]))
            #print(torch.Tensor(np.column_stack([t1[m2],t2[m2]])))
            if m1.sum() > 0:
                #print(t1[m1])
                #print(torch.Tensor(t1[m1].reshape((t1[m1].shape[0],1))).shape)
                #print(torch.Tensor(t1[m1]).shape)
                #print(f1.log_prob(torch.Tensor(t1[m1]).reshape((-1,1))).shape)
                #sys.exit(0)
                #print(f1.log_prob(torch.Tensor(t1[m1].reshape((t1[m1].shape[0],1)))).shape)
                llh[m1] = np.log(m1prob) + f1.log_prob(torch.Tensor(t1[m1].reshape((t1[m1].shape[0],1)))).detach().numpy().flatten()
            if m2.sum() > 0:
                #print(torch.Tensor(np.column_stack([t1[m2],t2[m2]])))
                llh[m2] = np.log(1-m1prob) + f2.log_prob(torch.Tensor(np.column_stack([t1[m2],t2[m2]]))).detach().numpy().flatten()
        return llh
    def getModelIdentifier_old(self):
        """ overrides the built-in method, returns the k variable """
        # TODO make a method called setModelIdentifyingParameters()
        ids = []
        # append the nblocks identifier
        ids.append('k')
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(id)
        if len(ids)==0:
            return None
        else:
            return ids
        
# TODO Explicitly define nested models as combinations of columns in a ParametricModel

if len(sys.argv) > 3:
    shelffilename=sys.argv[3]
else:
    shelffilename='./RJMCMC_sinharcsinh_nf_shelve.out'

NPARTICLES = 50000
if len(sys.argv) < 2 or sys.argv[1]=="run":
    #proptypes = ['affine','nf','perfect']
    proptypes = ['affine']
    mymodel = ToyModel()
    nd=50
    calibrate_draws = mymodel.draw_perfect(50000)
    m1idx = calibrate_draws[:,1]==0
    m2idx = calibrate_draws[:,1]==1
    m1theta = calibrate_draws[m1idx]
    print("m1theta",m1theta[:10])
    m2theta = calibrate_draws[m2idx]
    sastf = SinArcSinhTransform(ep[0],dp[0])
    #p_u = np.linspace(1e-5,1-1e-5,nd)
    if False:
        p_u = uniform(0,1).rvs(nd)
        p_n = norm(0,1).ppf(p_u)
        p_sas_1d,__ = sastf.forward(torch.Tensor(p_n))
        p_sas_1d = p_sas_1d.detach().numpy()
    else:
        sasnf = Flow(n.transforms.InverseTransform(sastf),StandardNormal((1,)))
        p_sas_1d = sasnf.sample(nd)
        p_n,__ = sastf.inverse(p_sas_1d)
        p_u = norm(0,1).cdf(p_n.detach().numpy())
        print(p_u)
        p_sas_1d = p_sas_1d.detach().numpy()
    p_sas = np.column_stack([p_sas_1d,np.zeros((nd,2))])
    print(p_sas)
    #samp = np.random.choice(m1theta.shape[0],size=nd)
    pt_prop_m1theta = {}
    for pt in proptypes:
        mymodel = ToyModel(pt)
        rjmcmc = RJMCMC(mymodel,calibrate_draws=calibrate_draws) #,starting_distribution=mymodel.starting_dist)
        # just draw one proposal
        #pt_prop_m1theta[pt], prop_m1lpq, prop_m1id = rjmcmc.pmodel.propose(m1theta[samp],nd)
        ptl = []
        for i in range(1):
            run_idx=i
            prop_m1theta, prop_m1lpq, prop_m1id = rjmcmc.pmodel.propose(p_sas,nd)
            ptl.append(prop_m1theta)
        pt_prop_m1theta[pt] = np.vstack(ptl)

    
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')
    f,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,4))
    ax = ax.flatten()
    ax[0].set_title('Source points')
    #ax[0].hist(m1theta[:,0],bins=50,color='lightblue',density=True)
    kde_1D(ax[0],m1theta[:,0],bw=0.25,plotline=False)
    ax[0].set_yticklabels([])
    #ax[0].vlines(m1theta[samp,0],ymin=0,ymax=0.02,color='r',linewidth=0.5)
    #ax[0].vlines(p_sas,ymin=0,ymax=0.02,color='r',linewidth=0.5)
    ax[0].scatter(p_sas_1d,np.zeros_like(p_u),c=p_u,marker='x',cmap='brg')
    pt_titles={'affine':'Affine TRJ','nf':'RQMA TRJ','perfect':'Perfect TRJ'}
    for i,pt in enumerate(proptypes):
        kde_joint(ax[i+1],m2theta[:,[0,2]],cmap="Blues",alpha=1,bw=0.1,maxz_scale=1.5)
        #ax[i+1].scatter(m2theta[:,0],m2theta[:,2],s=0.1,color='lightblue')
        ax[i+1].scatter(pt_prop_m1theta[pt][:,0],pt_prop_m1theta[pt][:,2],marker='x',alpha=1,c=p_u,cmap='brg')
        ax[i+1].set_xlim([-5,15])
        ax[i+1].set_ylim([-7,0.5])
        ax[i+1].set_title("Proposed points")
    #plt.suptitle('Sinh Arcsinh 1D 2D Jump 1->2 PLMA Transform')
    plt.tight_layout()
    plt.show()
                       

   
    if False:
        final_theta, llh,log_prior, ar = rjmcmc.run(NPARTICLES,np.array([1,0,0]))
    
        my_shelf = shelve.open(shelffilename,'n') # 'n' for new
        
        #for key in dir():
        for key in {'final_theta','llh','log_prior','ar'}:
            try:
                my_shelf[key] = globals()[key]
            except TypeError:
                #
                # __builtins__, my_shelf, and imported modules can not be shelved.
                #
                print('ERROR shelving: {0}'.format(key))
        my_shelf.close()
else:

    import matplotlib.pyplot as plt
    with shelve.open(shelffilename) as my_shelf:
        for key in my_shelf:
            globals()[key]=my_shelf[key]

    if len(sys.argv)>=3 and sys.argv[2]=='n':
        final_theta = final_theta[10000:]
        kcol = 1
        nrows = final_theta.shape[1]
        m0idx = final_theta[:,kcol]==0
        m1idx = final_theta[:,kcol]==1
        print(m0idx.sum()/40000.)
        if PLOT_PROGRESS:
            f,ax = plt.subplots(ncols=2)
            ax[0].hist(final_theta[m0idx,0],bins=50)
            ax[1].scatter(final_theta[m1idx,0],final_theta[m1idx,2])
            plt.show()
        sys.exit(0)
        #print(model_tle)
        #for mk,mkidx in model_key_indices.items():
        #for mk,tle in model_tle.items():
        #    count = 0
        #    if mk in model_key_indices.keys():
        #        mkidx = model_key_indices[mk]
        #        count = final_theta[mkidx].shape[0]
        #    print(mk,";", count * 1./NPARTICLES,";", tle)

    # get acceptance rates
    ar_by_temp = {}
    count_by_temp = {}
    nprops_at_temp = {}
    particle_count_by_temp = {}
    Z_by_temp = {}
    Zprob_by_temp = {}
    Zsmc_by_temp = {}
    Zprobsmc_by_temp = {}
    for rbar in rbar_list:
        if rbar.power not in ar_by_temp: 
            ar_by_temp[rbar.power] = {}
            count_by_temp[rbar.power] = {}
            nprops_at_temp[rbar.power] = {}
        #print("gamma_t = ",rbar.power)
        #print("E(ar)=",np.exp(logsumexp(rbar.lar)-np.log(rbar.lar.shape[0])))
        unm, unmidx = np.unique(np.column_stack([rbar.pre_model_ids,rbar.post_model_ids]),return_inverse = True, axis=0)
        for i,unn in enumerate(unm):
            un = tuple(unn)
            #un = unn
            keylength = int(unn.shape[0]/2)
            if un not in ar_by_temp[rbar.power].keys():
                ar_by_temp[rbar.power][un] = []
            mlar = rbar.lar[unmidx==i]
            #print("Count for proposal {}".format(unm[i]),mlar.shape[0])
            ear = min(1.,np.exp(logsumexp(np.minimum(0.,mlar))-np.log(mlar.shape[0])))
            ar_all = np.exp(np.minimum(0.,mlar))
            #print("E(ar({}))=".format(unm[i]),np.exp(logsumexp(mlar)-np.log(mlar.shape[0])))
            #ar_by_temp[rbar.power][un].append(ear)
            ar_by_temp[rbar.power][un].append(ar_all)
            un2 = tuple(unn[0:keylength])
            if un not in count_by_temp[rbar.power].keys():
                count_by_temp[rbar.power][un]=0
            if un not in nprops_at_temp[rbar.power].keys():
                nprops_at_temp[rbar.power][un]=0
            count_by_temp[rbar.power][un]+=mlar.shape[0]
            nprops_at_temp[rbar.power][un]+=1
    probtemps = pp_target.getTemperatures()
    mkeys = pp_target.getModelKeys()
    rssize = 1000
    for t in probtemps:
        Z_by_temp[t] = {}
        Zsmc_by_temp[t] = {}
        Zprob_by_temp[t] = {}
        Zprobsmc_by_temp[t] = {}
        particle_count_by_temp[t] = {}
        temp_target,temp_targetw = pp_target.getOriginalParticleDensityForTemperature(t,resample=True,resample_max_size=rssize)
        mkdict,rev = mymodel.enumerateModels(temp_target)
        for mk in mkeys:
            Z_by_temp[t][mk] = pp_target.getlogZForModelAndTemperature(mk,t)
            Zsmc_by_temp[t][mk] = pp_target.getSMCLogZForModelAndTemperature(mk,t)
            particle_count_by_temp[t][mk] = temp_target[mkdict[mk]].shape[0]*1./rssize
        lZ = []
        lZsmc = []
        for mk in mkeys:
            lZ.append(Z_by_temp[t][mk])
            lZsmc.append(Zsmc_by_temp[t][mk])
        lZsum = logsumexp(np.array(lZ))
        lZsumsmc = logsumexp(np.array(lZsmc))
        for mk in mkeys:
            Zprob_by_temp[t][mk] = np.exp(Z_by_temp[t][mk]-lZsum)
            Zprobsmc_by_temp[t][mk] = np.exp(Zsmc_by_temp[t][mk]-lZsumsmc)
    # this is messy but it doesn't matter
    Zprob_by_mk = {}
    Zprobsmc_by_mk = {}
    particle_prob_by_mk = {}
    for mk in mkeys:
        Zprob_by_mk[mk] = []
        Zprobsmc_by_mk[mk] = []
        particle_prob_by_mk[mk] = []
        for t in probtemps:
            Zprob_by_mk[mk].append(Zprob_by_temp[t][mk])
            Zprobsmc_by_mk[mk].append(Zprobsmc_by_temp[t][mk])
            particle_prob_by_mk[mk].append(particle_count_by_temp[t][mk])


    
    
    temps = list(ar_by_temp.keys())
    model_key_indices, m_tuple_indices = mymodel.enumerateModels(final_theta)

    
    if len(sys.argv)<3 or sys.argv[2]=='p':
    
        # plot posteriors for each model
        n_posterior = np.sum(blocksizes)
        #for mk, m_theta_indices in model_key_indices.items():
        #    m_theta = final_theta[m_theta_indices]
        for mk in pp_target.getModelKeys():
            m_theta, m_theta_w = pp_target.getParticleDensityForModelAndTemperature(mk,1,resample=True,resample_max_size=2000)
            fig, ax = plt.subplots(nrows=n_posterior,ncols=n_posterior)
            #for prow in range(n_posterior):
            for prow in range(n_posterior):
                for pcol in range(n_posterior):
                    rowidx = 1+getblock(prow)+prow
                    colidx = 1+getblock(pcol)+pcol
                    print("prow",prow,"pcol",pcol,"rowidx",rowidx,"colidx",colidx)
                    if prow==pcol:
                        #marginal
                        #print("mtheta shape",m_theta.shape)
                        #print(m_theta)
                        if mk[getblock(prow)]>0:
                            ax[prow,pcol].hist(m_theta[:,rowidx],bins=50)
                        else:
                            ax[prow,pcol].hist(np.zeros_like(m_theta[:,rowidx]),bins=50)
                    elif prow>pcol:
                        #print("mtheta shape",m_theta.shape)
                        x = m_theta[:,colidx]
                        y = m_theta[:,rowidx]
                        if mk[getblock(pcol)]==0:
                            x = np.zeros_like(x)
                        if mk[getblock(prow)]==0:
                            y = np.zeros_like(y)
                        ax[prow,pcol].scatter(x,y,s=0.1)
                    else:
                        # delete axes
                        fig.delaxes(ax[prow,pcol])
                    if pcol>0:
                        ax[prow,pcol].yaxis.set_visible(False)
                    if prow<n_posterior-1:
                        ax[prow,pcol].xaxis.set_visible(False)
            plt.suptitle("Model {}".format(mk))
            plt.show()
        #now plot performance metrics
        #fig = plt.figure(figsize=(16, 8))
        #outer = gridspec.GridSpec(4, ncols, wspace=0.2, hspace=0.2)
        #bk = final_theta[:,0]
        #vk = final_theta[:,0]
        #kidx = [i*(blocksize+1) for i in range(nblocks)]
        #k = final_theta[:,kidx]

    else:
        mk2idx = 0
        msize=1.0
        marker_circle = [dict(markerfacecolor='red', marker='.', markersize=msize, markeredgecolor='red'),
                         dict(markerfacecolor='orange', marker='.',markersize=msize, markeredgecolor='orange'),
                         dict(markerfacecolor='green', marker='.', markersize=msize,markeredgecolor='green'),
                         dict(markerfacecolor='cyan', marker='.',markersize=msize, markeredgecolor='cyan'),
                         dict(markerfacecolor='blue', marker='.',markersize=msize, markeredgecolor='blue'),
                         dict(markerfacecolor='darkblue', marker='.',markersize=msize, markeredgecolor='darkblue'),
                         dict(markerfacecolor='indigo', marker='.',markersize=msize, markeredgecolor='indigo'),
                         dict(markerfacecolor='darkviolet', marker='.',markersize=msize, markeredgecolor='darkviolet'),
                         dict(markerfacecolor='magenta', marker='.',markersize=msize, markeredgecolor='magenta'),
                         dict(markerfacecolor='yellow', marker='.',markersize=msize, markeredgecolor='yellow')]
        def makekeypair(mk1,mk2):
            return tuple(list(mk1)+list(mk2))
        #fig, ax = plt.subplots(nrows=3,ncols=len(model_key_indices))
        #for mk, m_theta in model_key_indices.items():

        NPinv = 1./NPARTICLES
        mklist = pp_target.getModelKeys()
        fig, ax = plt.subplots(nrows=len(mklist),ncols=len(mklist)+1)
        for mk2 in mklist:
            mkidx = 0
            mcount = [1./len(mklist)] + [0 for t in temps]
            mcount_temps = [0] + [t for t in temps]
            for mk in mklist:
                for i, (temp, ar_m) in enumerate(ar_by_temp.items()):
                    pos = temps.index(temp)
                    #print(count_by_temp)
                    if makekeypair(mk,mk2) in count_by_temp[temp].keys():
                        mcount[pos+1]+=float(count_by_temp[temp][makekeypair(mk,mk2)])/float(nprops_at_temp[temp][makekeypair(mk,mk2)])*NPinv
            #ax[mk2idx,-1].plot(mcount_temps,mcount)
            #ax[mk2idx,-1].plot(np.arange(len(mcount)),mcount)
            print(np.arange(len(probtemps)),particle_prob_by_mk[mk2])
            ax[mk2idx,-1].plot(np.arange(len(probtemps)),Zprob_by_mk[mk2],'r',label='Z prob OPR')
            ax[mk2idx,-1].plot(np.arange(len(probtemps)),Zprobsmc_by_mk[mk2],'b',label='Z prob SMC')
            ax[mk2idx,-1].plot(np.arange(len(probtemps)),particle_prob_by_mk[mk2],'g',label='pi prob')
            ax[mk2idx,-1].legend(loc='upper right')
            #ax[mk2idx,-1].set_xlabel('Power Posterior Density')
            #ax[mk2idx,-1].set_xticks(np.arange(1+len(temps)))
            ax[mk2idx,-1].set_xticks(np.arange(len(probtemps)))
            #ax[mk2idx,-1].set_xticklabels(["{:.2e}".format(t) for t in [0.] + temps],rotation=90)
            ax[mk2idx,-1].set_xticklabels(["{:.2e}".format(t) for t in probtemps],rotation=90)
            if mk2idx<len(mklist)-1:
                    ax[mk2idx,-1].set(xticklabels=[]) 
            #ax[mk2idx,-1].set_ylim([-0.1,np.max(mcount)+0.1])
            ax[mk2idx,-1].set_ylim([0.,1.])
            ax[mk2idx,-1].yaxis.tick_right()
            for mk in mklist:
                for i, (temp, ar_m) in enumerate(ar_by_temp.items()):
                    pos = temps.index(temp)
                    #plt.boxplot([ar_m[1],ar_m[2]],positions=[2*i,2*i+1])
                    if makekeypair(mk,mk2) not in ar_m.keys():
                        print(mk,mk2,"not in acceptance rate keys for temp",temp)
                        continue
                    ax[mkidx,mk2idx].boxplot(np.concatenate(ar_m[makekeypair(mk,mk2)]),positions=[pos],flierprops=marker_circle[pos])
                    #ax[mkidx,mk2idx].boxplot(ar_m[makekeypair(mk,mk2)],positions=[pos])
                ax[mkidx,mk2idx].set_xticks(np.arange(len(temps)))
                ax[mkidx,mk2idx].set_xticklabels(["{:.2e}".format(t) for t in temps],rotation=90)
                if mkidx<len(mklist)-1:
                    ax[mkidx,mk2idx].set(xticklabels=[]) 
                ax[mkidx,mk2idx].set_ylim([-0.1,1.1])
                if mk2idx>0:
                    ax[mkidx,mk2idx].set(yticklabels=[]) 
                ax[mkidx,0].set_ylabel('{}'.format(mk), rotation=0,va="center", labelpad=40) #'Jump Proposal Acceptance Rate')
                mkidx += 1
            ax[0,mk2idx].set_title('{}'.format(mk2))
            mk2idx+=1
        ax[0,-1].set_title('Probability')
        plt.show()

# 
# 
#         fig = plt.figure(figsize=(16, 8))
#         outer = gridspec.GridSpec(4, ncols, wspace=0.2, hspace=0.2)
#         inner = gridspec.GridSpecFromSubplotSpec(nblocks*blocksize, nblocks*blocksize,
#                     subplot_spec=outer[0,block], wspace=0.1, hspace=0.1)
#             
#         for block in range(nblocks):
#         for blockvar in range(blocksize)
#             axt = plt.Subplot(fig, inner[j])
#             ax[0,colidx].scatter(final_theta[k[block]==1,final]
#     ax[0,0].hist(final_theta[k==1,1],bins=25)
#     ax[0,1].scatter(final_theta[k==2,1],final_theta[k==2,2])
#     num_steps = {}
#     mcount = {}
#     num_steps[0] = []
#     num_steps[1] = []
#     mcount[0] = [0.5]
#     mcount[1] = [0.5]
#     maxns = 0
#     for i, (temp, ar_m) in enumerate(ar_by_temp.items()):
#         print(ar_m)
#         #plt.boxplot([ar_m[1],ar_m[2]],positions=[2*i,2*i+1])
#         ax[1,0].boxplot(ar_m[0],positions=[i])
#         ax[1,1].boxplot(ar_m[1],positions=[i])
#         num_steps[0].append(len(ar_m[0]))
#         num_steps[1].append(len(ar_m[1]))
#         print(count_by_temp)
#         #mcount[0].append((count_by_temp[temp][0]+count_by_temp[temp][1])*0.001)
#         #mcount[1].append((count_by_temp[temp][2]+count_by_temp[temp][3])*0.001)
#         NPinv = 1./NPARTICLES
#         mcount[0].append((count_by_temp[temp][0])*NPinv)
#         mcount[1].append((count_by_temp[temp][1])*NPinv)
#         maxns = max(maxns,len(ar_m[1]))
#     #plt.xticks(np.arange(len(temps))*2+1,temps)
#     for i in range(2):
#         #ax[1,i].set_xticks(np.arange(len(temps)),temps)
#         ax[1,i].set_xticks(np.arange(len(temps)))
#         ax[1,i].set_xticklabels(temps)
#         ax[2,i].set_xticks(np.arange(len(temps)))
#         ax[2,i].set_xticklabels(temps)
#         ax[3,i].set_xticks(np.arange(1+len(temps)))
#         ax[3,i].set_xticklabels([0.] + temps)
#         ax[1,i].set_ylim([0,1])
#         ax[1,i].set_xlabel('Power Posterior Density')
#         ax[2,i].plot(num_steps[i])
#         ax[2,i].set_ylabel('# Mutation steps')
#         ax[2,i].set_ylim([0,maxns+2])
#         ax[3,i].plot(mcount[i])
#         ax[3,i].set_ylabel('Model Probability')
#         ax[3,i].set_ylim([0,1])
#         if i==0:
#             ax[3,i].plot(np.full(len(mcount[i]),m1prob),':k')
#         else:
#             ax[3,i].plot(np.full(len(mcount[i]),1-m1prob),':k')
#     ax[1,0].set_ylabel('Jump Proposal Acceptance Rate')
#     ax[0,0].set_title('Model 1')
#     ax[0,1].set_title('Model 2')
#     
#     plt.show()
#     
if False:
    ar_by_temp = {}
    count_by_temp = {}
    for rbar in rbar_list:
        if rbar.power not in ar_by_temp: 
            ar_by_temp[rbar.power] = {}
            count_by_temp[rbar.power] = {}
        print("gamma_t = ",rbar.power)
        print("E(ar)=",np.exp(logsumexp(rbar.lar)-np.log(rbar.lar.shape[0])))
        unm, unmidx = np.unique(np.column_stack([rbar.pre_model_ids,rbar.post_model_ids]),return_inverse = True, axis=0)
        #print(unm,unmidx)
        for i,un in enumerate(unm):
            if i not in ar_by_temp[rbar.power]:
                ar_by_temp[rbar.power][i] = []
            mlar = rbar.lar[unmidx==i]
            print("Count for proposal {}".format(unm[i]),mlar.shape[0])
            ear = np.exp(logsumexp(mlar)-np.log(mlar.shape[0]))
            print("E(ar({}))=".format(unm[i]),np.exp(logsumexp(mlar)-np.log(mlar.shape[0])))
            ar_by_temp[rbar.power][i].append(ear)
            count_by_temp[rbar.power][i]=mlar.shape[0]
    
    import matplotlib.pyplot as plt
    temps = list(ar_by_temp.keys())
    fig, ax = plt.subplots(nrows=4,ncols=2)
    k = final_theta[:,0]
    ax[0,0].hist(final_theta[k==1,1],bins=25)
    ax[0,1].scatter(final_theta[k==2,1],final_theta[k==2,2])
    num_steps = {}
    mcount = {}
    num_steps[0] = []
    num_steps[1] = []
    mcount[0] = [0.5]
    mcount[1] = [0.5]
    maxns = 0
    for i, (temp, ar_m) in enumerate(ar_by_temp.items()):
        print(ar_m)
        #plt.boxplot([ar_m[1],ar_m[2]],positions=[2*i,2*i+1])
        ax[1,0].boxplot(ar_m[0],positions=[i])
        ax[1,1].boxplot(ar_m[1],positions=[i])
        num_steps[0].append(len(ar_m[0]))
        num_steps[1].append(len(ar_m[1]))
        print(count_by_temp)
        #mcount[0].append((count_by_temp[temp][0]+count_by_temp[temp][1])*0.001)
        #mcount[1].append((count_by_temp[temp][2]+count_by_temp[temp][3])*0.001)
        NPinv = 1./NPARTICLES
        mcount[0].append((count_by_temp[temp][0])*NPinv)
        mcount[1].append((count_by_temp[temp][1])*NPinv)
        maxns = max(maxns,len(ar_m[1]))
    #plt.xticks(np.arange(len(temps))*2+1,temps)
    for i in range(2):
        #ax[1,i].set_xticks(np.arange(len(temps)),temps)
        ax[1,i].set_xticks(np.arange(len(temps)))
        ax[1,i].set_xticklabels(temps)
        ax[2,i].set_xticks(np.arange(len(temps)))
        ax[2,i].set_xticklabels(temps)
        ax[3,i].set_xticks(np.arange(1+len(temps)))
        ax[3,i].set_xticklabels([0.] + temps)
        ax[1,i].set_ylim([0,1])
        ax[1,i].set_xlabel('Power Posterior Density')
        ax[2,i].plot(num_steps[i])
        ax[2,i].set_ylabel('# Mutation steps')
        ax[2,i].set_ylim([0,maxns+2])
        ax[3,i].plot(mcount[i])
        ax[3,i].set_ylabel('Model Probability')
        ax[3,i].set_ylim([0,1])
        if i==0:
            ax[3,i].plot(np.full(len(mcount[i]),m1prob),':k')
        else:
            ax[3,i].plot(np.full(len(mcount[i]),1-m1prob),':k')
    ax[1,0].set_ylabel('Jump Proposal Acceptance Rate')
    ax[0,0].set_title('Model 1')
    ax[0,1].set_title('Model 2')
    
    plt.show()
    
