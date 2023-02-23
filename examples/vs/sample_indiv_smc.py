import sys
from rjlab.samplers import *
from rjlab.variables import *
from rjlab.proposals.standard import *
from rjlab.proposals.vs_proposals import *
from rjlab.transforms import *
import nflows as n
from nflows.distributions.uniform import *
from nflows.transforms import *
import shelve

PLOT_PROGRESS = False


class RobustBlockVSModelIndiv(ParametricModelSpace):
    def __init__(self,k,rj_only=False,is_naive=False):
        random_variables = {}
        self.nblocks = 3
        self.blocksizes = [1,1,2]
        if k is None:
            self.minblockcount = [1,0,0]
            self.maxblockcount = [1,1,1]
        else:
            self.minblockcount = k
            self.maxblockcount = k

        self.blocknames = blocknames = ['b{}'.format(i) for i in range(self.nblocks)]
        self.betanames = betanames = ['beta{}{}'.format(i,j) for i in range(self.nblocks) for j in range(self.blocksizes[i])]
        self.gammanames = gammanames = ['gamma{}'.format(i) for i in range(self.nblocks)]
        for i in range(self.nblocks):
            cbs = int(np.array(self.blocksizes[:i]).sum())
            block_rvs = {betanames[cbs+j]:NormalRV(0,10) for j in range(self.blocksizes[i])}
            block_rvs[gammanames[i]]=UniformIntegerRV(self.minblockcount[i],self.maxblockcount[i])
            random_variables[blocknames[i]]=RandomVariableBlock(block_rvs)
        proposal = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        super(RobustBlockVSModelIndiv, self).__init__(random_variables,proposal)

    def compute_llh(self,theta):
        """
        target <- function(x){
          p <- length(x)
          a <- X%*%x
          mn <- exp(-(y - a)^2/2) + exp(-(y - a)^2/200)/10 # Normal mix part
          phi_0 <- log(mn)   ## Log likelihood
          log_q <- sum(phi_0)  + sum(x^2/200)  ## Add a N(0,10) prior
          return(list(log_q = log_q))
        }
        """
        global m1prob,x_data,y_data
        cols = self.generateRVIndices()
        betas_stack = []
        for bn in self.betanames:
            if len(cols[bn]) > 0:
                betas_stack.append(theta[:,cols[bn]])
            else:
                #betas_stack.append(np.zeros((theta.shape[0],self.blocksizes[i])))
                betas_stack.append(np.zeros((theta.shape[0],1)))
        betas = np.column_stack(betas_stack)
        gammas = np.column_stack([theta[:,cols[i]] for i in self.gammanames])
        # get model keys (indices)
        model_enumeration,rev = self.enumerateModels(theta)
        # likelihood is as follows
        # for i in gammas
        # y_i ~ Bern(p_i)
        # p_i = exp(x^T beta)/(1+exp(x^T beta))
        log_like = np.zeros(theta.shape[0])
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            gamma_vec = gammas[mk_row_idx][0]
            x_data_active = x_data[:,gamma_vec.astype(bool).repeat(self.blocksizes)] # n x p
            betas_active = betas[mk_row_idx][:,gamma_vec.astype(bool).repeat(self.blocksizes)]
            a = np.dot(betas_active,x_data_active.T) # nrows x p dot p x ndata = nrows x ndata #n x p dot p x 1 = n x 1
            # true likelihood is N(mu=0,var=5)
            log_like[mk_row_idx] = np.log(np.exp(-(y_data-a)**2/2) + np.exp(-(y_data-a)**2/200)/10).sum(axis=1)
        return log_like
    def getModelIdentifier(self):
        """ overrides the built-in method, returns the k variable """
        # TODO make a method called setModelIdentifyingParameters()
        ids = self.gammanames
        for name in self.rv_names:
            i = self.rv[name].getModelIdentifier()
            if i is not None:
                ids.append(id)
        if len(ids)==0:
            return None
        else:
            return ids
        
nplist = [1000,2000,4000,8000]
NPARTICLES = nplist[int(sys.argv[3])]
k_list = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputdir = 'output/smc'
if len(sys.argv) < 2 or sys.argv[1]=="run":
    if len(sys.argv) >= 3:
        run_no=int(sys.argv[2])
    else:
        run_no=0
    # TODO Explicitly define nested models as combinations of columns in a ParametricModel
    all_data = np.loadtxt('six_dim_rr.csv',delimiter=',',skiprows=1,usecols=[1,2,3,4,5])
    import os
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    os.chdir(outputdir)
    # convert data to required arrays
    x_data = all_data[:,1:]
    y_data = all_data[:,0]
    # run SMC for each model
    theta_list = []
    for k in k_list:
        mymodel = RobustBlockVSModelIndiv(k)
        smc = SMC1(mymodel)
        shelffilename='./SMC2_RobustBlockVS_2block_saturated_shelve_k1{}{}_run{}.out'.format(k[1],k[2],run_no)
        final_theta, smc_quantities, llh,log_prior, pp_target, rbar_list = smc.run(NPARTICLES,ess_threshold=0.5)
        theta_list.append(final_theta)
        np.save('SMC2_RobustBlockVS_d4_2block_saturated_k1{}{}_run{}_N{}.npy'.format(k[1],k[2],run_no,NPARTICLES),final_theta)
        if False:
            my_shelf = shelve.open(shelffilename,'n') # 'n' for new
            for key in {'final_theta','smc_quantities','llh','log_prior','pp_target','rbar_list'}:
                try:
                    my_shelf[key] = globals()[key]
                except TypeError:
                    print('ERROR shelving: {0}'.format(key))
            my_shelf.close()
