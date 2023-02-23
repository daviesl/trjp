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


class RobustBlockVSModel(ParametricModelSpace):
    def __init__(self,k=None,rj_only=False,is_naive=False):
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
            random_variables[blocknames[i]]=TransDimensionalBlock(block_rvs,nblocks_name=gammanames[i],minimum_blocks=self.minblockcount[i],maximum_blocks=self.maxblockcount[i],nblocks_position='last')
        mep = ModelEnumerateProposal(subproposal=EigDecComponentwiseNormalProposalTrial(proposals=self.betanames))
        if k is None:
            if rj_only:
                proposal = RJZGlobalBlockVSProposalIndiv(self.blocksizes,self.blocknames,self.gammanames,self.betanames,mep,affine=True)
            else:
                proposal = SystematicChoiceProposal([RJZGlobalBlockVSProposalIndiv(self.blocksizes,self.blocknames,self.gammanames,self.betanames,mep,affine=True),mep])
        else:
            proposal = mep
        super(RobustBlockVSModel, self).__init__(random_variables,proposal)

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
        #for i in range(self.nblocks):
        for bn in self.betanames:
            #bn = self.betanames[i]
            if len(cols[bn]) > 0:
                betas_stack.append(theta[:,cols[bn]])
            else:
                #betas_stack.append(np.zeros((theta.shape[0],self.blocksizes[i])))
                betas_stack.append(np.zeros((theta.shape[0],1)))
        betas = np.column_stack(betas_stack)
        #betas = np.column_stack([theta[:,cols[i]] for i in self.betanames])
        gammas = np.column_stack([theta[:,cols[i]] for i in self.gammanames])
        # get model keys (indices)
        #mkeys = self.pmodel.getModelKeys()
        model_enumeration,rev = self.enumerateModels(theta)
        # likelihood is as follows
        # for i in gammas
        # y_i ~ Bern(p_i)
        # p_i = exp(x^T beta)/(1+exp(x^T beta))
        log_like = np.zeros(theta.shape[0])
        for mk, mk_row_idx in model_enumeration.items():
            mk_theta = theta[mk_row_idx]
            #params = self.proposal.explodeParameters(theta,mk)
            #betas = np.column_stack([params[xn] for xn in self.betanames]) # should only return active blocks
            #gammas = np.column_stack([int(params[xn])==1 for xn in self.gammanames]) # should return all gammas. Rows should be identical.
            gamma_vec = gammas[mk_row_idx][0]
            #x_data_active = x_data[:,gamma_vec.astype(bool)] # n x p
            x_data_active = x_data[:,gamma_vec.astype(bool).repeat(self.blocksizes)] # n x p
            betas_active = betas[mk_row_idx][:,gamma_vec.astype(bool).repeat(self.blocksizes)]
            a = np.dot(betas_active,x_data_active.T) # nrows x p dot p x ndata = nrows x ndata #n x p dot p x 1 = n x 1
            #a = np.dot(betas[mk_row_idx],x_data.T) # nrows x p dot p x ndata = nrows x ndata #n x p dot p x 1 = n x 1
            #expa = np.exp(a)
            #log_like[mk_row_idx] = (y_data*a - np.log(1+expa)).sum(axis=1)
            # true likelihood is N(mu=0,var=5)
            log_like[mk_row_idx] = np.log(np.exp(-(y_data-a)**2/2) + np.exp(-(y_data-a)**2/200)/10).sum(axis=1)
            # what do I need the gammas for?
        return log_like
        
nplist = [1000,2000,4000,8000]
NPARTICLES = nplist[int(sys.argv[3])]
k_list = [[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
if sys.argv[1]=="reload":
    all_data = np.loadtxt('six_dim_rr.csv',delimiter=',',skiprows=1,usecols=[1,2,3,4,5])
    outputdir = 'output/smc'
    import os
    os.chdir(outputdir)
    if len(sys.argv) >= 3:
        run_no=int(sys.argv[2])
    else:
        run_no=0
    theta_list = []
    for k in k_list:
        theta_list.append(np.load('SMC2_RobustBlockVS_d4_2block_saturated_k1{}{}_run{}_N{}.npy'.format(k[1],k[2],run_no,NPARTICLES)))

    x_data = all_data[:,1:]
    y_data = all_data[:,0]
    # now run the bridge
    theta=np.vstack(theta_list)
    train_idx = np.random.choice(theta.shape[0],size=int(theta.shape[0]/2),replace=False)
    test_idx = test_idx = np.delete(np.arange(theta.shape[0]),train_idx)
    train_theta = theta[train_idx]
    test_theta = theta[test_idx]
    if True:
        mymodel = RobustBlockVSModel(rj_only=True,is_naive=True)
        rjb = RJBridge(mymodel,train_theta)
        log_p_mk_dict = rjb.estimate_log_p_mk(test_theta,2000)
        #print(log_p_mk_dict)
        for mk,lp in log_p_mk_dict.items():
            print("Bartolucci RJBridge prob for ",mk," is ",np.exp(lp))
        with open("results_robustvs_d4_2block_affine_N{}.txt".format(NPARTICLES), "a") as file_object:
            file_object.write("BE_lp;")
            for mk,log_p_mk in log_p_mk_dict.items():
                file_object.write(str(mk)+';'+str(log_p_mk)+";")
            file_object.write("0\n")


