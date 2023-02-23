import sys
import numpy as np
import matplotlib.pyplot as plt
from fa_model import *

import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from rjlab.distributions import *
from rjlab.samplers.smc import *
from rjlab.utils.kde import *
from rjlab.transforms import *
plt.style.use('seaborn-white')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

kdegriddim = 128 # 256 # 1024

SAVEPLOTS = True


class RJPropTest(object):
    def __init__(self,parametric_model,target_draws):
        """
        target_draws should be an enumerated model list perhaps. TODO determine this.
        This class uses equation (16) from Bartolucci et al (2006) 
        """
        self.pmodel = parametric_model
        self.nmutations = 0
        self.use_mmmpd = True
        assert(isinstance(self.pmodel,ParametricModelSpace))
        # init future things
        self.init_more()
        self.target_draws = target_draws
        if not self.use_mmmpd:
            self.pmodel.calibrateProposalsUnweighted(calibrate_draws,calibrate_draws.shape[0],1)
        else:
            self.pmodel.setStartingDistribution(self.pmodel)

            # create the MMMPD
            mmmpd = SingleModelMPD(self.pmodel)

            # add prior draws to avoid an exception
            init_draws = self.pmodel.draw(target_draws.shape[0])
            init_llh = self.pmodel.compute_llh(init_draws)
            init_log_prior = self.pmodel.compute_prior(init_draws)
            init_pppd = PowerPosteriorParticleDensity(self.pmodel,None,init_llh,init_log_prior,init_draws,0.,np.log(np.ones_like(init_llh)*1./init_draws.shape[0]))
            mmmpd.addComponent(init_pppd)

            # add target draws
            llh = self.pmodel.compute_llh(target_draws)
            log_prior = self.pmodel.compute_prior(target_draws)
            pppd = PowerPosteriorParticleDensity(self.pmodel,None,llh,log_prior,target_draws,1.0,np.log(np.ones_like(llh)*1./target_draws.shape[0]))
            mmmpd.addComponent(pppd)
            self.pmodel.calibrateProposalsMMMPD(mmmpd,target_draws.shape[0],1)
    def init_more(self):
        pass
    def propose(self,target_draws,blocksize=None):
        theta = target_draws
        N = theta.shape[0]
        if blocksize is None:
            blocksize = N
        llh = np.zeros(N) # log likelihood
        cur_prior = np.zeros(N) # log prior 
        prop_theta = np.zeros_like(theta)
        log_acceptance_ratio = np.zeros(N)
        prop_llh = np.full(N,np.NINF)
        prop_prior = np.zeros(N)
        prop_id = np.zeros(N)
        prop_lpqratio = np.zeros(N)
        # clean up theta if necessary
        theta = self.pmodel.sanitise(theta)
        # get indices for computation blocks
        nblocks = int(np.ceil((1. * N)/blocksize))
        blocks = [np.arange(i*blocksize,min(N,(i+1)*blocksize)) for i in range(nblocks)]
        #print("blocks\n",blocks)
        # TODO reuse this computation from constructor for mmmpd
        for bidx in blocks:
            print("proposing for block shape",bidx.shape,"index ",bidx.min()," to ",bidx.max())
            llh[bidx] = self.pmodel.compute_llh(theta[bidx])
            cur_prior[bidx] = self.pmodel.compute_prior(theta[bidx])
            prop_theta[bidx], prop_lpqratio[bidx], prop_id[bidx] = self.pmodel.propose(theta[bidx],N)
            prop_prior[bidx] = self.pmodel.compute_prior(prop_theta[bidx])
        ninfprioridx = np.where(~np.isfinite(cur_prior)) # is this used?
        # sanitise again
        prop_theta = self.pmodel.sanitise(prop_theta)
        # only compute likelihoods of models that have non-zero prior support
        valid_theta = np.logical_and(np.isfinite(prop_prior),np.isfinite(prop_lpqratio))
        prop_llh[valid_theta] = self.pmodel.compute_llh(prop_theta[valid_theta,:])

        #print("acceptance ratio",prop_lpqratio,prop_llh,llh,cur_prior,prop_prior) # TODO implement
        log_acceptance_ratio[:] = self.pmodel.compute_lar(theta,prop_theta,prop_lpqratio,prop_llh,llh,cur_prior,prop_prior,1) # TODO implement
        return prop_theta
        if False:
            print("theta first 10:",theta[:10])
            print("prop theta first 10:",prop_theta[:10])
            print("Log AR first 10:",log_acceptance_ratio[:10])
            print("prior_first 10:",cur_prior[:10])
            print("prop prior_first 10:",prop_prior[:10])
            print("llh 10:",llh[:10])
            print("prop llh 10:",prop_llh[:10])
            print("Log AR last 10:",log_acceptance_ratio[-10:])
            print("prior_last 10:",cur_prior[-10:])
            print("prop prior_last 10:",prop_prior[-10:])
            print("llh last 10:",llh[-10:])
            print("prop llh last 10:",prop_llh[-10:])
            print("non-fininte Log AR:",np.where(~np.isfinite(log_acceptance_ratio))[0],np.sum(~np.isfinite(log_acceptance_ratio)))
        if False:
            import seaborn as sns
            import pandas as pd
            import matplotlib.pyplot as plt
            idxX8=[0,1,3,4,6,7,9,10]
            idxk8=[2,5,8,11]
            idxX4=[0,1,3,4]
            idxk4=[2,5,6,7]
            idxXFA=list(range(0,15))+list(range(16,22))
            idxkFA=15
            k0=[1] #[1,0,0,0] #[1,1,1,0]
            k1=[2] #[1,1,0,0] #[1,1,1,1]
            X = theta[:,idxXFA]
            k = theta[:,idxkFA]
            propX = prop_theta[:,idxXFA]
            propk = prop_theta[:,idxkFA]
            print("k==k0",k==k0)
            print("propk==k1",propk==k1)
            #k_up = np.logical_and(np.all(k==k0,axis=1),np.all(propk==k1,axis=1))
            #k_down = np.logical_and(np.all(k==k1,axis=1),np.all(propk==k0,axis=1))
            k_up = np.logical_and(k==k0,propk==k1)
            k_down = np.logical_and(k==k1,propk==k0)
            print("kup",k_up)
            labels = ['beta{}'.format(i) for i in range(X.shape[1])]
            labels = labels + ['Proposed']
            #didx = np.random.choice(X[ki].shape[0],size=1000,replace=False)
            #data = np.vstack([np.column_stack([propX[didx],np.ones(1000)+ki[didx]]),np.column_stack([X[didx],np.zeros(1000)+propki[didx]])])
            data_lower = np.vstack([np.column_stack([propX[k_down],np.ones((k_down).sum())]),np.column_stack([X[k_up],np.zeros((k_up).sum())])])
            #didx = np.random.choice(X[~ki].shape[0],size=1000,replace=False)
            data_upper = np.vstack([np.column_stack([propX[k_up],np.ones((k_up).sum())]),np.column_stack([X[k_down],np.zeros((k_down).sum())])])
            df=pd.DataFrame(data_lower,columns=labels)
            g = sns.PairGrid(df,hue=labels[-1],palette="Paired")
            g.map_lower(sns.scatterplot,s=1)
            g.map_diag(sns.distplot)
            g.add_legend()
            plt.show()
            # again for down
            df=pd.DataFrame(data_upper,columns=labels)
            g = sns.PairGrid(df,hue=labels[-1],palette="Paired")
            g.map_lower(sns.scatterplot,s=1)
            g.map_diag(sns.distplot)
            g.add_legend()
            plt.show()


proptypelist = ['lw','af','rq']
if __name__ == "__main__":
    Y = np.load('FA_data.npy')

    mk_theta_train = {}
    mk_theta_test = {}
    mk_theta_mode = {}
    for k in range(2,4):
        mk_theta = np.load('gold_m{}.npy'.format(k))
        train_idx = np.zeros(mk_theta.shape[0]).astype(bool)
        train_idx[::2]=True
        test_idx = ~train_idx
        mk_theta_train[k] = mk_theta[train_idx]
        mk_theta_test[k] = mk_theta[test_idx]
        # get one point, Max likelihood and hopefully the mode.
        #findmode = mk_theta_test[k][1::10][:,list(range(11))]
        #h = gaussian_kde(mk_theta_test[k][::10][:,list(range(1,2))]).pdf(mk_theta_test[k][1::10][:,list(range(1,2))])
        #gkde = gaussian_kde(findmode.T)
        #h = gkde.pdf(findmode.T)
        #h_as = np.argsort(h)[::-1][:10]
        #print("mode idxs",h_as)
        ##mode = mk_theta_test[k][1::10][np.argmax(h)]
        #modes = mk_theta_test[k][1::10][h_as]
        #print(k,"mode",mode)
        #mk_theta_mode[k] = np.tile(modes,(100,1))
        mk_theta_mode[k] = mk_theta_test[k][0::100]


    prop_theta = {}
    train_theta = np.vstack([th for mk,th in mk_theta_train.items()])
    test_theta = np.vstack([th for mk,th in mk_theta_test.items()])

    # todo do all proposal types
    #ft = proptypelist[int(sys.argv[1])]
    for ft in proptypelist:
        prop_theta[ft] = {}
        fa_k1k2_model = FactorAnalysisModel(Y,ft=ft,k_min=1,k_max=2)

        proptest = RJPropTest(fa_k1k2_model,train_theta[::10])
        for i,k in enumerate(range(2,4)):
            prop_theta[ft][k] = proptest.propose(mk_theta_mode[k])
            #prop_theta[ft][k] = proptest.propose(mk_theta_test[k])
            # plot
            if False:
                labels = ['beta{}'.format(i) for i in range(mk_theta_test[k].shape[1])]
                df=pd.DataFrame(mk_theta_test[k][::250],columns=labels)
                g = sns.PairGrid(df,hue=labels[-1],palette="Paired")
                g.map_lower(sns.scatterplot,s=1)
                g.map_diag(sns.distplot)
                g.add_legend()
                plt.show()
            elif False:
                plot_bivariates_scatter(mk_theta_test[k][::50])

    f, ax = plt.subplots(nrows=2,ncols=2,figsize=(0.9*6.,0.9*7.))
    ax = ax.flatten()
    #sns.kdeplot(mk_theta_test[2][:,[9,10]],ax=ax[0])
    mktt2 = mk_theta_test[2]
    mktt3 = mk_theta_test[3]
    #df=pd.DataFrame([mktt2[:,[9,10]]])
    #sns.kdeplot(df,ax=ax[0])
    #sns.kdeplot(mk_theta_test[3][:,[14,15]],ax=ax[1])
    #sns.kdeplot(pd.DataFrame([mktt3[:,[14,15]]]),ax=ax[1])
    kde_joint(ax[0],mktt2[:,[9,10]],cmap="Blues",alpha=1,bw=0.05,n_grid_points=kdegriddim)
    for i in range(1,4):
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([-1,1])
        kde_joint(ax[i],mktt3[:,[9,13]],cmap="Blues",alpha=1,bw=0.05,n_grid_points=kdegriddim)
    for i,ft in enumerate(proptypelist):
        ax[i+1].scatter(prop_theta[ft][2][:,9],prop_theta[ft][2][:,13],color='red',s=1,alpha=0.7)
        print("prop type",ft)
        print(prop_theta[ft][2][:10,[9,13]])
    #ax[0].scatter(mk_theta_mode[2][0,9],mk_theta_mode[2][0,10],color='red',s=100, marker='+')
    ax[0].scatter(mk_theta_mode[2][:,9],mk_theta_mode[2][:,10],color='red',s=0.3, alpha=0.5)
    #ax[0].scatter(mktt2[::100,9],mktt2[::100,10],color='red',s=1, alpha=0.4)
    ax[0].title.set_text('Source\n2-Factor Model')
    ax[0].set_xlabel(r'$\beta_{2,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[0].set_ylabel(r'$\beta_{2,5}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[1].title.set_text('Target\n3-Factor Model, L&W')
    ax[1].set_xlabel(r'$\beta_{2,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[1].set_ylabel(r'$\beta_{3,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[2].title.set_text('Target\n3-Factor Model, Affine')
    ax[2].set_xlabel(r'$\beta_{2,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[2].set_ylabel(r'$\beta_{3,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[3].title.set_text('Target\n3-Factor Model, RQMA')
    ax[3].set_xlabel(r'$\beta_{2,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    ax[3].set_ylabel(r'$\beta_{3,4}$') # beta1 1,2,3,4,5,6 beta2 2,3,4,5,6
    for i in range(4):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
    plt.tight_layout()
    if SAVEPLOTS:
        plt.savefig('fa_proposal.pdf')
    else:
        plt.show()
