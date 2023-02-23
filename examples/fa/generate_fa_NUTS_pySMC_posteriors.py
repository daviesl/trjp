import sys
import numpy as np
import matplotlib.pyplot as plt
from examples.fa.fa_model import *
import os

outputdir = 'output'
proptypelist = ['lw','af','rq']
if __name__ == "__main__":
    Y = np.load('FA_data.npy')
    NP_list = [2000,4000,8000,16000] # TODO use this list via arg
    NPARTICLES=NP_list[int(sys.argv[1])]
    run_no=int(sys.argv[2])

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    os.chdir(outputdir)

    if sys.argv[3]=='run':
        import pymc as pm
        import aesara
        #aesara.config.mode = 'FAST_COMPILE'
        #aesara.config.optimizer_verbose = True
        #aesara.config.optimizer='None'
        aesara.config.exception_verbosity='high'
        from aesara import tensor as at
        import arviz as az
        CORES = 2
        CHAINS = 2
        SEED = 123456789 + CHAINS * run_no
        DRAWS = int(NPARTICLES)
        
        SMC_SAMPLE_KWARGS = {
            'cores': CORES,
            'random_seed': [SEED + i for i in range(CHAINS)],
            'draws':DRAWS,
            'chains':CHAINS,
            'return_inferencedata': True
        }
        NUTS_SAMPLE_KWARGS = {
            'cores': CORES,
            'init': 'adapt_diag',
            'random_seed': [SEED + i for i in range(CHAINS)],
            'draws':DRAWS,
            'tune':1000,
            'chains':CHAINS,
            'return_inferencedata': True
        }
        # generate SMC2 posterior for 2-factor model
        mk_theta_train = {}
        mk_theta_test = {}
        k = 1
        fa2f_model = pm.Model()
    
        with fa2f_model:
            betaii = pm.HalfNormal("betaii", sigma=1., shape=2)
            betaij = pm.Normal("betaij", mu=0., sigma=1., shape=5+4)
            lambdaii = pm.InverseGamma("lambdaii", 1.1, 0.05, shape=(6))
            mu = at.zeros(6)
            beta1 = at.stack([betaii[0]]+[betaij[i] for i in range(5)],axis=0)
            beta2 = at.stack([0, betaii[1]]+[betaij[i] for i in range(5,9)],axis=0)
            #beta = at.stack([beta1,beta2,at.zeros(6),at.zeros(6),at.zeros(6),at.zeros(6)],axis=1)
            beta = at.stack([beta1,beta2],axis=1)
            sigma = at.diag(lambdaii) + at.dot(beta, beta.T)
            obs = pm.MvNormal("obs", mu=mu,cov=sigma, observed=Y)

        with fa2f_model:
            fa2f_trace = pm.sample_smc(**SMC_SAMPLE_KWARGS)
            #fa2f_trace.to_netcdf('fa_m2.nc')
            #if PLOT_PROGRESS:
            #    fa2f_trace = az.from_netcdf('fa_m2.nc')
            #    pm.plot_trace(fa2f_trace, var_names=['betaii','betaij','lambdaii']) 
            #    plt.show()
            #    az.rcParams["plot.max_subplots"] = 500
            #    az.plot_pair(fa2f_trace,var_names=['betaii','betaij','lambdaii'],scatter_kwargs={'alpha': 0.5,'ms':0.2})
            #    plt.show()
            # arrange parameters in format for RJBridge
            p = fa2f_trace['posterior']
            betaii = p['betaii']
            betaij = p['betaij']
            lambdaii = p['lambdaii']
            for gid,group in enumerate(['test','train']):
                thlist = []
                for i in range(int(gid*CHAINS/2),int((gid+1)*CHAINS/2)):
                    thlist.append(np.column_stack([betaii[i,:,0],betaij[i,:,0:5],betaii[i,:,1],betaij[i,:,5:9],np.zeros((DRAWS,4)),np.ones(DRAWS)*k,lambdaii[i]]))
                m2theta = np.vstack(thlist)
                np.save('FA_pyMC_k{}_N{}_run{}_{}.npy'.format(k,NPARTICLES,run_no,group),m2theta)
                if group=='train':
                    mk_theta_train[k] = m2theta
                else:
                    mk_theta_test[k] = m2theta

        # generate NUTS posterior for 3-factor model
        k=2
        fa3f_model = pm.Model()
    
        with fa3f_model:
            betaii = pm.HalfNormal("betaii", sigma=1., shape=3)
            betaij = pm.Normal("betaij", mu=0., sigma=1., shape=5+4+3)
            lambdaii = pm.InverseGamma("lambdaii", 1.1, 0.05, shape=(6))
            mu = at.zeros(6)
            beta1 = at.stack([betaii[0]]+[betaij[i] for i in range(5)],axis=0)
            beta2 = at.stack([0, betaii[1]]+[betaij[i] for i in range(5,9)],axis=0)
            beta3 = at.stack([0,0,betaii[2]]+[betaij[i] for i in range(9,12)],axis=0)
            #beta = at.stack([beta1,beta2,beta3,at.zeros(6),at.zeros(6),at.zeros(6)],axis=1)
            beta = at.stack([beta1,beta2,beta3],axis=1)
            sigma = at.diag(lambdaii) + at.dot(beta, beta.T)
            obs = pm.MvNormal("obs", mu=mu,cov=sigma, observed=Y)
        
        
        with fa3f_model:
            fa3f_trace = pm.sample(**NUTS_SAMPLE_KWARGS)
            #fa3f_trace.to_netcdf('fa_m3.nc')
            #if PLOT_PROGRESS:
            #    fa3f_trace = az.from_netcdf('fa_m3.nc')
            #    pm.plot_trace(fa3f_trace, var_names=['betaii','betaij','lambdaii']) 
            #    plt.show()
            #    az.rcParams["plot.max_subplots"] = 500
            #    az.plot_pair(fa3f_trace,var_names=['betaii','betaij','lambdaii'],scatter_kwargs={'alpha': 0.5,'ms':0.2})
            #    plt.show()
            # arrange parameters in format for RJBridge
            p = fa3f_trace['posterior']
            betaii = p['betaii']
            betaij = p['betaij']
            lambdaii = p['lambdaii']
            for gid,group in enumerate(['test','train']):
                thlist = []
                for i in range(int(gid*CHAINS/2),int((gid+1)*CHAINS/2)):
                    thlist.append(np.column_stack([betaii[i,:,0],betaij[i,:,0:5],betaii[i,:,1],betaij[i,:,5:9],betaii[i,:,2],betaij[i,:,9:],np.ones(DRAWS)*k,lambdaii[i]]))
                m3theta = np.vstack(thlist)
                np.save('FA_pyMC_k{}_N{}_run{}_{}.npy'.format(k,NPARTICLES,run_no,group),m3theta)
                if group=='train':
                    mk_theta_train[k] = m3theta
                else:
                    mk_theta_test[k] = m3theta
    elif sys.argv[3]=='bridge':
        mk_theta_train = {}
        mk_theta_test = {}
        for k in range(1,3):
            mk_theta_train[k] = np.load('FA_pyMC_k{}_N{}_run{}_{}.npy'.format(k,NPARTICLES,run_no,'train'))
            mk_theta_test[k] = np.load('FA_pyMC_k{}_N{}_run{}_{}.npy'.format(k,NPARTICLES,run_no,'test'))

        # run the RJBridge
        ft = proptypelist[int(sys.argv[4])]
        fa_k1k2_model = FactorAnalysisModel(Y,ft=ft,k_min=1,k_max=2)
        train_theta = np.vstack([th for mk,th in mk_theta_train.items()])
        test_theta = np.vstack([th for mk,th in mk_theta_test.items()])
        print("train theta size",train_theta.shape)
        print("test theta size",test_theta.shape)
        for i in range(100):
            rjb = RJBridge(fa_k1k2_model,train_theta)
            log_p_mk_dict = rjb.estimate_log_p_mk(test_theta,1000)
            for mk,lp in log_p_mk_dict.items():
                print("Bartolucci RJBridge prob for ",mk," is ",np.exp(lp))
            #with open("results_pySMC_NUTS_RJMCMC_FA_N{}_run{}_{}_tt.txt".format(NPARTICLES,run_no,ft), "a") as file_object:
            #    # Trialling with initial whitening instead of marginal standardisation
            with open("results_pySMC_NUTS_RJMCMC_FA_N{}_run{}_{}_tt3.txt".format(NPARTICLES,run_no,ft), "a") as file_object:
                file_object.write("BE_lp;")
                for mk,log_p_mk in log_p_mk_dict.items():
                    file_object.write(str(mk)+';'+str(log_p_mk)+";")
                file_object.write("{}\n".format(run_no))
   
