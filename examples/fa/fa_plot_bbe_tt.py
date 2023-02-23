import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

SAVEPLOTS = True

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


pfx = 'output/'

nsamp = [2000,16000]
#nsamp = [4000,16000]
#nsamp = [2000,4000,8000]

f,ax = plt.subplots(nrows=1,ncols=2,figsize=(6,2.5),sharey=True )
ax = ax.flatten()
pts = ['lw','af','rq']
pts_dict = {'lw':'Lopes\n& West','af':'Affine TRJ','rq':'RQMA-NF TRJ'}
#pts = ['lw','af','rq']
#pfx='/media/laurence/DATA/fa_simres/tt/'

m1probs = {}
m2probs = {}

# pre-process output, load all into a dict
for pi,proptype in enumerate(pts):
    m1probs[proptype] = {}
    m2probs[proptype] = {}
    for ni,i in enumerate(nsamp):
        m1probs[proptype][i] = []
        m2probs[proptype][i] = []
        for runno in range(10):
            #if proptype=='rq' or proptype=='lw':
            m = np.loadtxt('{}results_pySMC_NUTS_RJMCMC_FA_N{}_run{}_{}_tt3.txt'.format(pfx,i,runno,proptype),usecols=[2,4],delimiter=';')
            #else:
            #    m = np.loadtxt('{}results_pySMC_NUTS_RJMCMC_FA_N{}_run{}_{}_tt.txt'.format(pfx,i,runno,proptype),usecols=[2,4],delimiter=';')
            # m will have 100 lines
            m1probs[proptype][i].append(np.exp(m[:,0]))
            m2probs[proptype][i].append(np.exp(m[:,1]))

for pi,proptype in enumerate(pts):
    for ni,i in enumerate(nsamp):
        #m = np.loadtxt('{}allres_{}_{}_tt'.format(pfx,i,proptype),usecols=[2,4],delimiter=';')
        #ax.violinplot(np.exp(m[:,0]),positions=[ni*len(pts)+pi],)
        #ax[ni].violinplot(np.exp(m[:,0]),positions=[pi],showmeans=True)
        m = np.hstack(m1probs[proptype][i])
        print("m1 probs for prop {} mean={} std={}".format(proptype,m.mean(),np.std(m)))
        ax[ni].violinplot(m,positions=[pi],showmeans=True)
        #if ni==3:
        #    hline1 = ax[ni].hlines(0.88,-0.5,len(pts)-0.5,linewidth=0.5,label='Ground Truth',color='black')
        #    ax[ni].legend(loc='lower right',handles=[hline1])
        #else:
        #    ax[ni].hlines(0.88,-0.5,len(pts)-0.5,linewidth=0.5,color='black',label='Ground Truth')
        ax[ni].set_xticks(np.arange(len(pts)))
        ax[ni].set_xticklabels([pts_dict[pt] for pt in pts],rotation=45)
        ax[ni].set_title('N={} Per Model'.format(int(i)))
        #if ni==1:
        #    ax[ni].set_yticklabels([])
        ax[ni].set_ylim([0.5,1.01])
        if ni==1:
            hline1 = ax[ni].hlines(0.88,-0.5,len(pts)-0.5,linewidth=0.5,color='black',label='Ground Truth')
            ax[ni].legend(loc='lower right',handles=[hline1],fancybox=True, frameon=True,framealpha=0.7)
        else:
            ax[ni].hlines(0.88,-0.5,len(pts)-0.5,linewidth=0.5,color='black',label='Ground Truth')
ax[0].set_ylabel('2-Factor Model\nProbability Estimate')
#plt.suptitle('Probability for 2-factor model')
#plt.legend(loc='lower right')
plt.tight_layout()
if SAVEPLOTS:
    plt.savefig('fa_variability_violin_loglw.pdf')
else:
    plt.show()
