import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import shelve
import os

SAVEPLOTS=True

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


lz = np.loadtxt('SMCLogZ_BlockVS_indiv.txt',delimiter=';',usecols=[0,3])
p2lz = [0,2,3,6]
nmodels = len(p2lz)
#klabel=['0,0','0,1','1,0','1,1']
klabel=['1,0,0,0','1,0,1,1','1,1,0,0','1,1,1,1']

outputdir = 'output/smc'
os.chdir(outputdir)

nplist = [1000,2000,4000,8000]
p_cnf={}
p_af={}
p_na={}
p_inf={}
for nn in nplist:
    lp_ss_cnf = np.loadtxt('results_robustvs_d4_2block_cnf_N{}.txt'.format(nn),delimiter=';',usecols=[2,4,6,8])
    p_cnf[nn] = np.exp(lp_ss_cnf)
    lp_ss_af = np.loadtxt('results_robustvs_d4_2block_affine_N{}.txt'.format(nn),delimiter=';',usecols=[2,4,6,8])
    p_af[nn] = np.exp(lp_ss_af)
    lp_ss_na = np.loadtxt('results_robustvs_d4_2block_naive_N{}.txt'.format(nn),delimiter=';',usecols=[2,4,6,8])
    p_na[nn] = np.exp(lp_ss_na)
    lp_ss_nf = np.loadtxt('results_robustvs_d4_2block_inf_N{}.txt'.format(nn),delimiter=';',usecols=[2,4,6,8])
    p_inf[nn] = np.exp(lp_ss_nf)

lzdict = {}
for pi,lzi in enumerate(p2lz):
    lzdict[pi]=lz[lz[:,0]==lzi,1]
# compute probs from lz
lzmeans = np.zeros(nmodels)
for i,lz in lzdict.items():
    lzmeans[i] = logsumexp(lz) - np.log(lz.shape[0])
goldlogprobmean = lzmeans - logsumexp(lzmeans)
goldlogprobdict = {}
for i,lz in lzdict.items():
    goldlogprobdict[i] = lz - logsumexp(lzmeans)
# plot
fig,ax = plt.subplots(nrows=4,ncols=4,figsize=(8, 8.5))
for j,nn in enumerate(nplist):
    for i,glp in goldlogprobdict.items():
        #ax[i].violinplot(np.exp(glp),positions=[1],  vert=1) #whis=1.5)
        ax[j,i].hlines(np.exp(glp).mean(),xmin=-0.5, xmax=3.5,linewidth=0.5,color='black',label='Ground Truth')
        #ax[i].violinplot(probnfdict[i],positions=[0],  vert=1)#, whis=1.5)
        #ax[i].violinplot(probafdict[i],positions=[1],  vert=1)#, whis=1.5)
        #ax[i].violinplot(probnrwdict[i],positions=[2],  vert=1)#, whis=1.5)
        ax[j,i].violinplot(p_na[nn][:,i],positions=[0],  vert=1,showmeans=True)#, whis=1.5)
        ax[j,i].violinplot(p_af[nn][:,i],positions=[1],  vert=1,showmeans=True)#, whis=1.5)
        ax[j,i].violinplot(p_inf[nn][:,i],positions=[2],  vert=1,showmeans=True)#, whis=1.5)
        ax[j,i].violinplot(p_cnf[nn][:,i],positions=[3],  vert=1,showmeans=True)#, whis=1.5)
        #ax[i].set_xticks([0,1], ['Gold', 'NFSMC1'])
        #xtickNames = plt.setp(ax[i], xticklabels=['Gold','NFSMC1','AFSMC1','NRWSMC1'])
        #labels=['Indiv_NF_10k_SMC1OPR','Affine_10k_SMC1OPR','Naive_SS_10k_SMC1OPR','CNF_SS_BBE_10kEach']
        labels=['Standard','A-TRJ','RQMA-TRJ','RQMA-CTRJ']
        ax[j,i].set_xticks(np.arange(0, len(labels)))
        if j==3:
            ax[j,i].set_xticklabels(labels,rotation=90,fontsize=8)
        else:
            ax[j,i].set_xticklabels([])
        #plt.setp(xtickNames, rotation=45, fontsize=8)
        ax[j,i].set_ylim([0,1])
        if i>0:
            ax[j,i].set_yticklabels([])
        if j==0:
            ax[j,i].set_title(r'$k=({})$'.format(klabel[i]))
    ax[j,0].set_ylabel('N={}\n\nEstimated Model Probability'.format(int(nn/2)))
plt.tight_layout()
if SAVEPLOTS:
    plt.savefig('vs_variability_violin.pdf')
else:
    plt.show()
