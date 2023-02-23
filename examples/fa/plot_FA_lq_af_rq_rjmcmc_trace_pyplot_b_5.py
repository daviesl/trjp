import shelve
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os.path
import sys
from scipy.special import logsumexp

SAVEPLOTS = True

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def logmean(log_x):
    n = log_x.shape[0]
    print("logmean n=",n)
    return logsumexp(log_x) - np.log(n)

#chains = []
if len(sys.argv) > 1:
    pfx = sys.argv[1] 
else:
    pfx = './fa_traces_200k/'
m1problist = {}
proptypes=['lw','af','rq']
for pt in proptypes:
    m1problist[pt]=[]
for pt in proptypes:
    count = 0
    for r in range(100):
        #ms= shelve.open('{}RJMCMC_RobustBlockVS_mp_2block_saturated_shelve_run{}.out'.format(pfx,r)) 
        if not os.path.isfile('{}results_FA_theta_{}_{}_16000_0b.npy'.format(pfx,pt,r)):
            print("skipping run {} for pt {}".format(r,pt))
            continue
        if count >= 5:
            print("limit reached")
            continue
        count += 1
        theta = np.load('{}results_FA_theta_{}_{}_16000_0b.npy'.format(pfx,pt,r))
        #ptheta = np.load('{}results_FA_ptheta_{}_{}_16000_0b.npy'.format(pfx,pt,r))
        #ar = np.load('{}results_FA_ar_{}_{}_16000_0b.npy'.format(pfx,pt,r))
        # get model keys per step
        # systematic proposal, so every other prop is a jump
        k = theta[:,15][1::2]
        #pk = ptheta[:,15][1::2]
        #ar_jump = ar[1::2]
        # empirical
        l = (k==1).cumsum()/np.arange(1,k.shape[0]+1)
        m1problist[pt].append(l)

f, ax=plt.subplots(nrows=1,ncols=1,figsize=(5,3))
cols = {'lw':'blue','af':'darkorange','rq':'green'}
labels = {'lw':'Lopes & West','af':'Affine TRJ','rq':'RQMA-NF TRJ'}
for pt in proptypes:
    stack = np.column_stack(m1problist[pt]).T
    steps = np.arange(stack.shape[1])
    if False:
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.plot(steps,mean,color=cols[pt],linewidth=0.5,label=pt)
        ax.fill_between(steps,mean-std,mean+std,alpha=0.3,facecolor=cols[pt])
    else:
        ax.plot(steps,stack.T,color=cols[pt],linewidth=1,alpha=0.5,label=labels[pt])
    #df = pd.DataFrame(np.vstack(m1problist[pt]),columns=['chain','step','prob'])
    #sns.lineplot(data=df, x="step", y="prob")
#plt.xscale('log')
ax.hlines(0.882,xmin=-5000,xmax=105000, color='black',linewidth=1, linestyle='solid', label='Ground Truth')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(),loc='lower right')
plt.legend(by_label.values(), by_label.keys(),loc='lower right',fancybox=True, frameon=True,framealpha=0.7)
plt.xlabel('RJMCMC Step')
plt.ylabel('2-Factor Model Probability Estimate')
plt.tight_layout()
if SAVEPLOTS:
    plt.savefig('fa_running_mp_trace.pdf')
else:
    plt.show()
