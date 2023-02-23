import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import sys
SAVEPLOTS=True
plt.style.use('seaborn-whitegrid')
f,ax = plt.subplots(nrows=1,ncols=1,figsize=(5,3))
proptypes = ['affine','nf','perfect']
pt_titles={'affine':'Affine TRJ','nf':'RQMA TRJ','perfect':'Perfect TRJ'}
cols={'affine':'darkorange','nf':'green','perfect':'darkmagenta'}
alphas={'affine':1,'nf':0.9,'perfect':0.7}
fpfx={'affine':'af','nf':'rqma','perfect':'perfect'}
steps = np.arange(1,10001)
if len(sys.argv) > 1:
    pfx = sys.argv[1]
else:
    pfx = 'sas_rjmcmc_chains/'
for i,pt in enumerate(proptypes):
    for run_no in range(5):
        pt_m1theta = np.load('./{}{}_sas_rjmcmc_theta_run{}.npy'.format(pfx,fpfx[pt],run_no))
        trace=pt_m1theta[:,1].cumsum()/steps
        ax.plot(steps,trace,color=cols[pt],linewidth=0.5,label=pt_titles[pt],alpha=alphas[pt])
#plt.suptitle('Sinh Arcsinh 1D 2D Jump 1->2 PLMA Transform')
ax.hlines(0.75,xmin=-500,xmax=10500,color='black',linewidth=1,linestyle='solid',label='Ground Truth')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),loc='lower right',fancybox=True, frameon=True,framealpha=0.7)
#plt.legend(loc="lower right")
ax.set_ylabel(r'$k=2$ SAS Model Probability Estimate')
ax.set_xlabel('RJMCMC Step')
ax.set_ylim([0.55,0.85])
plt.tight_layout()
if SAVEPLOTS:
    plt.savefig('sas_running_mp_trace.pdf')
else:
    plt.show()

