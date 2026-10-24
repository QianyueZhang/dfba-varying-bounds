"""Render neural learning/control results with the direct KKT check."""
from control_experiments import *
R=O/'revision_validation'
def main():
 n=json.loads((R/'neural_extension.json').read_text());d=json.loads((R/'direct_polished.json').read_text())
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});fig,ax=plt.subplots(1,2,figsize=(10.5,4.1));sel=n['selected'];sizes=[r['training_size'] for r in sel]
 ax[0].plot(sizes,[r['validation']['log_ratio_rmse'] for r in sel],'o-',label='Validation (150 controls)');ax[0].plot(sizes,[r['fresh_audit']['log_ratio_rmse'] for r in sel],'s-',label='Independent audit (200 controls)');ax[0].set(xlabel='Training trajectories',ylabel='Log-ratio RMSE (dimensionless)');ax[0].legend(fontsize=8)
 for r in n['hybrid']:ax[1].semilogy([0,1],[100*np.sqrt(r['before']['objective']),100*np.sqrt(r['after']['objective'])],'o-',label=str(r['seed']))
 ax[1].scatter([1.05],[d['absolute_ratio_error_percent']],marker='*',s=160,color='black',zorder=5);ax[1].annotate('Direct KKT check',(1.05,d['absolute_ratio_error_percent']),xytext=(.1,.003),textcoords='data',arrowprops={'arrowstyle':'->'},fontsize=8)
 ax[1].set(xticks=[0,1],xticklabels=['Surrogate proposal','Mechanistic correction'],ylabel='Absolute terminal ratio error (%)',xlim=(-.08,1.15));ax[1].legend(title='Search seed',fontsize=7,loc='upper right');plt.tight_layout();plt.savefig(F/'revision_neural_extension.png',dpi=240);plt.savefig(F/'revision_neural_extension.pdf');plt.close()

if __name__=='__main__':main()
