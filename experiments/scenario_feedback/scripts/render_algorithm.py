from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
r=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(10,5));ax.set(xlim=(0,10),ylim=(0,5));ax.axis('off')
def box(x,y,w,h,text,color='#edf3f6'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=color,ec='#557080',lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b,label=None,rad=0):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color='#506273',connectionstyle=f'arc3,rad={rad}'))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',va='bottom',fontsize=9)
box(.1,3.9,2.2,.75,'Observed past intervals\nBounded affinity fit')
box(3.0,3.9,3.4,.75,'Finite scenarios + common policy\nFreeze reference and sensitivity basis')
box(7.2,3.9,2.5,.75,'Local affine proposal\nTrust-region linear program')
box(7.2,2.2,2.5,.8,'Nonlinear scenario check\nComposition and\nincumbent quality','#e5f1ee')
box(3.1,2.2,2.9,.8,'Recenter and contract\nAt most two corrections','#fff4df')
box(.1,2.2,2.2,.8,'Matched direct solve\nSame model\nand objective','#f4e9e9')
box(3.2,.3,3.6,.85,'Execute first feed with volume guard\nObserve and replan after 2 h','#e5f1ee')
arrow((2.3,4.27),(3.,4.27));arrow((6.4,4.27),(7.2,4.27));arrow((8.45,3.9),(8.45,3.0))
arrow((7.2,2.6),(6.,2.6),'Reject');arrow((4.5,3.),(7.2,4.1),rad=-.12)
arrow((3.1,2.6),(2.3,2.6),'Limit');arrow((1.2,2.2),(3.2,.72),'Fallback candidate',rad=.08)
arrow((8.45,2.2),(6.8,.72),'Accept',rad=-.08)
ax.text(.2,.4,'Whole-solver exception:\nzero-feed physical backup',fontsize=9,color='#8b3f45')
ax.text(5.,0.,'Finite-scenario acceptance is model-relative; terminal process feasibility is evaluated separately.',ha='center',fontsize=9,color='#485460')
fig.savefig(r/'figures/algorithm.pdf',bbox_inches='tight');fig.savefig(r/'figures/algorithm.png',dpi=180,bbox_inches='tight')
