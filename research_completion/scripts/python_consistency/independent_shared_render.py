"""Render the three validated full-horizon resource cases from saved arrays only."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research_completion/runs/python_consistency/independent_shared/supported'
OUT=BASE/'figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'axes.titlesize':11.5,'axes.labelsize':11,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
C=['#0072BD','#D95319'];styles=['-','--'];markers=['o','s'];items=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(key):
 d=BASE/key;a=json.loads((d/'audit.json').read_text());check=json.loads((d/'differentiated_verification.json').read_text())
 assert a['full_horizon_completed']and a['validation']['passed']and check['adoption_ready']
 data=np.load(d/'differentiated_refined.npz')
 return {'key':key,'d':d,'case':a['case_definition'],'t':data['time_h'],'y':data['concentrations'],'v':data['flux']}
def setup(ax,d,letter,title,unit):
 ax.set_title(f'({letter}) {title}',loc='left',pad=9);ax.set_xlabel('Time (h)');ax.set_ylabel(unit)
 ax.set_xlim(0,d['t'][-1]);ax.grid(color='.88',lw=.55);ax.set_axisbelow(True)
def pair(ax,d,values,labels=('Species 1','Species 2')):
 for i in range(2):
  ax.plot(d['t'],values[:,i],color=C[i],ls=styles[i],marker=markers[i],markevery=45+5*i,ms=3.5,mfc='white',mew=.8,lw=1.7,label=labels[i])
 ax.legend(frameon=False)
def biomass(ax,d,k):setup(ax,d,k,'Biomass concentrations','Biomass (gDW L$^{-1}$)');pair(ax,d,d['y'][:,1:3]);ax.set_ylim(bottom=0)
def amounts(ax,d,k):setup(ax,d,k,'Biomass amounts','Biomass (gDW)');pair(ax,d,d['y'][:,1:3]*d['y'][:,0:1]);ax.set_ylim(bottom=0)
def volume(ax,d,k):setup(ax,d,k,'Liquid volume','Volume (L)');ax.plot(d['t'],d['y'][:,0],color='.15',lw=1.7);ax.set_ylim(bottom=0)
def substrate(ax,d,k):
 setup(ax,d,k,'Shared glucose'if d['case']['shared']else'Separate glucose pools','Glucose (g L$^{-1}$)')
 if d['case']['shared']:ax.plot(d['t'],d['y'][:,3],color='.15',lw=1.7)
 else:pair(ax,d,d['y'][:,3:5],('Pool 1','Pool 2'))
 ax.set_ylim(bottom=0)
def bounds(ax,d,i,k):
 G=d['y'][:,3+(0 if d['case']['shared']else i)];lower=-d['case']['vmax'][i]*G/(1+G);upper=np.full_like(G,1.0001)
 setup(ax,d,k,f'Species {i+1}: glucose exchange','Flux (mmol gDW$^{-1}$ h$^{-1}$)')
 ax.fill_between(d['t'],lower,upper,color='.93',zorder=0)
 ax.plot(d['t'],lower,color='.35',ls='--',lw=1.05);ax.plot(d['t'],upper,color='.35',ls=':',lw=1.05)
 ax.plot(d['t'],d['v'][:,i,27],color=C[i],ls='-',marker=markers[i],markevery=45,ms=3.2,mfc='white',mew=.7,lw=1.6)
 ax.set_ylim(lower.min()-.4,1.5)
def bound_legend(fig):
 fig.legend(handles=[Line2D([0],[0],color='.2',marker='o',mfc='white',lw=1.5,label='Exchange flux'),Line2D([0],[0],color='.35',ls='--',label='Lower bound'),Line2D([0],[0],color='.35',ls=':',label='Upper bound'),Patch(facecolor='.93',label='Scalar interval')],loc='outside lower center',ncol=4,frameon=False)
def save(fig,num,d,caption):
 if num in [16,17,18]:caption+=' Exchange and lower-bound curves nearly coincide at the displayed scale.'
 stem=OUT/f'figure_{num:02d}_full_horizon';png=stem.with_suffix('.png');svg=stem.with_suffix('.svg');fig.savefig(png,dpi=260,bbox_inches='tight');fig.savefig(svg,bbox_inches='tight');plt.close(fig)
 grey=OUT/f'figure_{num:02d}_greyscale.png';Image.open(png).convert('L').save(grey)
 items.append({'old_figure':num,'case':d['key'],'asset':str(png.relative_to(ROOT)),'svg':str(svg.relative_to(ROOT)),'greyscale':str(grey.relative_to(ROOT)),'caption':caption,'source':str((d['d']/'differentiated_refined.npz').relative_to(ROOT)),'source_sha256':sha(d['d']/'differentiated_refined.npz'),'sha256':sha(png),'horizon_h':float(d['t'][-1]),'data_policy':'Complete saved 501-point differentiated-KKT arrays. No smoothing, clipping, digitisation or point deletion.'})
d=load('independent_identical')
f,ax=plt.subplots(1,2,figsize=(7.8,3.35),layout='constrained');biomass(ax[0],d,'a');substrate(ax[1],d,'b')
save(f,14,d,'Independent glucose pools with identical uptake capacities over 5 h: (a) biomass concentrations and (b) glucose concentrations. Equal initial conditions and parameters produce coincident trajectories. Each population uses only its own glucose pool; both experience the same volume and dilution law.')
f,ax=plt.subplots(1,2,figsize=(7.8,3.35),layout='constrained');volume(ax[0],d,'a');amounts(ax[1],d,'b')
save(f,15,d,'Identical independent-resource case: (a) liquid volume and (b) biomass amounts. The inlet rate is 1 L h⁻¹ and the initial volume is 5 L. The increase in biomass amount is distinguished from its concentration in the expanding liquid volume.')
f,ax=plt.subplots(1,2,figsize=(7.8,3.6),layout='constrained');bounds(ax[0],d,0,'a');bounds(ax[1],d,1,'b');bound_legend(f)
save(f,16,d,'Glucose exchange in the identical independent-resource case: (a) Species 1 and (b) Species 2. Negative exchange denotes uptake. The lower bound is −vmax,i Gi/(1+Gi), with Gi in g L⁻¹; the upper bound is 1.0001 mmol gDW⁻¹ h⁻¹. Pale shading denotes the scalar interval; all other intracellular constraints are also enforced.')
for num,key in [(17,'independent_different'),(18,'shared_competition')]:
 d=load(key);f,ax=plt.subplots(3,2,figsize=(7.8,8.6),layout='constrained')
 volume(ax[0,0],d,'a');amounts(ax[0,1],d,'b');biomass(ax[1,0],d,'c');substrate(ax[1,1],d,'d');bounds(ax[2,0],d,0,'e');bounds(ax[2,1],d,1,'f');bound_legend(f)
 title='Independent glucose pools with unequal uptake capacities (Species 1: 5; Species 2: 10 mmol gDW⁻¹ h⁻¹), over 10 h.'if num==17 else'Shared glucose with unequal uptake capacities (Species 1: 10; Species 2: 5 mmol gDW⁻¹ h⁻¹), over 5 h.'
 save(f,num,d,title+' Panels show (a) liquid volume, (b) biomass amounts, (c) biomass concentrations, (d) glucose, and (e,f) glucose exchange with its scalar bounds. The inlet rate is 1 L h⁻¹. Negative exchange denotes uptake; pale intervals do not represent the complete metabolic feasible set.')
# Contact sheets are internal visual checks, never thesis figures.
for grey in [False,True]:
 paths=[ROOT/(x['greyscale']if grey else x['asset'])for x in items];thumbs=[]
 for p in paths:
  im=Image.open(p).convert('RGB');im.thumbnail((960,1050));thumbs.append(im)
 sheet=Image.new('RGB',(1920,3*1090),'white')
 for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*960,(i//2)*1090))
 sheet.save(OUT/('contact_greyscale.png'if grey else'contact_colour.png'))
manifest={'purpose':'Replacement Figures 14–18 for the same three resource case studies under the explicitly specified substrate-supported inlet rate. No extra displayed cases.','figures':items,'adoption_ready':True,'source_preservation':'Original-feed diagnostics and all original materials retained in records only.','script_sha256':sha(Path(__file__))}
(BASE/'adoption_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(manifest,indent=2,ensure_ascii=False))
