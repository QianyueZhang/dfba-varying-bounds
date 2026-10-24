"""Render main-text Edwards figures from validated differentiated KKT data."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'research_completion/runs/python_consistency/edwards'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.labelsize':10.5,'axes.titlesize':11,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.linewidth':.8})
COL={'bio':'#202020','D':'#0072B2','E':'#b75a00','bound':'#496b80'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 with Image.open(OUT/f'{name}.png') as im:im.convert('L').save(OUT/f'{name}_greyscale.png')
 plt.close(fig)

def state_plot(kind,number):
 p=OUT/f'differentiated_{kind}_refined.npz';d=np.load(p);t=d['time'];s=d['state'];v=d['flux'];cap=d['glucose_capacity']
 fig,axs=plt.subplots(2,2,figsize=(8.0,6.6),sharex=True,layout='constrained')
 a,b,c,e=axs.flat
 for idx,label,key,style,marker in [(0,'Biomass','bio','-',None),(2,'Product D','D','--','s'),(3,'Product E','E','-.','^')]:a.plot(t,s[idx],label=label,color=COL[key],ls=style,lw=1.6,marker=marker,markevery=50,markersize=3.2,mfc='white',mew=.9)
 a.set(title='A   Biomass and products',ylabel='Concentration (g L$^{-1}$)',ylim=(0,4.2));a.legend(loc='upper left',frameon=False)
 b.plot(t,s[1],color=COL['bio'],lw=1.6);b.set(title='B   Glucose',ylabel='Concentration (g L$^{-1}$)',ylim=(0,10.5))
 for ax,flux,ub,label in [(c,v[:,0],cap,'C   Glucose uptake'),(e,v[:,9],np.full(t.shape,15.),'D   Oxygen uptake')]:
  # A prescribed glucose flux is an equality, so no interval is shaded.
  prescribed=kind=='prescribed' and ax is c
  if not prescribed:ax.fill_between(t,0,ub,color='#a6adb3',alpha=.13,zorder=0,label='Scalar bound interval')
  ax.plot(t,ub,color=COL['bound'],ls='--',lw=1.15,label='Prescribed rate' if prescribed else 'Upper bound',zorder=2)
  if not prescribed:ax.axhline(0,color='#666666',ls=':',lw=1.05,label='Lower bound',zorder=2)
  ax.plot(t,flux,color='#111111',lw=1.65,marker='o',markevery=50,markersize=3,mfc='white',mew=.8,label='Uptake flux',zorder=3)
  ax.set(title=label,ylabel='Flux (mmol gDW$^{-1}$ h$^{-1}$)',xlabel='Time (h)',ylim=(-.35,10.5 if ax is c else 16.0))
  legendloc='upper right' if prescribed else 'lower left'
  ax.legend(loc=legendloc,frameon=False,fontsize=8)
 for ax in axs.flat:ax.set_xlim(0,7);ax.set_xticks(range(8));ax.grid(axis='y',color='#dddddd',lw=.45,alpha=.6)
 save(fig,f'figure_{number:02}_validated_{kind}')
 np.savetxt(OUT/f'figure_{number:02}_data.csv',np.c_[t,s.T,v,cap],delimiter=',',header='time_h,biomass_g_L,glucose_g_L,product_D_g_L,product_E_g_L,'+','.join('v'+str(i+1) for i in range(13))+',glucose_capacity',comments='')
 return {'figure':number,'png':str(OUT/f'figure_{number:02}_validated_{kind}.png'),'data_npz':str(p),'data_sha256':sha(p),'model':kind,'terminal_X_G_D_E':s[:,-1].tolist(),'shading':'The scalar uptake interval only; joint intracellular feasibility additionally requires Nv=0. No interval is shaded for prescribed glucose.'}

def phase_plot():
 d=np.load(OUT/'phase_grid.npz');g=d['glucose'];o=d['oxygen'];z=d['optimal_biomass_D_E_ATP_dissipation'][:,:,0]
 # Pale monotonic lightness plus numerical contours preserves grayscale use.
 colors=plt.get_cmap('Blues')(np.linspace(.05,.45,256));cmap=ListedColormap(colors)
 fig,ax=plt.subplots(figsize=(7.8,6.4),layout='constrained');ax.set_facecolor('#f1f1f1')
 mesh=ax.pcolormesh(g,o,z,cmap=cmap,shading='auto',rasterized=True,vmin=0,vmax=5)
 cb=fig.colorbar(mesh,ax=ax,pad=.025,aspect=32);cb.set_label('LP-optimal biomass flux (mmol gDW$^{-1}$ h$^{-1}$)')
 levels=[1,2,3,4];cs=ax.contour(g,o,z,levels=levels,colors='#626262',linewidths=.55);ax.clabel(cs,fmt='%g',fontsize=8,inline=True)
 for slope in [1,2,3,13]:
  xx=np.array([0,min(10,20/slope)]);ax.plot(xx,slope*xx,color='#747474',ls=':',lw=.85)
 xx=np.array([0,5]);ax.plot(xx,4*xx,color='#9b6e00',ls='--',lw=1.6,label='Maximum-yield ridge (LO)')
 summary=[]
 for kind,col,ls,lw,marker in [('capacity','#0072B2','-',2.2,'o'),('prescribed','#a63716','-.',1.5,'s')]:
  src=OUT/f'differentiated_{kind}_refined.npz';data=np.load(src);v=data['flux'];lab='Glucose capacity' if kind=='capacity' else 'Prescribed glucose'
  ax.plot(v[:,0],v[:,9],color=col,ls=ls,lw=lw,marker=marker,markevery=50,markersize=3.6,mfc='white',mew=.9,label=lab,path_effects=[pe.Stroke(linewidth=lw+1.6,foreground='white'),pe.Normal()])
  ax.scatter(v[0,0],v[0,9],marker=marker,s=40,color=col,edgecolors='white',zorder=5)
  ax.scatter(v[-1,0],v[-1,9],marker='x',s=35,color=col,zorder=5)
  textpos=(6.9,18.6) if kind=='capacity' else (8.9,17.2)
  ax.annotate(lab+' start',xy=(v[0,0],v[0,9]),xytext=textpos,fontsize=8.5,ha='center',arrowprops={'arrowstyle':'->','lw':.8,'color':col})
  summary.append({'kind':kind,'source_npz':str(src),'source_sha256':sha(src),'trajectory_axes':'actual glucose and oxygen flux; scalar background is LP-optimal biomass flux at those exact uptake values'})
 for label,x,y in [('P1',2.0,17),('P2',4.5,16.5),('P3',6.0,16.5),('P4',8.9,12.6)]:ax.text(x,y,label,fontsize=11,weight='bold',bbox={'facecolor':'white','alpha':.75,'edgecolor':'none','pad':1.8})
 ax.text(.45,14,'Infeasible input pairs',rotation=75,fontsize=8,color='#666666');ax.text(6.7,3.4,'Infeasible input pairs',fontsize=8.5,color='#666666')
 ax.set(xlim=(0,10.5),ylim=(0,20),xlabel='Glucose uptake (mmol gDW$^{-1}$ h$^{-1}$)',ylabel='Oxygen uptake (mmol gDW$^{-1}$ h$^{-1}$)')
 ax.legend(loc='lower right',framealpha=.94,fontsize=9)
 save(fig,'figure_08_validated_phase_plane')
 return {'figure':8,'png':str(OUT/'figure_08_validated_phase_plane.png'),'phase_grid_sha256':sha(OUT/'phase_grid.npz'),'paths':summary,'definition':'At each pair, both uptake fluxes are fixed; all other fluxes have bounds0..100, Nv=0, objective maximum biomass. Grey cells denote infeasible fixed pairs, not zero-growth solutions. The trajectory policies differ only in the glucose constraint; oxygen is capacity-bounded in both.'}

if __name__=='__main__':
 records=[state_plot('capacity',7),phase_plot(),state_plot('prescribed',9)]
 for r in records:r['png_sha256']=sha(Path(r['png']))
 (OUT/'adoption_figure_manifest.json').write_text(json.dumps({'figures':records,'artwork':'New Python artwork drawn from numerical arrays, no screenshot tracing or borrowed graphic background. Network and phenotype-phase-plane concept remain attributed to Edwards et al.2002.','colour_and_monochrome':'Distinct solid/dashed/dash-dot lines and markers; pale scalar-interval shading; corresponding greyscale PNGs produced for visual check.'},indent=2)+'\n')
 print(json.dumps(records,indent=2))
