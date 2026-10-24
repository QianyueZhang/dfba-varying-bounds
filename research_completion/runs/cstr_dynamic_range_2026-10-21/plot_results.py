"""Readable colour-first scientific figures from fixed saved trajectories."""
from pathlib import Path
import json,hashlib
import numpy as np
import benchmark as b
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import loadmat
OUT=Path(__file__).resolve().parent;ROOT=b.ROOT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','svg.hashsalt':'cstr-three-diel-cycles'})
D={n:np.load(OUT/('nominal_'+n+'.npz')) for n in ['light','dark']};ids={}
for name,file in [('SC','iND750.mat'),('RC','iRC1080.mat')]:
 raw=next(v for k,v in loadmat(ROOT/'data/models/cstr'/file,simplify_cells=True).items()if not k.startswith('__'));ids[name]={str(x):i for i,x in enumerate(raw['rxns'])}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[];figmeta=[]
def register(ax,t,y,case,variable,source):
 line=ax.lines[-1];checks.append({'case':case,'variable':variable,'source':source,'matches_saved_values':bool(np.array_equal(line.get_xdata(),t) and np.array_equal(line.get_ydata(),y))})
def shade(ax):
 for d in range(3):
  ax.axvspan(24*d,24*d+5,color='.88',alpha=.6,zorder=-5);ax.axvspan(24*d+19,24*d+24,color='.88',alpha=.6,zorder=-5)
 ax.set_xlim(0,72);ax.set_xticks([0,24,48,72]);ax.grid(axis='y',color='.85',lw=.5);ax.tick_params(direction='out')
def annotate(ax,letter):ax.text(.005,1.015,'('+letter+')',transform=ax.transAxes,fontweight='bold',va='bottom')
def export(fig,stem,grey,dimensions):
 suffix='_greyscale'if grey else '';fig.canvas.draw();child=sum(len(a.child_axes) for a in fig.axes)
 assert child==0
 for ext in ['png','pdf','svg']:
  options={'metadata':{'Date':None}}if ext=='svg'else {'metadata':{'CreationDate':None,'ModDate':None}}if ext=='pdf'else {}
  fig.savefig(OUT/(stem+suffix+'.'+ext),dpi=320,**options)
 figmeta.append({'stem':stem+suffix,'dimensions_mm':dimensions,'axes':len(fig.axes),'insets':child,'font_pt':8.5,'png':str((OUT/(stem+suffix+'.png')).relative_to(ROOT))})
 plt.close(fig)
for grey in [False,True]:
 blue,orange,green=('#111111','#777777','#444444')if grey else('#176b9a','#c15a18','#29805c')
 # All states retain a common time coordinate. Only carbon substrates use
 # separate, explicitly coloured concentration axes; no data scaling is used.
 fig,axes=plt.subplots(6,1,figsize=(155/25.4,180/25.4),sharex=True,gridspec_kw={'height_ratios':[.85,1.08,1.05,1,1,1]},layout='constrained')
 for i,ax in enumerate(axes):shade(ax);annotate(ax,chr(97+i))
 t=D['light']['time_h'];axes[0].plot(t,[b.IRRADIANCE*b.light_fraction(q)for q in t],color=blue);axes[0].set_ylabel('Light\n(μmol m$^{-2}$ s$^{-1}$)');axes[0].set_yticks([0,60]);axes[0].set_ylim(-3,100)
 styles=[('light','-'),('dark','--')]
 for name,style in styles:
  dd=D[name];t=dd['time_h']
  for col,label,colr in [(1,r'$\mathit{S.\ cerevisiae}$',blue),(2,r'$\mathit{C.\ reinhardtii}$',orange)]:
   axes[1].plot(t,dd['states'][:,col],style,color=colr,lw=1.25,label=label+(' — light'if name=='light'else' — dark'));register(axes[1],t,dd['states'][:,col],name,label,'states')
 axes[1].set_ylabel('Biomass\n(gDW L$^{-1}$)');axes[1].legend(loc='upper left',ncol=2,fontsize=6.4,framealpha=.85)
 twin=axes[2].twinx();twin.spines['right'].set_visible(True)
 for name,style in styles:
  dd=D[name];t=dd['time_h']
  axes[2].plot(t,dd['states'][:,3],style,color=blue,lw=1.25);register(axes[2],t,dd['states'][:,3],name,'glucose','states[:,3]')
  twin.plot(t,dd['states'][:,4],style,color=orange,lw=1.25);register(twin,t,dd['states'][:,4],name,'acetate','states[:,4]')
 axes[2].set_ylabel('Glucose\n(mmol L$^{-1}$)',color=blue);axes[2].tick_params(axis='y',colors=blue);twin.set_ylabel('Acetate\n(mmol L$^{-1}$)',color=orange);twin.tick_params(axis='y',colors=orange)
 for axis,col,label in [(axes[3],5,'Oxygen'),(axes[4],6,'Carbon dioxide'),(axes[5],7,'Ethanol')]:
  for name,style in styles:
   dd=D[name];axis.plot(dd['time_h'],dd['states'][:,col],style,color=blue if col!=6 else orange,lw=1.25);register(axis,dd['time_h'],dd['states'][:,col],name,label,'states')
  axis.set_ylabel(label+'\n(mmol L$^{-1}$)')
  if col in [5,6]:axis.axhline(b.O_SAT if col==5 else b.C_SAT,color='.45',ls=':',lw=.9)
 axes[0].legend(handles=[Line2D([],[],color='0.2',ls='-',label='Light cycle'),Line2D([],[],color='0.2',ls='--',label='Dark reference')],ncol=2,loc='upper right',fontsize=6.5)
 axes[-1].set_xlabel('Time (h)');export(fig,'cstr_states',grey,[155,180])
 # Yeast: three legible stacked panels.
 fig,axes=plt.subplots(3,1,figsize=(155/25.4,120/25.4),sharex=True,layout='constrained');sp=ids['SC']
 for i,ax in enumerate(axes):shade(ax);annotate(ax,chr(97+i))
 for name,style,colr in [('light','-',blue),('dark','--',orange)]:
  dd=D[name];t=dd['time_h'];v=dd['sc_flux'];l=dd['sc_lower'];u=dd['sc_upper'];g=v[:,sp['BIOMASS_SC4_bal']]
  axes[0].plot(t,g,style,color=colr,lw=1.3,label='Light cycle'if name=='light'else'Dark reference');register(axes[0],t,g,name,'growth','sc_flux')
  for a,key,label in [(1,'EX_glc__D_e','Glucose'),(2,'EX_o2_e','Oxygen')]:
   axes[a].plot(t,v[:,sp[key]],style,color=colr,lw=1.3);register(axes[a],t,v[:,sp[key]],name,key,'sc_flux')
   axes[a].plot(t,l[:,sp[key]],':',color=colr,lw=1.0);register(axes[a],t,l[:,sp[key]],name,key+'_lower','sc_lower');axes[a].set_ylabel(label+' flux\n(mmol gDW$^{-1}$ h$^{-1}$)')
   if a==1:axes[a].plot(t,u[:,sp[key]],':',color=colr,lw=1.0);register(axes[a],t,u[:,sp[key]],name,key+'_upper','sc_upper')
 axes[0].axhline(b.D,color='.45',ls=':',label='Dilution');axes[0].set_ylabel('Specific growth\n(h$^{-1}$)');axes[0].set_ylim(.0075,.0205);axes[0].legend(ncol=3,loc='upper right',fontsize=7);axes[1].set_ylim(-.31,.01);axes[1].set_yticks([-.3,-.2,-.1,0]);axes[-1].set_xlabel('Time (h)');export(fig,'cstr_sc_fluxes',grey,[155,120])
 # Alga: a full-width growth row and three paired rows. Separate ordinary
 # oxygen and uptake-bound axes expose both scales without an inset.
 fig=plt.figure(figsize=(155/25.4,180/25.4),layout='constrained');grid=fig.add_gridspec(4,2,height_ratios=[.85,1,1,1]);axes=[fig.add_subplot(grid[0,:])]
 for row in [1,2,3]:
  for col in [0,1]:axes.append(fig.add_subplot(grid[row,col],sharex=axes[0]))
 for i,ax in enumerate(axes):shade(ax);annotate(ax,chr(97+i))
 sp=ids['RC'];labels=['Specific growth\n(h$^{-1}$)','Acetate exchange','Carbon dioxide exchange','Oxygen exchange','Oxygen uptake bound','Oxygen components','Solar photon flux']
 for i,ax in enumerate(axes):
  ax.set_ylabel(labels[i]+(''if i==0 else '\n(mmol gDW$^{-1}$ h$^{-1}$)'))
  if i<5:ax.tick_params(labelbottom=False)
  else:ax.set_xlabel('Time (h)')
 for name,style,colr in [('light','-',blue),('dark','--',orange)]:
  dd=D[name];t=dd['time_h'];v=dd['rc_flux'];l=dd['rc_lower'];u=dd['rc_upper'];axes[0].plot(t,v[:,sp['BIOMASS_Chlamy_auto']],style,color=colr,lw=1.3,label='Light cycle'if name=='light'else'Dark reference');register(axes[0],t,v[:,sp['BIOMASS_Chlamy_auto']],name,'growth','rc_flux')
  for panel,key in [(1,'EX_ac_e'),(2,'EX_co2_e'),(3,'EX_o2_e')]:
   axes[panel].plot(t,v[:,sp[key]],style,color=colr,lw=1.3);register(axes[panel],t,v[:,sp[key]],name,key,'rc_flux')
   if panel!=3:axes[panel].plot(t,l[:,sp[key]],':',color=colr,lw=.9);register(axes[panel],t,l[:,sp[key]],name,key+'_lower','rc_lower')
   if panel==1:axes[panel].plot(t,u[:,sp[key]],':',color=colr,lw=.9);register(axes[panel],t,u[:,sp[key]],name,key+'_upper','rc_upper')
  axes[4].plot(t,l[:,sp['EX_o2_e']],':'if name=='light'else'-.',color=colr,lw=1.3);register(axes[4],t,l[:,sp['EX_o2_e']],name,'oxygen_lower','rc_lower')
  net=v[:,sp['EX_o2_e']]+v[:,sp['DM_o2D_u']];photo=v[:,sp['DM_o2D_u']]
  axes[5].plot(t,net,style,color=colr,lw=1.3);register(axes[5],t,net,name,'net_oxygen','sum of ordinary and photo flux')
  axes[5].plot(t,photo,':'if name=='light'else'-.',color=colr,lw=1.0);register(axes[5],t,photo,name,'photosynthetic_oxygen','rc_flux')
  idx=sp['PRISM_solar_litho'];axes[6].plot(t,v[:,idx],style,color=colr,lw=1.3);register(axes[6],t,v[:,idx],name,'photons','rc_flux');axes[6].plot(t,u[:,idx],':',color=colr,lw=.9);register(axes[6],t,u[:,idx],name,'photon_upper','rc_upper')
 axes[0].axhline(b.D,color='.45',ls=':',label='Dilution');axes[0].set_ylim(.0048,.024);axes[0].legend(ncol=3,loc='upper right',fontsize=7)
 axes[1].set_ylim(-.31,.01);axes[1].set_yticks([-.3,-.2,-.1,0]);axes[1].ticklabel_format(axis='y',useOffset=False,style='plain')
 axes[5].axhline(0,color='.55',lw=.6);axes[5].set_ylim(-.42,1.03);axes[5].legend(handles=[Line2D([],[],color='.3',ls='-',label='Net'),Line2D([],[],color='.3',ls=':',label='Photosynthetic')],loc='upper left',ncol=2,fontsize=6.2,handlelength=1.8,columnspacing=.9)
 export(fig,'cstr_rc_fluxes',grey,[155,180])
report={'figures':figmeta,'curve_checks':checks,'all_curve_checks_passed':all(c['matches_saved_values']for c in checks),'input_hashes':{str((OUT/('nominal_'+n+'.npz')).relative_to(ROOT)):sha(OUT/('nominal_'+n+'.npz'))for n in ['light','dark']},'outputs':{str(p.relative_to(ROOT)):sha(p)for p in OUT.glob('cstr_*')if p.suffix in ['.png','.pdf','.svg']},'scope':'Same unscaled numerical arrays; labelled dual y axes only for glucose/acetate. No inset axes. Specified canvas dimensions preserve text size when inserted at155mm width.'}
(OUT/'PLOT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'figures':len(figmeta),'curves':len(checks),'passed':report['all_curve_checks_passed']},indent=2))
