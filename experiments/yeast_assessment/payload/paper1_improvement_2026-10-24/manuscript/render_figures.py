"""Regenerate original manuscript figures from packaged numerical records."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D, OUT = HERE / "data", HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, ORANGE, GREEN, GRAY = "#1769aa", "#c65f13", "#34856b", "#636b73"
def csv(name):
    return np.genfromtxt(D / name, delimiter=",", names=True)
def finish(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(OUT / (name + ".png"), dpi=180)
    plt.close(fig)
def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", fontsize=10, pad=8)
    ax.grid(alpha=.15)

a = csv("analytic_reference.csv")
s = csv("analytic_sensitivities.csv")
val = json.loads((D / "analytic_validation.json").read_text())
fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.2), constrained_layout=True)
t = a["time_h"]
ax[0,0].fill_between(t, a["lower"], a["upper"], color=BLUE, alpha=.10)
ax[0,0].plot(t, a["lower"], "--", color=GRAY, label="Bounds")
ax[0,0].plot(t, a["upper"], "--", color=GRAY)
ax[0,0].plot(t, a["analytic_v1"], color=BLUE, label=r"$v_1=v_2=q$")
ax[0,0].set(xlabel="Time (h)", ylabel="Dimensionless flux")
ax[0,0].legend(fontsize=8)
panel(ax[0,0], "a", "Simultaneous bound motion")
ax[0,1].plot(t, s["dq_dlower"]*s["lower_rate"], color=ORANGE, label="Lower-bound contribution")
ax[0,1].plot(t, s["dq_dupper"]*s["upper_rate"], color=BLUE, label="Upper-bound contribution")
ax[0,1].plot(t, s["analytic_rate"], "--", color=GRAY, label="Total")
ax[0,1].set(xlabel="Time (h)", ylabel=r"Flux rate (h$^{-1}$)")
ax[0,1].legend(fontsize=7)
panel(ax[0,1], "b", "Separate chain-rule terms")
for key, color, label in [("complete",BLUE,"Complete equation"),("omit_lower",ORANGE,"Omit lower term"),("omit_upper",GRAY,"Omit upper term")]:
    arr=np.load(D/f"analytic_{key}.npz")
    ax[1,0].plot(arr["time_h"],np.max(np.abs(arr["state"][:,:2]-arr["reference"][:,:2]),axis=1),color=color,label=label)
ax[1,0].set(xlabel="Time (h)", ylabel="Maximum absolute flux error", yscale="symlog", ylim=(0,.35))
ax[1,0].set_yscale("symlog",linthresh=1e-13)
ax[1,0].legend(fontsize=7)
panel(ax[1,0], "c", "Deliberate missing-term diagnostics")
rows=val["barrier_refinement"]
ax[1,1].loglog([r["mu"] for r in rows],[r["maximum_LP_objective_gap"] for r in rows],"o-",color=BLUE,label="Maximum LP gap")
ax[1,1].loglog([r["mu"] for r in rows],[r["nominal_primal_dual_gap_bound"] for r in rows],"--",color=GRAY,label=r"$4\mu$ bound")
ax[1,1].set(xlabel=r"Barrier parameter $\mu$",ylabel="Dimensionless objective gap")
ax[1,1].legend(fontsize=8)
panel(ax[1,1], "d", "Barrier refinement; fixed bounds")
finish(fig,"figure1_analytic")

fig,ax=plt.subplots(2,2,figsize=(7.1,5.0),constrained_layout=True)
for fname,color,style,label in [("reduced_capacity.csv",BLUE,"-","Capacity uptake"),("reduced_prescribed.csv",ORANGE,"--","Prescribed uptake")]:
    b=csv(fname); t=b["time_h"]
    for aa,col in [(ax[0,0],"biomass_g_L"),(ax[0,1],"glucose_g_L"),(ax[1,0],"product_E_g_L")]:
        aa.plot(t,b[col],style,color=color,label=label)
    ax[1,1].plot(t,b["v1"],style,color=color,label=label)
    ax[1,1].plot(t,b["glucose_capacity"],":",color=color,alpha=.7)
for aa,letter,title,y in [(ax[0,0],"a","Biomass","Biomass (g L$^{-1}$)"),(ax[0,1],"b","Substrate A","Substrate (g L$^{-1}$)"),(ax[1,0],"c","Product E","Product E (g L$^{-1}$)"),(ax[1,1],"d","Uptake and available capacity","Uptake (mmol gDW$^{-1}$ h$^{-1}$)")]:
    aa.set(xlabel="Time (h)",ylabel=y);panel(aa,letter,title)
ax[0,0].legend(fontsize=8)
ax[1,1].text(.04,.08,"Dotted: kinetic capacity",transform=ax[1,1].transAxes,fontsize=8)
finish(fig,"figure2_reduced")

fig,ax=plt.subplots(2,2,figsize=(7.1,5.0),constrained_layout=True)
for case,color,label in [("independent_identical",GREEN,"Equal capacities; separate pools"),("independent_different",ORANGE,"Unequal capacities; separate pools"),("shared_competition",BLUE,"Unequal capacities; shared pool")]:
    b=csv(case+".csv"); t=b["time_h"]
    style={"independent_identical":"--","independent_different":"-","shared_competition":"-."}[case]
    marker="o" if case=="independent_identical" else None
    ax[0,0].plot(t,b["X1_gDW_L"],color=color,ls=style,marker=marker,markevery=50,ms=3,mfc="white",label=label,zorder=4 if marker else 2)
    ax[0,1].plot(t,b["X2_gDW_L"],color=color,ls=style,marker=marker,markevery=50,ms=3,mfc="white",label=label,zorder=4 if marker else 2)
shared=csv("shared_competition.csv")
for idx,color in [(1,BLUE),(2,ORANGE)]:
    ax[1,0].plot(shared["time_h"],shared[f"X{idx}_gDW_L"]*shared["V_L"],color=color,ls="-" if idx==1 else "--",label=f"Population {idx}")
ax[1,1].plot(shared["time_h"],shared["G1_g_L"],color=GREEN)
for aa,letter,title,y in [(ax[0,0],"a","Population 1","Biomass (gDW L$^{-1}$)"),(ax[0,1],"b","Population 2","Biomass (gDW L$^{-1}$)"),(ax[1,0],"c","Shared-pool biomass amounts","Biomass (gDW)"),(ax[1,1],"d","Shared glucose","Glucose (g L$^{-1}$)")]:
    aa.set(xlabel="Time (h)",ylabel=y);panel(aa,letter,title)
ax[0,1].legend(fontsize=6.5,loc="lower right")
ax[1,0].legend(fontsize=8)
finish(fig,"figure3_communities")

fig,ax=plt.subplots(1,2,figsize=(7.1,2.8),constrained_layout=True)
rr=val["finite_difference_sensitivity"]
for key,color,label in [("lower_sensitivity_max_abs_error",ORANGE,"Lower bound"),("upper_sensitivity_max_abs_error",BLUE,"Upper bound")]:
    ax[0].loglog([r["step"] for r in rr],[r[key] for r in rr],"o-",color=color,label=label)
ax[0].set(xlabel="Central-difference step",ylabel="Maximum sensitivity error")
panel(ax[0],"a","Independent derivative check");ax[0].legend(fontsize=8)
rr=val["tolerance_refinement"]
ax[1].loglog([r["rtol"] for r in rr],[r["max_abs_flux_error"] for r in rr],"o-",color=BLUE)
ax[1].set(xlabel="ODE relative tolerance",ylabel="Maximum absolute flux error")
panel(ax[1],"b",r"Integration refinement; $\mu=0.2$")
finish(fig,"figureS1_refinement")

# Original, source-generated TOC graphic at the JACS artwork dimensions.
fig = plt.figure(figsize=(3.25, 1.65), facecolor="white")
left=fig.add_axes([.09,.27,.26,.43])
left.fill_between(a["time_h"],a["lower"],a["upper"],color=BLUE,alpha=.12)
left.plot(a["time_h"],a["analytic_v1"],color=BLUE,lw=1)
left.plot(a["time_h"],a["lower"],color=GRAY,lw=.6,ls="--")
left.plot(a["time_h"],a["upper"],color=GRAY,lw=.6,ls="--")
left.tick_params(labelsize=5,length=2,pad=1)
left.set_xlabel("Time",fontsize=6,labelpad=1)
left.set_ylabel("Flux",fontsize=6,labelpad=1)
right=fig.add_axes([.72,.27,.23,.43])
for fname,col,ls in [("reduced_capacity.csv",BLUE,"-"),("reduced_prescribed.csv",ORANGE,"--")]:
    b=csv(fname);right.plot(b["time_h"],b["biomass_g_L"],color=col,lw=1,ls=ls)
right.tick_params(labelsize=5,length=2,pad=1)
right.set_xlabel("Time",fontsize=6,labelpad=1)
right.set_ylabel("")
fig.text(.18,.85,"Moving capacities",ha="center",fontsize=7)
fig.text(.52,.85,"Consistent KKT",ha="center",fontsize=7)
fig.text(.83,.85,"Dynamic response",ha="center",fontsize=7)
fig.text(.83,.75,"Biomass (g/L)",ha="center",fontsize=6)
fig.text(.52,.50,r"$D\dot{v}+A^T\dot{\lambda}$"+"\n"+r"$=D_L\dot{\ell}+D_U\dot{u}$",ha="center",va="center",fontsize=7)
fig.text(.37,.50,r"$\rightarrow$",ha="center",fontsize=10)
fig.text(.645,.50,r"$\rightarrow$",ha="center",fontsize=10)
fig.text(.52,.06,"Analytic and direct-solution verification",ha="center",fontsize=7,color=GRAY)
fig.savefig(OUT/"toc_workflow.pdf",bbox_inches=None)
fig.savefig(OUT/"toc_workflow.png",dpi=300,bbox_inches=None)
plt.close(fig)
print("Rendered four figures and a TOC graphic from packaged data; no simulations rerun.")
