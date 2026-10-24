"""Correctly differentiated varying-bound KKT system in null-space coordinates.

Flux coordinates keep all independent stoichiometric equalities satisfied.
The full sparse KKT matrix supplies derivatives; no flux is clipped. All
donor-floor chain-rule terms are included. Direct algebraic runs are an
independent reference, not used to correct the differentiated trajectory.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from scipy.sparse import bmat,diags
from scipy.sparse.linalg import spsolve
from resource_diversion import Model,parameters,HERE,digest

class Differential:
    def __init__(self,case,horizon,mu=1e-8):
        self.case=case;self.p=parameters(case,.1,horizon);self.mu=mu
        self.models=[Model(case,s,.1,mu)for s in ['S1','S2']]
        x=self.p['initial_biomass_gDW_L'];V=5.
        self.initial_physical=np.r_[V,V*x,V*x,100.,0.,np.zeros(9)]
        env=self.initial_physical[1:5]/V
        self.base=[m.flux(env)for m in self.models]
        self.Z=[null_space(m.A.toarray())for m in self.models]
        assert all(z.shape==(95,28)for z in self.Z)
        self.initial=np.r_[self.initial_physical,np.zeros(56)]
        self.calls=0;self.mintrial=np.inf
        self.lam_recover=[np.linalg.pinv(m.A.toarray().T)for m in self.models]

    def unpack(self,y):
        return [self.base[i]+self.Z[i]@y[14+28*i:14+28*(i+1)]for i in range(2)]

    def rhs(self,t,y):
        self.calls+=1;V=y[0];env=y[1:5]/V;x1,x2,G,P=env
        F=self.p['feed_L_h'];Gin=self.p['feed_glucose_g_L'];kd=self.p['death_h_inv']
        vv=self.unpack(y)
        grow=np.array([vv[i][12]*y[1+i]for i in range(2)]);loss=kd*y[1:3]
        gg=np.array([.18*vv[i][27]*y[1+i]for i in range(2)])
        pp=np.array([.088*vv[i][37]*y[1+i]for i in range(2)])
        physical=np.r_[F,grow-loss,gg.sum()+F*Gin,pp.sum(),grow,loss,gg,pp,F*Gin]
        x1dot,x2dot,Gdot,Pdot=(physical[1:5]-env*F)/V
        coords=[]
        for i,m in enumerate(self.models):
            l,u=m.bounds(env);v=vv[i];sl=v-l;su=u-v
            self.mintrial=min(self.mintrial,float(min(sl.min(),su.min())))
            dl=np.zeros(95);dl[27]=-10*Gdot/(1+G)**2
            if i==0:
                dl[37]=.1*(Gdot*x1*x2+G*x1dot*x2+G*x1*x2dot)
            else:
                dl[37]=-50*Pdot/(5+P)**2
            yl=self.mu/sl**2;yu=self.mu/su**2
            K=bmat([[diags(yl+yu),m.A.T],[m.A,None]],format='csc')
            sol=spsolve(K,np.r_[yl*dl,np.zeros(67)])
            coords.append(self.Z[i].T@sol[:95])
        return np.r_[physical,*coords]

    def output(self,Y):
        flux=[];lam=[];lower=[];upper=[];res=[];gaps=[]
        for y in Y:
            env=y[1:5]/y[0];vs=self.unpack(y);rr=[];la=[];ll=[];uu=[];gap=[]
            for i,m in enumerate(self.models):
                v=vs[i];l,u=m.bounds(env);sl=v-l;su=u-v
                target=m.c+self.mu/sl-self.mu/su;mult=self.lam_recover[i]@target
                eq=float(abs(m.rawA@v).max());stat=float(abs(m.A.T@mult-target).max())
                rr.append([eq,stat,float(min(sl.min(),su.min()))]);la.append(mult);ll.append(l);uu.append(u)
                lp=m.lp(env);assert lp.success,lp.message;gap.append(float(-lp.fun-v[12]))
            flux.append(vs);lam.append(la);lower.append(ll);upper.append(uu);res.append(rr);gaps.append(gap)
        return dict(flux=np.array(flux),lambda_=np.array(lam),lower=np.array(lower),upper=np.array(upper),
                    residuals=np.array(res),LP_growth_gaps=np.array(gaps))

def run(case,horizon,label,rtol=1e-10,atol=1e-12):
    m=Differential(case,horizon);start=time.perf_counter()
    s=solve_ivp(m.rhs,(0,horizon),m.initial,method='Radau',rtol=rtol,atol=atol,max_step=.02,dense_output=True)
    assert s.success,s.message
    times=np.linspace(0,horizon,501);Y=s.sol(times).T;out=m.output(Y)
    path=HERE/label;path.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path/'trajectory.npz',time_h=times,amounts=Y[:,:14],null_coordinates=Y[:,14:],**out)
    r=out['residuals'];g=out['LP_growth_gaps']
    report=dict(case=case,parameters=m.p,method='Correct differentiated KKT; Radau; null-space flux coordinates',
                mu=m.mu,rtol=rtol,atol=atol,max_step=.02,elapsed_s=time.perf_counter()-start,
                rhs_calls=m.calls,nfev=s.nfev,njev=s.njev,nlu=s.nlu,
                maximum_stoichiometric_residual=float(r[:,:,0].max()),maximum_explicit_stationarity_residual=float(r[:,:,1].max()),
                minimum_output_slack=float(r[:,:,2].min()),minimum_trial_slack=m.mintrial,
                max_growth_gap_to_LP=float(g.max()),min_growth_gap_to_LP=float(g.min()),barrier_gap_bound=190*m.mu,
                terminal_concentrations=(Y[-1,1:5]/Y[-1,0]).tolist(),
                donor_floor_derivative='.1*(Gdot*X1*X2+G*X1dot*X2+G*X1*X2dot)',
                no_flux_clipping=True,script_sha256=digest(Path(__file__)))
    (path/'audit.json').write_text(json.dumps(report,indent=2)+'\n');(path/'run_script.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--horizon',required=True,type=float);p.add_argument('--name',required=True);p.add_argument('--tight',action='store_true')
    a=p.parse_args();run(a.case,a.horizon,a.name,rtol=1e-12 if a.tight else 1e-10,atol=1e-14 if a.tight else 1e-12)
