"""Deterministic finite-game decisions; no scenario identity is an input.

Biomass maximization subject to recovery is distinct from minimax ratio loss.
All ties use ascending action index after the declared primary ordering.
"""
import numpy as np
import engine as e
INNER=e.BAND-2e-5
TOL=1e-10
PHYSICAL=-2e-7

def observable(states):
    return states[...,1]-states[...,2],np.exp(states[...,1])+np.exp(states[...,2])

def choose(errors,biomass,valid):
    errors=np.where(valid,errors,np.inf)
    safe=np.flatnonzero(valid & (errors<=INNER+TOL))
    minimax=min(range(len(errors)),key=lambda i:(errors[i],-biomass[i],i))
    economic=min(safe,key=lambda i:(-biomass[i],errors[i],i)) if len(safe) else minimax
    return int(economic),int(minimax),safe

def one_values(m,target,nominal=False):
    z,b=observable(m['terminal']);err=np.abs(z-np.log(target))
    err=err[:,0] if nominal else err.max(axis=1)
    b=b[:,0] if nominal else b.min(axis=1)
    valid=np.isfinite(err)&np.isfinite(b)&(m['min_S'].min(axis=1)>=PHYSICAL)&(m['terminal'][...,0].max(axis=1)<=e.CAP)
    idx,jidx,safe=choose(err,b,valid)
    return dict(errors=err,biomass=b,valid=valid,index=idx,minimax_index=jidx,safe=safe,J=float(err[jidx]) if valid[jidx] else float('inf'),W=float(b[idx]) if len(safe) else None)

def two_values(t,target):
    n=len(t['branches']); q=np.empty(n);wb=np.zeros(n);safe0=np.zeros(n,bool)
    olerror=[];olbio=[];olvalid=[];plans=[];nomerror=[];nombio=[];continuations=[]
    for i,branch in enumerate(t['branches']):
        z,b=observable(branch['terminal']);er=np.abs(z-np.log(target)) # K0,A1,K1
        valid=(branch['min_S'].min(axis=2)>=PHYSICAL)&(branch['terminal'][...,0].max(axis=2)<=e.CAP)
        valid &= t['first']['min_S'][i,:,None]>=PHYSICAL
        errs=er.max(axis=2);bs=b.min(axis=2)
        ids=[];js=[];ws=[];oks=[]
        for k in range(len(e.K)):
            idx,jidx,ss=choose(errs[k],bs[k],valid[k]);ids.append(idx);js.append(float(errs[k,jidx]) if valid[k,jidx] else float('inf'));ws.append(float(bs[k,idx]));oks.append(bool(len(ss)))
        q[i]=max(js);safe0[i]=all(oks);wb[i]=min(ws) if safe0[i] else 0.
        continuations.append(ids)
        for j,u1 in enumerate(branch['actions']):
            plans.append([float(t['first']['actions'][i]),float(u1)])
            olerror.append(float(er[:,j,:].max()));olbio.append(float(b[:,j,:].min()));olvalid.append(bool(valid[:,j].all()))
            nomerror.append(float(er[0,j,0]));nombio.append(float(b[0,j,0]))
    # q is a minimax composition value, wb the worst feasible economic value.
    didx,dj,dsafe=choose(q,wb,np.isfinite(q))
    assert np.array_equal(safe0,np.isin(np.arange(n),dsafe))
    oi,oj,os=choose(np.array(olerror),np.array(olbio),np.array(olvalid))
    ni,nj,ns=choose(np.array(nomerror),np.array(nombio),np.array(olvalid))
    assert q[dj] <= olerror[oj]+1e-10
    return dict(q=q,W_by_action=wb,safe=dsafe,index=didx,minimax_index=dj,J=float(q[dj]),W=float(wb[didx]) if len(dsafe) else None,continuations=continuations,
        open=dict(index=oi,minimax_index=oj,plan=plans[oi],minimax_plan=plans[oj],J=float(olerror[oj]),selected_error=olerror[oi],W=olbio[oi] if len(os) else None,safe_count=len(os)),
        nominal=dict(plan=plans[ni],J=float(nomerror[nj]),safe_count=len(ns)),
        plans=plans,open_errors=olerror,open_biomass=olbio)

def proposal(x,args,target,h):
    """Two nominal constant-feed probes define an affine terminal-ratio surrogate."""
    cap=max(0.,min(.2,(e.CAP-x[0]-e.GUARD)/(e.DT*h)))
    zs=[]
    for feed in (0.,cap):
        y=x.copy()
        for _ in range(h):y=e.flow(y,feed,e.K[0],args)[0]
        zs.append(float(y[1]-y[2]))
    slope=(zs[1]-zs[0])/cap if cap>0 else 0.
    raw=float(np.clip((np.log(target)-zs[0])/slope,0,cap)) if abs(slope)>1e-12 else 0.
    alphabet=e.actions(x);j=int(np.argmin(np.abs(alphabet-raw)))
    return dict(raw=raw,quantized=float(alphabet[j]),index=j,probe_terminal_z=zs,probe_transitions=2*h)

def filtered_index(values,proposal_index,alphabet):
    safe=values['safe'];errors=values.get('q',values.get('errors'));b=values.get('W_by_action',values.get('biomass'))
    if proposal_index in safe:return int(proposal_index),'accept'
    if len(safe):
        idx=min(safe,key=lambda i:(abs(alphabet[i]-alphabet[proposal_index]),-b[i],errors[i],int(i)))
        return int(idx),'correct'
    return int(values['minimax_index']),'fallback'

def decision(x,args,target,h,method,transition_map=None):
    m=(e.tree(x,args) if h==2 else e.one_step(x,args)) if transition_map is None else transition_map
    v=two_values(m,target) if h==2 else one_values(m,target,nominal=method=='nominal_replan')
    alphabet=m['first']['actions'] if h==2 else m['actions'];status='direct';p=None
    if h==2 and method in ('open_grid','open_replan','nominal_replan'):
        plan=v['nominal' if method=='nominal_replan' else 'open']['plan'];feed=plan[0];idx=int(np.argmin(abs(alphabet-feed)))
    else:
        idx=v['index']
        if method=='recovery_filter':
            p=proposal(x,args,target,h);idx,status=filtered_index(v,p['index'],alphabet)
        feed=float(alphabet[idx]);plan=None
    return dict(feed=float(feed),index=int(idx),status=status,proposal=p,plan=plan,J=v['J'],recoverable=bool(len(v['safe'])))
