"""Run only after development checks; creates an immutable local protocol."""
import hashlib,json,time
import numpy as np
from model import ROOT
from run_study import METHODS,hashes,save

def build():
    rng=np.random.default_rng(2026102417); cases=[];blocks=[]
    for family in ('table','analytic'):
        for block in range(4):
            cluster=family+'_'+str(block)
            initial_ratio=float(rng.uniform(.85,1.15));glucose=float(rng.uniform(18.,23.))
            target=[.75,.95,1.15,1.35][block]
            magnitude=float(rng.uniform(.12,.18))
            specs=[('nominal',[1.,1.],[1.,1.],99,None,0)]
            for sign in (-1,1):
                k=(1+sign*magnitude*np.array([1.,-1.])).tolist()
                specs.append((f'static_{sign}',k,k,99,None,0))
                for stage in (3,4):specs.append((f'step_{stage}_{sign}',[1.,1.],k,stage,None,0))
            for i,ki in enumerate(([180.,260.],[260.,180.])):
                specs.append((f'inhibition_{i}',[1.,1.],[1.,1.],99,ki,0))
            for seed in (0,1):
                sign=1 if (block+seed)%2 else -1
                k=(1+sign*magnitude*np.array([1.,-1.])).tolist()
                specs.append((f'noisy_{seed}',[1.,1.],k,3+seed,None,1+seed))
            blocks.append({'cluster':cluster,'initial_ratio':initial_ratio,'initial_glucose':glucose,'target_ratio':target,'magnitude':magnitude})
            for name,before,after,shift,ki,noiseid in specs:
                noise=np.zeros((8,3))
                if noiseid:
                    noise_rng=np.random.default_rng(2026102450+100*int(family=='analytic')+10*block+noiseid)
                    noise=noise_rng.normal(size=(8,3))*[.0025,.0025,.025]
                case=dict(id=cluster+'_'+name,cluster=cluster,family=family,regime=name.split('_')[0],
                    initial_ratio=initial_ratio,initial_glucose=glucose,target_ratio=target,
                    kappa_before=before,kappa_after=after,change_stage=shift,inhibition=ki,noise=noise.tolist())
                cases.append(case)
    return {'title':'Causal scenario feedback and surrogate management under coculture model mismatch',
        'status':'Local freeze after declared development; not public preregistration',
        'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'methods':METHODS,'blocks':blocks,'cases':cases,
        'design':{'clusters':8,'cases':len(cases),'episodes':len(cases)*len(METHODS),'families':['table','analytic'],
            'interpretation':'Four independently assigned IC/target blocks per mathematical family; not biological replicates'},
        'controller':{'iterations':{'ce_limited':5,'other_direct':12},'gain_bounds':[0,2],
            'scenario_radius_by_observed_intervals':[.15,.05,.025],'future_shift_magnitude':.16,'future_shift_stages':[3,4],
            'scenario_box':[.75,1.25],'scenario_combination':'union, not Cartesian product','terminal_design_band':.0075,
            'inward_numerical_margin':2e-6,'acceptance_band':float(np.log(1.01)),
            'surrogate_feed_radius':.035,'surrogate_corrections':2,'objective_slack_penalty':100},
        'primary_comparisons':['scenario_feedback vs ce_direct: formulation','scenario_managed vs scenario_feedback: computation and achieved quality'],
        'other_comparisons':['ce_limited: strong practical baseline','scenario_open: feedback-family ablation'],
        'endpoints':['actual composition and physical feasibility; all cases in denominator','operational success excluding controller exceptions',
            'absolute log-ratio error','terminal biomass','total online time including estimation/checks/corrections/fallback',
            'jointly feasible paired biomass/time; complete totals also reported','remaining physical volume by stage'],
        'domain_bound':{'max_initial_S':23.,'max_kappa':1.25,'max_noise_S':max(abs(np.array(c['noise'])[:,2]).max() for c in cases),
            'note':'No uptake upper bound (0.5*23+100*1.3)/1.8=78.6112; effective≤98.264; finite noise margin verified in audit'},
        'inference':'Descriptive paired/cluster-resampling across fixed mathematical conditions only; no biological reliability confidence statement',
        'numerical_audit_thresholds':'audit/campaign_audit_protocol.json (hash-locked with independent replay source)',
        'timing':'Main effectiveness campaign: 4 isolated processes, wall times diagnostic only. Separate prespecified serial panel after workers exit; rotated method order, one BLAS thread; shared compiled initialization separately recorded.',
        'serial_timing_case_ordinals':[0,5,22,25,44,49,66,69],
        'failure_policy':'Failed candidates logged; managed verification failure triggers matched direct fallback; whole controller exception applies zero feed with explicit flag',
        'development':['pilot_v1 (interrupted)','pilot_v2 (interrupted)','pilot_v3'],
        'limitations':['Finite scenarios do not cover all model errors','Future policy composition observations ideal in design',
            'No guaranteed recursive feasibility','Analytic family is designed, not a calibrated second organism','Public data prediction study separate']}
if __name__=='__main__':
    if (ROOT/'LOCK.json').exists():raise RuntimeError('Already locked')
    protocol=build();save(ROOT/'protocol.json',protocol)
    h=hashes();h['protocol.json']=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
    save(ROOT/'LOCK.json',{'hashes':h,'created_utc':protocol['created_utc']})
    print(protocol['design'])
