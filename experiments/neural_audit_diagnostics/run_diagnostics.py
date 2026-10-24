"""Recompute diagnostics from already published, fixed neural audit predictions.

No network fitting, pickle loading, reactor integration or model MAT input is
needed. Run against this repository's three NPZ fixtures and neural run record.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import csv
import json
import os
import sys

HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'mpl-cache'))
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def metrics(y,pred):
    residual=pred-y
    sse=float(np.sum(residual**2));sst=float(np.sum((y-y.mean())**2))
    if sst<=0:raise ValueError('R2 is undefined for a constant reference')
    mse=sse/len(y)
    return {'n':len(y),'MSE':mse,'RMSE':float(np.sqrt(mse)),
            'MAE':float(np.mean(abs(residual))),'R2':1-sse/sst,
            'SSE':sse,'SST':sst,'reference_mean':float(y.mean()),
            'maximum_absolute_error':float(np.max(abs(residual)))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,help='Repository root containing research_completion/runs')
    p.add_argument('--output',type=Path,default=HERE/'results')
    args=p.parse_args()
    root=args.repository
    if root is None:
        root=next((x for x in HERE.parents if (x/'research_completion/runs/surrogate_dataset.npz').is_file()),None)
    if root is None:raise SystemExit('Supply --repository with the public repository root')
    runs=root/'research_completion/runs';rev=runs/'revision_validation'
    paths=[runs/'surrogate_dataset.npz',rev/'neural_fresh_audit.npz',rev/'neural_audit_predictions.npz',rev/'neural_extension.json']
    D,A,P=[np.load(x,allow_pickle=False)for x in paths[:3]]
    E=json.loads(paths[3].read_text());checks=[]
    def check(name,ok,detail=None):checks.append({'check':name,'passed':bool(ok),'detail':detail})
    controls=A['controls'];z=P['target_log_ratio'];ref=P['refined_target_log_ratio']
    check('Independent audit shapes',controls.shape==(200,7)and z.shape==(200,)and P['predictions'].shape==(3,200))
    check('All stored values are finite',all(np.isfinite(x).all()for x in [controls,z,ref,P['predictions']]))
    parts=[set(map(int,D[k]))for k in ['train','validation','test']]
    check('Training validation test split is disjoint and complete',list(map(len,parts))==[700,150,150]
          and set.union(*parts)==set(range(1000)) and not any(parts[i]&parts[j]for i,j in [(0,1),(0,2),(1,2)]))
    check('Audit controls are unique and separate from fitting data',len(set(map(tuple,controls)))==200
          and not set(map(tuple,controls))&set(map(tuple,D['controls'])))
    check('Audit labels align across both fixtures',np.array_equal(A['target_log_ratio'],z)
          and np.array_equal(A['refined_target_log_ratio'],ref))
    check('Declared training sizes',P['training_sizes'].tolist()==[175,350,700])
    rows=[]
    for i,size in enumerate(P['training_sizes']):
        size=int(size);seed=int(P['selected_seeds'][i]);pred=P['predictions'][i]
        selected=next(x for x in E['selected']if x['training_size']==size)
        best=min((x for x in E['candidates']if x['training_size']==size),key=lambda x:x['validation']['log_ratio_rmse'])
        check(f'{size}: stored seed follows validation selection',seed==selected['selected_seed']==best['seed'])
        for target,truth,estimate,oldkey in [('log_ratio',z,pred,'log_ratio_rmse'),
                                          ('squared_ratio_objective',np.expm1(z)**2,np.expm1(pred)**2,'objective_rmse')]:
            stat=metrics(truth,estimate)
            check(f'{size}/{target}: independent metric implementation',
                  abs(stat['MSE']-mean_squared_error(truth,estimate))<2e-15
                  and abs(stat['R2']-r2_score(truth,estimate))<2e-15)
            check(f'{size}/{target}: RMSE matches published run',abs(stat['RMSE']-selected['fresh_audit'][oldkey])<1e-14)
            rows.append({'training_size':size,'selected_seed':seed,'target':target,**stat})
    report={'scope':'Descriptive statistics from fixed saved predictions on 200 independently sampled simulator controls. No retraining, network deserialisation or reactor rerun. This assesses simulator-surrogate agreement, not experimental validity or the accuracy of an optimised policy.',
            'definitions':{'z':'ln(X_SC(tf)/X_KP(tf))','J':'expm1(z)^2','MSE':'mean((prediction-reference)^2)',
                           'RMSE':'sqrt(MSE)','R2':'1-SSE/SST, SST centred on the evaluation reference mean','units':'All responses and statistics are dimensionless.'},
            'sources':[{'path':str(x.relative_to(root)),'sha256':sha256(x.read_bytes()).hexdigest()}for x in paths],
            'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'models':rows,
            'reference_refinement_log_ratio_RMSE':float(np.sqrt(np.mean((ref-z)**2))),
            'environment':{'python':sys.version,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__},
            'checks':checks,'checks_passed':sum(x['passed']for x in checks),'checks_total':len(checks),
            'passed':all(x['passed']for x in checks)}
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'neural_statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    with(out/'neural_statistics.csv').open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if not report['passed']:raise SystemExit('\n'.join(x['check']for x in checks if not x['passed']))
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    pred=P['predictions'][2];fig,ax=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
    limits=(min(z.min(),pred.min())-.04,max(z.max(),pred.max())+.04)
    ax[0].plot(limits,limits,'--',color='#666666',lw=1)
    ax[0].scatter(z,pred,s=15,color='#376f85',alpha=.7,edgecolor='white',linewidth=.3)
    ax[0].set(xlim=limits,ylim=limits,xlabel=r'Simulator reference $z$',ylabel=r'Prediction $\widehat z$')
    ax[0].set_title('(a) Prediction agreement',loc='left');ax[0].set_aspect('equal','box')
    ax[1].axhline(0,color='#666666',ls='--',lw=1)
    ax[1].scatter(z,pred-z,s=15,color='#5f8c84',alpha=.7,edgecolor='white',linewidth=.3)
    ax[1].set(xlim=limits,xlabel=r'Simulator reference $z$',ylabel=r'Residual $\widehat z-z$')
    ax[1].set_title('(b) Prediction residuals',loc='left')
    ax[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,2),useMathText=True)
    for a in ax:a.grid(axis='y',color='#e0e5e7',lw=.5)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'neural_audit_diagnostics.{suffix}',dpi=400,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:report[k]for k in ['passed','checks_passed','checks_total','reference_refinement_log_ratio_RMSE']},indent=2))
    print(json.dumps([x for x in rows if x['training_size']==700],indent=2))


if __name__=='__main__':main()
