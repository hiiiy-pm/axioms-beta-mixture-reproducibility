import json
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from score_limit_audit import ExactScore,ROOT,IH
from finite_bulk import FiniteBulk,joint_reference
from joint_score_test import simulate as joint_simulate


def scalar_simulate(obj,n,reps,seed,cuts):
    rng=np.random.default_rng(seed)
    L=np.linalg.cholesky([[obj.v,obj.kappa],[obj.kappa,obj.w]])
    z=[];check=None
    for lo in range(0,reps,100):
        size=min(100,reps-lo)
        bc=rng.standard_normal((size,n,2))@L.T
        p=expit(obj.alpha+bc[:,:,0]);phi=obj.phi*np.exp(bc[:,:,1]);aa=p*phi;bb=(1-p)*phi
        x=np.log(rng.gamma(aa+1))+np.log(rng.random(aa.shape))/aa
        x-=np.log(rng.gamma(bb+1))+np.log(rng.random(bb.shape))/bb
        sc=obj.interp(x);bad=~np.isfinite(sc)
        if bad.any():sc[bad]=obj.efficient(x[bad])
        z.extend(((sc-obj.mean).sum(axis=1)/np.sqrt(n*obj.info)).tolist())
        if check is None:check=x.ravel()[:500]
    z=np.array(z)
    methods={}
    for method,(left,right) in cuts.items():
        pl=float(np.mean(z<left));pr=float(np.mean(z>right));total=pl+pr
        methods[method]=dict(cuts=[left,right],left=pl,right=pr,size=total,
            mcse=float(np.sqrt(total*(1-total)/reps)),
            left_mcse=float(np.sqrt(pl*(1-pl)/reps)),right_mcse=float(np.sqrt(pr*(1-pr)/reps)))
    interpolation_error=float(np.max(abs(obj.efficient(check)-obj.interp(check)))/np.sqrt(obj.info))
    return dict(phi=obj.phi,tau=obj.tau,a=obj.a,H=obj.H,n=n,reps=reps,seed=seed,
        info=obj.info,eta=obj.parameters()['rare_fraction'],methods=methods,
        quantiles=np.quantile(z,[.025,.975]).tolist(),interpolation_error=interpolation_error)


def main():
    refs=json.loads((ROOT/'finite_bulk.json').read_text())
    out={'joint_primary':[],'joint_nonzero_covariance':[],'scalar_grid':[],'numerical_checks':[]}
    tau=.01;D0=9*tau*tau/16+IH/64;a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
    obj=ExactScore(.0001,tau=tau,a=a,gh=20,gl=48)
    for n,reps in [(200,20000),(1000,16000),(10000,4000)]:
        critical=next(x for x in refs['joint_reference'] if x['n']==n)
        row=joint_simulate(obj,n,reps,critical,seed=65300+n)
        row['reference_reps']=critical['reps'];out['joint_primary'].append(row)
        print('JOINT',json.dumps(row),flush=True)
        (ROOT/'calibration_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for idx,r in enumerate(refs['grid']):
        if r['n']!=1000:continue
        obj=ExactScore(r['phi'],tau=r['tau'],a=r['a'],H=r['H'],gh=20,gl=48)
        cuts={'normal':[float(norm.ppf(.025)),float(norm.ppf(.975))],
            'gaussian_poisson':obj.limit_quantiles(1000*obj.phi),
            'finite_bulk':r['finite_bulk_cuts']}
        row=scalar_simulate(obj,1000,16000,seed=829000+idx,cuts=cuts)
        row['direction']=r['direction'];out['scalar_grid'].append(row)
        print('SCALAR',json.dumps(row),flush=True)
        (ROOT/'calibration_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for idx,h in enumerate([(1.,.4,1.2),(.7,-.25,1.4)]):
        tau=.3;D0=IH/64+tau*tau*(9*h[0]**2/16+3*h[1]**2)
        a=2*(tau*h[1]-np.sqrt(D0/1.5));obj=ExactScore(.0001,tau=tau,a=a,H=h,gh=20,gl=48)
        ref=joint_reference(obj,1000,reps=200000,seed=932100+idx)
        row=joint_simulate(obj,1000,16000,ref,seed=953100+idx)
        row['reference']=ref;out['joint_nonzero_covariance'].append(row)
        print('JOINT-COV',json.dumps(row),flush=True)
        (ROOT/'calibration_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    obj=ExactScore(.0001,tau=.01,a=2*np.sqrt((IH/64+9*.01**2/16)*2/1.5),gh=20,gl=48)
    for n in [200,1000]:
        low=FiniteBulk(obj,n);high=FiniteBulk(obj,n,order=128,torder=1400,tmax=65)
        q0=low.quantiles();q1=high.quantiles()
        out['numerical_checks'].append(dict(n=n,default=q0,refined=q1,
            max_quantile_change=float(max(abs(np.array(q0)-q1)))))
    (ROOT/'calibration_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')


if __name__=='__main__':main()
