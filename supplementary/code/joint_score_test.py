import json
import numpy as np
from scipy.special import expit,zeta
from scipy.stats import chi2
from scipy.interpolate import PchipInterpolator
from score_limit_audit import ExactScore,ROOT,IH


def critical_limit(lam,D0,reps=1000000,seed=52900):
    rng=np.random.default_rng(seed);omega=IH/(64*D0);M=24.
    num=rng.poisson(lam*M/2,size=reps)
    x=rng.uniform(-M,M,size=int(num.sum()))
    f=np.logaddexp(0.,-x)*np.logaddexp(0.,x)
    sums=np.bincount(np.repeat(np.arange(reps),num),weights=f,minlength=reps)
    weak=rng.normal(0,np.sqrt(1-omega),reps)-(sums-lam*zeta(3,1)/2)/(4*np.sqrt(lam*D0))
    q=rng.chisquare(4,reps)+weak*weak
    cuts=np.quantile(q,[.9,.95,.975,.99])
    return dict(lambda_=lam,D0=D0,omega=omega,reps=reps,seed=seed,levels=[.9,.95,.975,.99],critical_values=cuts.tolist())


def simulate(obj,n,reps,critical,seed):
    mass=obj.mass/sum(obj.mass)
    basis=np.column_stack([obj.S[:,[0,1,2,4]],obj.S[:,5]/obj.scale])
    means=mass@basis
    G=basis.T@(mass[:,None]*basis)
    invL=np.linalg.inv(np.linalg.cholesky(G)).T
    W=(basis-means)@invL
    interp=PchipInterpolator(obj.x,W,axis=0,extrapolate=False)
    rng=np.random.default_rng(seed);L=np.linalg.cholesky(np.array([[obj.v,obj.kappa],[obj.kappa,obj.w]]))
    vals=[];check=None
    for start in range(0,reps,50):
        size=min(50,reps-start)
        bc=rng.standard_normal((size,n,2))@L.T
        p=expit(obj.alpha+bc[:,:,0]);ph=obj.phi*np.exp(bc[:,:,1]);aa=p*ph;bb=(1-p)*ph
        x=np.log(rng.gamma(aa+1))+np.log(rng.random(aa.shape))/aa
        x-=np.log(rng.gamma(bb+1))+np.log(rng.random(bb.shape))/bb
        scores=interp(x)
        bad=~np.all(np.isfinite(scores),axis=2)
        if bad.any():
            _,S=obj.evaluate(x[bad]);B=np.column_stack([S[:,[0,1,2,4]],S[:,5]/obj.scale])
            scores[bad]=(B-means)@invL
        ss=scores.sum(axis=1)/np.sqrt(n)
        vals.extend(np.sum(ss*ss,axis=1).tolist())
        if check is None:check=x.ravel()[:500]
    vals=np.asarray(vals)
    _,S=obj.evaluate(check);B=np.column_stack([S[:,[0,1,2,4]],S[:,5]/obj.scale])
    error=(B-means)@invL-interp(check)
    usual=float(chi2.ppf(.95,5));new=critical['critical_values'][1]
    p0=float(np.mean(vals>usual));p1=float(np.mean(vals>new))
    return dict(phi=obj.phi,tau=obj.tau,a=obj.a,n=n,n_phi=n*obj.phi,reps=reps,seed=seed,
        chi_square_critical=usual,derived_critical=new,usual_size=p0,derived_size=p1,
        usual_mcse=float(np.sqrt(p0*(1-p0)/reps)),derived_mcse=float(np.sqrt(p1*(1-p1)/reps)),
        actual_quantiles=np.quantile(vals,[.9,.95,.975,.99]).tolist(),
        max_whitened_interpolation_error=float(np.max(abs(error))),
        gram_condition=float(np.linalg.cond(G)),whitened_mean=(means@invL).tolist())


if __name__=='__main__':
    tau=.01;D0=9*tau*tau/16+IH/64;a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
    obj=ExactScore(.0001,tau=tau,a=a,gh=20,gl=48)
    out={'null':dict(phi=obj.phi,s=obj.s,alpha=obj.alpha,v=obj.v,w=obj.w,kappa=obj.kappa),
        'critical_laws':[],'simulation':[]}
    for n,reps in [(200,20000),(1000,16000),(10000,4000)]:
        critical=critical_limit(n*obj.phi,D0,seed=52900+n)
        out['critical_laws'].append(critical)
        row=simulate(obj,n,reps,critical,seed=65300+n)
        out['simulation'].append(row);print(json.dumps(row),flush=True)
        (ROOT/'joint_score_test.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
