import json
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from score_limit_audit import ExactScore,ROOT,IH

tau=.01;D0=9*tau*tau/16+IH/64;a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
obj=ExactScore(.0001,tau=tau,a=a,gh=12,gl=24);p=obj.parameters()
rng=np.random.default_rng(84029);M=24.;reps=200000
out={'parameters':p,'window':[-M,M],'replications':reps,'rows':[]}
for lam in [.02,.1,1.]:
    count=rng.poisson(lam*M/2,size=reps)
    x=rng.uniform(-M,M,size=int(count.sum()))
    f=np.logaddexp(0.,-x)*np.logaddexp(0.,x)
    ix=np.repeat(np.arange(reps),count)
    marks=np.bincount(ix,weights=f,minlength=reps)
    from scipy.special import zeta
    jump=p['b']/(4*np.sqrt(lam*p['info']))*(marks-lam/4*(2*zeta(3,1)))
    vals=rng.normal(0,np.sqrt(1-p['rare_fraction']),reps)+jump
    lo,hi=obj.limit_quantiles(lam)
    row=dict(lambda_=lam,numerical_quantiles=[lo,hi],simulation_quantiles=np.quantile(vals,[.025,.975]).tolist(),
        left_probability=float(np.mean(vals<lo)),right_probability=float(np.mean(vals>hi)),
        mean=float(vals.mean()),variance=float(vals.var()),mcse_tail=float(np.sqrt(.025*.975/reps)))
    out['rows'].append(row);print(row,flush=True)
(ROOT/'verify_poisson_limit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
