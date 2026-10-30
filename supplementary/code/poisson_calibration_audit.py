import json
import numpy as np
from score_limit_audit import ExactScore,ROOT,IH

tau=.01;D0=9*tau*tau/16+IH/64;a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
out={'tau':tau,'a':a,'characteristics':[],'simulation':[]}
ts=np.array([.5,1.,1.5,2.,3.])
for phi in [.003,.001,.0003,.0001]:
    obj=ExactScore(phi,tau=tau,a=a,gh=20,gl=48)
    for lam in [.02,.1,1.]:
        n=max(2,round(lam/phi));actual=obj.characteristic(n,ts);pred=obj.limiting_characteristic(n*phi,ts)
        row=dict(phi=phi,n=n,n_phi=n*phi,info=obj.info,
            error_hybrid=float(max(abs(actual-pred))),error_normal=float(max(abs(actual-np.exp(-ts*ts/2)))))
        out['characteristics'].append(row);print('CF',row,flush=True)
obj=ExactScore(.0001,tau=tau,a=a,gh=20,gl=48)
for n,reps in [(200,20000),(1000,16000),(10000,4000)]:
    row=obj.simulate(n,reps,seed=74600+n+1)
    out['simulation'].append(row);print('CALIBRATION',row,flush=True)
    (ROOT/'poisson_calibration_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
