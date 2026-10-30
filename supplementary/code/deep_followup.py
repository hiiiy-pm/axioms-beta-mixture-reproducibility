import json
import time
import numpy as np
import mpmath as mp
from numpy.polynomial.hermite import hermgauss
from score_limit_audit import ExactScore, ROOT, IH


def independent_score_check():
    rows=[]
    for phi in [.03,.001]:
        obj=ExactScore(phi,gh=12,gl=36)
        hn,hw=hermgauss(12)
        zz=[(mp.mpf(float(np.sqrt(2)*x)),mp.mpf(float(np.sqrt(2)*y)),mp.mpf(float(wx*wy/np.pi)))
            for x,wx in zip(hn,hw) for y,wy in zip(hn,hw)]
        with mp.workdps(60):
            theta=list(map(mp.mpf,[obj.alpha,np.log(phi),obj.v,obj.w,obj.kappa]))
            def density(t,x):
                al,ga,v,w,k=t
                total=mp.mpf(0)
                for z1,z2,ww in zz:
                    b=mp.sqrt(v)*z1;c=k/mp.sqrt(v)*z1+mp.sqrt(w-k*k/v)*z2
                    p=1/(1+mp.exp(-al-b));ph=mp.exp(ga+c)
                    ly=-mp.log(1+mp.exp(-x));ln=-mp.log(1+mp.exp(x))
                    total+=ww*mp.exp(p*ph*ly+(1-p)*ph*ln-mp.log(mp.beta(p*ph,(1-p)*ph)))
                return total
            for xf in [0.,1.,-2/phi,4/phi]:
                x=mp.mpf(xf);den=density(theta,x)
                grad=[]
                for j in range(5):
                    def fun(t):
                        tt=theta.copy();tt[j]=t;return density(tt,x)
                    grad.append(float(mp.diff(fun,theta[j])/den))
                den0,S=obj.evaluate(np.array([xf]))
                rows.append(dict(phi=phi,x=xf,mp_scores=grad,price_scores=S[0,:5].tolist(),
                    max_absolute_difference=float(np.max(abs(np.array(grad)-S[0,:5]))),
                    relative_density_difference=float(den0[0]/float(den)-1)))
                print('independent',rows[-1],flush=True)
    return rows


def main():
    out={'independent_scores':independent_score_check(),'characteristics':[],'equal_information':[],'simulation':[]}
    ts=np.array([.5,1.,1.5,2.,3.])
    for phi in [.003,.001,.0003,.0001]:
        obj=ExactScore(phi,gh=18,gl=48)
        for lam in [.1,1.,10.]:
            n=round(lam/phi);actual=obj.characteristic(n,ts);limit=obj.limiting_characteristic(n*phi,ts)
            row=dict(phi=phi,n=n,n_phi=n*phi,ts=ts.tolist(),
                actual=[[z.real,z.imag] for z in actual],predicted=[[z.real,z.imag] for z in limit],
                max_error_to_hybrid=float(max(abs(actual-limit))),
                max_error_to_standard_normal=float(max(abs(actual-np.exp(-ts*ts/2)))))
            out['characteristics'].append(row);print('CF',row,flush=True)
    for tau in [.01,1.]:
        D0=9*tau*tau/16+IH/64
        a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
        obj=ExactScore(.0001,tau=tau,a=a,gh=20,gl=48)
        out['equal_information'].append(obj.diagnostics())
        refined=ExactScore(.0001,tau=tau,a=a,gh=28,gl=64)
        out['equal_information'][-1]['refinement_relative_difference']=refined.info/obj.info-1
        for n,reps in [(200,20000),(1000,16000),(10000,4000)]:
            row=obj.simulate(n,reps,seed=74600+n+round(100*tau))
            row['tau']=tau;row['a']=a;row['I_exact']=obj.info;row['I_limit']=.5
            out['simulation'].append(row);print('SIM',row,flush=True)
            (ROOT/'deep_followup.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    (ROOT/'deep_followup.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
