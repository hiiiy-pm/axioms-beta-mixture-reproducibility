import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import zeta
from scipy.optimize import brentq
from score_limit_audit import ExactScore, ROOT, IH


G4=np.array([[.5,0,0,-.5],[0,1,-.5,0],[0,-.5,5/16,0],[-.5,0,0,2.]])
WHITEN=np.linalg.inv(np.linalg.cholesky(G4)).T


def polynomials(u,sign):
    A=sign*u/2
    B=1-u
    V=(u*u-2)/8
    K=A*(2-u)
    e=A*(3-u)
    R=(u**3-9*u*u+18*u-6)/8
    Q=sign*(-u**3+8*u*u-12*u)/4
    return A,B,V,K,e,R,Q


class FiniteBulk:
    def __init__(self,obj,n,order=64,torder=900,tmax=55):
        self.obj=obj;self.n=n;self.pars=obj.parameters();self.lam=n*obj.phi
        self.order=order;self.torder=torder;self.tmax=tmax
        gx,gw=leggauss(order)
        def nodes(cuts):
            xx=np.concatenate([(lo+hi)/2+(hi-lo)*gx/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
            ww=np.concatenate([(hi-lo)*gw/2 for lo,hi in zip(cuts[:-1],cuts[1:])])
            return xx,ww
        u,wu=nodes([0,.25,.5,1,2,4,8,12,16,24,32,48,64])
        U=np.tile(u,2);S=np.repeat([-1,1],len(u))
        self.wg=np.tile(wu*np.exp(-u)/2,2)
        _,_,_,_,e,R,Q=polynomials(U,S)
        p=self.pars;h11,h12,_=obj.H
        self.g=p['ae']*e-p['b']*obj.tau*(h11*R+h12*Q)
        x,wx=nodes([0,1,2,4,8,16,32,64])
        self.F=np.logaddexp(0.,-x)*np.logaddexp(0.,x)
        self.wf=2*wx
        tg,tw=leggauss(torder)
        self.ts=(tg+1)*tmax/2;self.tw=tw*tmax/2
        self.cf=self.characteristic(self.ts)

    def characteristic(self,ts):
        out=[];p=self.pars
        for lo in range(0,len(ts),100):
            t=np.asarray(ts[lo:lo+100]);a=t/np.sqrt(self.n*p['info'])
            one=1+np.expm1(1j*a[:,None]*self.g[None,:])@self.wg
            bulk=one**self.n
            jump=t[:,None]*p['b']*self.F[None,:]/(4*np.sqrt(self.lam*p['info']))
            rare=np.exp(self.lam/4*((np.expm1(1j*jump)-1j*jump)@self.wf))
            out.extend((bulk*rare).tolist())
        return np.array(out)

    def cdf(self,x):
        return float(.5-np.sum(self.tw*np.imag(np.exp(-1j*self.ts*x)*self.cf)/self.ts)/np.pi)

    def quantiles(self,probs=(.025,.975)):
        return [float(brentq(lambda x:self.cdf(x)-q,-40,40,xtol=2e-9)) for q in probs]


def joint_reference(obj,n,reps=100000,seed=8801,levels=(.9,.95,.975,.99)):
    rng=np.random.default_rng(seed);p=obj.parameters();D0=p['D0'];lam=n*obj.phi
    h11,h12,_=obj.H;M=24.
    vals=[];scalar=[]
    for lo in range(0,reps,100):
        size=min(100,reps-lo)
        u=rng.exponential(size=(size,n));sign=2*rng.integers(0,2,size=(size,n))-1
        A,B,V,K,e,R,Q=polynomials(u,sign)
        strong=np.column_stack([a.sum(axis=1) for a in [A,B,V,K]])@WHITEN/np.sqrt(n)
        h=obj.tau*(h11*R+h12*Q)
        weak=h.sum(axis=1)/np.sqrt(n*D0)
        eff=(p['ae']*e-p['b']*h).sum(axis=1)/np.sqrt(n*p['info'])
        num=rng.poisson(lam*M/2,size=size)
        x=rng.uniform(-M,M,size=int(num.sum()))
        f=np.logaddexp(0.,-x)*np.logaddexp(0.,x)
        sums=np.bincount(np.repeat(np.arange(size),num),weights=f,minlength=size)
        compensated=sums-lam*zeta(3,1)/2
        weak-=compensated/(4*np.sqrt(lam*D0))
        eff+=p['b']*compensated/(4*np.sqrt(lam*p['info']))
        vals.extend((np.sum(strong**2,axis=1)+weak**2).tolist())
        scalar.extend(eff.tolist())
    vals=np.array(vals);scalar=np.array(scalar)
    return dict(phi=obj.phi,tau=obj.tau,a=obj.a,H=obj.H,n=n,n_phi=lam,reps=reps,seed=seed,
        levels=list(levels),critical_values=np.quantile(vals,levels).tolist(),
        scalar_quantiles=np.quantile(scalar,[.025,.975]).tolist(),
        joint_mean=float(vals.mean()),joint_variance=float(vals.var()),
        scalar_mean=float(scalar.mean()),scalar_variance=float(scalar.var()))


def main():
    ts=np.array([.5,1.,1.5,2.,3.]);grid=json.loads((ROOT/'critical_law_grid.json').read_text())
    out={'grid':[],'joint_reference':[]}
    for old in grid['rows']:
        if old['phi']!=.0001:continue
        obj=ExactScore(old['phi'],tau=old['tau'],a=old['a'],H=old['H'],gh=18,gl=48)
        for check in old['cf_checks']:
            n=check['n'];ref=FiniteBulk(obj,n)
            actual=obj.characteristic(n,ts)
            err=float(max(abs(actual-ref.characteristic(ts))))
            cuts=ref.quantiles()
            row=dict(phi=obj.phi,tau=obj.tau,a=obj.a,H=obj.H,n=n,direction=old['direction'],
                eta=obj.parameters()['rare_fraction'],normal_error=check['normal_error'],
                gaussian_poisson_error=check['hybrid_error'],finite_bulk_error=err,
                finite_bulk_cuts=cuts,cf_at_cutoff=float(abs(ref.characteristic(np.array([55.]))[0])))
            out['grid'].append(row);print(json.dumps(row),flush=True)
            (ROOT/'finite_bulk.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    tau=.01;D0=9*tau*tau/16+IH/64;a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
    obj=ExactScore(.0001,tau=tau,a=a,gh=20,gl=48)
    for n,reps in [(200,200000),(1000,200000),(10000,50000)]:
        row=joint_reference(obj,n,reps=reps,seed=721000+n)
        ref=FiniteBulk(obj,n)
        row['inverted_scalar_quantiles']=ref.quantiles()
        out['joint_reference'].append(row);print('REFERENCE',json.dumps(row),flush=True)
        (ROOT/'finite_bulk.json').write_text(json.dumps(out,indent=2),encoding='utf-8')


if __name__=='__main__':main()
