import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, betaln, logsumexp
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
from joint_information import conditional_constants, data_quadrature, IH

ROOT = Path(__file__).resolve().parents[1] / 'results'


class ExactScore:
    def __init__(self, phi, tau=0.15, a=0.6, H=(1., 0., 1.), gh=18, gl=40):
        self.phi, self.tau, self.a, self.H = phi, tau, a, H
        self.s = tau*phi**2.5
        self.alpha = a*phi**2.5
        h11,h12,h22=H
        self.v,self.kappa,self.w=self.s*np.array(H)
        hn,hw=hermgauss(gh)
        zz=np.stack(np.meshgrid(np.sqrt(2)*hn,np.sqrt(2)*hn,indexing='ij'),axis=-1).reshape(-1,2)
        L=np.linalg.cholesky(np.array([[self.v,self.kappa],[self.kappa,self.w]]))
        bc=zz@L.T
        self.weights=(hw[:,None]*hw[None,:]/np.pi).ravel()
        self.p=expit(self.alpha+bc[:,0]);self.d=self.p*(1-self.p)
        self.z=1-2*self.p;self.ph=phi*np.exp(bc[:,1])
        self.const=conditional_constants(self.p,self.ph)
        p0=expit(self.alpha);self.d0=p0*(1-p0);self.z0=1-2*p0
        cc=conditional_constants(np.array([p0]),np.array([phi]))
        self.ca0,self.cb0=float(cc[5][0]),float(cc[6][0])
        D=9*self.v**2/16+3*self.kappa**2+IH*phi**5/64
        self.scale=np.sqrt(D)
        x,dx=data_quadrature(phi,gl=gl,central_cut=64,tail_cut=120)
        self.x,self.dx=x,dx
        self.density,self.S=self.evaluate(x)
        self.mass=dx*self.density
        nuisance=np.column_stack([self.S[:,:3],(self.S[:,5]+self.z0*self.S[:,4])/self.scale])
        sw=np.sqrt(self.mass)
        self.coef=np.linalg.lstsq(sw[:,None]*nuisance,sw*self.S[:,4],rcond=None)[0]
        self.eff=self.S[:,4]-nuisance@self.coef
        self.mean=float(self.mass@self.eff)
        self.info=float(self.mass@(self.eff**2))
        self.interp=PchipInterpolator(x,self.eff,extrapolate=False)

    def evaluate(self,x):
        x=np.asarray(x); density=[];scores=[]
        p,d,z,ph=self.p,self.d,self.z,self.ph
        eu,ev,C,J,R,ca,cb,sef=self.const
        for start in range(0,len(x),400):
            xx=x[start:start+400,None]
            ly,ln=-np.logaddexp(0.,-xx),-np.logaddexp(0.,xx)
            logf=(p*ph)[None,:]*ly+((1-p)*ph)[None,:]*ln-betaln(p*ph,(1-p)*ph)[None,:]
            lm=logsumexp(logf+np.log(self.weights)[None,:],axis=1)
            post=np.exp(logf+np.log(self.weights)[None,:]-lm[:,None])
            A=d[None,:]*(ph[None,:]*xx-eu[None,:]+ev[None,:])
            B=(p*ph)[None,:]*ly+((1-p)*ph)[None,:]*ln-(p*eu+(1-p)*ev)[None,:]
            V=(A*A+z[None,:]*A-C[None,:])/2
            W=(B*B+B-J[None,:])/2
            K=A*B+A-R[None,:]
            F=-d[None,:]*(ph[None,:]**2*ly*ln-sef[None,:])
            F+=(self.ca0-ca)[None,:]*A+(self.cb0-cb)[None,:]*B
            F+=2*(d-self.d0)[None,:]*W+(z-self.z0)[None,:]*K
            scores.append(np.stack([np.sum(post*t,axis=1) for t in [A,B,V,W,K,F]],axis=1))
            density.append(np.exp(lm))
        return np.concatenate(density),np.concatenate(scores)

    def efficient(self,x):
        _,S=self.evaluate(np.asarray(x))
        N=np.column_stack([S[:,:3],(S[:,5]+self.z0*S[:,4])/self.scale])
        return S[:,4]-N@self.coef

    def parameters(self):
        h11,h12,h22=self.H
        c0=IH/64; dh=self.tau**2*(9*h11*h11/16+3*h12*h12)
        D0=c0+dh; m=self.tau*h12-self.a/2; den=D0+1.5*m*m
        b=1.5*m/den; ae=D0/den
        info=1.5*D0/den
        eta=b*b*c0/info
        return dict(D0=D0,m=m,b=b,ae=ae,info=info,rare_fraction=eta,bulk_variance_fraction=1-eta)

    def diagnostics(self,ns=(100,1000,10000)):
        z=(self.eff-self.mean)/np.sqrt(self.info)
        pars=self.parameters()
        central=np.abs(self.x)<=10
        F=np.logaddexp(0.,-self.x)*np.logaddexp(0.,self.x)
        U=np.abs(self.phi*self.x)/2;sign=np.sign(self.x)
        e=sign*U*(3-U)/2
        R=(U**3-9*U**2+18*U-6)/8
        Q=sign*(-U**3+8*U**2-12*U)/4
        h11,h12,_=self.H
        g=pars['ae']*e-pars['b']*self.tau*(h11*R+h12*Q)
        g-=np.dot(self.mass,g)/sum(self.mass)
        rare=pars['b']*(F-np.dot(self.mass,F)/sum(self.mass))/(4*np.sqrt(self.phi))
        approx=g+rare
        return dict(phi=self.phi,tau=self.tau,a=self.a,H=self.H,
            mass=float(sum(self.mass)),score_mean=float(self.mean),info=self.info,prediction=pars,
            central_probability=float(sum(self.mass[central])),
            central_information_fraction=float(sum(self.mass[central]*z[central]**2)),
            decomposition_relative_L2_error=float(np.sqrt(self.mass@((self.eff-approx)**2)/self.info)),
            third_moment=float(self.mass@z**3),fourth_moment=float(self.mass@z**4),
            phi_fourth_moment=float(self.phi*(self.mass@z**4)),
            normality=[dict(n=n,n_phi=n*self.phi,skewness=float(self.mass@z**3/np.sqrt(n)),
                excess_kurtosis=float((self.mass@z**4-3)/n),
                lindeberg_eps_1=float(np.sum(self.mass*z*z*(np.abs(z)>np.sqrt(n))))) for n in ns])

    def simulate(self,n,reps,seed):
        rng=np.random.default_rng(seed)
        L=np.linalg.cholesky(np.array([[self.v,self.kappa],[self.kappa,self.w]]))
        vals=[];check_x=[]
        for lo in range(0,reps,100):
            batch=min(100,reps-lo)
            bc=rng.standard_normal((batch,n,2))@L.T
            p=expit(self.alpha+bc[:,:,0]);ph=self.phi*np.exp(bc[:,:,1])
            aa,bb=p*ph,(1-p)*ph
            lg1=np.log(rng.gamma(aa+1))+np.log(rng.random(aa.shape))/aa
            lg2=np.log(rng.gamma(bb+1))+np.log(rng.random(bb.shape))/bb
            x=lg1-lg2
            sc=self.interp(x)
            bad=~np.isfinite(sc)
            if bad.any(): sc[bad]=self.efficient(x[bad])
            vals.extend(((sc-self.mean).sum(axis=1)/np.sqrt(n*self.info)).tolist())
            if lo==0:check_x=x.ravel()[:500]
        vals=np.asarray(vals)
        exact=self.efficient(check_x);interp=self.interp(check_x)
        diff=(exact-interp)/np.sqrt(self.info)
        p_reject=float(np.mean(abs(vals)>norm.ppf(.975)))
        lo,hi=self.limit_quantiles(n*self.phi)
        hrej=float(np.mean((vals<lo)|(vals>hi)))
        return dict(phi=self.phi,n=n,n_phi=n*self.phi,reps=reps,seed=seed,
            mean=float(vals.mean()),variance=float(vals.var()),
            oracle_score_rejection=p_reject,monte_carlo_se=float(np.sqrt(p_reject*(1-p_reject)/reps)),
            left_tail=float(np.mean(vals<norm.ppf(.025))),right_tail=float(np.mean(vals>norm.ppf(.975))),
            hybrid_critical_values=[lo,hi],hybrid_oracle_rejection=hrej,
            hybrid_mcse=float(np.sqrt(hrej*(1-hrej)/reps)),
            hybrid_left_tail=float(np.mean(vals<lo)),hybrid_right_tail=float(np.mean(vals>hi)),
            quantiles=dict(zip(['.01','.025','.5','.975','.99'],np.quantile(vals,[.01,.025,.5,.975,.99]).tolist())),
            interpolation_max_normalized_error=float(np.nanmax(abs(diff))),
            interpolation_rms_normalized_error=float(np.sqrt(np.nanmean(diff**2))))

    def characteristic(self,n,ts):
        z=(self.eff-self.mean)/np.sqrt(self.info)
        w=self.mass/sum(self.mass)
        return np.array([np.exp(n*np.log1p(np.dot(w,np.expm1(1j*t*z/np.sqrt(n))))) for t in ts])

    def limiting_characteristic(self,lam,ts):
        p=self.parameters();eta=p['rare_fraction'];b=p['b'];info=p['info']
        if lam==0:return np.exp(-.5*(1-eta)*np.asarray(ts)**2).astype(complex)
        if np.isinf(lam):return np.exp(-.5*np.asarray(ts)**2).astype(complex)
        vals=[]
        for t in ts:
            def integrand(x):
                F=np.logaddexp(0.,-x)*np.logaddexp(0.,x)
                z=t*b*F/(4*np.sqrt(lam*info))
                return np.expm1(1j*z)-1j*z
            val=2*quad(lambda x:integrand(x).real,0,80,epsabs=1e-12)[0]
            val+=2j*quad(lambda x:integrand(x).imag,0,80,epsabs=1e-12)[0]
            vals.append(np.exp(-.5*(1-eta)*t*t+lam*val/4))
        return np.array(vals)

    def limit_quantiles(self,lam,probs=(.025,.975)):
        p=self.parameters();eta=p['rare_fraction'];b=p['b'];info=p['info']
        gx,gw=leggauss(48);xx=[];ww=[]
        cuts=[0,1,2,4,8,16,32,64]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            xx.extend(((lo+hi)/2+(hi-lo)*gx/2).tolist())
            ww.extend(((hi-lo)*gw).tolist())
        xx=np.array(xx);ww=np.array(ww)
        F=np.logaddexp(0.,-xx)*np.logaddexp(0.,xx)
        tg,tw=leggauss(640);tmax=np.sqrt(90/(1-eta))
        tt=(tg+1)*tmax/2;tw=tw*tmax/2
        zz=tt[:,None]*b*F[None,:]/(4*np.sqrt(lam*info))
        phi=np.exp(-(1-eta)*tt*tt/2+lam/4*((np.expm1(1j*zz)-1j*zz)@ww))
        def cdf(x):return .5-np.sum(tw*np.imag(np.exp(-1j*tt*x)*phi)/tt)/np.pi
        return [float(brentq(lambda x:cdf(x)-q,-40,40,xtol=2e-10)) for q in probs]


def poisson_constants():
    def F(x):return np.logaddexp(0.,-x)*np.logaddexp(0.,x)
    return {str(k):2*quad(lambda x:F(x)**k,0,100,epsabs=1e-13)[0] for k in [1,2,3,4]}


def main():
    out={'integrals':poisson_constants(),'diagnostics':[],'simulation':[]}
    for phi in [.03,.01,.003,.001,.0003]:
        obj=ExactScore(phi)
        row=obj.diagnostics()
        out['diagnostics'].append(row)
        print(json.dumps(row),flush=True)
        (ROOT/'score_limit_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    obj=ExactScore(.001,gh=18,gl=48)
    for n,reps in [(100,20000),(1000,12000),(10000,4000)]:
        row=obj.simulate(n,reps,seed=629001+n)
        out['simulation'].append(row)
        print(json.dumps(row),flush=True)
        (ROOT/'score_limit_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
