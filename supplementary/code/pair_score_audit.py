import json
import numpy as np
from scipy.special import expit,betaln,logsumexp
from scipy.interpolate import RectBivariateSpline
from scipy.stats import norm
from score_limit_audit import ExactScore,ROOT,IH


class PairScore(ExactScore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        p,d,z,ph=self.p,self.d,self.z,self.ph
        eu,ev,C,J,R,ca,cb,sef=self.const
        x=self.x[:,None]
        ly,ln=-np.logaddexp(0.,-x),-np.logaddexp(0.,x)
        lf=(p*ph)[None,:]*ly+((1-p)*ph)[None,:]*ln-betaln(p*ph,(1-p)*ph)[None,:]
        lm=logsumexp(lf+np.log(self.weights)[None,:],axis=1)
        F=np.exp(lf-lm[:,None])
        A=d[None,:]*(ph[None,:]*x-eu[None,:]+ev[None,:])
        B=(p*ph)[None,:]*ly+((1-p)*ph)[None,:]*ln-(p*eu+(1-p)*ev)[None,:]
        AF=A*F;BF=B*F
        VF=(A*A+z[None,:]*A-C[None,:])*F
        WF=(B*B+B-J[None,:])*F
        KF=(A*B+A-R[None,:])*F
        mul=lambda X,Y:(X*self.weights[None,:])@Y.T
        P=mul(F,F)
        DA=mul(AF,F)+mul(F,AF)
        DB=mul(BF,F)+mul(F,BF)
        DV=(mul(VF,F)+mul(F,VF))/2+mul(AF,AF)
        DW=(mul(WF,F)+mul(F,WF))/2+mul(BF,BF)
        DK=mul(KF,F)+mul(F,KF)+mul(AF,BF)+mul(BF,AF)
        deriv=np.stack([DA,DB,DV,DW,DK],axis=-1)
        measure=self.mass[:,None]*self.mass[None,:]
        scores=deriv/P[:,:,None]
        mat=np.einsum('ij,ija,ijb->ab',measure*P,scores,scores)
        co=np.linalg.solve(mat[:4,:4],mat[:4,4])
        eff=scores[:,:,4]-scores[:,:,:4]@co
        self.pair_info=float(np.sum(measure*P*eff*eff))
        self.pair_mass=float(np.sum(measure*P))
        self.pair_mean=float(np.sum(measure*P*eff))
        self.pair_coeff=co
        self.pair_mat=mat
        self.pair_interp=RectBivariateSpline(self.x,self.x,eff,kx=3,ky=3,s=0)

    def efficient_pair(self,x):
        x=np.asarray(x);p,d,z,ph=self.p,self.d,self.z,self.ph
        eu,ev,C,J,R,ca,cb,sef=self.const
        out=[]
        for start in range(0,len(x),200):
            xx=x[start:start+200,:,None]
            ly,ln=-np.logaddexp(0.,-xx),-np.logaddexp(0.,xx)
            lf=(p*ph)[None,None,:]*ly+((1-p)*ph)[None,None,:]*ln-betaln(p*ph,(1-p)*ph)[None,None,:]
            lf=lf.sum(axis=1)
            lm=logsumexp(lf+np.log(self.weights)[None,:],axis=1)
            post=np.exp(lf+np.log(self.weights)[None,:]-lm[:,None])
            A=d[None,None,:]*(ph[None,None,:]*xx-eu[None,None,:]+ev[None,None,:])
            B=(p*ph)[None,None,:]*ly+((1-p)*ph)[None,None,:]*ln-(p*eu+(1-p)*ev)[None,None,:]
            AA,BB=A.sum(axis=1),B.sum(axis=1)
            V=(AA*AA+z[None,:]*AA-2*C[None,:])/2
            W=(BB*BB+BB-2*J[None,:])/2
            K=AA*BB+AA-2*R[None,:]
            S=np.stack([np.sum(post*t,axis=1) for t in [AA,BB,V,W,K]],axis=1)
            out.extend((S[:,4]-S[:,:4]@self.pair_coeff).tolist())
        return np.asarray(out)

    def simulate_pair(self,n,reps,seed):
        rng=np.random.default_rng(seed)
        L=np.linalg.cholesky(np.array([[self.v,self.kappa],[self.kappa,self.w]]))
        vals=[];check=None
        for lo in range(0,reps,100):
            batch=min(100,reps-lo)
            bc=rng.standard_normal((batch,n,2))@L.T
            p=expit(self.alpha+bc[:,:,0]);ph=self.phi*np.exp(bc[:,:,1])
            aa,bb=np.repeat((p*ph)[:,:,None],2,axis=2),np.repeat(((1-p)*ph)[:,:,None],2,axis=2)
            lg1=np.log(rng.gamma(aa+1))+np.log(rng.random(aa.shape))/aa
            lg2=np.log(rng.gamma(bb+1))+np.log(rng.random(bb.shape))/bb
            x=lg1-lg2;flat=x.reshape(-1,2)
            sc=self.pair_interp.ev(flat[:,0],flat[:,1]).reshape(batch,n)
            bad=(np.max(abs(x),axis=2)>self.x[-1])
            if bad.any():sc[bad]=self.efficient_pair(x[bad])
            vals.extend(((sc-self.pair_mean).sum(axis=1)/np.sqrt(n*self.pair_info)).tolist())
            if check is None:check=flat[:500]
        vals=np.asarray(vals)
        diff=(self.efficient_pair(check)-self.pair_interp.ev(check[:,0],check[:,1]))/np.sqrt(self.pair_info)
        reject=float(np.mean(abs(vals)>norm.ppf(.975)))
        return dict(phi=self.phi,n_clusters=n,n_records=2*n,reps=reps,seed=seed,
            info_per_cluster=self.pair_info,mass=self.pair_mass,score_mean=self.pair_mean,
            mean=float(vals.mean()),variance=float(vals.var()),oracle_score_rejection=reject,
            mcse=float(np.sqrt(reject*(1-reject)/reps)),
            left_tail=float(np.mean(vals<norm.ppf(.025))),right_tail=float(np.mean(vals>norm.ppf(.975))),
            quantiles=np.quantile(vals,[.025,.5,.975]).tolist(),
            interpolation_max_normalized_error=float(max(abs(diff))),interpolation_rms_error=float(np.sqrt(np.mean(diff**2))))


if __name__=='__main__':
    tau=.01;D0=9*tau*tau/16+IH/64;a=2*np.sqrt(D0*(1.5/.5-1)/1.5)
    obj=PairScore(.0001,tau=tau,a=a,gh=12,gl=24)
    ref=PairScore(.0001,tau=tau,a=a,gh=18,gl=36)
    out={'I_default':obj.pair_info,'I_refined':ref.pair_info,'relative_difference':ref.pair_info/obj.pair_info-1,
        'matrix':obj.pair_mat.tolist(),'simulations':[]}
    print('INFORMATION',out,flush=True)
    for n,reps in [(100,20000),(500,16000),(5000,4000)]:
        row=obj.simulate_pair(n,reps,seed=84900+n)
        out['simulations'].append(row);print('PAIR',row,flush=True)
        (ROOT/'pair_score_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
