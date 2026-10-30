import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,time,math
import numpy as np
from scipy.special import expit,betaln,betainc,digamma,polygamma,logsumexp
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]

def latent(alpha,v,w,kappa,gh):
    if v==0:
        assert kappa==0;xx,ww=hermgauss(gh);b=np.zeros(gh);c=np.sqrt(2*w)*xx;weights=ww/np.sqrt(np.pi)
    else:
        xx,ww=hermgauss(gh);zz=np.stack(np.meshgrid(xx*np.sqrt(2),xx*np.sqrt(2),indexing='ij'),-1).reshape(-1,2)
        b=np.sqrt(v)*zz[:,0];c=kappa/np.sqrt(v)*zz[:,0]+np.sqrt(max(w-kappa*kappa/v,0))*zz[:,1]
        weights=(ww[:,None]*ww[None,:]/np.pi).ravel()
    eta=alpha+b;p=expit(eta);q=expit(-eta)
    return p,q,c,weights

def information(alpha=-1.,phi=.1,v=.1,w=.1,kappa=0.,model='AL',gh=24,gl=7,score_method='latent'):
    tic=time.time();p,q,c,weights=latent(alpha,v,w,kappa,gh);d=p*q;z=q-p;ph=np.exp(c)*(phi if model=='Beta' else 1.)
    a=p*ph;b=q*ph;logB=betaln(a,b)
    if model=='AL':C=1-2*d;J=np.ones_like(d);R=z;eu=ev=None
    else:
        eu=ph*(digamma(a)-digamma(ph));ev=ph*(digamma(b)-digamma(ph))
        ta=ph*ph*polygamma(1,a);tb=ph*ph*polygamma(1,b);tt=ph*ph*polygamma(1,ph)
        C=d*d*(ta+tb);J=p*p*ta+q*q*tb-tt;R=d*(p*ta-q*tb)
    def tail(distance,positive):
        if model=='AL':return float(np.dot(weights,(p if positive else q)*np.exp(-ph*(q if positive else p)*distance)))
        aa,bb=(b,a) if positive else (a,b)
        if distance<700:return float(np.dot(weights,betainc(aa,bb,expit(-distance))))
        logcdf=-aa*distance-np.log(aa)-logB
        return float(np.dot(weights,np.exp(np.minimum(logcdf,0))))
    Fzero=tail(0.,False);surzero=tail(0.,True);assert abs(Fzero+surzero-1)<1e-8
    gx,gw=leggauss(gl);xdata=[];mass=[]
    for positive,branch in [(False,Fzero),(True,surzero)]:
        for lo,hi in zip([-36,-20,-12,-8,-4,0,4,8,12,20],[-20,-12,-8,-4,0,4,8,12,20,36]):
            for x,weight in zip(gx,gw):
                t=(lo+hi)/2+(hi-lo)/2*x;u=expit(t);target=branch*u
                bound=1.
                while tail(bound,positive)>target:
                    bound*=3
                    if bound>1e18:raise ArithmeticError('marginal tail bracket unresolved')
                dist=brentq(lambda val:tail(val,positive)-target,0.,bound,xtol=2e-12,rtol=3e-13)
                xdata.append(dist if positive else -dist)
                mass.append(branch*expit(t)*expit(-t)*(hi-lo)/2*weight)
    x=np.array(xdata)[:,None];quadr=np.array(mass);ax=abs(x);sg=np.sign(x)
    if model=='AL':
        U=ph[None,:]*np.where(sg>0,q[None,:],p[None,:])*ax
        logpdf=np.log(ph*d)[None,:]-U
        A=z[None,:]+sg*ph[None,:]*d[None,:]*ax;B=1-U
    else:
        ly=-np.logaddexp(0.,-x);ln=-np.logaddexp(0.,x)
        logpdf=a[None,:]*ly+b[None,:]*ln-logB[None,:]
        A=d[None,:]*(ph[None,:]*x-eu[None,:]+ev[None,:])
        B=a[None,:]*ly+b[None,:]*ln-p[None,:]*eu[None,:]-q[None,:]*ev[None,:]
    Haa=z[None,:]*A-C[None,:];Hcc=B-J[None,:];Hac=A-R[None,:]
    lm=logsumexp(logpdf+np.log(weights)[None,:],axis=1)
    F=np.exp(logpdf-lm[:,None]);AF=A*F;BF=B*F
    VV=(A*A+Haa)*F;WW=(B*B+Hcc)*F;KK=(A*B+Hac)*F
    S=np.stack([AF@weights,BF@weights,.5*(VV@weights),.5*(WW@weights),KK@weights],-1)
    I1=S.T@(quadr[:,None]*S)
    mul=lambda X,Y:(X*weights[None,:])@Y.T
    P=mul(F,F);DA=mul(AF,F)+mul(F,AF);DB=mul(BF,F)+mul(F,BF)
    DV=.5*(mul(VV,F)+mul(F,VV))+mul(AF,AF)
    DW=.5*(mul(WW,F)+mul(F,WW))+mul(BF,BF)
    DK=mul(KK,F)+mul(F,KK)+mul(AF,BF)+mul(BF,AF)
    deriv=np.stack([DA,DB,DV,DW,DK],-1);Q=quadr[:,None]*quadr[None,:];positive=P>1e-300
    G=deriv[positive]/np.sqrt(P[positive])[:,None];I2=G.T@(Q[positive,None]*G)
    def result(I,r):
        eig=np.linalg.eigvalsh(I)
        try:
            coeff=np.linalg.solve(I[:4,:4],I[:4,4]);eff=float(I[4,4]-I[4,:4]@coeff)
        except np.linalg.LinAlgError:eff=None
        if eff is not None and eff<1e-11 and model=='AL' and v==0 and w==0:eff=None
        oracle=1/(v*w+kappa*kappa) if v*w>0 else None
        return dict(r=r,efficient_kappa=eff,eigenvalues=eig.tolist(),matrix=I.tolist(),
                    latent_Gaussian_upper=oracle,ratio_to_oracle=(eff/oracle if eff is not None and oracle else None))
    mean2=np.sum(Q[...,None]*deriv,axis=(0,1))
    price_information=[result(I1,1),result(I2,2)]
    if score_method=='latent' and v*w-kappa*kappa>0:
        bb=np.log(p/q)-alpha;cc=c;inv=np.linalg.inv(np.array([[v,kappa],[kappa,w]]))
        loc=np.stack([bb,cc],-1)@inv
        gs=np.stack([loc[:,0],loc[:,1],.5*(loc[:,0]**2-inv[0,0]),.5*(loc[:,1]**2-inv[1,1]),loc[:,0]*loc[:,1]-inv[0,1]],-1)
        S=(F*weights[None,:])@gs;I1=S.T@(quadr[:,None]*S)
        deriv=np.stack([(F*(weights*gs[:,j])[None,:])@F.T for j in range(5)],-1)
        G=deriv[positive]/np.sqrt(P[positive])[:,None];I2=G.T@(Q[positive,None]*G)
        mean2=np.sum(Q[...,None]*deriv,axis=(0,1))
    out=dict(alpha=alpha,phi=phi,v=v,w=w,kappa=kappa,model=model,gh=gh,gl=gl,data_nodes=len(quadr),
       univariate_mass=float(quadr.sum()),joint_mass=float(np.sum(P*Q)),mean_score_r1=(quadr@S).tolist(),mean_score_r2=mean2.tolist(),
       information=[result(I1,1),result(I2,2)],price_information=price_information,
       score_method=score_method,seconds=time.time()-tic)
    return out

if __name__=='__main__':
    rows=[]
    for model,phi,s in [('AL',1.,.001),('AL',1.,.1),('AL',1.,1.),('Beta',5.,.1)]:
        out=information(model=model,phi=phi,v=s,w=s,gh=24,gl=7);rows.append(out)
        print(json.dumps({k:v for k,v in out.items() if k not in ['information','mean_score_r1','mean_score_r2']} | {'I':[r['efficient_kappa'] for r in out['information']]}),flush=True)
    (ROOT/'results/matrix_information_pilot.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
