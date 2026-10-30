import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, betaln, digamma, polygamma, logsumexp, zeta

ROOT = Path(__file__).resolve().parents[1] / 'results'
IH = float(16*zeta(5, 1)-8*zeta(2, 1)*zeta(3, 1))

def conditional_constants(p, ph):
    q = 1-p
    a, b = p*ph, q*ph
    eu = ph*(digamma(a)-digamma(ph))
    ev = ph*(digamma(b)-digamma(ph))
    if np.max(ph) < .65:
        du = np.zeros_like(p)
        dv = np.zeros_like(p)
        correction = np.zeros_like(p)
        for n in range(1, 95):
            coeff = (-1)**(n+1)*zeta(n+1, 1)*ph**(n+1)
            du += coeff*(p**n-1)
            dv += coeff*(q**n-1)
            correction += n*coeff
        u0, v0 = -q/p, -p/q
        scaled_ef = u0*dv+v0*du+du*dv-correction
        eu, ev = u0+du, v0+dv
    else:
        import mpmath as mp
        with mp.workdps(50):
            scaled_ef = np.array([
                float(f*f*((mp.digamma(pp*f)-mp.digamma(f))*
                            (mp.digamma((1-pp)*f)-mp.digamma(f))-
                            mp.polygamma(1, f)))
                for pp, f in zip(map(mp.mpf, p), map(mp.mpf, ph))])
    ta, tb, tt = [ph*ph*polygamma(1, x) for x in (a,b,ph)]
    d = p*q
    C = d*d*(ta+tb)
    J = p*p*ta+q*q*tb-tt
    R = d*(p*ta-q*tb)
    ca, cb = p*eu-q*ev, d*(1-eu-ev)
    return eu, ev, C, J, R, ca, cb, scaled_ef

def data_quadrature(phi, gl=32, central_cut=48, tail_cut=100):
    gx, gw = leggauss(gl)
    endpoints = [0,1,2,4,8,16,32,central_cut]
    endpoints = sorted(set(x for x in endpoints if x <= central_cut))
    scaled_start = phi*central_cut/2
    tail = sorted(set([scaled_start]+[t for t in [.01,.025,.05,.1,.25,.5,1,2,4,8,16,32,64,tail_cut] if t>scaled_start]))
    pieces = [(lo, hi) for lo,hi in zip(endpoints[:-1],endpoints[1:])]
    pieces += [(2*lo/phi, 2*hi/phi) for lo,hi in zip(tail[:-1],tail[1:])]
    positive = np.concatenate([(lo+hi)/2+(hi-lo)*gx/2 for lo,hi in pieces])
    weights = np.concatenate([(hi-lo)*gw/2 for lo,hi in pieces])
    return np.concatenate([-positive[::-1],positive]), np.concatenate([weights[::-1],weights])

def information(phi, s, alpha, gh=14, gl=24, central_cut=48, tail_cut=100, H=(1.,0.,1.)):
    tic = time.time()
    h11,h12,h22=H
    if h11 <= 0 or h11*h22-h12*h12 <= 0:
        raise ValueError('H must be positive definite')
    v,kappa,w=s*h11,s*h12,s*h22
    if s == 0:
        b, c, weights = np.zeros(1), np.zeros(1), np.ones(1)
    else:
        hn, hw = hermgauss(gh)
        z1,z2 = np.meshgrid(np.sqrt(2)*hn,np.sqrt(2)*hn,indexing='ij')
        b=np.sqrt(v)*z1.ravel()
        c=np.sqrt(s)*(h12/np.sqrt(h11)*z1+np.sqrt(h22-h12*h12/h11)*z2).ravel()
        weights = (hw[:,None]*hw[None,:]/np.pi).ravel()
    p = expit(alpha+b)
    q = 1-p
    d, z = p*q, q-p
    ph = phi*np.exp(c)
    eu,ev,C,J,R,ca,cb,scaled_ef = conditional_constants(p,ph)
    p0 = expit(alpha)
    d0, z0 = p0*(1-p0),1-2*p0
    ca0,cb0 = [float(x[0]) for x in conditional_constants(np.array([p0]),np.array([phi]))[5:7]]
    x, dx = data_quadrature(phi,gl,central_cut,tail_cut)
    matrices=[]
    for start in range(0,len(x),400):
        xx=x[start:start+400,None]
        ly, ln = -np.logaddexp(0.,-xx), -np.logaddexp(0.,xx)
        logf = (p*ph)[None,:]*ly+(q*ph)[None,:]*ln-betaln(p*ph,q*ph)[None,:]
        lm=logsumexp(logf+np.log(weights)[None,:],axis=1)
        posterior=np.exp(logf+np.log(weights)[None,:]-lm[:,None])
        A=d[None,:]*(ph[None,:]*xx-eu[None,:]+ev[None,:])
        B=(p*ph)[None,:]*ly+(q*ph)[None,:]*ln-(p*eu+q*ev)[None,:]
        V=.5*(A*A+z[None,:]*A-C[None,:])
        W=.5*(B*B+B-J[None,:])
        K=A*B+A-R[None,:]
        F=-d[None,:]*(ph[None,:]**2*ly*ln-scaled_ef[None,:])
        F+=(ca0-ca)[None,:]*A+(cb0-cb)[None,:]*B
        F+=2*(d-d0)[None,:]*W+(z-z0)[None,:]*K
        S=np.stack([np.sum(posterior*t,axis=1) for t in [A,B,V,W,K,F]],axis=1)
        matrices.append(np.column_stack([np.exp(lm)*dx[start:start+400], S]))
    matrix=np.concatenate(matrices)
    measure,S=matrix[:,0],matrix[:,1:]
    D=9*v*v/16+3*kappa*kappa+IH*phi**5/64
    scale=np.sqrt(D)
    T=S[:,[0,1,2]]
    nuisance=np.column_stack([T,(S[:,5]+z0*S[:,4])/scale])
    weighted=nuisance*np.sqrt(measure)[:,None]
    target=S[:,4]*np.sqrt(measure)
    Q,qr=np.linalg.qr(weighted,mode='reduced')
    eff=float(np.sum((target-Q@(Q.T@target))**2))
    Qt,_=np.linalg.qr(T*np.sqrt(measure)[:,None],mode='reduced')
    def residual(j):
        t=S[:,j]*np.sqrt(measure)
        return t-Qt@(Qt.T@t)
    er,fr=residual(4),residual(5)
    Fnorm=float(fr@fr)
    cross=float(fr@er)
    info=S[:,:5].T@(measure[:,None]*S[:,:5])
    directweak=ca0*S[:,0]+cb0*S[:,1]+2*S[:,2]-2*d0*S[:,3]-z0*S[:,4]
    residual_four_norm=float(np.sum((fr-cross/(er@er)*er)**2))
    efficient_information_gram=float((er@er)*residual_four_norm/np.sum((fr+z0*er)**2))
    result=dict(phi=phi,s=s,alpha=alpha,H=H,v=v,w=w,kappa=kappa,gh=gh,gl=gl,
        nodes=len(x),mass=float(measure.sum()),score_mean=(measure@S).tolist(),
        efficient_information=eff,efficient_information_gram=efficient_information_gram,
        prediction=1/(2/3+(kappa-alpha/2)**2/D),
        weak_norm=Fnorm,weak_norm_prediction=D+1.5*kappa*kappa,
        residual_four_norm=residual_four_norm,weak_ratio=residual_four_norm/D,
        residual_four_prediction=D,
        strong_norm=float(er@er),weak_strong_cross=cross,
        cross_normalized=cross/np.sqrt(Fnorm*(er@er)),
        weak_strong_cross_prediction=1.5*kappa,
        nuisance_rotation=z0+cross/(er@er),
        nuisance_rotation_prediction=kappa-alpha/2,
        stabilized_identity_error=float(np.max(np.abs(directweak-S[:,5]))),
        nuisance_condition=float(np.linalg.cond(qr)),
        information_matrix=info.tolist(),seconds=time.time()-tic)
    result['relative_error']=eff/result['prediction']-1
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='pilot.json')
    parser.add_argument('--gh',type=int,default=14)
    parser.add_argument('--gl',type=int,default=24)
    args=parser.parse_args()
    rows=[]
    for phi in [.1,.03,.01,.003,.001]:
        for ratio in [0,.03,.15,1,10]:
            s=ratio*phi**2.5
            D=9*s*s/16+IH*phi**5/64
            for t in [0,1,2]:
                alpha=2*t*np.sqrt(D)
                row=information(phi,s,alpha,gh=args.gh,gl=args.gl)
                row.update(s_phi_ratio=ratio,normalized_mean=t)
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ['information_matrix','score_mean']}),flush=True)
                (ROOT/args.output).write_text(json.dumps(dict(IH=IH,rows=rows),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
