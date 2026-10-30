from pathlib import Path
from functools import lru_cache
import math,json
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
def information(mu,phi,r):
    p=mp.mpf(mu);f=mp.mpf(phi);d=p*(1-p);z=1-2*p;a=p*f;b=(1-p)*f;h=f*d
    def cumulant(j,k):
        n=j+k
        if n<=1:return mp.mpf(0)
        value=h**j*(a**k*mp.polygamma(n-1,a)+(-1)**j*b**k*mp.polygamma(n-1,b))
        if j==0:value-=f**k*mp.polygamma(n-1,f)
        return r*value
    @lru_cache(None)
    def moment(j,k):
        if j+k==0:return mp.mpf(1)
        if j:
            return sum(mp.mpf(math.comb(j-1,u)*math.comb(k,v))*cumulant(u+1,v)*moment(j-1-u,k-v)
                       for u in range(j) for v in range(k+1))
        return sum(mp.mpf(math.comb(k-1,v))*cumulant(0,v+1)*moment(0,k-1-v) for v in range(k))
    C=cumulant(2,0)/r;J=cumulant(0,2)/r;R=cumulant(1,1)/r
    polys=[{(1,0):mp.mpf(1)},{(0,1):mp.mpf(1)},
       {(2,0):mp.mpf('.5'),(1,0):z/2,(0,0):-r*C/2},
       {(0,2):mp.mpf('.5'),(0,1):mp.mpf('.5'),(0,0):-r*J/2},
       {(1,1):mp.mpf(1),(1,0):mp.mpf(1),(0,0):-r*R}]
    I=mp.matrix(5)
    for j,P in enumerate(polys):
      for k,Q in enumerate(polys):
        I[j,k]=sum(c1*c2*moment(x1+x2,y1+y2) for (x1,y1),c1 in P.items() for (x2,y2),c2 in Q.items())
    eff=1/(I**-1)[4,4]
    mu_u=mp.digamma(a)-mp.digamma(f);mu_v=mp.digamma(b)-mp.digamma(f)
    ca=f*(p*mu_u-(1-p)*mu_v);cb=d*(1-f*(mu_u+mu_v));hv=mp.matrix([ca,cb,2,-2*d,-z])
    return I,eff,hv

def identity_checks():
    rows=[]
    for p0,f0,y0 in [('.2','.02','.3'),('.7','2','.05'),('.5','.2','.9')]:
        p,f,y=map(mp.mpf,[p0,f0,y0]);d=p*(1-p);z=1-2*p;a=p*f;b=(1-p)*f;h=f*d
        eu=mp.digamma(a)-mp.digamma(f);ev=mp.digamma(b)-mp.digamma(f);u=mp.log(y);v=mp.log(1-y)
        A=h*((u-eu)-(v-ev));B=a*(u-eu)+b*(v-ev)
        C=h*h*(mp.polygamma(1,a)+mp.polygamma(1,b));J=a*a*mp.polygamma(1,a)+b*b*mp.polygamma(1,b)-f*f*mp.polygamma(1,f)
        R=h*(a*mp.polygamma(1,a)-b*mp.polygamma(1,b));V=(A*A+z*A-C)/2;W=(B*B+B-J)/2;K=A*B+A-R
        ca=f*(p*eu-(1-p)*ev);cb=d*(1-f*(eu+ev));left=ca*A+cb*B+2*V-2*d*W-z*K
        right=-f*f*d*(u*v-(eu*ev-mp.polygamma(1,f)))
        rows.append(dict(mu=float(p),phi=float(f),y=float(y),absolute_error=mp.nstr(abs(left-right),8)))
    return rows
if __name__=='__main__':
    mp.mp.dps=100;Ih=16*mp.zeta(5)-8*mp.zeta(2)*mp.zeta(3);rows=[]
    for p0 in ['.2','.35','.5']:
      for r in [1,2,6]:
       for f0 in ['.1','.02','.002','.00002']:
        I,eff,hv=information(p0,f0,r);p=mp.mpf(p0);f=mp.mpf(f0);d=p*(1-p);z=1-2*p
        pred=f**5*d**3*Ih/z**2 if z and r==1 else None
        row=dict(mu=float(p),phi=float(f),r=r,efficient_kappa=mp.nstr(eff,30),
          prediction=mp.nstr(pred,30) if pred else None,ratio=float(eff/pred) if pred else None)
        rows.append(row)
    cross=[]
    for lam in ['0','.1','1','3']:
      for f0 in ['.02','.002','.00002']:
        f=mp.mpf(f0);z=mp.mpf(lam)*f**mp.mpf('2.5');p=(1-z)/2;_,eff,_=information(p,f,1)
        pred=1/(mp.mpf(2)/3+64*mp.mpf(lam)**2/Ih)
        cross.append(dict(lambda_value=float(lam),phi=float(f),information=float(eff),limit=float(pred),ratio=float(eff/pred)))
    out=dict(identity=identity_checks(),fixed_means=rows,crossover=cross)
    (ROOT/'results/general_information_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for row in rows:
        if row['phi']==.00002:print(json.dumps(row),flush=True)
    for row in cross:
        if row['phi']==.00002:print(json.dumps(row),flush=True)
