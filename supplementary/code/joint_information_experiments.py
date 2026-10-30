import json
import sys
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import expit
from joint_information import information, IH, ROOT

def symbolic_calculations():
    import sympy as s
    p, ph, ly, lz, my, mz, ta, tb, tf = s.symbols('p ph ly lz my mz ta tb tf')
    d = p * (1 - p)
    z = 1 - 2 * p
    A = ph * d * (ly - lz - my + mz)
    B = ph * (p * (ly - my) + (1 - p) * (lz - mz))
    V = (A * A + z * A - ph ** 2 * d * d * (ta + tb)) / 2
    W = (B * B + B - ph ** 2 * (p * p * ta + (1 - p) ** 2 * tb - tf)) / 2
    K = A * (B + 1) - ph ** 2 * d * (p * ta - (1 - p) * tb)
    ca = ph * (p * my - (1 - p) * mz)
    cb = d * (1 - ph * (my + mz))
    residual = s.factor(ca * A + cb * B + 2 * V - 2 * d * W - z * K + ph ** 2 * d * (ly * lz - my * mz + tf))
    u, sg = s.symbols('u sg')

    def avg(expr):
        return s.expand(sum((co * s.factorial(i) * (0 if j % 2 else 1) for (i, j), co in s.Poly(s.expand(expr), u, sg).terms())))
    A = sg * u / 2
    B = 1 - u
    V = (u * u - 2) / 8
    K = A * (2 - u)
    e = A * (3 - u)
    R = (u ** 3 - 9 * u * u + 18 * u - 6) / 8
    Q = sg * (-u ** 3 + 8 * u * u - 12 * u) / 4
    gs = s.Matrix([[avg(x * y) for y in [A, B, V, K]] for x in [A, B, V, K]])
    res = s.Matrix([[avg(x * y) for y in [e, R, Q]] for x in [e, R, Q]])
    curves = []
    t = s.symbols('t')
    for sign in [-1, 1]:
        l1 = 1 - 2 * p + d * t
        l2 = 1 + (p - s.Rational(1 + sign, 2)) * t
        D1 = lambda g: s.expand(d * s.diff(g, p) + l1 * g)
        D2 = lambda g: s.expand(t * s.diff(g, t) + l2 * g)
        h = s.Rational(3, 4) * D2(1) + D1(D1(1)) - D2(D2(1)) / 4
        vals = [s.factor(x.subs(p, s.Rational(1, 2)).subs(t, 2 * sign * u)) for x in [D1(D1(h)) / 2, D1(D2(h)), D2(D2(h)) / 2]]
        curves.append(dict(sign=sign, expressions=[str(x) for x in vals]))
    u1, u2, s1, s2 = s.symbols('u1 u2 s1 s2')

    def avg2(expr):
        return s.expand(sum((co * s.factorial(i) * s.factorial(j) * (0 if k % 2 or l % 2 else 1) for (i, j, k, l), co in s.Poly(s.expand(expr), u1, u2, s1, s2).terms())))
    A = (s1 * u1 + s2 * u2) / 2
    B = 2 - u1 - u2
    V = (A * A - 1) / 2
    W = (B * B + B - 2) / 2
    K = A * (B + 1)
    mat = s.Matrix([[avg2(x * y) for y in [A, B, V, W, K]] for x in [A, B, V, W, K]])
    out = dict(weak_identity_residual=str(residual), strong_gram=[[str(v) for v in row] for row in gs.tolist()], residual_gram=[[str(v) for v in row] for row in res.tolist()], heat_curvatures=curves, replicated_information=[[str(v) for v in row] for row in mat.tolist()], determinant=str(mat.det()), nuisance_determinant=str(mat[:4, :4].det()), replicated_eff_fourth_moment=str(avg2((K + A) ** 4)))
    return out

def vertex_checks():
    from general_information import information as exact_moments
    rows=[]
    with mp.workdps(90):
        for phi in [.1,.01,.001]:
            for t in [0,1,2]:
                alpha=2*t*np.sqrt(IH*phi**5/64)
                row=information(phi,0,alpha)
                _,exact,_=exact_moments(mp.mpf(float(expit(alpha))),mp.mpf(phi),1)
                rows.append(dict(phi=phi,alpha=alpha,exact=str(exact),quadrature=row['efficient_information'],
                    relative_difference=float(mp.mpf(row['efficient_information'])/exact-1)))
    return rows

def refinements():
    rows=[]
    for phi in [.03,.001,.0001]:
        for H,ratio,t in [((1,0,1),.15,1),((1,.5,1),1,-1),((1,-.75,1),.15,1),((.4,.3,2),1,0)]:
            s=ratio*phi**2.5
            D=9*(s*H[0])**2/16+3*(s*H[1])**2+IH*phi**5/64
            alpha=2*s*H[1]+2*t*np.sqrt(D)
            a=information(phi,s,alpha,H=H)
            b=information(phi,s,alpha,H=H,gh=26,gl=44,central_cut=64,tail_cut=150)
            rows.append(dict(phi=phi,H=H,ratio=ratio,t=t,default=a,refined=b,
                relative_difference=b['efficient_information']/a['efficient_information']-1))
    return rows

def examples():
    balanced=[]
    for phi in [.1,.03,.01,.003,.001,.0003,.0001]:
        s=np.sqrt(IH)*phi**2.5/6
        alpha=1.5*s
        row=information(phi,s,alpha,gh=22,gl=36)
        row.update(joint_limit=6/7,AL_only_limit=.6,vertex_only_limit=.6)
        balanced.append(row)
    shifted=[]
    for H in [(1,.5,1),(1,-.75,1)]:
        phi=.001;s=10*phi**2.5
        for alpha in [0,2*s*H[1]]:
            shifted.append(information(phi,s,alpha,H=H,gh=22,gl=36))
    return dict(balanced=balanced,shifted=shifted)

def fixed_mean_checks():
    rows=[]
    for alpha in [-1.,1.5]:
        p=expit(alpha);d=p*(1-p);z=1-2*p
        M=np.array([[3-8*d-4*d*d,z*(3+2*d)],[z*(3+2*d),3+4*d]])
        for H in [(1,0,1),(1,.5,1)]:
            h=np.array(H[:2]);CH=12*d*d*(h@M@h)
            for phi in [.03,.01,.003,.001]:
                for ratio in [.03,.15,1.]:
                    s=ratio*phi**2.5
                    row=information(phi,s,alpha,H=H,gh=22,gl=36)
                    pred=(CH*s*s+d**3*IH*phi**5)/(z*z)
                    row['fixed_mean_prediction']=pred
                    row['fixed_mean_relative_error']=row['efficient_information_gram']/pred-1
                    row['qr_gram_relative_difference']=row['efficient_information']/row['efficient_information_gram']-1
                    rows.append(row)
    return rows

def main():
    out=dict(symbolic=symbolic_calculations(),vertex=vertex_checks(),refinements=refinements(),examples=examples(),fixed_mean=fixed_mean_checks())
    out['summary']=dict(
        vertex_max_relative_difference=max(abs(x['relative_difference']) for x in out['vertex']),
        refinement_max_relative_difference=max(abs(x['relative_difference']) for x in out['refinements']),
        refinement_max_relative_difference_phi_at_least_001=max(abs(x['relative_difference']) for x in out['refinements'] if x['phi']>=.001),
        max_mass_error=max(abs(x['refined']['mass']-1) for x in out['refinements']),
        max_mean_score_error=max(abs(y) for x in out['refinements'] for y in x['refined']['score_mean'][:5]))
    (ROOT/'validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['summary'],indent=2))
    for row in out['examples']['balanced']:
        print('balanced',row['phi'],row['efficient_information'],row['relative_error'])
    for row in out['examples']['shifted']:
        print('shifted',row['H'],row['alpha'],row['efficient_information'],row['prediction'])

if __name__=='__main__':
    main()
