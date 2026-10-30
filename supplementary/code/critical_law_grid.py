import json
import numpy as np
from score_limit_audit import ExactScore, ROOT, IH


def main():
    hs = [(1., 0., 1.), (1., .4, 1.2), (.7, -.25, 1.4)]
    taus = [.03, .3]
    signs = [-1, 0, 1]
    phis = [.001, .0001]
    ts = np.array([.5, 1., 1.5, 2., 3.])
    rows = []
    for h in hs:
        for tau in taus:
            D0 = IH/64 + tau*tau*(9*h[0]**2/16+3*h[1]**2)
            for direction in signs:
                m = direction*np.sqrt(D0/1.5)
                a = 2*(tau*h[1]-m)
                for phi in phis:
                    obj = ExactScore(phi, tau=tau, a=a, H=h, gh=18, gl=48)
                    row = obj.diagnostics(ns=())
                    row['direction'] = direction
                    row['relative_information_error'] = obj.info/obj.parameters()['info']-1
                    row['cf_checks'] = []
                    for lam in [.1, 1.]:
                        n = round(lam/phi)
                        actual = obj.characteristic(n, ts)
                        hybrid = obj.limiting_characteristic(n*phi, ts)
                        row['cf_checks'].append(dict(n=n, n_phi=n*phi,
                            hybrid_error=float(max(abs(actual-hybrid))),
                            normal_error=float(max(abs(actual-np.exp(-ts*ts/2))))))
                    rows.append(row)
                    print(json.dumps(dict(H=h,tau=tau,direction=direction,phi=phi,
                        relative_information_error=row['relative_information_error'],
                        cf_checks=row['cf_checks'])), flush=True)
                    (ROOT/'critical_law_grid.json').write_text(json.dumps(dict(
                        ts=ts.tolist(),rows=rows),indent=2),encoding='utf-8')
    refinements=[]
    for idx in [7,19,31]:
        row=rows[idx]
        obj=ExactScore(row['phi'],tau=row['tau'],a=row['a'],H=row['H'],gh=24,gl=64)
        refinements.append(dict(row_index=idx,information_relative_change=obj.info/row['info']-1,
            cf_max_change=float(max(abs(obj.characteristic(1000,ts)-
                ExactScore(row['phi'],tau=row['tau'],a=row['a'],H=row['H'],gh=18,gl=48).characteristic(1000,ts))))))
    (ROOT/'critical_law_grid.json').write_text(json.dumps(dict(ts=ts.tolist(),rows=rows,
        refinements=refinements),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
