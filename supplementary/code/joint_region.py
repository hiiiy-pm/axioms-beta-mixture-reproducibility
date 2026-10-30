import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
import json
import time
from itertools import product
import numpy as np
from scipy.special import zeta
from applicability_experiments import ScoreMap, simulate_logits, path_a, reference_statistics
from score_limit_audit import ExactScore, ROOT, IH
from finite_bulk import polynomials, WHITEN
from point_null_benchmark import quantile_resamples


class CommonReference:
    def __init__(self, n, reps=20000, seed=2026504000):
        self.n, self.reps, self.seed = n, reps, seed
        rng = np.random.default_rng(seed)
        self.strong, self.rr, self.qq = [np.empty(reps) for _ in range(3)]
        for start in range(0, reps, 100):
            end = min(start + 100, reps)
            u = rng.exponential(size=(end - start, n))
            sign = 2 * rng.integers(0, 2, size=(end - start, n)) - 1
            a, b, v, k, _, rr, qq = polynomials(u, sign)
            strong = np.column_stack([x.sum(axis=1) for x in [a, b, v, k]]) @ WHITEN / np.sqrt(n)
            self.strong[start:end] = np.sum(strong ** 2, axis=1)
            self.rr[start:end], self.qq[start:end] = rr.sum(axis=1) / np.sqrt(n), qq.sum(axis=1) / np.sqrt(n)
        self.poisson, self.critical = {}, {}

    def cutoff(self, obj):
        phi, vv, kk = obj.phi, obj.v / obj.phi ** 2.5, obj.kappa / obj.phi ** 2.5
        key = (phi, vv, kk)
        if key in self.critical:
            return self.critical[key]
        if phi not in self.poisson:
            rng = np.random.default_rng(self.seed + int(round(phi * 1e9)))
            lam = self.n * phi
            counts = rng.poisson(12 * lam, self.reps)
            x = rng.uniform(-24, 24, int(counts.sum()))
            f = np.logaddexp(0, -x) * np.logaddexp(0, x)
            jumps = np.bincount(np.repeat(np.arange(self.reps), counts), weights=f, minlength=self.reps)
            self.poisson[phi] = (jumps - lam * zeta(3, 1) / 2) / (4 * np.sqrt(lam))
        d = IH / 64 + 9 * vv ** 2 / 16 + 3 * kk ** 2
        values = self.strong + (vv * self.rr + kk * self.qq - self.poisson[phi]) ** 2 / d
        self.critical[key] = float(np.quantile(values, .95))
        return self.critical[key]


def make_object(phi0, phi_factor, alpha_factor, v_factor, w_factor, rho, gh=14, gl=32):
    phi = phi0 * phi_factor
    unit = phi0 ** 2.5
    v, w = .3 * v_factor * unit, .3 * w_factor * unit
    kappa = rho * np.sqrt(v * w)
    obj = ExactScore(phi, tau=.3 * unit / phi ** 2.5,
                     a=alpha_factor * unit / phi ** 2.5,
                     H=(v_factor, kappa / (.3 * unit), w_factor), gh=gh, gl=gl)
    return obj


def main():
    start = time.perf_counter()
    phi0, n, seed = .0001, 1000, 2026503000
    true = ExactScore(phi0, tau=.3, a=path_a(.3), gh=18, gl=48)
    x = simulate_logits(true, np.random.default_rng(seed), 1, n)[0]
    reference = CommonReference(n)
    cache = {}
    grids = []
    for name, phis, alphas, rhos in [
            ('coarse', np.linspace(.8, 1.2, 5), [-2., 0., 2.], [-.6, 0., .6]),
            ('refined', np.linspace(.8, 1.2, 9), [-2., -1., 0., 1., 2.], [-.6, -.3, 0., .3, .6])]:
        indices = []
        for factors in product(phis, alphas, [.5, 1., 2.], [.5, 1., 2.], rhos):
            key = tuple(round(float(t), 12) for t in factors)
            if key not in cache:
                obj = make_object(phi0, *key)
                score = ScoreMap(obj)
                total = score.exact(x).sum(axis=0) / np.sqrt(n)
                statistic = float(total @ total)
                critical = reference.cutoff(obj)
                cache[key] = dict(index=len(cache), factors=list(key), phi=obj.phi, alpha=obj.alpha,
                                  v=obj.v, w=obj.w, kappa=obj.kappa, statistic=statistic,
                                  critical=critical, margin=critical-statistic, accepted=statistic <= critical)
                if len(cache) % 100 == 0:
                    print('CANDIDATES', len(cache), 'SECONDS', round(time.perf_counter()-start, 1), flush=True)
            indices.append(cache[key]['index'])
        rows = {row['index']: row for row in cache.values()}
        accepted = [rows[i] for i in indices if rows[i]['accepted']]
        grids.append(dict(name=name, count=len(indices), accepted=len(accepted), indices=indices,
                          phi_grid=list(phis), alpha_grid=alphas, variance_factor_grid=[.5, 1., 2.], rho_grid=rhos,
                          kappa_hull=[min(r['kappa'] for r in accepted), max(r['kappa'] for r in accepted)] if accepted else None,
                          phi_hull=[min(r['phi'] for r in accepted), max(r['phi'] for r in accepted)] if accepted else None))
        print('GRID', name, grids[-1]['accepted'], grids[-1]['kappa_hull'], flush=True)
    rows = list(cache.values())
    witnesses = []
    unit = phi0 ** 2.5
    for endpoint in [-.36 * unit, .36 * unit]:
        candidates = [r for r in rows if abs(r['kappa']-endpoint) < 1e-20]
        row = max(candidates, key=lambda r:r['margin'])
        obj = make_object(phi0, *row['factors'], gh=24, gl=64)
        score = ScoreMap(obj)
        total = score.exact(x).sum(axis=0) / np.sqrt(n)
        refined_statistic = float(total @ total)
        verification_seed = 2026509000 + len(witnesses)
        values = reference_statistics(obj, n, 50000, verification_seed)
        draws = quantile_resamples(values, np.random.default_rng(verification_seed + 50000), reps=4000)
        critical = float(np.quantile(values, .95))
        ci = np.quantile(draws, [.025, .975]).tolist()
        witnesses.append(dict(candidate_index=row['index'], factors=row['factors'], kappa=row['kappa'],
                              statistic=row['statistic'], refined_statistic=refined_statistic,
                              statistic_difference=refined_statistic-row['statistic'],
                              critical=row['critical'], independent_critical=critical, critical_ci95=ci,
                              reference_reps=50000, reference_seed=verification_seed,
                              lower_ci_acceptance_margin=ci[0]-refined_statistic))
    obj = ExactScore(phi0, tau=.3, a=path_a(.3), gh=24, gl=64)
    total = ScoreMap(obj).exact(x).sum(axis=0) / np.sqrt(n)
    truth = dict(phi=phi0, alpha=true.alpha, v=true.v, w=true.w, kappa=0., statistic=float(total @ total), critical=reference.cutoff(obj))
    out = dict(design=dict(n=n, phi0=phi0, unit=unit, data_seed=seed, reference_seed=reference.seed,
                           reference_reps=reference.reps, fine_quadrature=[24, 64], grid_quadrature=[14, 32],
                           phi_factor_bounds=[.8, 1.2], alpha_factor_bounds=[-2, 2],
                           variance_factor_bounds=[.5, 2], rho_bounds=[-.6, .6],
                           kappa_domain=[-.36 * unit, .36 * unit],
                           interpretation='Grid inversion is an inner approximation. Feasible endpoint witnesses establish the full domain hull only when both domain endpoints are accepted; no grid coverage claim is made.'),
               truth=truth, logit_data=x.tolist(), grids=grids, candidates=rows, endpoint_witnesses=witnesses,
               seconds=time.perf_counter()-start)
    (ROOT / 'joint_region.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('WITNESSES', witnesses, flush=True)


if __name__ == '__main__':
    main()
