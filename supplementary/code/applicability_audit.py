import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
import json
import numpy as np
from applicability_experiments import ScoreMap, simulate_logits, path_a
from score_limit_audit import ExactScore, ROOT
from pair_score_audit import PairScore


def main():
    results = json.loads((ROOT / 'applicability_experiments.json').read_text())
    rows = []
    seen = set()
    for row in results['precision'] + results['stress']:
        key = row['phi'], row['tau'], row['a']
        if key in seen:
            continue
        seen.add(key)
        base = ExactScore(*key, gh=18, gl=48)
        fine = ExactScore(*key, gh=24, gl=64)
        basic, refined = ScoreMap(base), ScoreMap(fine)
        x = simulate_logits(base, np.random.default_rng(row['seed'] + 7000000), 20, 1000, **row['generation'])
        interpolated = basic.statistic(x)
        raw = basic.exact(x.ravel()).reshape(20, 1000, 5)
        exact = np.sum(raw.sum(axis=1) ** 2, axis=1) / 1000
        accurate = refined.exact(x.ravel()).reshape(20, 1000, 5)
        checked = np.sum(accurate.sum(axis=1) ** 2, axis=1) / 1000
        rows.append(dict(phi=base.phi, tau=base.tau, alpha=base.alpha, label=row['label'],
                         seed=row['seed'] + 7000000, datasets=20, n=1000,
                         coarse_mass=float(base.mass.sum()), refined_mass=float(fine.mass.sum()),
                         maximum_statistic_interpolation_error=float(max(abs(interpolated-exact))),
                         maximum_statistic_refinement_difference=float(max(abs(checked-exact))),
                         rms_statistic_refinement_difference=float(np.sqrt(np.mean((checked-exact)**2))),
                         normalized_score_mean_maximum=float(max(abs(refined.means @ refined.whitening))),
                         gram_condition_number=float(np.linalg.cond(refined.factor))))
        print('AUDIT', base.phi, row['label'], rows[-1]['maximum_statistic_refinement_difference'], flush=True)
    pairs = []
    for phi in [.001, .1]:
        base = PairScore(phi, tau=.3, a=path_a(.3), gh=14, gl=28)
        fine = PairScore(phi, tau=.3, a=path_a(.3), gh=18, gl=40)
        for index, dependence in enumerate([0., .5]):
            seed = 2026603000 + 100 * int(phi == .1) + index
            x = simulate_logits(base, np.random.default_rng(seed), 20, 500, r=2, dependence=dependence)
            flat = x.reshape(-1, 2)
            interp = base.pair_interp.ev(flat[:, 0], flat[:, 1])
            exact = base.efficient_pair(flat)
            refined = fine.efficient_pair(flat)
            zi = (interp.reshape(20, 500)-base.pair_mean).sum(axis=1)/np.sqrt(500*base.pair_info)
            ze = (exact.reshape(20, 500)-base.pair_mean).sum(axis=1)/np.sqrt(500*base.pair_info)
            zr = (refined.reshape(20, 500)-fine.pair_mean).sum(axis=1)/np.sqrt(500*fine.pair_info)
            pairs.append(dict(phi=phi, dependence=dependence, seed=seed, datasets=20, n_clusters=500,
                              maximum_statistic_interpolation_error=float(max(abs(zi-ze))),
                              maximum_statistic_refinement_difference=float(max(abs(zr-ze))),
                              relative_information_difference=float(fine.pair_info/base.pair_info-1)))
    out = dict(joint=rows, pair=pairs, joint_orders=[[18,48],[24,64]], pair_orders=[[14,28],[18,40]])
    (ROOT/'applicability_audit.json').write_text(json.dumps(out, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
