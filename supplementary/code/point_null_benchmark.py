import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
import json
import platform
import time
import numpy as np
from scipy.special import expit, zeta
from scipy.interpolate import PchipInterpolator
from scipy.stats import chi2, binomtest
from score_limit_audit import ExactScore, ROOT, IH
from finite_bulk import polynomials, WHITEN


def prepare(obj):
    mass = obj.mass / sum(obj.mass)
    basis = np.column_stack([obj.S[:, [0, 1, 2, 4]], obj.S[:, 5] / obj.scale])
    means = mass @ basis
    gram = basis.T @ (mass[:, None] * basis)
    whitening = np.linalg.inv(np.linalg.cholesky(gram)).T
    interp = PchipInterpolator(obj.x, (basis - means) @ whitening, axis=0, extrapolate=False)
    mixing = np.linalg.cholesky([[obj.v, obj.kappa], [obj.kappa, obj.w]])
    return means, whitening, interp, mixing


def beta_statistics(obj, prepared, n, reps, seed, checkpoint=None):
    means, whitening, interp, mixing = prepared
    rng = np.random.default_rng(seed)
    values = []
    times = {}
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    for start in range(0, reps, 50):
        size = min(50, reps - start)
        bc = rng.standard_normal((size, n, 2)) @ mixing.T
        p = expit(obj.alpha + bc[:, :, 0])
        phi = obj.phi * np.exp(bc[:, :, 1])
        aa, bb = p * phi, (1 - p) * phi
        x = np.log(rng.gamma(aa + 1)) + np.log(rng.random(aa.shape)) / aa
        x -= np.log(rng.gamma(bb + 1)) + np.log(rng.random(bb.shape)) / bb
        scores = interp(x)
        bad = ~np.all(np.isfinite(scores), axis=2)
        if bad.any():
            _, score = obj.evaluate(x[bad])
            basis = np.column_stack([score[:, [0, 1, 2, 4]], score[:, 5] / obj.scale])
            scores[bad] = (basis - means) @ whitening
        total = scores.sum(axis=1) / np.sqrt(n)
        values.extend(np.sum(total * total, axis=1).tolist())
        if checkpoint is not None and start + size == checkpoint:
            times = dict(cpu_seconds=time.process_time() - start_cpu,
                         wall_seconds=time.perf_counter() - start_wall)
    return np.asarray(values), times


def pp_statistics(obj, n, reps, seed):
    rng = np.random.default_rng(seed)
    pars = obj.parameters()
    d0, lam = pars['D0'], n * obj.phi
    h11, h12, _ = obj.H
    values = []
    for start in range(0, reps, 100):
        size = min(100, reps - start)
        u = rng.exponential(size=(size, n))
        signs = 2 * rng.integers(0, 2, size=(size, n)) - 1
        a, b, v, k, _, r, q = polynomials(u, signs)
        strong = np.column_stack([s.sum(axis=1) for s in [a, b, v, k]]) @ WHITEN / np.sqrt(n)
        weak = (obj.tau * (h11 * r + h12 * q)).sum(axis=1) / np.sqrt(n * d0)
        counts = rng.poisson(lam * 12, size=size)
        x = rng.uniform(-24, 24, size=int(counts.sum()))
        f = np.logaddexp(0, -x) * np.logaddexp(0, x)
        jumps = np.bincount(np.repeat(np.arange(size), counts), weights=f, minlength=size)
        weak -= (jumps - lam * zeta(3, 1) / 2) / (4 * np.sqrt(lam * d0))
        values.extend((np.sum(strong ** 2, axis=1) + weak ** 2).tolist())
    return np.asarray(values)


def gp_statistics(obj, n, reps, seed):
    rng = np.random.default_rng(seed)
    d0, lam = obj.parameters()['D0'], n * obj.phi
    counts = rng.poisson(lam * 12, size=reps)
    x = rng.uniform(-24, 24, size=int(counts.sum()))
    f = np.logaddexp(0, -x) * np.logaddexp(0, x)
    jumps = np.bincount(np.repeat(np.arange(reps), counts), weights=f, minlength=reps)
    weak = rng.normal(0, np.sqrt(1 - IH / (64 * d0)), reps)
    weak -= (jumps - lam * zeta(3, 1) / 2) / (4 * np.sqrt(lam * d0))
    return rng.chisquare(4, reps) + weak * weak


def cutoff(values, rank=False):
    if rank:
        index = len(values) - int(np.floor(.05 * (len(values) + 1)))
        return float(np.partition(values, index)[index])
    return float(np.quantile(values, .95))


def quantile_resamples(values, rng, reps=4000, rank=False):
    ordered = np.sort(values)
    size = len(values)
    if rank:
        k = size - int(np.floor(.05 * (size + 1))) + 1
        uniforms = rng.beta(k, size + 1 - k, size=reps)
        return ordered[np.minimum((size * uniforms).astype(int), size - 1)]
    position = .95 * (size - 1)
    k, weight = int(np.floor(position)) + 1, position % 1
    gaps = rng.dirichlet([k, 1, size - k], size=reps)
    low = ordered[np.minimum((size * gaps[:, 0]).astype(int), size - 1)]
    high = ordered[np.minimum((size * gaps[:, :2].sum(axis=1)).astype(int), size - 1)]
    return (1 - weight) * low + weight * high


def rejection(values, critical):
    count = int(np.count_nonzero(values > critical))
    rate = count / len(values)
    interval = binomtest(count, len(values)).proportion_ci(method='wilson')
    return dict(rejections=count, size=rate, mcse=float(np.sqrt(rate * (1 - rate) / len(values))),
                wilson95=[interval.low, interval.high])


def paired(values, first_cut, second_cut, first_resamples, second_resamples, rng):
    first, second = values > first_cut, values > second_cut
    differences = first.astype(int) - second.astype(int)
    discordant = [int(np.sum(first & ~second)), int(np.sum(second & ~first))]
    conditional_se = float(differences.std(ddof=1) / np.sqrt(len(values)))
    ordered = np.sort(values)
    left = np.searchsorted(ordered, first_resamples, side='right')
    right = np.searchsorted(ordered, second_resamples, side='right')
    probabilities = np.abs(right - left) / len(values)
    resampled_differences = np.sign(right - left) * rng.binomial(len(values), probabilities) / len(values)
    return dict(difference=float(differences.mean()), conditional_mcse=conditional_se,
                discordant_counts=discordant,
                mcnemar_exact_p=float(binomtest(discordant[0], sum(discordant)).pvalue) if sum(discordant) else 1.0,
                joint_resampling_ci95=np.quantile(resampled_differences, [.025, .975]).tolist(),
                joint_resampling_sd=float(resampled_differences.std(ddof=1)))


def main():
    archived = json.loads((ROOT / 'calibration_validation.json').read_text())['joint_primary']
    gp_old = json.loads((ROOT / 'joint_score_test.json').read_text())['simulation']
    pp_old = json.loads((ROOT / 'finite_bulk.json').read_text())['joint_reference']
    setup_cpu, setup_wall = time.process_time(), time.perf_counter()
    tau = .01
    d0 = IH / 64 + 9 * tau ** 2 / 16
    a = 2 * np.sqrt(d0 * 2 / 1.5)
    obj = ExactScore(.0001, tau=tau, a=a, gh=20, gl=48)
    prepared = prepare(obj)
    output = dict(environment=dict(python=platform.python_version(), platform=platform.platform(),
                                  processor=platform.processor(), logical_processors=os.cpu_count(),
                                  blas_threads=1),
                  shared_score_setup=dict(cpu_seconds=time.process_time() - setup_cpu,
                                          wall_seconds=time.perf_counter() - setup_wall),
                  null=dict(phi=obj.phi, tau=tau, a=a, H=list(obj.H)), settings=[],
                  quantile_uncertainty='4000 nonparametric bootstrap quantiles from exact uniform order-statistic simulation of each independent reference sample',
                  paired_uncertainty='common validation resampling plus independent reference-quantile resampling; percentile intervals; conditional discordance summaries also retained',
                  timing='single-process CPU and wall time for reference generation and cutoff selection; shared score setup reported separately; validation and uncertainty calculations excluded',
                  null_mc='known-parameter simulation; rank cutoff gives unconditional rejection probability floor(0.05*(B+1))/(B+1) for a continuous statistic; numerical score mapping is fixed for all samples')
    for original, old_gp, old_pp in zip(archived, gp_old, pp_old):
        n, reps, reference_reps = original['n'], original['reps'], old_pp['reps']
        print('START', n, reference_reps, flush=True)
        validation, _ = beta_statistics(obj, prepared, n, reps, original['seed'])
        assert np.mean(validation > chi2.ppf(.95, 5)) == original['usual_size']
        assert np.mean(validation > original['derived_critical']) == original['derived_size']
        row = dict(n=n, validation_reps=reps, validation_seed=original['seed'], reference_reps=reference_reps,
                   references={}, original_methods={})
        pool, samples = {}, {}
        for method in ['pp', 'gp', 'null_mc']:
            seed = old_pp['seed'] if method == 'pp' else 129000 + n if method == 'gp' else 202610300 + n
            start_cpu, start_wall = time.process_time(), time.perf_counter()
            if method == 'pp':
                values = pp_statistics(obj, n, reference_reps, seed)
                assert abs(cutoff(values) - old_pp['critical_values'][1]) < 1e-10
            elif method == 'gp':
                values = gp_statistics(obj, n, reference_reps, seed)
            else:
                values, partial_time = beta_statistics(obj, prepared, n, reference_reps, seed, checkpoint=10000)
            critical = cutoff(values, rank=method == 'null_mc')
            elapsed = dict(cpu_seconds=time.process_time() - start_cpu,
                           wall_seconds=time.perf_counter() - start_wall)
            rng = np.random.default_rng(369000 + n + ['pp', 'gp', 'null_mc'].index(method))
            draws = quantile_resamples(values, rng, rank=method == 'null_mc')
            entry = dict(seed=seed, reps=reference_reps, critical_value=critical, **elapsed,
                         quantile_variance=float(draws.var(ddof=1)), quantile_mcse=float(draws.std(ddof=1)),
                         **rejection(validation, critical))
            row['references'][method] = entry
            pool[method], samples[method] = values, draws
            print('REFERENCE', n, method, json.dumps(entry), flush=True)
            if method == 'null_mc':
                small = values[:10000]
                tc, tw = time.process_time(), time.perf_counter()
                small_cut = cutoff(small, rank=True)
                partial_time['cpu_seconds'] += time.process_time() - tc
                partial_time['wall_seconds'] += time.perf_counter() - tw
                draws_small = quantile_resamples(small, rng, rank=True)
                row['references']['null_mc_10000'] = dict(seed=seed, reps=10000, nested=True,
                    critical_value=small_cut, **partial_time, quantile_variance=float(draws_small.var(ddof=1)),
                    quantile_mcse=float(draws_small.std(ddof=1)), **rejection(validation, small_cut))
        for method, critical in [('chi_square', chi2.ppf(.95, 5)),
                                 ('gp', old_gp['derived_critical']), ('pp', original['derived_critical'])]:
            row['original_methods'][method] = dict(critical_value=float(critical), **rejection(validation, critical))
        row['paired_pp_minus_null_mc'] = paired(validation, row['references']['pp']['critical_value'],
            row['references']['null_mc']['critical_value'], samples['pp'], samples['null_mc'],
            np.random.default_rng(976000 + n))
        output['settings'].append(row)
        (ROOT / 'point_null_benchmark.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
        print('DONE', n, json.dumps(row['paired_pp_minus_null_mc']), flush=True)


if __name__ == '__main__':
    main()
