import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
import json
import platform
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, ndtr, zeta
from scipy.stats import chi2, norm
from score_limit_audit import ExactScore, ROOT, IH
from finite_bulk import polynomials, WHITEN
from pair_score_audit import PairScore
from point_null_benchmark import rejection, cutoff, quantile_resamples, paired


def path_a(tau):
    return 2 * np.sqrt((IH / 64 + 9 * tau ** 2 / 16) * 2 / 1.5)


class ScoreMap:
    def __init__(self, obj):
        self.obj = obj
        self.weights = obj.mass / obj.mass.sum()
        basis = np.column_stack([obj.S[:, [0, 1, 2, 4]], obj.S[:, 5] / obj.scale])
        self.means = self.weights @ basis
        centered = basis - self.means
        _, self.factor = np.linalg.qr(np.sqrt(self.weights)[:, None] * centered)
        self.whitening = np.linalg.inv(self.factor)
        self.interp = PchipInterpolator(obj.x, centered @ self.whitening, axis=0, extrapolate=False)

    def exact(self, x):
        _, scores = self.obj.evaluate(np.asarray(x))
        basis = np.column_stack([scores[:, [0, 1, 2, 4]], scores[:, 5] / self.obj.scale])
        return (basis - self.means) @ self.whitening

    def values(self, x):
        values = self.interp(x)
        bad = ~np.all(np.isfinite(values), axis=-1)
        if bad.any():
            values[bad] = self.exact(x[bad])
        return values

    def statistic(self, x):
        return np.sum(self.values(x).sum(axis=-2) ** 2, axis=-1) / x.shape[-1]


def simulate_logits(obj, rng, size, n, r=1, mixture=False, dependence=0., copy_shift=0., covariance=None):
    covariance = np.array([[obj.v, obj.kappa], [obj.kappa, obj.w]]) if covariance is None else covariance
    z = rng.standard_normal((size, n, 2))
    if mixture:
        scale = np.where(rng.random((size, n, 1)) < .1, 3., 1.) / np.sqrt(1.8)
        z = z * scale
    bc = z @ np.linalg.cholesky(covariance).T
    if r == 1:
        aa = expit(obj.alpha + bc[:, :, 0]) * obj.phi * np.exp(bc[:, :, 1])
        bb = expit(-obj.alpha - bc[:, :, 0]) * obj.phi * np.exp(bc[:, :, 1])
    else:
        offsets = np.linspace(-copy_shift, copy_shift, r)
        pp = expit(obj.alpha + bc[:, :, 0, None] + offsets)
        ph = obj.phi * np.exp(bc[:, :, 1, None])
        aa, bb = pp * ph, (1 - pp) * ph
    def log_gamma(shape):
        if dependence:
            shared = rng.standard_normal((size, n, 1))
            noise = rng.standard_normal(shape.shape)
            uniforms = ndtr(np.sqrt(dependence) * shared + np.sqrt(1 - dependence) * noise)
            uniforms = np.clip(uniforms, np.nextafter(0., 1.), np.nextafter(1., 0.))
        else:
            uniforms = rng.random(shape.shape)
        return np.log(rng.gamma(shape + 1)) + np.log(uniforms) / shape
    return log_gamma(aa) - log_gamma(bb)


def actual_statistics(obj, scoremap, n, reps, seed, **kwargs):
    rng = np.random.default_rng(seed)
    values = np.empty(reps)
    check = None
    for start in range(0, reps, 50):
        stop = min(start + 50, reps)
        x = simulate_logits(obj, rng, stop - start, n, **kwargs)
        values[start:stop] = scoremap.statistic(x)
        if check is None:
            check = x.ravel()[:1200]
    errors = scoremap.values(check) - scoremap.exact(check)
    return values, dict(maximum=float(np.max(abs(errors))), rms=float(np.sqrt(np.mean(errors ** 2))))


def reference_statistics(obj, n, reps, seed, gp=False):
    rng = np.random.default_rng(seed)
    d0, lam = obj.parameters()['D0'], n * obj.phi
    h11, h12, _ = obj.H
    output = np.empty(reps)
    for start in range(0, reps, 100):
        size = min(100, reps - start)
        if gp:
            strong = rng.chisquare(4, size)
            weak = rng.normal(0, np.sqrt(1 - IH / (64 * d0)), size)
        else:
            u = rng.exponential(size=(size, n))
            signs = 2 * rng.integers(0, 2, size=(size, n)) - 1
            a, b, v, k, _, rr, qq = polynomials(u, signs)
            vectors = np.column_stack([s.sum(axis=1) for s in [a, b, v, k]]) @ WHITEN / np.sqrt(n)
            strong = np.sum(vectors ** 2, axis=1)
            weak = obj.tau * (h11 * rr + h12 * qq).sum(axis=1) / np.sqrt(n * d0)
        counts = rng.poisson(lam * 12, size)
        x = rng.uniform(-24, 24, int(counts.sum()))
        f = np.logaddexp(0, -x) * np.logaddexp(0, x)
        jumps = np.bincount(np.repeat(np.arange(size), counts), weights=f, minlength=size)
        weak -= (jumps - lam * zeta(3, 1) / 2) / (4 * np.sqrt(lam * d0))
        output[start:start + size] = strong + weak ** 2
    return output


def run_joint(obj, n, reps, reference_reps, seed, label, generation=None):
    generation = generation or {}
    start = time.perf_counter()
    scoremap = ScoreMap(obj)
    actual, interpolation = actual_statistics(obj, scoremap, n, reps, seed, **generation)
    refs = {}
    uncertainty = {}
    for index, name in enumerate(['PP', 'GP', 'MC']):
        ref_seed = seed + 10000 * (index + 1)
        if name == 'MC':
            samples, _ = actual_statistics(obj, scoremap, n, reference_reps, ref_seed)
        else:
            samples = reference_statistics(obj, n, reference_reps, ref_seed, gp=name == 'GP')
        critical = cutoff(samples, rank=name == 'MC')
        draws = quantile_resamples(samples, np.random.default_rng(ref_seed + 800000), reps=4000, rank=name == 'MC')
        refs[name] = dict(seed=ref_seed, reps=reference_reps, critical=critical,
                         critical_mcse=float(draws.std(ddof=1)), critical_ci95=np.quantile(draws, [.025, .975]).tolist(),
                         **rejection(actual, critical))
        uncertainty[name] = draws
    refs['Chi-square'] = dict(critical=float(chi2.ppf(.95, 5)), **rejection(actual, chi2.ppf(.95, 5)))
    comparison = paired(actual, refs['PP']['critical'], refs['MC']['critical'], uncertainty['PP'],
                        uncertainty['MC'], np.random.default_rng(seed + 900000))
    return dict(label=label, phi=obj.phi, tau=obj.tau, alpha=obj.alpha, a=obj.a, H=list(obj.H),
                covariance=[[obj.v, obj.kappa], [obj.kappa, obj.w]], n=n, n_phi=n * obj.phi,
                n_phi5=n * obj.phi ** 5, eta=obj.parameters()['rare_fraction'], reps=reps, seed=seed,
                generation=generation, mass=float(obj.mass.sum()), information=obj.info,
                normalized_score_mean=(scoremap.means @ scoremap.whitening).tolist(), interpolation=interpolation,
                references=refs, paired_PP_minus_MC=comparison, seconds=time.perf_counter() - start)


def resolution(obj, digits, n=1000):
    delta = 10. ** (-digits)
    h = np.log((1 - delta / 2) / (delta / 2))
    x, w = leggauss(160)
    density, _ = obj.evaluate(h * x)
    interior = float(h * np.dot(w, density))
    x2, w2 = leggauss(240)
    refined, _ = obj.evaluate(h * x2)
    refined = float(h * np.dot(w2, refined))
    return dict(phi=obj.phi, digits=digits, half_bin=delta / 2, logit_threshold=h,
                interior_probability=refined, endpoint_probability=1 - refined,
                expected_interior_records=n * refined, n=n, integration_difference=abs(interior - refined))


def run_pair(phi, reps, reference_reps, seed):
    obj = PairScore(phi, tau=.3, a=path_a(.3), gh=14, gl=28)
    n = 500
    def samples(count, sample_seed, dependence=0., mixture=False, copy_shift=0.):
        rng = np.random.default_rng(sample_seed)
        values = []
        check = None
        for start in range(0, count, 50):
            size = min(50, count - start)
            x = simulate_logits(obj, rng, size, n, r=2, dependence=dependence, mixture=mixture, copy_shift=copy_shift)
            flat = x.reshape(-1, 2)
            scores = obj.pair_interp.ev(flat[:, 0], flat[:, 1]).reshape(size, n)
            bad = np.max(abs(x), axis=2) > obj.x[-1]
            if bad.any():
                scores[bad] = obj.efficient_pair(x[bad])
            values.extend(((scores - obj.pair_mean).sum(axis=1) / np.sqrt(n * obj.pair_info)).tolist())
            if check is None:
                check = flat[:1000]
        error = (obj.efficient_pair(check) - obj.pair_interp.ev(check[:, 0], check[:, 1])) / np.sqrt(obj.pair_info)
        return np.asarray(values), dict(maximum=float(max(abs(error))), rms=float(np.sqrt(np.mean(error ** 2))))
    reference, _ = samples(reference_reps, seed + 10000)
    cuts = np.quantile(reference, [.025, .975])
    rows = []
    settings = [('independent Gaussian', {}), ('uniform-copula correlation 0.25', {'dependence': .25}),
                ('uniform-copula correlation 0.50', {'dependence': .5}),
                ('10% scale-contaminated Gaussian', {'mixture': True}),
                ('opposite logit offsets 0.15', {'copy_shift': .15})]
    for index, (label, kwargs) in enumerate(settings):
        values, error = samples(reps, seed + index, **kwargs)
        rows.append(dict(label=label, seed=seed + index, generation=kwargs, reps=reps,
                         mean=float(values.mean()), variance=float(values.var(ddof=1)), interpolation=error,
                         normal=rejection(abs(values), norm.ppf(.975)),
                         normal_left=rejection(-values, norm.ppf(.975)),
                         normal_right=rejection(values, norm.ppf(.975)),
                         working_model_MC=rejection(((values < cuts[0]) | (values > cuts[1])).astype(float), .5)))
        print('PAIR', phi, label, rows[-1]['normal']['size'], flush=True)
    refined = PairScore(phi, tau=.3, a=path_a(.3), gh=18, gl=40)
    return dict(phi=phi, n_clusters=n, records=2 * n, tau=.3, a=path_a(.3), information=obj.pair_info,
                refined_information=refined.pair_info, relative_information_difference=refined.pair_info / obj.pair_info - 1,
                reference_reps=reference_reps, reference_seed=seed + 10000, reference_cuts=cuts.tolist(), rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['precision', 'stress', 'pair', 'power', 'resolution', 'all'], default='all')
    parser.add_argument('--reps', type=int, default=10000)
    parser.add_argument('--reference-reps', type=int, default=50000)
    args = parser.parse_args()
    target = ROOT / 'applicability_experiments.json'
    out = json.loads(target.read_text()) if target.exists() else {}
    out['design'] = dict(python=platform.python_version(), precision_grid=[.0001, .001, .01, .05, .1],
                         taus=[.01, .3], sample_sizes=[200, 1000], validation_reps=args.reps,
                         reference_reps=args.reference_reps, paired_bootstrap_reps=4000,
                         reference_scope='MC always uses the working Gaussian model; stress rejection is a reference-model sensitivity measure, not size under misspecification.',
                         uncertainty='Wilson intervals and MCSE condition on reference draws; paired intervals resample independent reference quantiles and common validation data.',
                         timing='single process and one BLAS thread; exploratory applicability study specified before results')
    def save():
        target.write_text(json.dumps(out, indent=2), encoding='utf-8')
    if args.part in ['precision', 'all']:
        out['precision'] = []
        for i, phi in enumerate(out['design']['precision_grid']):
            for j, tau in enumerate(out['design']['taus']):
                obj = ExactScore(phi, tau=tau, a=path_a(tau), gh=18, gl=48)
                for k, n in enumerate(out['design']['sample_sizes']):
                    seed = 2026103000 + 100 * i + 10 * j + k
                    row = run_joint(obj, n, args.reps, args.reference_reps, seed, 'balanced Gaussian')
                    out['precision'].append(row)
                    print('PRECISION', phi, tau, n, {m:round(v['size'], 5) for m,v in row['references'].items()}, flush=True)
                    save()
    if args.part in ['stress', 'all']:
        out['stress'] = []
        for i, phi in enumerate([.01, .1]):
            settings = [('mean logit 0.2', .3, .2 / phi ** 2.5, {}),
                        ('covariance exponent 2', .3 / np.sqrt(phi), path_a(.3), {}),
                        ('covariance exponent 3', .3 * np.sqrt(phi), path_a(.3), {}),
                        ('10% scale-contaminated Gaussian', .3, path_a(.3), {'mixture': True})]
            for j, (label, tau, a, generation) in enumerate(settings):
                obj = ExactScore(phi, tau=tau, a=a, gh=18, gl=48)
                seed = 2026203000 + 100 * i + j
                row = run_joint(obj, 1000, args.reps, args.reference_reps, seed, label, generation)
                out['stress'].append(row)
                print('STRESS', phi, label, {m:round(v['size'], 5) for m,v in row['references'].items()}, flush=True)
                save()
    if args.part in ['pair', 'all']:
        out['pair'] = []
        for i, phi in enumerate([.001, .1]):
            out['pair'].append(run_pair(phi, args.reps, args.reference_reps, 2026303000 + 100 * i))
            save()
    if args.part in ['resolution', 'all']:
        out['resolution'] = []
        for phi in out['design']['precision_grid']:
            obj = ExactScore(phi, tau=.3, a=path_a(.3), gh=20, gl=48)
            out['resolution'].extend(resolution(obj, digits) for digits in [2, 4, 6])
        save()
    if args.part in ['power', 'all']:
        out['power'] = []
        for i, phi in enumerate([.0001, .01, .1]):
            base = next(row for row in out['precision'] if row['phi'] == phi and row['tau'] == .3 and row['n'] == 1000)
            obj = ExactScore(phi, tau=.3, a=path_a(.3), gh=18, gl=48)
            covariance = np.array([[obj.v, .5 * np.sqrt(obj.v * obj.w)], [.5 * np.sqrt(obj.v * obj.w), obj.w]])
            seed = 2026403000 + i
            values, error = actual_statistics(obj, ScoreMap(obj), 1000, args.reps, seed, covariance=covariance)
            out['power'].append(dict(phi=phi, n=1000, n_phi5=1000 * phi ** 5, seed=seed, reps=args.reps,
                                     null_kappa=0., alternative_kappa=float(covariance[0, 1]), alternative_correlation=.5,
                                     interpolation=error, references={name:dict(null_size=entry['size'],
                                     **rejection(values, entry['critical'])) for name, entry in base['references'].items()}))
            print('POWER', phi, out['power'][-1]['references']['PP'], flush=True)
            save()


if __name__ == '__main__':
    main()
