# Reproducing the manuscript

*Information Concentration and Score Calibration in Mean-Precision Beta Mixtures*

## Manuscript and complete numerical records

The manuscript PDF includes all theoretical and numerical appendices. Appendices A-E retain the technical derivations, numerical implementation, full 18-setting scalar calibration grid, all three CpG diagnostics, and the source comparison. Appendices F-K provide the full 36-setting information/score grid, 36 PP characteristic-function comparisons, two additional joint settings, all 19 interpolation checks, eight direct differentiation checks, nine information-oracle comparisons, all six equal-record replication rows, and 36 additional normal/GP convergence comparisons.

Appendix L adds known-null Monte Carlo reference benchmarks, reference-cutoff uncertainty, and paired comparisons on the same validation datasets.

Appendix M supplies the extended finite-precision and assumption-sensitivity experiments, numerical refinements, and candidate-specific joint-region inversion. The main text states the local mean and covariance conditions explicitly and proves the conservative coverage of covariance projection from the continuous joint region. The grid implementation is distinguished from that continuous coverage result.

The complete machine-readable outputs remain in `results/`; the directory contains no separate supplementary PDF. The normal/GP and PP labels are defined by each script and experiment, not by a shared `derived_*` field name. Reference-law quantiles and actual-beta rejection rates are separate records. Primary scalar and full-parameter joint experiments use different statistics and simulation streams.

## Directory structure

The reproducibility package contains `supplementary/code/` for research scripts,
`supplementary/data/` for the extracted public-data subsets and their provenance,
and `supplementary/results/` for the archived numerical results.
`MANIFEST.sha256` records hashes of the supplementary files. Paths below are
relative to `supplementary/` unless a working directory is specified.

The accompanying manuscript, its appendices, and the cover letter are supplied
through the journal submission system. They are not required to execute the
research scripts. The figure scripts create a sibling `figures/` directory at
the package root. The root README explains how to regenerate the figures into
a separate output directory from the archived results.

## Environment

The computations used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, SymPy 1.14.0, mpmath 1.3.0, and Matplotlib 3.11.2. From this supplementary directory, install dependencies with:

```text
python -m pip install -r requirements.txt
```

Pillow 12.3.0 is pinned for the PNG image-export environment. It is not used by the core information calculations.

## Numerical experiments

Saved results can be inspected without repeating the simulations. Parameters, seeds, replication counts, reference cutoffs, and numerical settings are recorded in the scripts and JSON outputs. Reruns replace the corresponding files in `results/`; use a working copy when preserving the archived outputs.

Run from `supplementary/code/` in this order:

```text
python joint_information_experiments.py
python verify_repeats.py
python score_limit_audit.py
python deep_followup.py
python poisson_calibration_audit.py
python pair_score_audit.py
python joint_score_test.py
python critical_law_grid.py
python verify_poisson_limit.py
python finite_bulk.py
python calibration_validation.py
python interpolation_audit.py
python point_null_benchmark.py
python publication_figures.py
python reproduce_diagnostics.py
python applicability_experiments.py
python joint_region.py
python applicability_audit.py
python applicability_figures.py
```

`joint_information_experiments.py` also evaluates the exact weak-direction identity, exponential-moment Gram matrices, heat-curvature polynomials, and the two-repetition information matrix using SymPy. These mathematical quantities are saved in the `symbolic` section of `results/validation.json`. They are obtained from the score expressions and differential operators used in Appendix A.

The three information engines (`joint_information.py`, `general_information.py`, and `matrix_information.py`) are imported by the experiment scripts. Several reference simulations use hundreds of millions of exponential draws. Figure generation reads the saved simulation outputs and writes to the manuscript's `figures/` directory.

| Manuscript material | Scripts in `code/` | Saved files in `results/` |
|---|---|---|
| Joint information law and independent integration calculations | `joint_information_experiments.py`, `joint_information.py`, `general_information.py` | `validation.json` |
| Table 1: the information and score expansion grid | `critical_law_grid.py` | `critical_law_grid.json` |
| Figure 1: information concentration and limiting score densities | `publication_figures.py`, `score_limit_audit.py` | `publication_figures.json`, `score_limit_audit.json` |
| Figure 2 and Tables 2, 3, A1: finite-sample calibration | `finite_bulk.py`, `calibration_validation.py`, `joint_score_test.py` | `finite_bulk.json`, `calibration_validation.json`, `joint_score_test.json` |
| Appendix L: known-null reference, matched computation budgets, and paired uncertainty | `point_null_benchmark.py` | `point_null_benchmark.json` |
| Figure 3: repetition and equal-record comparisons | `pair_score_audit.py`, `deep_followup.py`, `publication_figures.py` | `pair_score_audit.json`, `deep_followup.json`, `publication_figures.json` |
| Appendix B: independent score, information, interpolation, and Poisson calculations | `deep_followup.py`, `verify_repeats.py`, `matrix_information.py`, `interpolation_audit.py`, `verify_poisson_limit.py` | Corresponding JSON outputs; `repeat_comparison.json` stores the repetition integrals |
| Earlier Gaussian-Poisson reference results | `poisson_calibration_audit.py` | `poisson_calibration_audit.json` |
| Appendix D and Table A2: all three CpGs and the conversion design | `reproduce_diagnostics.py` | Reads the archived EPICv2 and design-geometry reports and writes `diagnostic_recheck.json`; the optional matrix check also reads the downloaded GSE74167 series matrix |
| Figures 4-5 and Appendix M: finite precision, assumption sensitivity, recording resolution, and local power | `applicability_experiments.py`, `applicability_figures.py` | `applicability_experiments.json` |
| Figure 5D and Appendix M: joint-region inversion and endpoint witnesses | `joint_region.py` | `joint_region.json` |
| Appendix M: direct score, quadrature, and interpolation refinements | `applicability_audit.py` | `applicability_audit.json` |

The polynomial-Poisson reference is named `finite_bulk` in the code. Scalar simulations evaluate scores at the specified full parameter; joint tests use the full-parameter point null. Monte Carlo standard errors are conditional on independently computed reference cutoffs. The repetition comparison fixes the number of observation records. The known-null benchmark uses a rank-based Monte Carlo cutoff and reproduces the original PP quantiles and validation statistics before computing paired differences. Its supplementary variance estimates resample reference order statistics; the paired intervals additionally resample the common validation data. Timings exclude validation and uncertainty calculations and are specific to the recorded single-process run.

`applicability_experiments.py` uses 10,000 validation datasets and 50,000 reference draws by default. Its parts are `precision`, `stress`, `pair`, `resolution`, and `power`; they may be run separately with `--part`, sequentially in that order. The power part reads the corresponding null critical values from the precision part. Do not run parts concurrently because they update the same result file. The residual-dependence parameter is the correlation of latent normal variables used for the gamma-power uniform copula, not the Pearson correlation of the resulting beta observations. Working-model MC references always use conditionally independent observations with Gaussian shared effects. Rejection rates under model departures are sensitivity measures. Rounding calculations integrate the original continuous model; they do not fit a rounded-data likelihood.

`joint_region.py` fixes the data seed and the physical parameter domain before simulation. It evaluates 405 and 2025 grid candidates with 20,000 common PP draws, then checks extreme feasible covariance points using higher-order integration and independent 50,000-draw references. The resulting covariance hull fills the search domain. The grid is an inner approximation to the continuous region, and no grid-coverage claim is made. All generated logits and candidate evaluations are retained. `applicability_audit.py` checks 16 distinct joint working-model parameter points and four paired-score settings using independent data. Figure regeneration reads saved results and does not rerun experiments.

## Public-data inputs

`data/data_provenance.json` records GEO accessions, extraction definitions, source hashes, sample mappings, permutation seeds, and saved baseline-fit coefficients. The five CSV files contain the three prespecified CpG subsets, the conversion-design matrix, and the public sample sheet. All recorded diagnostic failures are retained. EPICv2 address repetitions and GSE74167 independent bisulfite conversions remain distinct experimental designs.

`reproduce_diagnostics.py` recalculates the permutation statistics, ranks, and sample mapping from these inputs. Precision ranges use the saved fit coefficients. For the optional full GSE74167 matrix calculation, download the file from the `series_matrix_url` in `data/data_provenance.json` and save it as `data/GSE74167_series_matrix.txt.gz`. This full matrix is an optional external input and is not needed for the supplied-subset diagnostics. Then run from `supplementary/code/`:

```text
python reproduce_diagnostics.py --series-matrix ../data/GSE74167_series_matrix.txt.gz
```

The supplied CSV subsets are the starting inputs for the default diagnostic recheck. The recheck compares newly calculated statistics with the archived reports and evaluates precision ranges from the saved baseline-fit coefficients; it does not refit those coefficients. It does not re-extract raw IDAT intensities, rerun the complete probe-screening pipeline, or regenerate the four original diagnostic reports from raw measurements. Extraction definitions, source hashes, and the saved coefficients are recorded in `data/data_provenance.json` and the archived diagnostic outputs. These public-data calculations are design-compatibility diagnostics, rather than a successful real-data validation of the low-precision asymptotic regime.

## Full result-file inventory

The `results/` directory contains 22 JSON files covering the theoretical checks, simulation comparisons, public-data diagnostics, and applicability experiments. Unfavorable comparisons and failed applicability diagnostics are retained. Additional numerical checks are distinguished from the specified evaluation grids.

| File in `results/` | Complete saved contents |
|---|---|
| `calibration_validation.json` | 3 primary joint settings, 2 correlated joint settings, all 18 scalar settings, and 2 inversion refinements |
| `critical_law_grid.json` | All 36 information/score settings, 72 nested characteristic-function comparisons, and 3 integration refinements |
| `finite_bulk.json` | All 36 PP comparisons and 3 joint reference simulations, including cutoffs, moments, and reference seeds |
| `interpolation_audit.json` | All 19 settings and full coordinate errors |
| `deep_followup.json` | 8 score checks, 12 convergence checks, 2 earlier equal-information integrations, and 6 scalar simulations |
| `joint_score_test.json` | 3 GP critical-law simulations and 3 actual-beta joint settings |
| `score_limit_audit.json` | Four integrals, five precision diagnostics and their nested moment checks, and three scalar simulations |
| `poisson_calibration_audit.json` | 12 characteristic-function checks and 3 scalar GP-calibration simulations |
| `verify_poisson_limit.json` | 3 independent reference-law checks |
| `validation.json` | Exact algebra, 9 vertex oracles, 12 refinements, balanced/shifted examples, and all 48 fixed-mean checks |
| `pair_score_audit.json` | The complete replicated information matrix and all 3 replication simulations |
| `repeat_comparison.json` | All 12 finite-precision information comparisons and saved refinements |
| `point_null_benchmark.json` | 3 primary paths; known-null reference with 10,000 and PP-matched sample sizes; matched PP/GP/null-MC cutoffs and CPU times; 4000 reference-quantile and paired uncertainty resamples |
| `publication_figures.json` | Refined figure values, window curves, limiting densities, and simulation points |
| `EPICv2_exchangeability_audit.json` | Copy statistics and permutation results for all 3 CpGs |
| `EPICv2_bead_diagnostic.json` | Bead statistics and permutation results for all 3 CpGs |
| `real_design_geometry.json` | All 3 probe designs, subset residual checks, panel ranks, and GSE74167 design calculations |
| `GSE74167_design_audit.json` | Matrix-quality counts, chip structure, and recorded design limitations |
| `diagnostic_recheck.json` | Reproduced diagnostics for all 3 probes and the GSE74167 donor mapping |
| `applicability_experiments.json` | 20 joint finite-precision settings; 8 path/distribution departures; 10 paired-observation settings; 15 rounding calculations; 3 local covariance alternatives; seeds, critical values, conditional and paired uncertainty |
| `joint_region.json` | Prespecified domain and dataset; both grid definitions; all 2025 candidates; accepted-node counts; covariance and precision hulls; two independently verified extreme covariance witnesses |
| `applicability_audit.json` | Direct-statistic and quadrature checks at 16 distinct joint parameter points and 4 paired settings, with independent data and recorded integration orders |

`publication_figures.json` contains the refined information/window values printed in Figure 1. `deep_followup.json` retains earlier quadrature calculations; these are not substitutes for the refined figure values. The 48 fixed-mean checks in `validation.json` evaluate a different mean regime from the main shrinking-mean theorem. Their generic comparison fields must be interpreted using the specific fixed-mean predictions in those rows.
