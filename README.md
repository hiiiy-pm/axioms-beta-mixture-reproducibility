# Mean-precision beta mixtures: reproducibility materials

Research materials accompanying **Information Concentration and Score Calibration
in Mean-Precision Beta Mixtures**, by Yongye Huang, Xiaopeng Yang, and Zhining Wang.

This repository contains the Python implementations, archived numerical results,
public-data subsets, and figure-generation workflow for the study. The manuscript
and cover letter are supplied separately through the journal submission system.

## Contents

| Path | Contents |
| --- | --- |
| `supplementary/code/` | 22 research scripts: information calculations, score limits, simulations, numerical refinements, public-data diagnostics, and figures |
| `supplementary/results/` | 22 archived JSON result files, including parameters, seeds, reference cutoffs, and all reported evaluation settings |
| `supplementary/data/` | Five CSV input subsets and `data_provenance.json` |
| `supplementary/REPRODUCE.md` | Detailed execution order, manuscript-to-script mapping, and result inventory |
| `reproduce.py` | Entry point for figure regeneration, supplied-data diagnostics, or the full supplied workflow |
| `CITATION.cff` | Software citation metadata |

## Environment

The recorded environment uses Python 3.12.14. Create an isolated Python 3.12
environment and install the pinned dependencies from the repository root:

```sh
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pinned dependencies are NumPy 2.3.5, SciPy 1.18.1, SymPy 1.14.0, mpmath 1.3.0,
Matplotlib 3.11.2, and Pillow 12.3.0. No API keys or paid services are required.

## Regenerate the five figures from archived results

```sh
python reproduce.py figures
```

This produces `figure1` through `figure5` as PDF and 600-dpi PNG files in
`reproduced_figures/`. It reads the archived simulation and Figure 1 integration
records, and evaluates the symbolic score moments used by Figure 3. It does not
rerun the Monte Carlo simulation campaigns or replace the archived result files.

Choose another output directory with:

```sh
python reproduce.py figures --output-dir path/to/figures
```

## Recalculate the supplied-data diagnostics

```sh
python reproduce.py diagnostics
```

This recalculates the permutation statistics, design ranks, and sample mappings
from the supplied CSV files and updates
`supplementary/results/diagnostic_recheck.json`. It uses the archived fitting
coefficients when reporting fitted precision ranges.

## Run the full supplied research workflow

```sh
python reproduce.py full
```

This executes the research scripts sequentially in the documented order. It is
computationally expensive: several reference simulations use hundreds of millions
of exponential draws. It replaces the corresponding files in
`supplementary/results/` and writes regenerated figures to `figures/`. Use a fresh
working copy to preserve the archived release. Do not run the stages concurrently,
because some stages update shared result files.

Individual scripts, experiment settings, output definitions, and the optional
GSE74167 full-matrix diagnostic are documented in
[the detailed reproduction guide](supplementary/REPRODUCE.md).

## Public-data sources and scope

The source studies are available from NCBI GEO under
[GSE229715](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229715) and
[GSE74167](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74167).
The supplied subsets, extraction definitions, sample mappings,
permutation seeds, and fitted coefficients are documented in
`supplementary/data/data_provenance.json`.

The supplied-data workflow starts from these extracted subsets. It does not
re-extract raw IDAT intensities, rerun the complete probe-screening pipeline, or
refit the archived baseline coefficients. The original raw-data reports are
preserved alongside the reproducible subset diagnostics. These are
design-compatibility diagnostics, not a claim that the real data lie in the
low-precision asymptotic regime. All reported settings and unfavorable comparisons
are retained in the archived results.

## Citation and licensing

Version 1.0.1 is archived in Zenodo: [10.5281/zenodo.23813141](https://doi.org/10.5281/zenodo.23813141).
The [GitHub release](https://github.com/hiiiy-pm/axioms-beta-mixture-reproducibility/releases/tag/v1.0.1) and Zenodo provide the same ZIP, matching the journal supplementary archive.

Huang, Y., Yang, X., and Wang, Z. (2026). *Information Concentration and Score Calibration in Mean-Precision Beta Mixtures: Reproducibility Materials* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23813141

Please cite the version of this software that you use; author and version
information is provided in `CITATION.cff`. The code is licensed under the MIT
License. Third-party data retain their original rights and source attribution;
see [DATA_SOURCES.md](DATA_SOURCES.md).
