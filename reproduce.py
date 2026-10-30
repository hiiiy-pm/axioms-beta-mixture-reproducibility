"""Run the figure, supplied-data, or full research reproduction workflow."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "supplementary" / "code"
FULL_WORKFLOW = (
    "joint_information_experiments.py",
    "verify_repeats.py",
    "score_limit_audit.py",
    "deep_followup.py",
    "poisson_calibration_audit.py",
    "pair_score_audit.py",
    "joint_score_test.py",
    "critical_law_grid.py",
    "verify_poisson_limit.py",
    "finite_bulk.py",
    "calibration_validation.py",
    "interpolation_audit.py",
    "point_null_benchmark.py",
    "publication_figures.py",
    "reproduce_diagnostics.py",
    "applicability_experiments.py",
    "joint_region.py",
    "applicability_audit.py",
    "applicability_figures.py",
)


def run_script(name):
    print(f"Running {name}", flush=True)
    subprocess.run([sys.executable, "-B", str(CODE / name)], cwd=CODE, check=True)


def figures(output):
    sys.path.insert(0, str(CODE))
    import publication_figures
    import applicability_figures

    output.mkdir(parents=True, exist_ok=True)
    publication_figures.OUT = output
    applicability_figures.OUT = output
    saved = json.loads(
        (ROOT / "supplementary" / "results" / "publication_figures.json").read_text(
            encoding="utf-8"
        )
    )
    publication_figures.mechanism_figure(rows=saved["mechanism"])
    publication_figures.calibration_figure()
    publication_figures.replication_figure()
    applicability_figures.main()
    print(f"Five figures written as PDF and PNG to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    figure_parser = commands.add_parser("figures", help="Regenerate figures from archived results")
    figure_parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced_figures")
    commands.add_parser("diagnostics", help="Recalculate supplied-data diagnostics")
    commands.add_parser("full", help="Run the full supplied workflow; replaces results and can be expensive")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.command == "figures":
        figures(args.output_dir.resolve())
    elif args.command == "diagnostics":
        run_script("reproduce_diagnostics.py")
    else:
        for script in FULL_WORKFLOW:
            run_script(script)


if __name__ == "__main__":
    main()
