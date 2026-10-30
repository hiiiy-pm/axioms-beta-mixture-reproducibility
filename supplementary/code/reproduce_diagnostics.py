from pathlib import Path
import argparse
import csv
import gzip
import json

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1] / "results"
DATA = Path(__file__).resolve().parents[1] / "data"
PROBES = ["cg00047338", "cg00121366", "cg00160522"]


def read_json(name):
    base = DATA if name == "data_provenance.json" else ROOT
    return json.loads((base / name).read_text(encoding="utf-8"))


def audit_matrix(path):
    opener = gzip.open if path.suffix == ".gz" else open
    rows = missing = endpoints = outside = 0
    minimum, maximum = float("inf"), float("-inf")
    active = False
    columns = None
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                active = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not active or line.startswith('"ID_REF"'):
                continue
            fields = line.rstrip().split("\t")[1:]
            values = np.array([float(x.strip('"')) if x.strip('"') not in
                               ["null", "NA", ""] else np.nan for x in fields])
            columns = len(values) if columns is None else columns
            assert len(values) == columns
            rows += 1
            missing += int(np.isnan(values).sum())
            endpoints += int(((values == 0) | (values == 1)).sum())
            outside += int(((values < 0) | (values > 1)).sum())
            minimum = min(minimum, float(np.nanmin(values)))
            maximum = max(maximum, float(np.nanmax(values)))
    assert rows > 0
    return dict(rows=rows, columns=columns, missing=missing, endpoints=endpoints,
                outside_unit_interval=outside, minimum=minimum, maximum=maximum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-matrix", type=Path)
    args = parser.parse_args()
    provenance = read_json("data_provenance.json")
    copy_rng = np.random.default_rng(provenance["seeds"]["exchangeability"])
    bead_rng = np.random.default_rng(provenance["seeds"]["bead_diagnostic"])
    permutations = provenance["permutations"]
    old_copy = {x["probe"]: x for x in read_json("EPICv2_exchangeability_audit.json")["rows"]}
    old_bead = {x["probe"]: x for x in read_json("EPICv2_bead_diagnostic.json")["rows"]}
    old_geometry = {x["probe"]: x for x in read_json("real_design_geometry.json")["EPICv2"]}
    results = []
    for name in PROBES:
        d = np.genfromtxt(DATA / f"EPICv2_{name}.csv", delimiter=",", names=True)
        m = int(d["specimen_index"].max())
        r = int(d["copy_index"].max())
        assert len(d) == m * r
        y, beads = d["beta"].reshape(m, r), d["beads"].reshape(m, r)
        ratio_error = float(np.max(np.abs(d["methylated"] /
                          (d["methylated"] + d["unmethylated"] + 100) - d["beta"])))
        assert ratio_error < 1e-14
        ranks = rankdata(y, axis=1)
        centered = ranks - (r + 1) / 2
        copy_stat = float(np.sum(centered.sum(axis=0) ** 2))
        sq = (y - y.mean(axis=1, keepdims=True)) ** 2
        sq /= sq.sum(axis=1, keepdims=True)
        log_beads = np.log(beads)
        log_beads -= log_beads.mean(axis=1, keepdims=True)
        bead_stat = float(np.sum(sq * log_beads))
        copy_count = bead_count = 0
        for start in range(0, permutations, 500):
            size = min(500, permutations - start)
            order = np.argsort(copy_rng.random((size, m, r)), axis=-1)
            ref = np.take_along_axis(np.broadcast_to(centered, (size, m, r)), order, axis=-1).sum(axis=1)
            copy_count += int(np.sum(np.sum(ref * ref, axis=-1) >= copy_stat - 1e-10))
            order = np.argsort(bead_rng.random((size, m, r)), axis=-1)
            ref = np.sum(np.take_along_axis(np.broadcast_to(sq, (size, m, r)), order, axis=-1)
                         * log_beads, axis=(1, 2))
            bead_count += int(np.sum(np.abs(ref) >= abs(bead_stat) - 1e-12))
        copy_p = (1 + copy_count) / (permutations + 1)
        bead_p = (1 + bead_count) / (permutations + 1)
        assert copy_p == old_copy[name]["p"]
        assert bead_p == old_bead[name]["p"]
        X = np.column_stack([d["X0"], d["X1"]]).reshape(m, r, 2)
        Z = np.column_stack([d["Z0"], d["Z1"]]).reshape(m, r, 2)
        within = [int(np.linalg.matrix_rank(np.column_stack([x, np.ones(r)]))) for x in X]
        augmented = [int(np.linalg.matrix_rank(np.column_stack([x, np.eye(r)]))) for x in X]
        subject_columns = np.repeat(np.eye(m), r, axis=0)
        copy_columns = np.tile(np.eye(r)[:, 1:], (m, 1))
        stacked = np.column_stack([subject_columns, copy_columns])
        logits = (np.log(y) - np.log1p(-y)).reshape(-1)
        residual = logits - stacked @ np.linalg.lstsq(stacked, logits, rcond=None)[0]
        gamma = np.array(provenance["baseline_fits"][name]["gamma"])
        precision = np.exp(Z @ gamma)
        precision_range = [float(precision.min()), float(precision.max())]
        assert within == old_geometry[name]["mean_within_rank"]
        assert augmented == old_geometry[name]["mean_plus_copy_fixed_rank"]
        assert np.allclose(precision_range, old_geometry[name]["fitted_baseline_phi_range"])
        results.append(dict(probe=name, clusters=m, copies=r, copy_statistic=copy_stat,
                            copy_p=copy_p, bead_statistic=bead_stat, bead_p=bead_p,
                            within_mean_rank=within, within_copy_augmented_rank=augmented,
                            stacked_shared_copy_rank=int(np.linalg.matrix_rank(stacked)),
                            stacked_records=m*r, stacked_logit_residual_norm=float(np.linalg.norm(residual)),
                            saved_fit_precision_range=precision_range, ratio_error=ratio_error))
    design = np.genfromtxt(DATA / "GSE74167_design.csv", delimiter=",", names=True)
    X = np.column_stack([design[f"X{i}"] for i in range(5)]).reshape(14, 3, 5)
    design_ranks = [int(np.linalg.matrix_rank(np.column_stack([x, np.ones(3)]))) for x in X]
    assert design_ranks == read_json("real_design_geometry.json")["GSE74167"]["within_mean_rank"]
    with (DATA / "GSE74167_sample_sheet.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    donors = {(x["group"], x["donor"]) for x in records}
    assert len(records) == 42 and len(donors) == 14
    assert all(len({x["chip"] for x in records if (x["group"], x["donor"]) == donor}) == 3
               for donor in donors)
    out = dict(passed=True, EPICv2=results, GSE74167=dict(donors=14, arrays=42,
               within_mean_rank=design_ranks, distinct_chips_per_donor=3),
               interpretation="Within-cluster rank and the rank of the shared-coefficient panel are different. No global likelihood-unboundedness claim follows from the former alone.")
    if args.series_matrix:
        out["GSE74167"]["full_matrix"] = audit_matrix(args.series_matrix)
        expected = read_json("GSE74167_design_audit.json")["full_matrix"]
        for key in ["rows", "columns", "missing", "endpoints", "outside_unit_interval", "minimum", "maximum"]:
            assert out["GSE74167"]["full_matrix"][key] == expected[key]
    (ROOT / "diagnostic_recheck.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
