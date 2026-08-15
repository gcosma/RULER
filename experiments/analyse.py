#!/usr/bin/env python3
"""Turn raw experiment results into the paper's tables.

    python experiments/analyse.py results/primary.csv --outdir results/tables

Produces:

* ``primary.csv``        -- Table 1, M2 and M4 by forget fraction and method
* ``output_level.csv``   -- Table 3, the output-level pass criteria
* ``per_dataset_mia.csv``-- Table 4, per-dataset MIA accuracy
* ``dataset_level.csv``  -- Table 10, dataset-level Wilcoxon tests
* ``pairwise.csv``       -- Section 5.3, post-hoc method comparisons

and prints a readable summary of the headline result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper.config import MIA_NULL, MIA_PASS_WINDOW  # noqa: E402
from paper.report import (  # noqa: E402
    dataset_level_table,
    output_level_table,
    pairwise_method_table,
    per_dataset_mia_table,
    primary_table,
)


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_summary(frame: pd.DataFrame, primary: pd.DataFrame, outputs: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("OUTPUT-LEVEL EVALUATION (Table 3)")
    print("=" * 78)
    print(
        f"{'ff':>5}  {'Method':<16}{'MIA':>8}{'Forget':>9}{'Retain':>9}"
        f"{'Test':>9}  Passes"
    )
    for _, row in outputs.iterrows():
        print(
            f"{row['forget_fraction']:>5.0%}  {row['method']:<16}"
            f"{row['mia_accuracy_mean']:>8.3f}{row['forget_accuracy_mean']:>9.3f}"
            f"{row['retain_accuracy_mean']:>9.3f}{row['test_accuracy_mean']:>9.3f}"
            f"  {'yes' if row['mia_passes'] else 'NO'}"
        )
    print(
        f"\nPass criterion: |mean MIA - {MIA_NULL:.2f}| < {MIA_PASS_WINDOW:.2f}"
    )

    print("\n" + "=" * 78)
    print("REPRESENTATION-LEVEL EVALUATION (Table 1)")
    print("=" * 78)
    print(
        f"{'ff':>5}  {'Method':<16}{'M2':>11}{'p (LMM)':>10}{'r_rb':>7}"
        f"{'M4':>8}{'p (LMM)':>10}{'ICC':>7}"
    )
    for _, row in primary.iterrows():
        print(
            f"{row['forget_fraction']:>5.0%}  {row['method']:<16}"
            f"{row['m2']:>+11.5f}{_format_p(row['m2_p_lmm']):>10}"
            f"{row['m2_r_rb']:>+7.2f}"
            f"{row['m4']:>8.3f}{_format_p(row['m4_p_lmm']):>10}"
            f"{row['m4_icc']:>7.2f}"
        )

    n_significant = int((primary["m2_p_lmm"] < 0.05).sum())
    n_conditions = int(primary["m2_p_lmm"].notna().sum())
    n_passing = int(outputs[outputs["method"] != "Oracle"]["mia_passes"].sum())
    n_methods = int((outputs["method"] != "Oracle").sum())

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(
        f"  {n_passing}/{n_methods} method-fraction conditions pass output-level "
        f"evaluation,"
    )
    if n_conditions:
        print(
            f"  yet M2 detects significant representation-level residuals in "
            f"{n_significant}/{n_conditions}."
        )
    else:
        print(
            "  but the mixed-effects test needs at least two datasets, so no "
            "p-values were computed."
        )
    if "pre_unlearning_m4" in frame.columns:
        pre = frame.groupby("forget_fraction")["pre_unlearning_m4"].mean()
        print("\n  Pre-unlearning M4 diagnostic (null 0.50):")
        for forget_fraction, value in pre.items():
            print(f"    ff={forget_fraction:.0%}: {value:.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", help="CSV written by run_primary.py")
    parser.add_argument("--outdir", default="results/tables", help="output directory")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.results)
    required = {"dataset", "method", "forget_fraction", "seed", "m1", "m2", "m3", "m4"}
    missing = sorted(required - set(frame.columns))
    if missing:
        parser.error(
            f"{args.results} is missing required column(s) {missing}; "
            "expected a CSV written by run_primary.py"
        )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tables = {
        "primary": primary_table(frame),
        "output_level": output_level_table(frame),
        "per_dataset_mia": per_dataset_mia_table(frame),
        "dataset_level": dataset_level_table(frame, "m2"),
        "pairwise": pairwise_method_table(frame),
    }
    for name, table in tables.items():
        table.to_csv(outdir / f"{name}.csv", index=False)

    print_summary(frame, tables["primary"], tables["output_level"])
    print(f"\nTables written to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
