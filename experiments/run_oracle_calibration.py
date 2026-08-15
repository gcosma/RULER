#!/usr/bin/env python3
"""Empirical calibration of the M2 null (paper Appendix A.5, Fig. 5).

M2's null hypothesis is that forget-set records attain the same similarity to
the oracle as retained records do.  This script checks that the null is
actually achievable rather than true only by construction: it applies M2 to
*pairs of independently retrained oracles*, with no unlearning applied to
either model.

Neither model in such a pair has seen the forget set, so any systematic
deviation from zero would be an artefact of the metric.  The resulting
distribution is approximately centred on zero, which is what licenses reading a
negative M2 after unlearning as residual memorisation.

With 10 training seeds this gives 45 oracle pairs per dataset and 450 values
pooled across the ten datasets.

    python experiments/run_oracle_calibration.py --output results/oracle_null.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper.config import FORGET_FRACTIONS, TRAIN_SEEDS  # noqa: E402
from paper.data import dataset_names, load_dataset  # noqa: E402
from ruler import m2  # noqa: E402
from paper.models import penultimate  # noqa: E402
from paper.config import M2_RETAIN_SUBSAMPLE  # noqa: E402
from paper.partition import make_partition  # noqa: E402
from paper.stats import wilcoxon_signed_rank  # noqa: E402
from paper.train import checkpoint_path, load_model  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=dataset_names())
    parser.add_argument("--forget-fraction", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output", default="results/oracle_null.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.forget_fraction not in FORGET_FRACTIONS:
        print(
            f"warning: no cached oracles for ff={args.forget_fraction:.0%}; "
            f"expected one of {[f'{f:.0%}' for f in FORGET_FRACTIONS]}",
            file=sys.stderr,
        )

    rows = []
    for name in args.datasets:
        dataset = load_dataset(name)
        partition = make_partition(dataset.n_train, args.forget_fraction)
        x_forget = dataset.x_train[partition.forget_idx]
        x_retain = dataset.x_train[partition.retain_idx]

        if len(x_retain) > M2_RETAIN_SUBSAMPLE:
            idx = np.random.RandomState(42).choice(
                len(x_retain), M2_RETAIN_SUBSAMPLE, replace=False
            )
            x_retain = x_retain[idx]

        embeddings = {}
        for seed in TRAIN_SEEDS:
            path = checkpoint_path(
                args.checkpoint_dir, dataset.key, seed, "oracle", args.forget_fraction
            )
            if not path.exists():
                continue
            oracle = load_model(path, dataset.input_dim, device=args.device)
            embeddings[seed] = (
                penultimate(oracle, x_forget, device=args.device),
                penultimate(oracle, x_retain, device=args.device),
            )

        # Each unordered pair contributes once. The pair is symmetric in the
        # sense that neither model has seen the forget set, so which one plays
        # the "unlearned" role is arbitrary.
        for left, right in itertools.combinations(sorted(embeddings), 2):
            forget_a, retain_a = embeddings[left]
            forget_b, retain_b = embeddings[right]
            gap = m2(forget_a, forget_b, retain_a, retain_b)
            rows.append(
                {
                    "dataset": dataset.key,
                    "dataset_name": dataset.display_name,
                    "forget_fraction": args.forget_fraction,
                    "seed_a": left,
                    "seed_b": right,
                    "m2": gap,
                }
            )
        print(f"{dataset.display_name}: {len(embeddings)} oracles")

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no oracle checkpoints found", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    dataset_means = frame.groupby("dataset")["m2"].mean()
    result = wilcoxon_signed_rank(dataset_means.to_numpy(), null=0.0)
    print(f"\n{len(frame)} oracle-oracle pairs across {frame['dataset'].nunique()} datasets")
    print(f"  mean M2   : {frame['m2'].mean():+.5f}   (null 0)")
    print(f"  median M2 : {np.median(frame['m2']):+.5f}")
    print(f"  dataset-level Wilcoxon p = {result.p_value:.3f}")
    print(
        "  A distribution centred on zero means M2 is well calibrated under "
        "correct retraining."
    )
    print(f"\nWrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
