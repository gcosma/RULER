#!/usr/bin/env python3
"""Run the primary tabular evaluation and write tidy per-condition results.

The default configuration is the paper's primary experiment: 10 datasets x 3
forget fractions x 10 training seeds x 4 unlearning methods, giving N = 100
observations per (method, forget fraction) condition.

Examples
--------
Primary experiment (Table 1, 3, 4)::

    python experiments/run_primary.py --output results/primary.csv

Bad Teacher, the fifth method with a different forgetting mechanism
(Section 5.4, Appendix A.11)::

    python experiments/run_primary.py --methods "Bad Teacher" \
        --output results/bad_teacher.csv

Robustness to mini-batch training at ff = 5% (Appendix A.6). This changes how
the models are trained, so it needs its own checkpoint directory::

    python experiments/run_primary.py --forget-fractions 0.05 --seeds 5 \
        --batch-size 128 --checkpoint-dir checkpoints_minibatch \
        --output results/minibatch.csv

Sensitivity to the unlearning learning rate (Appendix A.7). Only unlearning
changes here, so the cached checkpoints still apply::

    python experiments/run_primary.py --forget-fractions 0.05 \
        --unlearn-lr 1e-4 --output results/lr_1e-4.csv

Sensitivity to which records were selected for erasure (Appendix A.8). A new
forget set means a new retain set and therefore new oracles::

    python experiments/run_primary.py --forget-seed 1000 --seeds 5 \
        --checkpoint-dir checkpoints_fs1000 --output results/forget_seed_1000.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper.config import FORGET_FRACTIONS, METHODS, TRAIN_SEEDS  # noqa: E402
from paper.data import dataset_names  # noqa: E402
from paper.pipeline import run_experiment  # noqa: E402
from paper.unlearn import UNLEARN_METHODS  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=dataset_names(),
        choices=dataset_names(),
        metavar="NAME",
        help="datasets to evaluate (default: all ten)",
    )
    parser.add_argument(
        "--forget-fractions",
        nargs="+",
        type=float,
        default=list(FORGET_FRACTIONS),
        metavar="FF",
        help="forget fractions (default: 0.01 0.05 0.10)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=len(TRAIN_SEEDS),
        help="number of training seeds, starting at 0 (default: 10)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(UNLEARN_METHODS),
        metavar="METHOD",
        help="unlearning methods (default: the four approximate methods)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="cache for original models and oracles (default: checkpoints)",
    )
    parser.add_argument(
        "--output",
        default="results/primary.csv",
        help="destination CSV (default: results/primary.csv)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="mini-batch size; omit for full-batch training (Appendix A.6)",
    )
    parser.add_argument(
        "--unlearn-lr",
        type=float,
        default=None,
        help="override the unlearning learning rate (Appendix A.7)",
    )
    parser.add_argument(
        "--forget-seed",
        type=int,
        default=None,
        help="override the forget-set sampling seed (Appendix A.8)",
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device (default: cpu)"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.forget_seed is not None and args.checkpoint_dir == "checkpoints":
        print(
            "refusing to run: a different --forget-seed changes the retain set, so "
            "the oracles differ from the cached ones. Pass a separate "
            "--checkpoint-dir.",
            file=sys.stderr,
        )
        return 2
    if args.batch_size is not None and args.checkpoint_dir == "checkpoints":
        print(
            "refusing to run: --batch-size changes how the original models and "
            "oracles are trained. Pass a separate --checkpoint-dir.",
            file=sys.stderr,
        )
        return 2

    frame = run_experiment(
        datasets=args.datasets,
        forget_fractions=args.forget_fractions,
        seeds=range(args.seeds),
        methods=args.methods,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        unlearn_lr=args.unlearn_lr,
        forget_seed=args.forget_seed,
        device=args.device,
        train_kwargs={"batch_size": args.batch_size},
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"\nWrote {len(frame)} rows to {output}")
    print("Summarise with: python experiments/analyse.py " + str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
