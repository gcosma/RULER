#!/usr/bin/env python3
"""Run the oracle-free M4 diagnostic on any pre-computed embeddings.

M4 needs only penultimate-layer embeddings and the retain/forget split, so it
applies to any architecture. This script takes the embeddings as ``.npy`` files
and stays independent of how they were produced -- the image, clinical-text and
face-identity settings of Section 5.5 differ only in which layer is extracted
(Appendix Table 2), not in the metric.

    python experiments/run_m4_diagnostic.py --forget forget.npy --retain retain.npy

Each file is an ``(n, p)`` array of embeddings under a *single* model. Run it on
the original model for the pre-unlearning diagnostic and on each unlearned
model for the post-unlearning check.

Extracting the right layer, per Appendix Table 2:

    # Tabular MLP -- the second ReLU
    from paper.models import penultimate
    embeddings = penultimate(model, x)

    # ResNet-18 -- the post-global-average-pooling activation (512-d)
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    embeddings = backbone(images).flatten(1).cpu().numpy()

    # Three-layer CNN -- the second fully-connected layer (256-d)
    # BERT -- the [CLS] activation of the final transformer layer (768-d)
    embeddings = model.bert(**batch).last_hidden_state[:, 0, :].cpu().numpy()

Whatever the source, the model must be in evaluation mode so that dropout and
batch-norm updates are inactive: M4 compares records geometrically, which
requires each embedding to be deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from ruler import m4  # noqa: E402

M4_RETAIN_CAP = 2000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--forget", required=True, help="(n, p) .npy of forget embeddings")
    parser.add_argument("--retain", required=True, help="(m, p) .npy of retain embeddings")
    parser.add_argument(
        "--gallery-cap",
        type=int,
        default=M4_RETAIN_CAP,
        help=f"retain records used in the neighbour search (default {M4_RETAIN_CAP})",
    )
    parser.add_argument("--label", default="", help="name to print with the result")
    args = parser.parse_args(argv)

    forget = np.load(args.forget)
    retain = np.load(args.retain)
    for name, array in (("forget", forget), ("retain", retain)):
        if array.ndim != 2:
            print(
                f"{name} embeddings must be 2-D (n, p), got shape {array.shape}",
                file=sys.stderr,
            )
            return 2

    if args.gallery_cap and len(retain) > args.gallery_cap:
        idx = np.random.RandomState(42).choice(
            len(retain), args.gallery_cap, replace=False
        )
        retain = retain[idx]

    rank = m4(forget, retain)
    deviation = rank - 0.50
    if abs(deviation) <= 0.03:
        reading = "at the null: indistinguishable from retained records"
    elif deviation > 0:
        reading = "above the null: integration with the retain set"
    else:
        reading = "below the null: over-displacement from the retain distribution"

    if args.label:
        print(args.label)
    print(
        f"M4 = {rank:.4f} (null 0.50, |D_f| = {len(forget)}, "
        f"gallery = {len(retain)}): {reading}"
    )
    print(
        "\nRead against the null of 0.50: above indicates the forget records "
        "remain\nintegrated with the retain set; below indicates they have been "
        "displaced\nfurther than correct retraining would leave them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
