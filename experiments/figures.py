#!/usr/bin/env python3
"""Regenerate the paper's figures from a results CSV.

    python experiments/figures.py results/primary.csv --outdir figures

Produces:

* ``fig1_discordance.pdf``   -- MIA accuracy and M2 at ff = 5% (Fig. 1)
* ``fig3_m4_by_fraction.pdf``-- M4 across forget fractions (Fig. 3)
* ``fig7_m4_per_dataset.pdf``-- per-dataset M4 at ff = 5% (Fig. 7)

``--accessible`` swaps in the colour-vision-safe palette; the default
reproduces the published colours. Pass ``--oracle-null`` with the CSV from
``run_oracle_calibration.py`` to also draw Fig. 5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper.config import M2_NULL, M4_NULL, METHODS, MIA_NULL  # noqa: E402
from paper.plotting import (  # noqa: E402
    DOUBLE_COLUMN,
    NULL_COLOUR,
    apply_style,
    method_colours,
)


def _methods_present(frame: pd.DataFrame, include_oracle: bool = False) -> list[str]:
    present = set(frame["method"].unique())
    ordered = [m for m in METHODS if m in present]
    for extra in sorted(present - set(ordered) - {"Oracle"}):
        ordered.append(extra)
    if include_oracle and "Oracle" in present:
        ordered.append("Oracle")
    return ordered


def _boxplot(ax, frame, column, methods, edges, fills, jitter_seed=0):
    """Box plot with the underlying points overlaid.

    The points are shown because the effects are small relative to the spread:
    a box alone would hide how consistent the direction is across the 100
    observations behind each box.
    """
    groups = [frame[frame["method"] == m][column].dropna().to_numpy() for m in methods]
    artists = ax.boxplot(
        groups,
        positions=np.arange(len(methods)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )
    rng = np.random.RandomState(jitter_seed)
    for position, (method, values, box) in enumerate(
        zip(methods, groups, artists["boxes"], strict=True)
    ):
        box.set_facecolor(fills[method])
        box.set_edgecolor(edges[method])
        box.set_linewidth(0.9)
        if len(values):
            ax.scatter(
                position + rng.uniform(-0.16, 0.16, len(values)),
                values,
                s=3,
                color=edges[method],
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=20, ha="right")


def figure_discordance(frame, outdir, edges, fills, forget_fraction=0.05):
    """Fig. 1: all methods pass the output-level check, yet M2 is negative."""
    subset = frame[np.isclose(frame["forget_fraction"], forget_fraction)]
    if subset.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.9))

    mia_methods = _methods_present(subset, include_oracle=True)
    _boxplot(axes[0], subset, "mia_accuracy", mia_methods, edges, fills)
    axes[0].axhline(MIA_NULL, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
    axes[0].axhspan(MIA_NULL - 0.05, MIA_NULL + 0.05, color=NULL_COLOUR, alpha=0.07)
    axes[0].set_ylabel("MIA accuracy")
    axes[0].set_title("(a) Output level: all methods pass")

    m2_methods = _methods_present(subset)
    _boxplot(axes[1], subset, "m2", m2_methods, edges, fills)
    axes[1].axhline(M2_NULL, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("$M_2$: signed calibration gap")
    axes[1].set_title("(b) Representation level: residuals remain")

    fig.suptitle(
        f"Discordance at ff = {forget_fraction:.0%}   "
        f"(shaded band: $\\pm$0.05 pass window; dashed: null)",
        fontsize=8,
    )
    fig.tight_layout()
    path = outdir / "fig1_discordance.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_m4_by_fraction(frame, outdir, edges, fills):
    """Fig. 3: M4 for every method at each forget fraction."""
    fractions = sorted(frame["forget_fraction"].unique())
    fig, axes = plt.subplots(
        1, len(fractions), figsize=(DOUBLE_COLUMN, 2.9), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, forget_fraction in zip(axes, fractions, strict=False):
        subset = frame[np.isclose(frame["forget_fraction"], forget_fraction)]
        methods = _methods_present(subset)
        _boxplot(ax, subset, "m4", methods, edges, fills)
        ax.axhline(M4_NULL, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
        ax.set_title(f"ff = {forget_fraction:.0%}")
    axes[0].set_ylabel("$M_4$: percentile rank")
    fig.suptitle(
        "Oracle-free $M_4$ by forget fraction (dashed: null of 0.50)", fontsize=8
    )
    fig.tight_layout()
    path = outdir / "fig3_m4_by_fraction.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_m4_per_dataset(frame, outdir, edges, fills, forget_fraction=0.05):
    """Fig. 7: M4 per dataset, with the oracle as a within-dataset reference.

    M4's variance is dominated by dataset identity, so the per-dataset view is
    the one that carries meaning; the oracle panel entry shows what the same
    dataset's geometry produces under correct retraining.
    """
    subset = frame[np.isclose(frame["forget_fraction"], forget_fraction)]
    if subset.empty:
        return None
    datasets = list(dict.fromkeys(subset["dataset_name"]))
    methods = _methods_present(subset, include_oracle=True)

    columns = 5
    rows = int(np.ceil(len(datasets) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(DOUBLE_COLUMN, 2.1 * rows), squeeze=False
    )
    for index, dataset in enumerate(datasets):
        ax = axes[index // columns][index % columns]
        _boxplot(ax, subset[subset["dataset_name"] == dataset], "m4", methods, edges, fills)
        ax.axhline(M4_NULL, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
        ax.set_title(dataset, fontsize=7)
        if index % columns == 0:
            ax.set_ylabel("$M_4$")
    for empty in range(len(datasets), rows * columns):
        axes[empty // columns][empty % columns].axis("off")

    fig.suptitle(
        f"Per-dataset $M_4$ at ff = {forget_fraction:.0%} "
        f"(dashed: null of 0.50)",
        fontsize=8,
    )
    fig.tight_layout()
    path = outdir / "fig7_m4_per_dataset.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_m3_per_dataset(frame, outdir, edges, fills):
    """Fig. 6: representation shift M3 per dataset, across forget fractions.

    M3 asks whether unlearning moved forget records towards the oracle at all.
    Negative values mean it moved them further away, which is the dominant
    direction and the complement to M2's finding.
    """
    fractions = sorted(frame["forget_fraction"].unique())
    datasets = list(dict.fromkeys(frame["dataset_name"]))
    methods = _methods_present(frame)
    if not datasets:
        return None

    fig, axes = plt.subplots(
        len(fractions),
        len(datasets),
        figsize=(DOUBLE_COLUMN, 1.9 * len(fractions)),
        squeeze=False,
        sharey="row",
    )
    for row, forget_fraction in enumerate(fractions):
        for column, dataset in enumerate(datasets):
            ax = axes[row][column]
            subset = frame[
                np.isclose(frame["forget_fraction"], forget_fraction)
                & (frame["dataset_name"] == dataset)
            ]
            _boxplot(ax, subset, "m3", methods, edges, fills)
            ax.axhline(0.0, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
            ax.tick_params(labelbottom=(row == len(fractions) - 1))
            if row == 0:
                ax.set_title(dataset, fontsize=6)
            if column == 0:
                ax.set_ylabel(f"ff = {forget_fraction:.0%}\n$M_3$", fontsize=7)

    fig.suptitle(
        "Per-dataset representation shift $M_3$ (dashed: zero shift)", fontsize=8
    )
    fig.tight_layout()
    path = outdir / "fig6_m3_per_dataset.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_oracle_null(null_frame, frame, outdir, edges, fills, forget_fraction=0.05):
    """Fig. 5: the empirical null beside the unlearned distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN, 2.9))

    values = null_frame["m2"].dropna().to_numpy()
    box = axes[0].boxplot(
        [values], widths=0.5, patch_artist=True, showfliers=False,
        medianprops={"color": "#333333", "linewidth": 1.0},
    )
    box["boxes"][0].set_facecolor("#E6E6E6")
    box["boxes"][0].set_edgecolor(NULL_COLOUR)
    rng = np.random.RandomState(0)
    axes[0].scatter(
        1 + rng.uniform(-0.15, 0.15, len(values)), values,
        s=3, color=NULL_COLOUR, alpha=0.3, linewidths=0,
    )
    axes[0].axhline(M2_NULL, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
    axes[0].set_xticks([1])
    axes[0].set_xticklabels(["Oracle-Oracle"])
    axes[0].set_ylabel("$M_2$: signed calibration gap")
    axes[0].set_title(f"(a) Empirical null ({len(values)} oracle pairs)")

    subset = frame[np.isclose(frame["forget_fraction"], forget_fraction)]
    methods = _methods_present(subset)
    _boxplot(axes[1], subset, "m2", methods, edges, fills)
    axes[1].axhline(M2_NULL, color=NULL_COLOUR, linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("$M_2$: signed calibration gap")
    axes[1].set_title(f"(b) After unlearning (ff = {forget_fraction:.0%})")

    fig.suptitle(
        "$M_2$ is centred on zero under correct retraining, negative after "
        "unlearning",
        fontsize=8,
    )
    fig.tight_layout()
    path = outdir / "fig5_oracle_null.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", help="CSV written by run_primary.py")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument(
        "--oracle-null", help="CSV from run_oracle_calibration.py, to draw Fig. 5"
    )
    parser.add_argument(
        "--accessible",
        action="store_true",
        help="use the colour-vision-safe palette instead of the published colours",
    )
    args = parser.parse_args(argv)

    apply_style()
    edges, fills = method_colours(accessible=args.accessible)
    frame = pd.read_csv(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = [
        figure_discordance(frame, outdir, edges, fills),
        figure_m4_by_fraction(frame, outdir, edges, fills),
        figure_m3_per_dataset(frame, outdir, edges, fills),
        figure_m4_per_dataset(frame, outdir, edges, fills),
    ]
    if args.oracle_null:
        written.append(
            figure_oracle_null(
                pd.read_csv(args.oracle_null), frame, outdir, edges, fills
            )
        )

    for path in filter(None, written):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
