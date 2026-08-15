"""Shared plot style and palettes for the paper's figures.

Two palettes are provided.  ``PAPER_PALETTE`` reproduces the published figures
exactly and is the default.  ``ACCESSIBLE_PALETTE`` is a re-stepped version in
the same hue families that passes colour-vision-deficiency separation checks;
the published palette does not (its green and blue differ by less than the
readable threshold even for normal colour vision, and its marks fall below 3:1
contrast against the panel background).

In both palettes a colour is bound to a *method*, never to a series position,
so filtering the methods shown never repaints the survivors.  Every figure
also carries a legend, so identity is never conveyed by colour alone.
"""

from __future__ import annotations

import matplotlib as mpl

__all__ = [
    "PAPER_PALETTE",
    "ACCESSIBLE_PALETTE",
    "NULL_COLOUR",
    "SINGLE_COLUMN",
    "DOUBLE_COLUMN",
    "apply_style",
    "method_colours",
]

#: Colours used in the published figures.
PAPER_PALETTE = {
    "Gradient Ascent": "#E8857A",
    "NegGrad+": "#7BAFD4",
    "Fine-Tuning": "#82C49B",
    "SCRUB": "#B39DCA",
    "Bad Teacher": "#D9B45B",
    "Oracle": "#5A5A5A",
}

#: Same hue families, re-stepped to pass CVD separation and contrast checks.
ACCESSIBLE_PALETTE = {
    "Gradient Ascent": "#B8462F",
    "NegGrad+": "#2C6EA5",
    "Fine-Tuning": "#4E8F45",
    "SCRUB": "#8B5AA6",
    "Bad Teacher": "#A07A1F",
    "Oracle": "#4A4A4A",
}

#: Reference lines (metric nulls) are drawn in neutral ink, never a series hue,
#: so they cannot be mistaken for data.
NULL_COLOUR = "#5A5A5A"

#: Figure widths in inches for a two-column paper.
SINGLE_COLUMN = 3.5
DOUBLE_COLUMN = 7.16


def _lighten(hex_colour: str, amount: float = 0.55) -> str:
    """Blend towards white for box fills, keeping the edge as the identity."""
    rgb = mpl.colors.to_rgb(hex_colour)
    blended = tuple(channel + (1.0 - channel) * amount for channel in rgb)
    return mpl.colors.to_hex(blended)


def method_colours(accessible: bool = False) -> tuple[dict, dict]:
    """Return ``(edge_colours, fill_colours)`` keyed by method name."""
    palette = ACCESSIBLE_PALETTE if accessible else PAPER_PALETTE
    return dict(palette), {k: _lighten(v) for k, v in palette.items()}


def apply_style() -> None:
    """Serif, small-type style matching the paper's figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "#CCCCCC",
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
        }
    )
