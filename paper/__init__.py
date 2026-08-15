"""Experiments for the RULER paper.

This package reproduces the paper's results. It is *not* the library: it is one
consumer of it, importing the four metrics from :mod:`ruler` and adding
everything specific to the published experiments -- the ten datasets, the
tabular MLP of Fig. 2, the five unlearning methods, the statistical analysis,
and the table and figure generation.

Kept separate so the library stays small. If you are verifying your own models,
you want ``ruler``; this package is here so the paper's numbers can be
regenerated.

Needs the extras:  pip install -e '.[paper]'
"""
