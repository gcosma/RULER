"""RULER: representation-level verification of machine unlearning.

Four metrics that measure whether an unlearned model still encodes the records
it was asked to forget. You run your own experiment; the library takes the
resulting penultimate-layer embeddings and returns numbers.

    from ruler import m2, m4

    m4(forget_embeddings, retain_embeddings)          # null 0.50, no oracle needed
    m2(u_forget, o_forget, u_retain, o_retain)        # null 0, needs a retrain oracle

+--------+-----------------+------+---------------------------------------------+
| Metric | Needs an oracle | Null | Reading                                     |
+========+=================+======+=============================================+
| ``m1`` | yes             | --   | Mean similarity of forget records to the    |
|        |                 |      | oracle (Eq. 2)                              |
+--------+-----------------+------+---------------------------------------------+
| ``m2`` | yes             | 0    | Signed calibration gap; negative means      |
|        |                 |      | residual memorisation (Eq. 3)               |
+--------+-----------------+------+---------------------------------------------+
| ``m3`` | yes             | 0    | Did unlearning move records towards the     |
|        |                 |      | oracle? (Eq. 4)                             |
+--------+-----------------+------+---------------------------------------------+
| ``m4`` | **no**          | 0.50 | Percentile rank of forget records within    |
|        |                 |      | the retain set's own nearest-neighbour      |
|        |                 |      | distribution (Eqs. 5-7)                     |
+--------+-----------------+------+---------------------------------------------+

``all_metrics`` returns all four as a dict.

Depends only on numpy, so it works alongside PyTorch, JAX, TensorFlow or
scikit-learn without pulling in a framework.

The paper's own experiments live in ``paper/``, separate from the library.
"""

from .metrics import all_metrics, cosine_similarity, m1, m2, m3, m4

__version__ = "3.0.0"

__all__ = ["all_metrics", "cosine_similarity", "m1", "m2", "m3", "m4"]
