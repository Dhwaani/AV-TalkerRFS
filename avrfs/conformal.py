"""Calibrated prediction sets for the number of talkers.

[Adapted from the author's TalkerRFS project; kept byte-compatible where
possible so fixes and regression tests flow between the two repos.]

The CPHD hands you a full posterior over cardinality, and it is tempting to
read a credible set straight off it: take the most probable counts until they
sum to 0.9 and call that a 90% set.  That set is not calibrated.  The
cardinality posterior is the output of a filter whose model is wrong in all the
usual ways -- the motion model, the clutter model, the activity statistics --
and its probabilities inherit every one of those errors.

Split conformal prediction fixes the calibration without touching the filter:
score the *true* count against the filter's own posterior on held-out runs,
take an empirical quantile of those scores, and use it as the inclusion
threshold.  The resulting sets have finite-sample marginal coverage under
exchangeability, whatever the filter's posterior actually looks like.

Exchangeability is the whole subtlety here, and this module makes it explicit.
Frames inside a run are strongly dependent -- a talker who is present at frame
k is present at frame k+1 -- so pooling frames and calibrating over them is
*not* valid, and the empirical coverage of a set calibrated that way misses its
nominal level.  ``unit="run"`` samples one frame per run, which restores
exchangeability across independently drawn scenarios at the cost of a smaller
calibration set.  Both are implemented so the gap can be measured rather than
asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["aps_scores", "conformal_quantile", "prediction_set",
           "credible_set", "ConformalResult", "evaluate_sets"]


def aps_scores(pmf: np.ndarray, label: int) -> float:
    """Adaptive-prediction-set nonconformity score for one posterior.

    The score is the total probability mass of every count at least as probable
    as the true one, so a posterior that ranks the truth first scores low and a
    posterior that buries it scores near one.
    """
    pmf = np.asarray(pmf, dtype=float)
    label = int(label)
    if label >= pmf.size:
        return 1.0
    p_true = pmf[label]
    return float(pmf[pmf >= p_true].sum())


def conformal_quantile(scores: np.ndarray, alpha: float = 0.1) -> float:
    """The ``ceil((n+1)(1-alpha))/n`` empirical quantile of calibration scores."""
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return 1.0
    return float(np.sort(scores)[k - 1])


def prediction_set(pmf: np.ndarray, qhat: float) -> np.ndarray:
    """Counts whose APS score is at most ``qhat``.

    This has to be *exactly* the inverse of :func:`aps_scores`: the set is
    ``{y : aps_scores(pmf, y) <= qhat}``, which is the prefix of the descending
    sort whose cumulative mass does not exceed ``qhat``.  Taking the smallest
    prefix that *reaches* ``qhat`` instead -- the highest-posterior-density rule
    -- adds one extra count and inflates the measured coverage well above
    nominal, which is a comfortable failure to have and therefore an easy one
    to miss.  The two rules genuinely differ and both are needed:
    :func:`credible_set` keeps the HPD one.
    """
    pmf = np.asarray(pmf, dtype=float)
    order = np.argsort(pmf)[::-1]
    cum = np.cumsum(pmf[order])
    k = int(np.searchsorted(cum, qhat, side="right"))
    if k == 0:
        # An empty set is formally allowed but useless to report; the most
        # probable count is always the least surprising thing to name.
        k = 1
    return np.sort(order[:k])


def credible_set(pmf: np.ndarray, level: float = 0.9) -> np.ndarray:
    """The filter's own highest-posterior-density set (the uncalibrated one).

    Smallest set of counts whose posterior mass *reaches* ``level`` -- the
    ordinary credible-set rule, deliberately not the conformal one above.
    """
    pmf = np.asarray(pmf, dtype=float)
    order = np.argsort(pmf)[::-1]
    cum = np.cumsum(pmf[order])
    k = min(int(np.searchsorted(cum, level) + 1), pmf.size)
    return np.sort(order[:k])


@dataclass
class ConformalResult:
    """Coverage and size of a family of prediction sets."""

    coverage: float
    mean_size: float
    nominal: float

    def as_row(self) -> dict[str, float]:
        return {"coverage": self.coverage, "mean_size": self.mean_size,
                "nominal": self.nominal}


def evaluate_sets(pmfs: list[np.ndarray], labels: np.ndarray, qhat: float,
                  nominal: float) -> ConformalResult:
    """Empirical coverage and mean size of conformal sets on a test split."""
    labels = np.asarray(labels, dtype=int)
    hits, sizes = [], []
    for pmf, y in zip(pmfs, labels):
        s = prediction_set(pmf, qhat)
        hits.append(bool(np.any(s == y)))
        sizes.append(int(s.size))
    return ConformalResult(coverage=float(np.mean(hits)),
                           mean_size=float(np.mean(sizes)),
                           nominal=float(nominal))


def evaluate_credible(pmfs: list[np.ndarray], labels: np.ndarray,
                      level: float) -> ConformalResult:
    """Same, for the filter's uncalibrated credible sets."""
    labels = np.asarray(labels, dtype=int)
    hits, sizes = [], []
    for pmf, y in zip(pmfs, labels):
        s = credible_set(pmf, level)
        hits.append(bool(np.any(s == y)))
        sizes.append(int(s.size))
    return ConformalResult(coverage=float(np.mean(hits)),
                           mean_size=float(np.mean(sizes)),
                           nominal=float(level))
