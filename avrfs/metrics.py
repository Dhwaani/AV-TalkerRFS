"""Evaluation metrics.

Three families, deliberately separated:

* **OSPA** (Schuhmacher, Vo & Vo 2008) against two different ground truths --
  the set of talkers *present* and the set of talkers *actively talking*.  A
  filter that reports only who is talking right now scores well on the second
  and badly on the first; reporting both is what keeps the comparison honest.
* **Cardinality error**, again split present/active.
* **Continuity**, which is what a downstream system (beamformer steering,
  diarisation, camera framing) actually feels.  Measured without any label
  post-process: for each present talker and frame, is there an estimate inside
  a gate?  ``coverage`` is the fraction of frames answered yes and ``breaks``
  is the number of times the answer flips from yes to no while the talker is
  still in the room.

  ``breaks`` on its own is a trap, and is never reported without the other
  two: a filter that finds nobody scores zero breaks, and so does a perfect
  one, so the count peaks somewhere in the middle.  ``hold_time`` is the one to
  read -- the mean length in seconds of an unbroken stretch of coverage, which
  rises monotonically as a tracker gets better at keeping hold of a talker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .gm import wrap

__all__ = ["ospa", "TrackingScore", "evaluate", "coverage_and_breaks"]

DEFAULT_CUTOFF = np.deg2rad(15.0)


def ospa(x: np.ndarray, y: np.ndarray, cutoff: float = DEFAULT_CUTOFF,
         p: float = 1.0) -> float:
    """OSPA distance between two sets of azimuths (radians)."""
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    nx, ny = x.size, y.size
    if nx == 0 and ny == 0:
        return 0.0
    if nx == 0 or ny == 0:
        return float(cutoff)
    if nx > ny:
        x, y = y, x
        nx, ny = ny, nx
    d = np.minimum(np.abs(wrap(x[:, None] - y[None, :])), cutoff)
    r, c = linear_sum_assignment(d ** p)
    total = float((d[r, c] ** p).sum()) + (cutoff ** p) * (ny - nx)
    return float((total / ny) ** (1.0 / p))


def coverage_and_breaks(truth_per_frame: list[np.ndarray],
                        est_per_frame: list[np.ndarray],
                        gate: float = DEFAULT_CUTOFF,
                        truth_ids: list[np.ndarray] | None = None,
                        dt: float = 1.0
                        ) -> tuple[float, int, float]:
    """Continuity of coverage for each ground-truth object.

    Returns ``(coverage, breaks, hold_time)``.  ``truth_ids`` lets the
    per-object statistics be counted correctly; without it, objects are matched
    frame-to-frame by index, which is only right if the caller supplies a
    stable ordering.  ``dt`` converts the hold time from frames to seconds.
    """
    n_frames = len(truth_per_frame)
    covered: dict[int, list[tuple[int, bool]]] = {}
    hit = 0
    total = 0
    for k in range(n_frames):
        t = np.atleast_1d(np.asarray(truth_per_frame[k], dtype=float)).ravel()
        e = np.atleast_1d(np.asarray(est_per_frame[k], dtype=float)).ravel()
        ids = (truth_ids[k] if truth_ids is not None
               else np.arange(t.size))
        for i in range(t.size):
            ok = bool(e.size and np.min(np.abs(wrap(t[i] - e))) <= gate)
            hit += int(ok)
            total += 1
            covered.setdefault(int(ids[i]), []).append((k, ok))

    breaks = 0
    runs: list[int] = []
    for _, seq in covered.items():
        flags = [ok for _, ok in seq]
        # Count yes -> no transitions that are followed by another yes, i.e.
        # genuine interruptions rather than the tail of the track.
        for i in range(1, len(flags)):
            if flags[i - 1] and not flags[i] and any(flags[i:]):
                breaks += 1
        run = 0
        for ok in flags:
            if ok:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
    hold = float(np.mean(runs) * dt) if runs else 0.0
    return (hit / total if total else 0.0), breaks, hold


@dataclass
class TrackingScore:
    """Aggregate scores for one run (present set only -- this repo has no
    active/present split in its outputs; who is *speaking* is a latent the
    filter estimates, but the tracking target is who is *there*)."""

    ospa: float
    card_rmse: float
    card_bias: float
    coverage: float
    breaks: int
    hold: float
    mean_components: float

    def as_row(self) -> dict[str, float]:
        return {
            "OSPA_deg": float(np.rad2deg(self.ospa)),
            "card_RMSE": self.card_rmse,
            "card_bias": self.card_bias,
            "coverage": self.coverage,
            "breaks": float(self.breaks),
            "hold_s": self.hold,
            "mean_components": self.mean_components,
        }


def evaluate(world, out, cutoff: float = DEFAULT_CUTOFF) -> TrackingScore:
    """Score a FilterOutput against a World (present-talker ground truth)."""
    n = world.n_frames
    truth = [world.present_azimuths(k) for k in range(n)]
    ids = [np.array([t.tid for t in world.talkers if t.present(k)])
           for k in range(n)]
    est = [np.asarray(e)[:, 0] if np.size(e) else np.zeros(0)
           for e in out.estimates]

    o = float(np.mean([ospa(truth[k], est[k], cutoff) for k in range(n)]))
    np_true = np.array([world.n_present(k) for k in range(n)], dtype=float)
    np_est = np.array(out.n_present, dtype=float)
    cov, brk, hold = coverage_and_breaks(truth, est, cutoff, ids, world.dt)

    return TrackingScore(
        ospa=o,
        card_rmse=float(np.sqrt(np.mean((np_est - np_true) ** 2))),
        card_bias=float(np.mean(np_est - np_true)),
        coverage=cov, breaks=brk, hold=hold,
        mean_components=float(np.mean(out.n_components)),
    )
