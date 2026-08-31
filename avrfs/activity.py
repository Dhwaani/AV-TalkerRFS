"""Conversational speech-activity model.

[Adapted from the author's TalkerRFS project; kept byte-compatible where
possible so fixes and regression tests flow between the two repos.]

The whole point of TalkerRFS is that a talker's speech activity is *not* an
unknown nuisance parameter: conversational speech has well-characterised
talkspurt / pause statistics, and those statistics are exactly the prior that a
random-finite-set tracker needs in order to tell "this person stopped talking"
apart from "this person left the room".

We use the two-state (talkspurt / pause) alternating renewal process that
underlies ITU-T P.59 "Artificial conversational speech" and Brady's two-state
Markov model of conversation.  Nominal single-talker parameters are

    mean talkspurt   ~ 1.00 s
    mean pause       ~ 1.59 s
    activity factor  ~ 0.39

which are the values widely quoted from P.59 in the VoIP/traffic literature.
They are *defaults*, not assertions: everything downstream reads them from
``ActivityParams`` so a different corpus (AMI, CHiME, LOCATA) can be dropped in
by changing two numbers.  See ``docs/related-work.md`` for the provenance note.

Two samplers are provided:

``sample_markov``
    Memoryless (geometric holding times).  This is the model the *filter*
    assumes, and it is the one for which the mode-transition matrix is exact.

``sample_semi_markov``
    Log-normal holding times, which is a much better fit to measured
    conversational data (pause durations are famously heavy-tailed).  This is
    what the *simulator* uses by default, so the proposed filter is always
    evaluated under model mismatch rather than on its own generative model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ActivityParams",
    "transition_matrix",
    "stationary_activity",
    "sample_markov",
    "sample_semi_markov",
    "sample_activity",
]


@dataclass(frozen=True)
class ActivityParams:
    """Talkspurt / pause statistics for a single talker in conversation.

    Attributes
    ----------
    mean_talkspurt : float
        Mean duration of a talkspurt, in seconds.
    mean_pause : float
        Mean duration of a pause, in seconds.
    lognormal_sigma : float
        Shape parameter of the log-normal holding-time distribution used by the
        semi-Markov sampler.  ``0`` collapses it to deterministic durations.
    """

    mean_talkspurt: float = 1.004
    mean_pause: float = 1.587
    lognormal_sigma: float = 0.9

    @property
    def activity_factor(self) -> float:
        """Long-run fraction of time the talker is active."""
        return self.mean_talkspurt / (self.mean_talkspurt + self.mean_pause)


def transition_matrix(params: ActivityParams, dt: float) -> np.ndarray:
    """Two-state mode-transition matrix at frame rate ``dt``.

    Returns ``T`` with ``T[i, j] = P(mode_{k+1} = j | mode_k = i)`` and the
    convention ``0 = pause``, ``1 = active``.

    Derived as the matrix exponential of the continuous-time generator with
    rates ``1 / mean_talkspurt`` (active -> pause) and ``1 / mean_pause``
    (pause -> active).  Using the exact exponential rather than the usual
    first-order approximation ``lambda * dt`` matters here because the frame
    hop (10-100 ms) is not always negligible against the holding times, and
    because the exact form is guaranteed to stay a valid stochastic matrix.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    a = dt / params.mean_talkspurt   # active -> pause rate * dt
    b = dt / params.mean_pause       # pause  -> active rate * dt
    s = a + b
    e = np.exp(-s)
    # Closed-form exp of [[-b, b], [a, -a]] acting on (pause, active).
    t_pa = (b / s) * (1.0 - e)       # pause -> active
    t_ap = (a / s) * (1.0 - e)       # active -> pause
    return np.array([[1.0 - t_pa, t_pa],
                     [t_ap, 1.0 - t_ap]], dtype=float)


def stationary_activity(params: ActivityParams) -> float:
    """Stationary probability of the active mode (== the activity factor)."""
    return params.activity_factor


def sample_markov(n_frames: int, params: ActivityParams, dt: float,
                  rng: np.random.Generator, start_active: bool | None = None
                  ) -> np.ndarray:
    """Sample a binary activity sequence from the memoryless model."""
    T = transition_matrix(params, dt)
    p_active = stationary_activity(params)
    state = bool(rng.random() < p_active) if start_active is None else bool(start_active)
    out = np.empty(n_frames, dtype=bool)
    for k in range(n_frames):
        out[k] = state
        state = bool(rng.random() < T[int(state), 1])
    return out


def _lognormal_durations(mean: float, sigma: float, n: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Log-normal durations with the requested arithmetic mean."""
    if sigma <= 0:
        return np.full(n, mean)
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def sample_semi_markov(n_frames: int, params: ActivityParams, dt: float,
                       rng: np.random.Generator,
                       start_active: bool | None = None) -> np.ndarray:
    """Sample activity from log-normal holding times (model mismatch case)."""
    p_active = stationary_activity(params)
    state = bool(rng.random() < p_active) if start_active is None else bool(start_active)
    out = np.empty(n_frames, dtype=bool)
    k = 0
    # The first segment is the one in progress when observation starts, so it is
    # drawn from the *length-biased* law and then truncated uniformly.  Drawing
    # it from the ordinary law and halving it -- the obvious thing -- makes the
    # opening talkspurt and pause about half as long as they should be, and
    # every scenario starts in that transient.
    first = True
    while k < n_frames:
        mean = params.mean_talkspurt if state else params.mean_pause
        sigma = params.lognormal_sigma
        if first:
            # Size-biased lognormal: mu -> mu + sigma^2, i.e. the same law with
            # its arithmetic mean scaled by exp(sigma^2).
            dur = float(_lognormal_durations(mean * np.exp(sigma ** 2), sigma,
                                             1, rng)[0])
        else:
            dur = float(_lognormal_durations(mean, sigma, 1, rng)[0])
        n_hold = max(1, int(round(dur / dt)))
        if first:
            n_hold = max(1, int(round(n_hold * rng.random())))
            first = False
        n_hold = min(n_hold, n_frames - k)
        out[k:k + n_hold] = state
        k += n_hold
        state = not state
    return out


def sample_activity(n_frames: int, params: ActivityParams, dt: float,
                    rng: np.random.Generator, kind: str = "semi_markov",
                    start_active: bool | None = None) -> np.ndarray:
    """Dispatch to the requested activity sampler."""
    if kind == "markov":
        return sample_markov(n_frames, params, dt, rng, start_active)
    if kind == "semi_markov":
        return sample_semi_markov(n_frames, params, dt, rng, start_active)
    if kind == "always_on":
        return np.ones(n_frames, dtype=bool)
    raise ValueError(f"unknown activity model: {kind!r}")
