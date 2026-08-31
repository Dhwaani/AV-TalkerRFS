"""Regression tests inherited with the shared modules.

These guard defects that were found and fixed in the author's TalkerRFS
project, in code that this repository reuses verbatim (``avrfs.gm``,
``avrfs.activity``, ``avrfs.conformal``). They travel with the modules so a
fix cannot be silently undone by a copy-paste in either direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrfs import ActivityParams
from avrfs.activity import sample_activity
from avrfs.conformal import aps_scores, conformal_quantile, prediction_set
from avrfs.filters import FilterConfig
from avrfs.gm import GaussianMixture, wrap
from avrfs import make_filter


def test_merge_separations_are_conjunctive():
    """Two separation constraints must both hold, in one pass."""
    gm = GaussianMixture(
        np.array([0.9, 0.3]),
        np.array([[0.5, 0.0], [0.5, 0.0]]),
        np.stack([np.diag([0.001, 0.05])] * 2),
        {"pi": np.array([0.9, 0.9]), "v": np.array([0.8, 0.8]),
         "r": np.array([1.0, 0.35])})
    out = make_filter("fusion")._reduce(gm)
    assert out.n == 2, "a fresh birth was merged into a confirmed track"
    assert set(np.round(out.aux["r"], 2)) == {1.0, 0.35}


def test_cap_renormalisation_is_optional():
    rng = np.random.default_rng(0)
    n = 30
    gm = GaussianMixture(rng.random(n), rng.normal(0, 1, (n, 2)),
                         np.repeat(np.eye(2)[None], n, axis=0))
    assert gm.cap(10).mass == pytest.approx(gm.mass)
    kept = gm.cap(10, renormalise=False)
    assert kept.mass < gm.mass


def test_first_activity_segment_is_length_biased():
    """The segment in progress at t=0 comes from the length-biased law."""
    p = ActivityParams()
    dt = 0.064
    firsts_active, firsts_pause = [], []
    for seed in range(2000):
        rng = np.random.default_rng(seed)
        a = sample_activity(400, p, dt, rng, "semi_markov")
        run = int(np.argmax(a != a[0])) if (a != a[0]).any() else a.size
        (firsts_active if a[0] else firsts_pause).append(run * dt)
    for got, mean in ((np.mean(firsts_active), p.mean_talkspurt),
                      (np.mean(firsts_pause), p.mean_pause)):
        want = mean * np.exp(p.lognormal_sigma ** 2) / 2.0
        assert got == pytest.approx(want, rel=0.20), (got, want)


def test_steady_activity_statistics_are_unchanged():
    p = ActivityParams()
    rng = np.random.default_rng(0)
    a = sample_activity(300_000, p, 0.016, rng, "semi_markov")
    assert a.mean() == pytest.approx(p.activity_factor, abs=0.02)


def test_prediction_set_is_the_exact_inverse_of_the_score():
    """Using the HPD rule here instead inflates measured coverage."""
    rng = np.random.default_rng(12)
    for _ in range(400):
        pmf = rng.dirichlet(np.ones(int(rng.integers(2, 9))) * 0.7)
        qhat = float(rng.random())
        got = set(prediction_set(pmf, qhat).tolist())
        want = {y for y in range(pmf.size) if aps_scores(pmf, y) <= qhat}
        if not want:
            want = {int(np.argmax(pmf))}
        assert got == want


def test_conformal_sets_cover_at_the_nominal_rate():
    rng = np.random.default_rng(13)
    alpha = 0.1
    hits = []
    for _ in range(150):
        cal = []
        for _ in range(400):
            p = rng.dirichlet(np.ones(5))
            cal.append(aps_scores(p, int(rng.choice(5, p=p))))
        qhat = conformal_quantile(np.array(cal), alpha)
        p = rng.dirichlet(np.ones(5))
        y = int(rng.choice(5, p=p))
        hits.append(y in set(prediction_set(p, qhat).tolist()))
    cover = float(np.mean(hits))
    assert 1 - alpha - 0.06 <= cover <= 1 - alpha + 0.06
