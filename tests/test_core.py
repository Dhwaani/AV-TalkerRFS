"""Tests for the world, the sensors, and the shared mixture machinery.

Properties, not stored numbers. The regression cases inherited from TalkerRFS
live in ``test_regression.py`` and travel with the shared modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrfs import (ActivityParams, CameraParams, OcclusionParams, WorldConfig,
                   make_world, ospa, sense_audio, sense_video)
from avrfs.filters import _chain
from avrfs.gm import GaussianMixture, wrap
from avrfs.world import _sample_two_state


# ---------------- the two chains ----------------
def test_occlusion_chain_matrix_is_stochastic_and_stationary():
    for dt in (0.01, 0.064, 0.5):
        T = _chain(4.0, 2.0, dt)
        assert np.allclose(T.sum(axis=1), 1.0)
        assert (T >= 0).all() and (T <= 1).all()
        vals, vecs = np.linalg.eig(T.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        v = v / v.sum()
        assert v[1] == pytest.approx(4.0 / 6.0, abs=1e-9)


def test_occlusion_chain_composes_over_time():
    assert np.allclose(_chain(4.0, 2.0, 0.05) @ _chain(4.0, 2.0, 0.05),
                       _chain(4.0, 2.0, 0.10), atol=1e-12)


def test_two_state_sampler_hits_its_stationary_share():
    rng = np.random.default_rng(0)
    a = _sample_two_state(400_000, 4.0, 2.0, 0.016, rng)
    assert a.mean() == pytest.approx(4.0 / 6.0, abs=0.02)


# ---------------- the world ----------------
def test_speech_and_occlusion_are_independent():
    """The dual-blind-spot design requires the two processes be uncorrelated.

    Averaged over seeds, not measured on one. Both chains have multi-second
    holding times, so a single run of a few thousand frames carries only a few
    hundred *effective* samples and its sample correlation scatters by several
    hundredths -- a per-seed threshold tight enough to be meaningful would be
    flaky, and one loose enough to be stable would test nothing.
    """
    cfg = WorldConfig(duration=200.0, n_talkers=3, always_present=True)
    corrs = []
    for seed in range(20):
        w = make_world(cfg, seed=seed)
        spk = np.concatenate([t.speaking for t in w.talkers]).astype(float)
        occ = np.concatenate([t.occluded for t in w.talkers]).astype(float)
        corrs.append(float(np.corrcoef(spk, occ)[0, 1]))
    assert abs(float(np.mean(corrs))) < 0.02


def test_all_four_regimes_are_populated():
    cfg = WorldConfig(duration=200.0, n_talkers=3, always_present=True)
    w = make_world(cfg, seed=1)
    counts = {"both": 0, "audio": 0, "video": 0, "dark": 0}
    for t in w.talkers:
        for k in range(w.n_frames):
            a, v = w.audio_lit(t, k), w.video_lit(t, k)
            counts["both" if (a and v) else "audio" if a
                   else "video" if v else "dark"] += 1
    total = sum(counts.values())
    for name, c in counts.items():
        assert c / total > 0.03, f"regime {name} is too rare to test in"


def test_narrow_fov_camera_blinds_outside_its_arc():
    cfg = WorldConfig(duration=60.0, n_talkers=3, always_present=True,
                      camera=CameraParams(boresight=0.0,
                                          half_fov=np.deg2rad(45.0)))
    w = make_world(cfg, seed=2)
    for t in w.talkers:
        for k in range(w.n_frames):
            if w.video_lit(t, k):
                assert abs(wrap(t.azimuth[k])) <= np.deg2rad(45.0) + 1e-9


# ---------------- the sensors ----------------
def test_audio_sensor_is_blind_to_silence():
    w = make_world(WorldConfig(duration=60.0, n_talkers=3), seed=3)
    frames = sense_audio(w, seed=3)
    for k, f in enumerate(frames):
        for tid in f.origin:
            if tid >= 0:
                assert w.talkers[tid].speaking[k]


def test_video_sensor_is_blind_to_occlusion():
    w = make_world(WorldConfig(duration=60.0, n_talkers=3), seed=4)
    frames = sense_video(w, seed=4)
    for k, f in enumerate(frames):
        for tid in f.origin:
            if tid >= 0:
                assert not w.talkers[tid].occluded[k]


def test_video_clutter_stays_inside_the_field_of_view():
    cfg = WorldConfig(duration=60.0, n_talkers=2,
                      camera=CameraParams(boresight=0.5,
                                          half_fov=np.deg2rad(50.0)))
    w = make_world(cfg, seed=5)
    for f in sense_video(w, seed=5):
        for a in f.z:
            assert abs(wrap(a - 0.5)) <= np.deg2rad(50.0) + 1e-6


def test_audio_and_video_disagree_about_who_is_visible():
    """If the two sensors saw the same talkers, there would be nothing to fuse."""
    w = make_world(WorldConfig(duration=120.0, n_talkers=3,
                               always_present=True), seed=6)
    a_lit = np.array([[w.audio_lit(t, k) for k in range(w.n_frames)]
                      for t in w.talkers]).ravel()
    v_lit = np.array([[w.video_lit(t, k) for k in range(w.n_frames)]
                      for t in w.talkers]).ravel()
    disagree = float(np.mean(a_lit != v_lit))
    assert disagree > 0.25


# ---------------- metrics ----------------
def test_ospa_identity_and_wrap():
    x = np.array([0.1, -1.2, 2.0])
    assert ospa(x, x) == 0.0
    c = np.deg2rad(15.0)
    assert ospa(np.array([np.pi - 0.01]), np.array([-np.pi + 0.01]), c) == \
        pytest.approx(0.02, abs=1e-9)


def test_mixture_merge_across_the_seam():
    gm = GaussianMixture(np.array([0.6, 0.5]),
                         np.array([[np.pi - 0.01, 0.0], [-np.pi + 0.01, 0.0]]),
                         np.repeat(np.diag([0.01, 0.1])[None], 2, axis=0))
    out = gm.merge(4.0)
    assert out.n == 1
    assert out.w[0] == pytest.approx(1.1)
