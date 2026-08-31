"""Behavioural tests for the four variants.

The two that matter are ``test_camera_carries_a_silent_talker`` and
``test_array_carries_an_occluded_talker``: they are the dual-blind-spot claim
reduced to its smallest possible experiment, one sensor at a time, with
hand-built measurements so the test says exactly what it means.
"""

from __future__ import annotations

import numpy as np
import pytest

from avrfs import (FilterConfig, WorldConfig, evaluate, make_filter,
                   make_world, sense_audio, sense_video)
from avrfs.gm import wrap
from avrfs.sensors import SensorFrame

MODES = ["audio_only", "video_only", "naive_fusion", "fusion"]


def _frames(z_list):
    return [SensorFrame(z=np.atleast_1d(np.asarray(z, float)),
                        origin=np.full(np.size(z), -1, dtype=int))
            for z in z_list]


def _weight_near(gm, az, gate_deg=20.0):
    if gm.n == 0:
        return 0.0
    near = np.flatnonzero(np.abs(wrap(gm.m[:, 0] - az)) < np.deg2rad(gate_deg))
    return float(gm.w[near].sum()) if near.size else 0.0


@pytest.mark.parametrize("mode", MODES)
def test_empty_streams_invent_nobody(mode):
    f = make_filter(mode)
    out = f.run(_frames([[]] * 40), _frames([[]] * 40))
    assert all(np.isfinite(v) for v in out.n_present)
    assert max(out.n_present) < 1.0
    assert all(np.size(e) == 0 for e in out.estimates)


@pytest.mark.parametrize("mode", MODES)
def test_a_clean_talker_is_tracked(mode):
    """Both sensors see one steady talker: everything should find them."""
    cfg = FilterConfig()
    az = 0.6
    rng = np.random.default_rng(0)
    n = 120
    au = _frames([[az + rng.normal(0, cfg.audio.sigma)] for _ in range(n)])
    vi = _frames([[az + rng.normal(0, cfg.video.sigma)] for _ in range(n)])
    f = make_filter(mode)
    out = f.run(au, vi)
    assert _weight_near(out.mixtures[-1] if out.mixtures else f.gm, az) > 0.5 \
        or _weight_near(f.gm, az) > 0.5


def test_camera_carries_a_silent_talker():
    """Audio goes quiet; only video keeps reporting. The array-only filter
    must lose the talker and both fusion filters must not."""
    cfg = FilterConfig()
    az = 0.6
    rng = np.random.default_rng(1)
    n_on, n_quiet = 40, 40
    au = _frames([[az + rng.normal(0, cfg.audio.sigma)] for _ in range(n_on)]
                 + [[]] * n_quiet)
    vi = _frames([[az + rng.normal(0, cfg.video.sigma)]
                  for _ in range(n_on + n_quiet)])

    w = {}
    for mode in ("audio_only", "fusion"):
        f = make_filter(mode)
        f.run(au if mode != "video_only" else None, vi)
        w[mode] = _weight_near(f.gm, az)
    assert w["audio_only"] < 0.05, "audio-only should lose a silent talker"
    assert w["fusion"] > 0.5, "the camera should carry them"


def test_array_carries_an_occluded_talker():
    """The mirror image: video goes dark, audio keeps reporting."""
    cfg = FilterConfig()
    az = -0.4
    rng = np.random.default_rng(2)
    n_on, n_dark = 40, 40
    au = _frames([[az + rng.normal(0, cfg.audio.sigma)]
                  for _ in range(n_on + n_dark)])
    vi = _frames([[az + rng.normal(0, cfg.video.sigma)] for _ in range(n_on)]
                 + [[]] * n_dark)

    f = make_filter("video_only")
    f.run(None, vi)
    w_video = _weight_near(f.gm, az)
    g = make_filter("fusion")
    g.run(au, vi)
    w_fusion = _weight_near(g.gm, az)
    assert w_video < 0.05, "video-only should lose an occluded talker"
    assert w_fusion > 0.5, "the array should carry them"


def test_latent_modes_track_which_sensor_is_reporting():
    """pi follows the audio stream, v follows the video stream."""
    cfg = FilterConfig()
    az = 0.2
    rng = np.random.default_rng(3)
    n = 40
    au = _frames([[az + rng.normal(0, cfg.audio.sigma)] for _ in range(n)]
                 + [[]] * n)
    vi = _frames([[]] * n
                 + [[az + rng.normal(0, cfg.video.sigma)] for _ in range(n)])

    f = make_filter("fusion")
    out = f.run(au, vi, record=True)

    def modes(gm):
        near = np.flatnonzero(np.abs(wrap(gm.m[:, 0] - az)) < np.deg2rad(25))
        if near.size == 0:
            return np.nan, np.nan
        i = near[np.argmax(gm.w[near])]
        return float(gm.aux["pi"][i]), float(gm.aux["v"][i])

    pi_a, v_a = modes(out.mixtures[n - 1])     # audio reporting, video silent
    pi_b, v_b = modes(out.mixtures[-1])        # video reporting, audio silent
    assert pi_a > 0.7 and v_a < 0.6
    assert pi_b < 0.4 and v_b > 0.7


def test_fusion_beats_naive_on_a_fixed_world():
    w = make_world(WorldConfig(duration=40.0, n_talkers=3), seed=0)
    au, vi = sense_audio(w, seed=0), sense_video(w, seed=0)
    fus = evaluate(w, make_filter("fusion").run(au, vi))
    nai = evaluate(w, make_filter("naive_fusion").run(au, vi))
    assert fus.ospa <= nai.ospa
    assert fus.card_rmse <= nai.card_rmse


def test_fusion_beats_either_sensor_alone():
    w = make_world(WorldConfig(duration=40.0, n_talkers=3), seed=1)
    au, vi = sense_audio(w, seed=1), sense_video(w, seed=1)
    fus = evaluate(w, make_filter("fusion").run(au, vi))
    a = evaluate(w, make_filter("audio_only").run(au, None))
    v = evaluate(w, make_filter("video_only").run(None, vi))
    assert fus.ospa <= min(a.ospa, v.ospa)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        make_filter("magic")
