"""The simulated world: talkers with two independent ways of going dark.

Each talker carries two binary processes that the sensors, not the talker,
care about:

``speaking``   the P.59-style talkspurt/pause chain (``avrfs.activity``).
               The microphone array cannot detect a silent talker, however
               well lit they are.
``occluded``   a two-state occlusion chain plus a hard camera field-of-view
               test. The camera cannot detect an occluded or out-of-view
               talker, however loudly they are speaking.

The two processes are sampled independently, which is the deliberate design of
the whole experiment: audio and video have *dual* blind spots, and the value of
fusion is exactly the probability that a talker is dark to one sensor while lit
for the other. The residual case -- silent AND occluded at once -- is dark to
everything, and only the priors can carry a track across it.

Scope & known limitations note on the occlusion chain: unlike the talkspurt/pause statistics,
which come from a published standard, the occlusion holding times here are
hand-set (nothing like ITU-T P.59 exists for "how long a walking person stays
behind another person"). ``OcclusionParams`` is therefore a knob the
experiments sweep, not a claim about the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activity import ActivityParams, sample_activity

__all__ = ["OcclusionParams", "CameraParams", "Talker", "WorldConfig",
           "World", "make_world"]


def _wrap(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class OcclusionParams:
    """Two-state visible/occluded chain for a talker inside the camera FOV.

    The defaults are chosen by one stated criterion, fixed before any result
    was looked at: *both sensors must carry a meaningful share of the frames*,
    so that the experiment measures complementary blind spots rather than one
    sensor doing all the work. At these values a present talker is visible-only
    ~41% of frames, audible-only ~13%, both ~26%, and dark to everything ~20%.
    Push visibility to 0.8 and the camera alone sees 83% of frames, the audio
    contributes almost nothing, and the comparison stops being about fusion.

    Because this is a chosen operating point rather than a measured one, the
    whole (visibility x activity) surface is swept in
    ``experiments/exp3_complementarity.py`` and reported in full -- including
    the corners where the proposed filter has no advantage.
    """

    mean_visible: float = 4.0     # s, mean unbroken stretch of visibility
    mean_occluded: float = 2.0    # s, mean occlusion episode

    @property
    def visibility_factor(self) -> float:
        """Long-run fraction of in-FOV time the talker is visible."""
        return self.mean_visible / (self.mean_visible + self.mean_occluded)


@dataclass(frozen=True)
class CameraParams:
    """A camera co-located with the array.

    The default is a 360-degree camera (``half_fov = pi``), which is what a
    conferencing puck or smart speaker actually carries and, more importantly,
    is the configuration that *isolates the mechanism under test*. With a
    narrow fixed FOV the camera is blind to most of the room for purely
    geometric reasons, and a talker ends up dark to both sensors nearly half
    the time -- which confounds "the two sensors have complementary blind
    spots" with "the camera cannot see most of the room". Narrow-FOV cameras
    are still supported and are swept explicitly in
    ``experiments/exp3_complementarity.py``.
    """

    boresight: float = 0.0                 # rad, direction the camera faces
    half_fov: float = np.pi                # rad, half field of view (pi = 360)

    def in_fov(self, azimuth) -> np.ndarray:
        return np.abs(_wrap(np.asarray(azimuth) - self.boresight)) <= self.half_fov


@dataclass
class Talker:
    tid: int
    birth_frame: int
    death_frame: int              # exclusive
    azimuth: np.ndarray           # (n,) rad, NaN outside presence
    speaking: np.ndarray          # (n,) bool
    occluded: np.ndarray          # (n,) bool (meaningful only in FOV)

    def present(self, k: int) -> bool:
        return self.birth_frame <= k < self.death_frame


@dataclass
class WorldConfig:
    duration: float = 40.0
    dt: float = 0.064
    n_talkers: int = 3
    room: tuple[float, float] = (6.0, 5.0)
    array_xy: tuple[float, float] = (3.0, 2.5)
    speed: float = 0.35
    min_range: float = 0.8
    min_presence: float = 12.0
    always_present: bool = False
    activity: ActivityParams = field(default_factory=ActivityParams)
    occlusion: OcclusionParams = field(default_factory=OcclusionParams)
    camera: CameraParams = field(default_factory=CameraParams)


@dataclass
class World:
    cfg: WorldConfig
    talkers: list[Talker]
    n_frames: int

    @property
    def dt(self) -> float:
        return self.cfg.dt

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.cfg.dt

    def present_azimuths(self, k: int) -> np.ndarray:
        return np.array([t.azimuth[k] for t in self.talkers if t.present(k)])

    def n_present(self, k: int) -> int:
        return int(sum(t.present(k) for t in self.talkers))

    # -- per-sensor ground truth of who is detectable this frame ---------
    def audio_lit(self, t: Talker, k: int) -> bool:
        """Can the array possibly see this talker at frame k?"""
        return t.present(k) and bool(t.speaking[k])

    def video_lit(self, t: Talker, k: int) -> bool:
        """Can the camera possibly see this talker at frame k?"""
        return (t.present(k) and not bool(t.occluded[k])
                and bool(self.cfg.camera.in_fov(t.azimuth[k])))


def _sample_two_state(n: int, mean_on: float, mean_off: float, dt: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Exponential-holding two-state chain; True = the 'on' state.

    Starts from the stationary distribution, first segment drawn from the
    memoryless law (exact for exponential holding times -- the residual of an
    exponential is the same exponential, so no length-bias correction is
    needed here, unlike the log-normal speech sampler).
    """
    p_on = mean_on / (mean_on + mean_off)
    state = bool(rng.random() < p_on)
    out = np.empty(n, dtype=bool)
    k = 0
    while k < n:
        mean = mean_on if state else mean_off
        hold = max(1, int(round(rng.exponential(mean) / dt)))
        hold = min(hold, n - k)
        out[k:k + hold] = state
        k += hold
        state = not state
    return out


def _walk_azimuth(n: int, cfg: WorldConfig, rng: np.random.Generator
                  ) -> np.ndarray:
    """Azimuth track of a reflecting near-constant-velocity walk."""
    W, H = cfg.room
    ax, ay = cfg.array_xy
    for _ in range(200):
        p = np.array([rng.uniform(0.3, W - 0.3), rng.uniform(0.3, H - 0.3)])
        if np.hypot(*(p - np.array([ax, ay]))) > cfg.min_range + 0.4:
            break
    th = rng.uniform(-np.pi, np.pi)
    v = max(0.05, rng.normal(cfg.speed, 0.15)) * np.array([np.cos(th), np.sin(th)])
    az = np.empty(n)
    for k in range(n):
        az[k] = float(_wrap(np.arctan2(p[1] - ay, p[0] - ax)))
        turn = rng.normal(0.0, 0.25) * cfg.dt
        c, s = np.cos(turn), np.sin(turn)
        v = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
        p = p + v * cfg.dt
        for i, lim in enumerate(cfg.room):
            if p[i] < 0.2:
                p[i] = 0.4 - p[i]; v[i] = -v[i]
            elif p[i] > lim - 0.2:
                p[i] = 2 * (lim - 0.2) - p[i]; v[i] = -v[i]
        d = p - np.array([ax, ay]); r = np.hypot(*d)
        if r < cfg.min_range:
            p = np.array([ax, ay]) + d / max(r, 1e-9) * cfg.min_range
            v = -v
    return az


def make_world(cfg: WorldConfig | None = None, seed: int = 0) -> World:
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(seed)
    n = int(round(cfg.duration / cfg.dt))
    min_pres = int(round(cfg.min_presence / cfg.dt))

    talkers: list[Talker] = []
    for tid in range(cfg.n_talkers):
        if cfg.always_present or n <= min_pres + 2:
            b, d = 0, n
        else:
            b = int(rng.integers(0, max(1, n - min_pres)))
            d = int(rng.integers(b + min_pres, n + 1))
        az = _walk_azimuth(n, cfg, rng)
        speaking = sample_activity(n, cfg.activity, cfg.dt, rng, "semi_markov")
        occluded = ~_sample_two_state(n, cfg.occlusion.mean_visible,
                                      cfg.occlusion.mean_occluded, cfg.dt, rng)
        azm = az.copy()
        mask = np.ones(n, dtype=bool); mask[b:d] = False
        azm[mask] = np.nan
        speaking = speaking.copy(); speaking[mask] = False
        occluded = occluded.copy(); occluded[mask] = True
        talkers.append(Talker(tid, b, d, azm, speaking, occluded))
    return World(cfg=cfg, talkers=talkers, n_frames=n)
