"""One GM-PHD host, four sensor configurations.

The host is an iterated-corrector GM-PHD: predict once per frame, then apply
one PHD update per enabled sensor in sequence (audio first, then video --
the ordering ablation is in the experiments). All four variants below share
the host, the motion model, the birth model and the housekeeping, so the only
thing under test is what each one believes about detection.

``audio_only``   the pause-aware audio filter from TalkerRFS: p_D = r*pi*p_aud.
``video_only``   the mirror image: p_D = r*v*p_vid*1{in FOV}.
``naive_fusion`` both sensors, textbook constant detection probabilities set
                 to the practitioner's fix -- the *average* rates
                 p_aud*activity_factor and p_vid*visibility_factor. This is
                 the fair strawman: it uses both sensors and correct average
                 statistics, it just refuses to model WHEN each sensor is blind.
``fusion``       the proposed filter. Each component carries three latent
                 scalars: r (a real talker at all), pi (speaking now),
                 v (visible now). Then

                     p_D_audio = r * pi * p_aud
                     p_D_video = r * v  * p_vid * 1{in FOV(azimuth)}

                 pi is driven by the P.59 conversation chain and updated by
                 audio evidence; v by the occlusion chain and video evidence;
                 r by both. The failure modes are dual by construction: a
                 silent talker is carried by the camera, an occluded talker by
                 the array, and a talker dark to both is carried briefly by
                 the two chains' priors -- which is exactly as long as such a
                 track deserves to live.

The latent scalars are updated by the single-target Bernoulli recursion and
shared across every branch of a component -- never read off the mixture
weights. That lesson (the PHD's cardinality overshoot feeding back into the
mode posterior) cost TalkerRFS a week; it arrives here pre-learned, with the
regression test to hold it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activity import ActivityParams, transition_matrix
from .gm import GaussianMixture, kf_predict, kf_update_batch, wrap
from .sensors import AudioParams, SensorFrame, VideoParams
from .world import CameraParams, OcclusionParams

__all__ = ["FilterConfig", "AVFilter", "make_filter", "FilterOutput"]

#: Survival rate of a component that has not yet earned existence.
P_SURVIVAL_SPURIOUS = 0.55
R_BIRTH = 0.35


def _chain(mean_on: float, mean_off: float, dt: float) -> np.ndarray:
    """Exact 2-state transition matrix, rows = (off, on), cols = (off, on)."""
    a = dt / mean_on    # on -> off rate * dt
    b = dt / mean_off   # off -> on rate * dt
    s = a + b
    e = np.exp(-s)
    t_off_on = (b / s) * (1 - e)
    t_on_off = (a / s) * (1 - e)
    return np.array([[1 - t_off_on, t_off_on], [t_on_off, 1 - t_on_off]])


@dataclass
class FilterConfig:
    dt: float = 0.064
    sigma_a: float = 0.35             # rad/s^2 azimuth-rate process noise
    p_survival: float = 0.995
    activity: ActivityParams = field(default_factory=ActivityParams)
    occlusion: OcclusionParams = field(default_factory=OcclusionParams)
    camera: CameraParams = field(default_factory=CameraParams)
    audio: AudioParams = field(default_factory=AudioParams)
    video: VideoParams = field(default_factory=VideoParams)

    birth_rate: float = 0.03
    birth_weight_max: float = 0.10
    birth_sigma_az: float = np.deg2rad(4.0)
    birth_sigma_rate: float = 0.30
    birth_pi: float = 0.90            # a measurement-born component from audio
    birth_v: float = 0.80             # ... and from video

    prune_threshold: float = 1e-5
    merge_threshold: float = 4.0
    max_components: int = 100
    existence_merge_tol: float = 0.3
    #: Weight above which a component is reported. The default sits near the
    #: value ``experiments/exp0`` calibration picks for the fusion variant;
    #: every experiment overrides it per variant, because the four variants do
    #: not agree on the best threshold and a shared one would flatter whichever
    #: it happened to suit.
    extract_threshold: float = 0.2

    @property
    def F(self) -> np.ndarray:
        return np.array([[1.0, self.dt], [0.0, 1.0]])

    @property
    def Q(self) -> np.ndarray:
        G = np.array([[0.5 * self.dt ** 2], [self.dt]])
        return (G @ G.T) * self.sigma_a ** 2

    @property
    def H(self) -> np.ndarray:
        return np.array([[1.0, 0.0]])


@dataclass
class FilterOutput:
    estimates: list[np.ndarray]
    n_present: list[float]
    n_components: list[int]
    mixtures: list | None = None


class AVFilter:
    """The iterated-corrector host. ``mode`` picks the variant."""

    def __init__(self, cfg: FilterConfig | None = None, mode: str = "fusion"):
        if mode not in ("audio_only", "video_only", "naive_fusion", "fusion"):
            raise ValueError(f"unknown mode {mode!r}")
        self.cfg = cfg or FilterConfig()
        self.mode = mode
        c = self.cfg
        self.T_pi = transition_matrix(c.activity, c.dt)
        self.T_v = _chain(c.occlusion.mean_visible, c.occlusion.mean_occluded,
                          c.dt)
        self.gm = GaussianMixture(np.zeros(0), np.zeros((0, 2)),
                                  np.zeros((0, 2, 2)), self._empty_aux())
        self._pending_birth: dict[str, np.ndarray] = {}

        # Naive fusion's constant detection rates: the practitioner's fix.
        self.pd_naive_audio = c.audio.p_detect * c.activity.activity_factor
        fov_frac = c.camera.half_fov / np.pi   # fraction of the circle seen
        self.pd_naive_video = (c.video.p_detect
                               * c.occlusion.visibility_factor * fov_frac)

    # -- aux bookkeeping -------------------------------------------------
    def _empty_aux(self) -> dict[str, np.ndarray]:
        return {"r": np.zeros(0), "pi": np.zeros(0), "v": np.zeros(0)}

    def _birth_aux(self, n: int, sensor: str) -> dict[str, np.ndarray]:
        c = self.cfg
        if sensor == "audio":
            # Born from a sound: almost surely speaking; visibility unknown,
            # so it starts at the occlusion chain's stationary point.
            pi0, v0 = c.birth_pi, c.occlusion.visibility_factor
        else:
            # Born from a face: almost surely visible; speaking unknown.
            pi0, v0 = c.activity.activity_factor, c.birth_v
        return {"r": np.full(n, R_BIRTH), "pi": np.full(n, pi0),
                "v": np.full(n, v0)}

    # -- per-variant detection probabilities -----------------------------
    def _pd(self, pred: GaussianMixture, sensor: str) -> np.ndarray:
        c = self.cfg
        if sensor == "audio":
            if self.mode == "naive_fusion":
                return np.full(pred.n, self.pd_naive_audio)
            return pred.aux["r"] * pred.aux["pi"] * c.audio.p_detect
        # video
        fov = c.camera.in_fov(pred.m[:, 0]).astype(float)
        if self.mode == "naive_fusion":
            # The naive filter still knows the FOV -- pretending it does not
            # would be a strawman; what it refuses to model is occlusion and
            # per-frame visibility.
            return np.full(pred.n, c.video.p_detect
                           * c.occlusion.visibility_factor) * fov
        return pred.aux["r"] * pred.aux["v"] * c.video.p_detect * fov

    def _geom(self, pred: GaussianMixture, sensor: str) -> np.ndarray:
        """The geometric detection rate, latent modes stripped out."""
        c = self.cfg
        if sensor == "audio":
            return np.full(pred.n, c.audio.p_detect)
        return c.video.p_detect * c.camera.in_fov(pred.m[:, 0]).astype(float)

    # -- Bernoulli update of the latent scalars --------------------------
    def _aux_update(self, pred: GaussianMixture, sensor: str,
                    q: np.ndarray, kappa: float) -> dict[str, np.ndarray]:
        """Shared posterior for r and this sensor's mode; the other sensor's
        mode is untouched by this pass."""
        r = pred.aux["r"]
        mode_key = "pi" if sensor == "audio" else "v"
        m = pred.aux[mode_key]
        other_key = "v" if sensor == "audio" else "pi"
        g = self._geom(pred, sensor)

        m_miss = np.clip(m * (1 - g) / np.maximum(1 - m * g, 1e-12), 0, 1)
        r_miss = np.clip(r * (1 - m * g) / np.maximum(1 - r * m * g, 1e-12),
                         0, 1)
        if q.size == 0:
            out = {mode_key: m_miss, "r": r_miss,
                   other_key: pred.aux[other_key].copy()}
            return out

        lik = np.asarray(q).sum(axis=0) / max(kappa, 1e-12)
        u_det = r * m * g * lik
        u_miss = np.maximum(1 - r * m * g, 0)
        p_det = np.clip(u_det / np.maximum(u_det + u_miss, 1e-12), 0, 1)
        return {mode_key: np.clip(p_det + (1 - p_det) * m_miss, 0, 1),
                "r": np.clip(p_det + (1 - p_det) * r_miss, 0, 1),
                other_key: pred.aux[other_key].copy()}

    # -- recursion pieces -------------------------------------------------
    def _survival(self, gm: GaussianMixture) -> np.ndarray:
        if self.mode == "naive_fusion":
            return np.full(gm.n, self.cfg.p_survival)
        r = gm.aux["r"]
        return r * self.cfg.p_survival + (1 - r) * P_SURVIVAL_SPURIOUS

    def _predict(self) -> GaussianMixture:
        c = self.cfg
        pred = kf_predict(self.gm, c.F, c.Q)
        if pred.n:
            pred.w = pred.w * self._survival(pred)
            pred.aux["pi"] = (pred.aux["pi"] * self.T_pi[1, 1]
                              + (1 - pred.aux["pi"]) * self.T_pi[0, 1])
            pred.aux["v"] = (pred.aux["v"] * self.T_v[1, 1]
                             + (1 - pred.aux["v"]) * self.T_v[0, 1])
        return pred

    def _birth(self, sensor: str, z_prev: np.ndarray) -> GaussianMixture:
        c = self.cfg
        if z_prev is None or np.size(z_prev) == 0:
            return GaussianMixture(np.zeros(0), np.zeros((0, 2)),
                                   np.zeros((0, 2, 2)), self._empty_aux())
        z_prev = np.atleast_1d(z_prev)
        n = z_prev.size
        w = self._pending_birth.get(sensor, np.array([]))
        if w.size != n:
            w = np.full(n, c.birth_rate / max(n, 1))
        m = np.column_stack([wrap(z_prev), np.zeros(n)])
        P = np.repeat(np.diag([c.birth_sigma_az ** 2,
                               c.birth_sigma_rate ** 2])[None], n, axis=0)
        return GaussianMixture(w.copy(), m, P, self._birth_aux(n, sensor))

    def _sensor_update(self, pred: GaussianMixture, sensor: str,
                       z: np.ndarray) -> GaussianMixture:
        c = self.cfg
        if pred.n == 0:
            # Nothing to update; still record birth weights for next frame.
            kappa0 = (c.audio.clutter_density if sensor == "audio"
                      else c.video.clutter_density(c.camera.half_fov))
            self._pending_birth[sensor] = (
                np.minimum(np.full(np.size(z), c.birth_rate),
                           c.birth_weight_max) if np.size(z) else np.array([]))
            return pred
        R = np.array([[ (c.audio.sigma if sensor == "audio"
                         else c.video.sigma) ** 2 ]])
        kappa = (c.audio.clutter_density if sensor == "audio"
                 else c.video.clutter_density(c.camera.half_fov))
        pd = self._pd(pred, sensor)
        q, m_upd, P_upd, _ = kf_update_batch(pred, c.H, R, z)

        aux = (self._aux_update(pred, sensor, q, kappa)
               if self.mode != "naive_fusion"
               else {k: vv.copy() for k, vv in pred.aux.items()})

        parts = [GaussianMixture(pred.w * (1 - pd), pred.m.copy(),
                                 pred.P.copy(),
                                 {k: vv.copy() for k, vv in aux.items()})]
        rho = np.zeros(np.size(z))
        for j in range(np.size(z)):
            wj = pd * pred.w * q[j]
            rho[j] = float(wj.sum())
            parts.append(GaussianMixture(wj / (kappa + rho[j]), m_upd[j],
                                         P_upd,
                                         {k: vv.copy() for k, vv in aux.items()}))
        # Adaptive proportional birth weights for this sensor, next frame.
        if np.size(z):
            unexplained = kappa / (kappa + rho)
            self._pending_birth[sensor] = np.minimum(
                c.birth_rate * unexplained, c.birth_weight_max)
        else:
            self._pending_birth[sensor] = np.array([])

        out = parts[0]
        for p in parts[1:]:
            out = out.append(p)
        return out

    def _reduce(self, gm: GaussianMixture) -> GaussianMixture:
        c = self.cfg
        sep = None if self.mode == "naive_fusion" else {"r": c.existence_merge_tol}
        return gm.reduce(c.prune_threshold, c.merge_threshold,
                         c.max_components, separate=sep)

    # -- public interface --------------------------------------------------
    def step(self, audio_z: np.ndarray | None, video_z: np.ndarray | None,
             audio_prev: np.ndarray | None, video_prev: np.ndarray | None
             ) -> None:
        pred = self._predict()
        if self.mode != "video_only":
            pred = pred.append(self._birth("audio", audio_prev))
        if self.mode != "audio_only":
            pred = pred.append(self._birth("video", video_prev))

        if self.mode != "video_only" and audio_z is not None:
            pred = self._sensor_update(
                pred, "audio", np.atleast_1d(np.asarray(audio_z, float)))
        if self.mode != "audio_only" and video_z is not None:
            pred = self._sensor_update(
                pred, "video", np.atleast_1d(np.asarray(video_z, float)))
        self.gm = self._reduce(pred)

    def run(self, audio: list[SensorFrame] | None,
            video: list[SensorFrame] | None,
            record: bool = False) -> FilterOutput:
        n = len(audio) if audio is not None else len(video)
        est, npres, ncomp = [], [], []
        mixtures = [] if record else None
        a_prev = np.zeros(0); v_prev = np.zeros(0)
        for k in range(n):
            az = audio[k].z if audio is not None else None
            vz = video[k].z if video is not None else None
            self.step(az, vz, a_prev, v_prev)
            est.append(self.gm.extract(self.cfg.extract_threshold))
            npres.append(self.gm.mass)
            ncomp.append(self.gm.n)
            if record:
                mixtures.append(self.gm.copy())
            a_prev = az if az is not None else np.zeros(0)
            v_prev = vz if vz is not None else np.zeros(0)
        return FilterOutput(estimates=est, n_present=npres,
                            n_components=ncomp, mixtures=mixtures)


def make_filter(mode: str, cfg: FilterConfig | None = None) -> AVFilter:
    return AVFilter(cfg, mode)
