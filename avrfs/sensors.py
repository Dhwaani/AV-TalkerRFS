"""Two measurement-level front ends with dual blind spots.

``AudioSensor``   azimuth peaks from a mic-array localiser. Detects a talker
                  only while they are SPEAKING. Moderate angular noise, uniform
                  clutter over the full circle, reverberant ghosts anchored to
                  active talkers, finite angular resolution.

``VideoSensor``   azimuth of face detections from a fixed camera. Detects a
                  talker only while they are VISIBLE and inside the FOV.
                  Sharper angular noise, its own (sparser) clutter confined to
                  the FOV -- false face detections do not appear behind the
                  camera.

The deliberate symmetry: each sensor's detection probability is the product of
a geometric term (can the hardware see it at all) and a latent binary the other
sensor knows nothing about. Everything downstream keys on that structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import World, _wrap

__all__ = ["AudioParams", "VideoParams", "SensorFrame", "sense_audio",
           "sense_video"]


@dataclass(frozen=True)
class AudioParams:
    sigma: float = np.deg2rad(3.0)
    p_detect: float = 0.92          # P(detect | speaking)
    clutter_rate: float = 2.0       # Poisson mean per frame, uniform over 2*pi
    ghost_rate: float = 0.35        # per active talker
    ghost_sigma: float = np.deg2rad(12.0)
    resolution: float = np.deg2rad(8.0)

    @property
    def fov_extent(self) -> float:
        return 2.0 * np.pi

    @property
    def clutter_density(self) -> float:
        return self.clutter_rate / self.fov_extent


@dataclass(frozen=True)
class VideoParams:
    sigma: float = np.deg2rad(1.0)
    p_detect: float = 0.95          # P(detect | visible and in FOV)
    clutter_rate: float = 0.3       # Poisson mean per frame, uniform over FOV
    resolution: float = np.deg2rad(3.0)

    def fov_extent(self, half_fov: float) -> float:
        return 2.0 * half_fov

    def clutter_density(self, half_fov: float) -> float:
        return self.clutter_rate / self.fov_extent(half_fov)


@dataclass
class SensorFrame:
    """Measurements for one frame from one sensor."""

    z: np.ndarray        # (m,) azimuths, radians
    origin: np.ndarray   # (m,) talker id or -1, diagnostics only


def _merge_close(z: np.ndarray, o: np.ndarray, resolution: float):
    """Wrap-aware resolution merge (sorted input), as in TalkerRFS."""
    if z.size < 2 or resolution <= 0:
        return z, o
    keep_z = [float(z[0])]; keep_o = [int(o[0])]; count = 1
    for i in range(1, z.size):
        if abs(z[i] - keep_z[-1]) < resolution:
            keep_z[-1] = (keep_z[-1] * count + float(z[i])) / (count + 1)
            count += 1
            if keep_o[-1] == -1:
                keep_o[-1] = int(o[i])
        else:
            keep_z.append(float(z[i])); keep_o.append(int(o[i])); count = 1
    if len(keep_z) > 1 and abs(_wrap(keep_z[0] - keep_z[-1])) < resolution:
        merged = keep_z[0] + 0.5 * float(_wrap(keep_z[-1] - keep_z[0]))
        keep_z[0] = float(_wrap(merged))
        if keep_o[0] == -1:
            keep_o[0] = keep_o[-1]
        keep_z.pop(); keep_o.pop()
    return np.array(keep_z), np.array(keep_o, dtype=int)


def sense_audio(world: World, ap: AudioParams | None = None,
                seed: int = 0) -> list[SensorFrame]:
    """Audio measurement sequence: blind to silence, deaf to nothing else."""
    ap = ap or AudioParams()
    rng = np.random.default_rng(seed + 10_000)
    frames: list[SensorFrame] = []
    for k in range(world.n_frames):
        zs: list[float] = []; os_: list[int] = []
        for t in world.talkers:
            if not world.audio_lit(t, k):
                continue
            # Ghosts ride on radiating talkers, detected or not.
            for _ in range(rng.poisson(ap.ghost_rate)):
                zs.append(float(_wrap(t.azimuth[k]
                                      + rng.normal(0, ap.ghost_sigma))))
                os_.append(-1)
            if rng.random() <= ap.p_detect:
                zs.append(float(_wrap(t.azimuth[k] + rng.normal(0, ap.sigma))))
                os_.append(t.tid)
        for _ in range(rng.poisson(ap.clutter_rate)):
            zs.append(float(rng.uniform(-np.pi, np.pi))); os_.append(-1)
        z = np.array(zs); o = np.array(os_, dtype=int)
        if z.size:
            idx = np.argsort(z); z, o = z[idx], o[idx]
            z, o = _merge_close(z, o, ap.resolution)
        frames.append(SensorFrame(z=z, origin=o))
    return frames


def sense_video(world: World, vp: VideoParams | None = None,
                seed: int = 0) -> list[SensorFrame]:
    """Video measurement sequence: blind to occlusion and to everything
    outside the FOV, indifferent to whether anyone is speaking."""
    vp = vp or VideoParams()
    cam = world.cfg.camera
    rng = np.random.default_rng(seed + 20_000)
    lo = cam.boresight - cam.half_fov
    hi = cam.boresight + cam.half_fov
    frames: list[SensorFrame] = []
    for k in range(world.n_frames):
        zs: list[float] = []; os_: list[int] = []
        for t in world.talkers:
            if not world.video_lit(t, k):
                continue
            if rng.random() <= vp.p_detect:
                a = float(_wrap(t.azimuth[k] + rng.normal(0, vp.sigma)))
                if cam.in_fov(a):
                    zs.append(a); os_.append(t.tid)
        for _ in range(rng.poisson(vp.clutter_rate)):
            zs.append(float(_wrap(rng.uniform(lo, hi)))); os_.append(-1)
        z = np.array(zs); o = np.array(os_, dtype=int)
        if z.size:
            idx = np.argsort(z); z, o = z[idx], o[idx]
            z, o = _merge_close(z, o, vp.resolution)
        frames.append(SensorFrame(z=z, origin=o))
    return frames
