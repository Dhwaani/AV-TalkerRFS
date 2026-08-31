"""Gaussian-mixture container and the pruning / merging / capping machinery.

[Adapted from the author's TalkerRFS project; kept byte-compatible where
possible so fixes and regression tests flow between the two repos.]

Everything the filters do to an intensity function happens here.  Two details
are specific to bearing-only acoustic tracking:

* **Angle wrapping.**  The first state dimension is azimuth.  Innovations and
  component distances are wrapped to (-pi, pi] so a talker crossing the array's
  0/2pi seam does not spawn a duplicate track on the far side.
* **Auxiliary per-component variables.**  The pause-aware filter attaches a
  mode probability to every component and the Beta-Gaussian baseline attaches a
  Beta(s, t) belief over p_D.  Rather than fork the mixture class three ways,
  components carry an ``aux`` dict of arrays whose leading dimension tracks the
  component count, and prune/merge/cap carry them along.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GaussianMixture", "wrap", "kf_predict", "kf_update_batch"]


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


@dataclass
class GaussianMixture:
    """A weighted Gaussian mixture representing a PHD intensity."""

    w: np.ndarray                                    # (n,)
    m: np.ndarray                                    # (n, d)
    P: np.ndarray                                    # (n, d, d)
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.w = np.atleast_1d(np.asarray(self.w, dtype=float))
        m = np.asarray(self.m, dtype=float)
        P = np.asarray(self.P, dtype=float)
        if self.w.size == 0:
            d = m.shape[-1] if m.ndim >= 2 else (P.shape[-1] if P.ndim >= 3 else 2)
            self.m = np.zeros((0, d))
            self.P = np.zeros((0, d, d))
        else:
            self.m = m.reshape(self.w.size, -1)
            d = self.m.shape[1]
            self.P = P.reshape(self.w.size, d, d)

    # -- basics ---------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.w.size)

    @property
    def d(self) -> int:
        return int(self.m.shape[1]) if self.n else 0

    @property
    def mass(self) -> float:
        """Total mass == expected number of targets."""
        return float(self.w.sum())

    @classmethod
    def empty(cls, d: int = 2) -> "GaussianMixture":
        return cls(np.zeros(0), np.zeros((0, d)), np.zeros((0, d, d)))

    def copy(self) -> "GaussianMixture":
        return GaussianMixture(self.w.copy(), self.m.copy(), self.P.copy(),
                               {k: v.copy() for k, v in self.aux.items()})

    def take(self, idx) -> "GaussianMixture":
        idx = np.asarray(idx, dtype=int)
        return GaussianMixture(self.w[idx], self.m[idx], self.P[idx],
                               {k: v[idx] for k, v in self.aux.items()})

    def append(self, other: "GaussianMixture") -> "GaussianMixture":
        keys = set(self.aux) | set(other.aux)
        if self.n == 0 and other.n == 0:
            out = other.copy()
            out.aux = {k: np.zeros(0) for k in keys}
            return out
        aux = {}
        for k in keys:
            a = self.aux.get(k)
            b = other.aux.get(k)
            # An empty side legitimately carries no aux arrays (an empty birth,
            # a filter on its first frame); anything else is a real mismatch.
            if a is None:
                if self.n:
                    raise KeyError(f"aux key {k!r} missing from a non-empty mixture")
                a = np.zeros(0)
            if b is None:
                if other.n:
                    raise KeyError(f"aux key {k!r} missing from a non-empty mixture")
                b = np.zeros(0)
            aux[k] = np.concatenate([a, b], axis=0)
        if self.n == 0:
            out = other.copy()
            out.aux = aux
            return out
        if other.n == 0:
            out = self.copy()
            out.aux = aux
            return out
        return GaussianMixture(np.concatenate([self.w, other.w]),
                               np.vstack([self.m, other.m]),
                               np.concatenate([self.P, other.P], axis=0), aux)

    # -- reduction ------------------------------------------------------
    def prune(self, threshold: float) -> "GaussianMixture":
        keep = np.flatnonzero(self.w > threshold)
        return self.take(keep)

    def cap(self, jmax: int, renormalise: bool = True) -> "GaussianMixture":
        """Keep the ``jmax`` heaviest components.

        ``renormalise`` restores the pre-cap total mass, so a PHD's cardinality
        estimate is not silently biased low by a housekeeping step.  It must be
        left off for the CPHD family: there the cardinality distribution is
        propagated separately, and rescaling the intensity injects mass the
        cardinality recursion never saw, breaking the identity between the
        intensity's mass and the cardinality mean.
        """
        if self.n <= jmax:
            return self.copy()
        keep = np.argsort(self.w)[::-1][:jmax]
        out = self.take(np.sort(keep))
        total = self.mass
        if renormalise and out.mass > 0 and total > 0:
            out.w *= total / out.mass
        return out

    def merge(self, threshold: float, angle_dim: int = 0,
              separate: dict[str, float] | None = None) -> "GaussianMixture":
        """Merge components whose Mahalanobis distance is below ``threshold``.

        ``separate`` maps auxiliary-variable names to tolerances.  Each named
        variable acts as a soft discrete label: components whose values differ
        by more than the tolerance are never merged, even when they sit on top
        of each other in state space.  Every constraint is applied in the *same*
        pass and conjunctively -- running them as successive merges would let a
        later pass undo an earlier one's separation.

        The pause-aware filters separate on existence: a freshly born component
        sitting on a confirmed track is a different hypothesis, not the same
        talker seen twice, and averaging their existence probabilities pulls the
        confirmed track's survival probability down until it starts leaking
        mass.  Separating on the activity mode as well is available and is what
        multiple-model PHD practice would suggest, but is off by default because
        it measures worse here (see `experiments/exp7_ablations.py`).
        """
        if self.n == 0:
            return self.copy()
        remaining = set(range(self.n))
        ws, ms, Ps = [], [], []
        aux_out: dict[str, list] = {k: [] for k in self.aux}
        while remaining:
            idx = np.array(sorted(remaining))
            j = int(idx[np.argmax(self.w[idx])])
            Pj_inv = np.linalg.inv(self.P[j])
            diff = self.m[idx] - self.m[j]
            if angle_dim is not None:
                diff[:, angle_dim] = wrap(diff[:, angle_dim])
            dist = np.einsum("ij,jk,ik->i", diff, Pj_inv, diff)
            ok = dist <= threshold
            for key, tol in (separate or {}).items():
                if key in self.aux:
                    v = self.aux[key]
                    ok &= np.abs(v[idx] - v[j]) <= tol
            grp = idx[ok]
            wg = self.w[grp]
            wsum = float(wg.sum())
            if wsum <= 0:
                remaining -= set(grp.tolist())
                continue
            # Merge in a frame centred on the dominant component so wrapping is
            # handled correctly, then unwrap the result once.
            rel = self.m[grp] - self.m[j]
            if angle_dim is not None:
                rel[:, angle_dim] = wrap(rel[:, angle_dim])
            mrel = (wg[:, None] * rel).sum(0) / wsum
            mmerged = self.m[j] + mrel
            if angle_dim is not None:
                mmerged[angle_dim] = wrap(mmerged[angle_dim])
            dm = rel - mrel
            Pmerged = ((wg[:, None, None] * (self.P[grp]
                        + np.einsum("ni,nj->nij", dm, dm))).sum(0)) / wsum
            ws.append(wsum)
            ms.append(mmerged)
            Ps.append(Pmerged)
            for k, v in self.aux.items():
                # aux variables are 1-D per component; merge by weight average.
                aux_out[k].append(float((wg * v[grp]).sum() / wsum))
            remaining -= set(grp.tolist())
        aux = {k: np.array(v, dtype=float) for k, v in aux_out.items()}
        return GaussianMixture(np.array(ws), np.array(ms), np.array(Ps), aux)

    def reduce(self, prune_thresh: float, merge_thresh: float, jmax: int,
               separate: dict[str, float] | None = None,
               renormalise_cap: bool = True) -> "GaussianMixture":
        return (self.prune(prune_thresh)
                .merge(merge_thresh, separate=separate)
                .cap(jmax, renormalise=renormalise_cap))

    # -- state extraction -----------------------------------------------
    def extract(self, threshold: float = 0.5, weight_key: str | None = None
                ) -> np.ndarray:
        """Extract point estimates, repeating components with weight > 1.

        ``weight_key`` optionally scales each component's weight by an aux
        variable (used to extract *active* rather than *present* talkers).
        """
        if self.n == 0:
            return np.zeros((0, self.d if self.d else 2))
        w = self.w * (self.aux[weight_key] if weight_key else 1.0)
        out = []
        for i in range(self.n):
            reps = int(round(float(w[i]))) if w[i] >= threshold else 0
            reps = max(reps, 1) if w[i] >= threshold else 0
            for _ in range(reps):
                out.append(self.m[i])
        return np.array(out) if out else np.zeros((0, self.d))


# -- Kalman helpers ------------------------------------------------------
def kf_predict(gm: GaussianMixture, F: np.ndarray, Q: np.ndarray
               ) -> GaussianMixture:
    """Linear-Gaussian prediction of every component."""
    if gm.n == 0:
        return gm.copy()
    m = gm.m @ F.T
    m[:, 0] = wrap(m[:, 0])
    P = F @ gm.P @ F.T + Q
    return GaussianMixture(gm.w.copy(), m, P,
                           {k: v.copy() for k, v in gm.aux.items()})


def kf_update_batch(gm: GaussianMixture, H: np.ndarray, R: np.ndarray,
                    z: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """Update every component with every measurement.

    Returns ``(q, m_upd, P_upd, S)`` where ``q`` has shape ``(n_meas, n_comp)``
    and holds the measurement likelihoods, ``m_upd`` has shape
    ``(n_meas, n_comp, d)``, ``P_upd`` has shape ``(n_comp, d, d)`` (it does not
    depend on the measurement), and ``S`` is the per-component innovation
    covariance.
    """
    n, d = gm.n, gm.d
    z = np.atleast_1d(np.asarray(z, dtype=float))
    eta = gm.m @ H.T                                   # (n, dz)
    S = H @ gm.P @ H.T + R                             # (n, dz, dz)
    Sinv = np.linalg.inv(S)
    K = gm.P @ H.T @ Sinv                              # (n, d, dz)
    I = np.eye(d)
    P_upd = (I - K @ H) @ gm.P
    P_upd = 0.5 * (P_upd + np.transpose(P_upd, (0, 2, 1)))

    if z.size == 0:
        return (np.zeros((0, n)), np.zeros((0, n, d)), P_upd, S)

    # Bearing-only: dz == 1, wrap the innovation.
    nu = wrap(z[:, None] - eta[None, :, 0])            # (n_meas, n)
    s = S[:, 0, 0]
    q = np.exp(-0.5 * nu ** 2 / s) / np.sqrt(2 * np.pi * s)
    m_upd = gm.m[None, :, :] + nu[:, :, None] * K[None, :, :, 0]
    m_upd[:, :, 0] = wrap(m_upd[:, :, 0])
    return q, m_upd, P_upd, S
