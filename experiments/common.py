"""Shared harness: the four variants, threshold calibration, run helpers.

Extraction thresholds are calibrated per variant on seeds the evaluation never
sees, with one shared objective (mean OSPA). This matters because the four
variants do not agree on the best threshold — at the default operating point
they choose 0.15, 0.25, 0.15 and 0.20 — so any single fixed value would hand
one of them an advantage that has nothing to do with its detection model.

The grid deliberately extends well below the weight-one mark. An earlier
version started at 0.3 and every variant pinned to that floor, which is the
signature of a boundary artifact rather than a chosen optimum; the current grid
puts all four optima in its interior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avrfs import (WorldConfig, evaluate, make_filter, make_world,
                   sense_audio, sense_video)
from avrfs.metrics import DEFAULT_CUTOFF, ospa

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

MODES = ["audio_only", "video_only", "naive_fusion", "fusion"]
LABELS = {"audio_only": "Audio only (pause-aware)",
          "video_only": "Video only (occlusion-aware)",
          "naive_fusion": "Naive fusion (constant p_D)",
          "fusion": "Modeled fusion (proposed)"}

CAL_SEEDS = list(range(900, 906))
EVAL_SEEDS = list(range(0, 25))
THRESH_GRID = [0.10, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]


def run_mode(mode, world, au, vi, record=False, extract_threshold=None):
    f = make_filter(mode)
    if extract_threshold is not None:
        f.cfg.extract_threshold = extract_threshold
    return f.run(au if mode != "video_only" else None,
                 vi if mode != "audio_only" else None, record=record)


def calibrate(cfg: WorldConfig | None = None, seeds=CAL_SEEDS,
              verbose=True) -> dict[str, float]:
    """Best extraction threshold per variant, one shared OSPA objective."""
    cfg = cfg or WorldConfig()
    chosen: dict[str, float] = {}
    for mode in MODES:
        runs = []
        for s in seeds:
            w = make_world(cfg, seed=s)
            au, vi = sense_audio(w, seed=s), sense_video(w, seed=s)
            runs.append((w, run_mode(mode, w, au, vi, record=True)))
        best = None
        for thr in THRESH_GRID:
            vals = []
            for w, out in runs:
                est = [np.asarray(gm.extract(thr))[:, 0]
                       if gm.extract(thr).size else np.zeros(0)
                       for gm in out.mixtures]
                truth = [w.present_azimuths(k) for k in range(w.n_frames)]
                vals.append(np.mean([ospa(truth[k], est[k], DEFAULT_CUTOFF)
                                     for k in range(w.n_frames)]))
            m = float(np.mean(vals))
            if best is None or m < best[0]:
                best = (m, thr)
        chosen[mode] = best[1]
        if verbose:
            print(f"  {mode:13s} extract_threshold={best[1]} "
                  f"(cal OSPA {np.rad2deg(best[0]):.2f} deg)", flush=True)
    return chosen


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    print(f"wrote {path.relative_to(ROOT)}")


def load_json(path: Path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None
