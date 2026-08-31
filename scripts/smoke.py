"""Short end-to-end run for CI.

Checks the two claims the repository actually makes, on one seed:
  1. modeled fusion beats naive fusion on OSPA and head-count RMSE;
  2. in the audio-lit regime (speaking but occluded), modeled fusion recovers
     the talker far more often than naive fusion does.
The second is the mechanism; the first is the consequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avrfs import (WorldConfig, evaluate, make_filter, make_world,
                   sense_audio, sense_video)
from avrfs.gm import wrap
from avrfs.metrics import DEFAULT_CUTOFF

MODES = ["audio_only", "video_only", "naive_fusion", "fusion"]


def main() -> int:
    w = make_world(WorldConfig(duration=40.0), seed=0)
    au, vi = sense_audio(w, seed=0), sense_video(w, seed=0)

    scores, audio_cov = {}, {}
    print(f"{'variant':16s}{'OSPA':>8s}{'cRMSE':>8s}{'cover':>8s}"
          f"{'audio-lit cov':>15s}")
    for m in MODES:
        f = make_filter(m)
        out = f.run(au if m != "video_only" else None,
                    vi if m != "audio_only" else None)
        s = evaluate(w, out)
        scores[m] = s
        hit = tot = 0
        for k in range(w.n_frames):
            e = out.estimates[k]
            ez = np.asarray(e)[:, 0] if np.size(e) else np.zeros(0)
            for t in w.talkers:
                if not (t.present(k) and w.audio_lit(t, k)
                        and not w.video_lit(t, k)):
                    continue
                tot += 1
                hit += int(bool(ez.size and np.min(np.abs(wrap(ez - t.azimuth[k])))
                                <= DEFAULT_CUTOFF))
        audio_cov[m] = hit / max(tot, 1)
        print(f"{m:16s}{np.rad2deg(s.ospa):8.2f}{s.card_rmse:8.2f}"
              f"{s.coverage:8.2f}{audio_cov[m]:15.2f}")

    fails = []
    if scores["fusion"].ospa > scores["naive_fusion"].ospa:
        fails.append("OSPA worse than naive fusion")
    if scores["fusion"].card_rmse > scores["naive_fusion"].card_rmse:
        fails.append("head-count RMSE worse than naive fusion")
    # A real but modest margin. An earlier version demanded 1.5x, which only
    # held at a too-high extraction threshold -- a boundary artifact that
    # overstated the mechanism fivefold. See the README correction.
    if audio_cov["fusion"] < audio_cov["naive_fusion"] + 0.05:
        fails.append("no advantage in the audio-lit regime")

    if fails:
        print("\nFAILED: " + "; ".join(fails))
        return 1
    print("\nOK: modeled fusion leads, and the audio-lit regime is where.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
