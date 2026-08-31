"""Experiment 3 -- the whole operating surface, including where we lose.

The default occlusion rate is a chosen operating point, not a measured one, so
the honest thing is to report the entire grid rather than one cell. Two axes:

    activity factor    how much of the time a talker speaks (audio's blindness)
    visibility factor  how much of the time a talker is unoccluded (video's)

What the surface actually shows (and it is the opposite of what was predicted
before running it, which is why the prediction is recorded here rather than
quietly deleted):

    The advantage over naive fusion is positive in every cell, and it GROWS
    with sensor reliability -- ~52% at activity 0.65 / visibility 0.85, down to
    ~12-14% in the starved corner.

The original guess was that modeling would matter most when both sensors were
unreliable and the priors had to work hardest. The measured behaviour is the
reverse, and the reason is the same one that showed up in the TalkerRFS
activity sweep: when both sensors are starved there is little information for
ANY filter, every variant degrades toward the OSPA cutoff, and the differences
compress. Modeling *when* a sensor is blind only pays when that sensor has
something to contribute the rest of the time.

This is the more useful result for a deployment: the method earns its keep in
good conditions, not as a rescue for bad ones.
"""

from __future__ import annotations

import numpy as np

from common import (FIGURES, LABELS, MODES, RESULTS, calibrate, run_mode,
                    save_json)
from avrfs import (ActivityParams, OcclusionParams, WorldConfig, evaluate,
                   make_world, sense_audio, sense_video)

# Activity factor a = talk/(talk+pause); we vary the pause length.
PAUSES = [0.55, 1.587, 3.5]          # -> a ~ 0.65, 0.39, 0.22
OCCLUSIONS = [0.7, 2.0, 5.0]         # mean occluded s -> visibility 0.85/0.67/0.44
SEEDS = list(range(0, 8))


def main() -> None:
    grid = {}
    for pause in PAUSES:
        for occ in OCCLUSIONS:
            act = ActivityParams(mean_pause=pause)
            ocp = OcclusionParams(mean_visible=4.0, mean_occluded=occ)
            cfg = WorldConfig(duration=30.0, activity=act, occlusion=ocp)
            key = f"a{act.activity_factor:.2f}_v{ocp.visibility_factor:.2f}"
            print(f"activity {act.activity_factor:.2f}, "
                  f"visibility {ocp.visibility_factor:.2f} ...", flush=True)
            # Thresholds are re-calibrated per cell: the right threshold
            # genuinely depends on the operating point, and holding it fixed
            # would confound the sweep with a tuning artifact.
            thr = calibrate(cfg, seeds=[900, 901, 902], verbose=False)
            cell = {}
            for mode in MODES:
                rows = []
                for s in SEEDS:
                    w = make_world(cfg, seed=s)
                    au, vi = sense_audio(w, seed=s), sense_video(w, seed=s)
                    out = run_mode(mode, w, au, vi,
                                   extract_threshold=thr[mode])
                    rows.append(evaluate(w, out).as_row())
                cell[mode] = {k: float(np.mean([r[k] for r in rows]))
                              for k in rows[0]}
            cell["_thresholds"] = thr
            cell["_activity"] = act.activity_factor
            cell["_visibility"] = ocp.visibility_factor
            grid[key] = cell

    save_json({"pauses": PAUSES, "occlusions": OCCLUSIONS, "seeds": SEEDS,
               "grid": grid}, RESULTS / "complementarity.json")

    print(f"\n{'activity':>9s}{'visibility':>12s}"
          f"{'naive OSPA':>12s}{'fusion OSPA':>13s}{'gain %':>9s}")
    for key, cell in grid.items():
        n = cell["naive_fusion"]["OSPA_deg"]
        f = cell["fusion"]["OSPA_deg"]
        print(f"{cell['_activity']:9.2f}{cell['_visibility']:12.2f}"
              f"{n:12.2f}{f:13.2f}{100 * (1 - f / n):9.1f}")

    _figure(grid)


def _figure(grid) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acts = sorted({c["_activity"] for c in grid.values()})
    viss = sorted({c["_visibility"] for c in grid.values()})
    gain = np.full((len(viss), len(acts)), np.nan)
    for c in grid.values():
        i = viss.index(c["_visibility"]); j = acts.index(c["_activity"])
        gain[i, j] = 100 * (1 - c["fusion"]["OSPA_deg"]
                            / c["naive_fusion"]["OSPA_deg"])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lim = np.nanmax(np.abs(gain))
    im = ax.imshow(gain, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
    ax.set_xticks(range(len(acts)))
    ax.set_xticklabels([f"{a:.2f}" for a in acts])
    ax.set_yticks(range(len(viss)))
    ax.set_yticklabels([f"{v:.2f}" for v in viss])
    ax.set_xlabel("activity factor (audio sees more →)")
    ax.set_ylabel("visibility factor (video sees more →)")
    for i in range(len(viss)):
        for j in range(len(acts)):
            ax.text(j, i, f"{gain[i, j]:+.0f}%", ha="center", va="center",
                    fontsize=10,
                    color="white" if abs(gain[i, j]) > lim * 0.6 else "#101820")
    fig.colorbar(im, ax=ax, label="OSPA improvement over naive fusion (%)")
    ax.set_title("Where modeling the blind spots pays", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_complementarity.png", dpi=140,
                bbox_inches="tight")
    print("wrote results/figures/fig4_complementarity.png")


if __name__ == "__main__":
    main()
