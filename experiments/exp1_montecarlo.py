"""Experiment 1 -- the main Monte Carlo comparison.

Calibrates extraction thresholds on held-out seeds, then scores all four
variants on fresh scenarios with identical measurements. Writes
``results/montecarlo.json`` and a bar figure.
"""

from __future__ import annotations

import numpy as np

from common import (EVAL_SEEDS, FIGURES, LABELS, MODES, RESULTS, calibrate,
                    run_mode, save_json)
from avrfs import WorldConfig, evaluate, make_world, sense_audio, sense_video

KEYS = ["OSPA_deg", "card_RMSE", "card_bias", "coverage", "hold_s",
        "mean_components"]


def main() -> None:
    cfg = WorldConfig()
    print("calibrating extraction thresholds on held-out seeds ...")
    thresholds = calibrate(cfg)

    rows: dict[str, list[dict]] = {m: [] for m in MODES}
    for s in EVAL_SEEDS:
        w = make_world(cfg, seed=s)
        au, vi = sense_audio(w, seed=s), sense_video(w, seed=s)
        for mode in MODES:
            out = run_mode(mode, w, au, vi,
                           extract_threshold=thresholds[mode])
            rows[mode].append(evaluate(w, out).as_row())
        print(f"  seed {s} done", flush=True)

    agg = {}
    for mode in MODES:
        agg[mode] = {}
        for k in KEYS:
            v = np.array([r[k] for r in rows[mode]])
            agg[mode][k] = float(v.mean())
            agg[mode][k + "_se"] = float(v.std(ddof=1) / np.sqrt(v.size))
    save_json({"seeds": EVAL_SEEDS, "thresholds": thresholds,
               "aggregate": agg, "per_seed": rows},
              RESULTS / "montecarlo.json")

    print(f"\n{'variant':28s}{'OSPA':>8s}{'cRMSE':>8s}{'bias':>8s}"
          f"{'cover':>8s}{'hold_s':>8s}")
    for mode in MODES:
        a = agg[mode]
        print(f"{LABELS[mode]:28s}{a['OSPA_deg']:8.2f}{a['card_RMSE']:8.2f}"
              f"{a['card_bias']:+8.2f}{a['coverage']:8.2f}{a['hold_s']:8.2f}")

    _figure(agg)


def _figure(agg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"audio_only": "#eb6834", "video_only": "#eda100",
              "naive_fusion": "#e87ba4", "fusion": "#2a78d6"}
    panels = [("OSPA_deg", "OSPA (deg, lower better)"),
              ("card_RMSE", "Head-count RMSE (lower better)"),
              ("coverage", "Track coverage (higher better)"),
              ("hold_s", "Unbroken hold (s, higher better)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    y = np.arange(len(MODES))[::-1]
    for ax, (key, title) in zip(axes, panels):
        vals = [agg[m][key] for m in MODES]
        errs = [agg[m][key + "_se"] for m in MODES]
        ax.barh(y, vals, xerr=errs, height=0.6,
                color=[colors[m] for m in MODES],
                error_kw=dict(ecolor="#52514e", elinewidth=1, capsize=2.5))
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[m] for m in MODES] if ax is axes[0]
                           else [""] * len(MODES), fontsize=8.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        for yy, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yy, f"{v:.2f}", va="center",
                    fontsize=8.5, color="#52514e")
        ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Monte Carlo over {len(EVAL_SEEDS)} scenarios "
                 "(means ± standard errors; thresholds calibrated per variant "
                 "on held-out seeds)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_montecarlo.png", dpi=140, bbox_inches="tight")
    print("wrote results/figures/fig1_montecarlo.png")


if __name__ == "__main__":
    main()
