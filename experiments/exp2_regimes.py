"""Experiment 2 -- the claim, tested where it lives.

Every present-talker frame falls into one of four regimes:

    both      speaking AND visible   -- both sensors can see them
    audio     speaking, occluded     -- only the array can
    video     silent, visible        -- only the camera can
    dark      silent AND occluded    -- neither can; only the priors carry it

Aggregate metrics average over all four and hide the mechanism. This
experiment reports coverage *per regime*, which is the only place the
complementary-blind-spot argument can actually be checked:

* in ``audio`` the video-only baseline must fail and the audio-only one must not;
* in ``video`` the mirror image;
* in ``dark`` nothing can detect the talker, so the difference between the
  variants is entirely down to what their models believe a missed detection
  means -- and that is the proposed filter's whole thesis.

Writes ``results/regimes.json`` and two figures.
"""

from __future__ import annotations

import numpy as np

from common import (EVAL_SEEDS, FIGURES, LABELS, MODES, RESULTS, calibrate,
                    load_json, run_mode, save_json)
from avrfs import WorldConfig, make_world, sense_audio, sense_video
from avrfs.gm import wrap
from avrfs.metrics import DEFAULT_CUTOFF

REGIMES = ["both", "audio", "video", "dark"]
REGIME_LABEL = {"both": "speaking + visible", "audio": "speaking, occluded",
                "video": "silent, visible", "dark": "silent + occluded"}
SEEDS = EVAL_SEEDS[:15]
COLORS = {"audio_only": "#eb6834", "video_only": "#eda100",
          "naive_fusion": "#e87ba4", "fusion": "#2a78d6"}


def regime_of(world, t, k) -> str:
    a = world.audio_lit(t, k)
    v = world.video_lit(t, k)
    return "both" if (a and v) else "audio" if a else "video" if v else "dark"


def main() -> None:
    cfg = WorldConfig()
    cached = load_json(RESULTS / "montecarlo.json")
    if cached and "thresholds" in cached:
        thresholds = cached["thresholds"]
        print("reusing calibrated thresholds from montecarlo.json")
    else:
        print("calibrating ...")
        thresholds = calibrate(cfg)

    hits = {m: {r: [0, 0] for r in REGIMES} for m in MODES}
    for s in SEEDS:
        w = make_world(cfg, seed=s)
        au, vi = sense_audio(w, seed=s), sense_video(w, seed=s)
        for mode in MODES:
            out = run_mode(mode, w, au, vi, extract_threshold=thresholds[mode])
            for k in range(w.n_frames):
                e = out.estimates[k]
                ez = np.asarray(e)[:, 0] if np.size(e) else np.zeros(0)
                for t in w.talkers:
                    if not t.present(k):
                        continue
                    r = regime_of(w, t, k)
                    ok = bool(ez.size and np.min(np.abs(wrap(ez - t.azimuth[k])))
                              <= DEFAULT_CUTOFF)
                    hits[mode][r][0] += int(ok)
                    hits[mode][r][1] += 1
        print(f"  seed {s} done", flush=True)

    cov = {m: {r: (hits[m][r][0] / hits[m][r][1] if hits[m][r][1] else float("nan"))
               for r in REGIMES} for m in MODES}
    shares = {r: hits[MODES[0]][r][1] / sum(hits[MODES[0]][x][1] for x in REGIMES)
              for r in REGIMES}
    save_json({"seeds": SEEDS, "thresholds": thresholds, "coverage": cov,
               "regime_shares": shares, "counts": hits},
              RESULTS / "regimes.json")

    print("\nregime shares: " + "  ".join(f"{r}={shares[r]:.2f}" for r in REGIMES))
    print(f"\n{'variant':28s}" + "".join(f"{r:>12s}" for r in REGIMES))
    for m in MODES:
        print(f"{LABELS[m]:28s}" + "".join(f"{cov[m][r]:12.2f}" for r in REGIMES))

    _figure(cov, shares)
    _timeline(cfg, thresholds)


def _figure(cov, shares) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.0))
    x = np.arange(len(REGIMES))
    w = 0.2
    for i, m in enumerate(MODES):
        ax.bar(x + (i - 1.5) * w, [cov[m][r] for r in REGIMES], width=w,
               color=COLORS[m], label=LABELS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{REGIME_LABEL[r]}\n({shares[r]:.0%} of frames)"
                        for r in REGIMES], fontsize=9)
    ax.set_ylabel("track coverage")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, ncols=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Coverage by sensor regime — where each blind spot bites",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_regimes.png", dpi=140, bbox_inches="tight")
    print("wrote results/figures/fig2_regimes.png")


def _timeline(cfg, thresholds) -> None:
    """One talker, one run: regimes as bands, coverage as marks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = make_world(cfg, seed=3)
    au, vi = sense_audio(w, seed=3), sense_video(w, seed=3)
    t = max(w.talkers, key=lambda x: x.death_frame - x.birth_frame)
    band = {"both": "#dfe7ef", "audio": "#fbe0d3", "video": "#fdf0cf",
            "dark": "#c9d2db"}

    fig, ax = plt.subplots(figsize=(11, 3.4))
    times = w.times
    for k in range(t.birth_frame, t.death_frame):
        ax.axvspan(times[k], times[k] + w.dt, color=band[regime_of(w, t, k)],
                   lw=0)
    for i, m in enumerate(MODES):
        out = run_mode(m, w, au, vi, extract_threshold=thresholds[m])
        ys, xs = [], []
        for k in range(t.birth_frame, t.death_frame):
            e = out.estimates[k]
            ez = np.asarray(e)[:, 0] if np.size(e) else np.zeros(0)
            if ez.size and np.min(np.abs(wrap(ez - t.azimuth[k]))) <= DEFAULT_CUTOFF:
                xs.append(times[k]); ys.append(len(MODES) - 1 - i)
        ax.scatter(xs, ys, s=6, color=COLORS[m], marker="s")
    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([LABELS[m] for m in reversed(MODES)], fontsize=8.5)
    ax.set_xlabel("time (s)")
    ax.set_xlim(times[t.birth_frame], times[t.death_frame - 1])
    ax.set_ylim(-0.6, len(MODES) - 0.4)
    ax.set_title("One talker: a mark means the tracker had them. "
                 "Background = which sensors could see them.",
                 loc="left", fontsize=10.5)
    handles = [plt.Line2D([], [], marker="s", ls="", color=band[r],
                          label=REGIME_LABEL[r], markersize=10) for r in REGIMES]
    ax.legend(handles=handles, fontsize=8, ncols=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.42))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_timeline.png", dpi=140, bbox_inches="tight")
    print("wrote results/figures/fig3_timeline.png")


if __name__ == "__main__":
    main()
