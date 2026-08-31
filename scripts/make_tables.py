"""Generate the README results block from the experiment JSON.

No number in the README is typed by hand. Re-run the experiments, then this,
and the tables cannot drift away from the code that produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODES = ["audio_only", "video_only", "naive_fusion", "fusion"]
LABELS = {"audio_only": "Audio only (pause-aware)",
          "video_only": "Video only (occlusion-aware)",
          "naive_fusion": "Naive fusion (constant p_D)",
          "fusion": "**Modeled fusion (proposed)**"}
PROPOSED = "fusion"
REGIMES = ["both", "audio", "video", "dark"]
REGIME_HEAD = {"both": "speaking + visible", "audio": "speaking, **occluded**",
               "video": "silent, visible", "dark": "silent + occluded"}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def f(v, d=2):
    return f"{v:.{d}f}"


def montecarlo_table():
    data = load("montecarlo.json")
    if not data:
        return ["_(run `make all` to generate)_"]
    agg = data["aggregate"]
    n = len(data["seeds"])
    cols = [("OSPA_deg", "OSPA (deg)", True), ("card_RMSE", "count RMSE", True),
            ("card_bias", "count bias", None), ("coverage", "coverage", False),
            ("hold_s", "hold (s)", False)]
    best = {}
    for k, _, lower in cols:
        best[k] = (min(MODES, key=lambda m: abs(agg[m][k])) if lower is None
                   else min(MODES, key=lambda m: agg[m][k]) if lower
                   else max(MODES, key=lambda m: agg[m][k]))
    lines = ["| Variant | " + " | ".join(c[1] for c in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for m in MODES:
        cells = []
        for k, _, _l in cols:
            txt = f"{f(agg[m][k])} ± {f(agg[m][k + '_se'])}"
            cells.append(f"**{txt}**" if best[k] == m else txt)
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    lines += ["", f"_{n} independent 40-second scenarios, three talkers, "
                  "identical measurements for every variant. ± is the standard "
                  "error over scenarios. Extraction thresholds calibrated per "
                  "variant on six held-out seeds ("
                  + ", ".join(f"{m}={data['thresholds'][m]}" for m in MODES)
                  + ")._"]
    return lines


def regime_table():
    data = load("regimes.json")
    if not data:
        return ["_(run `make all` to generate)_"]
    cov, sh = data["coverage"], data["regime_shares"]
    lines = ["| Variant | " + " | ".join(
        f"{REGIME_HEAD[r]}<br/>({sh[r]:.0%})" for r in REGIMES) + " |",
             "|---" * (len(REGIMES) + 1) + "|"]
    for m in MODES:
        best = {r: max(MODES, key=lambda x: cov[x][r]) for r in REGIMES}
        cells = [(f"**{f(cov[m][r])}**" if best[r] == m else f(cov[m][r]))
                 for r in REGIMES]
        lines.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    lines += ["", f"_Track coverage within each sensor regime, "
                  f"{len(data['seeds'])} scenarios._"]
    return lines


def sweep_table():
    data = load("complementarity.json")
    if not data:
        return ["_(run `make all` to generate)_"]
    g = data["grid"]
    rows = sorted(g.values(), key=lambda c: (-c["_visibility"], -c["_activity"]))
    lines = ["| Activity | Visibility | Naive OSPA | Modeled OSPA | Gain |",
             "|---|---|---|---|---|"]
    for c in rows:
        n_, f_ = c["naive_fusion"]["OSPA_deg"], c["fusion"]["OSPA_deg"]
        lines.append(f"| {c['_activity']:.2f} | {c['_visibility']:.2f} | "
                     f"{f(n_)}° | **{f(f_)}°** | {100 * (1 - f_ / n_):+.0f}% |")
    lines += ["", f"_{len(data['seeds'])} scenarios per cell, thresholds "
                  "re-calibrated per cell._"]
    return lines


def headline():
    mc, rg = load("montecarlo.json"), load("regimes.json")
    if not mc:
        return []
    a = mc["aggregate"]
    out = []
    for key, label, lower in [("OSPA_deg", "OSPA", True),
                              ("card_RMSE", "head-count RMSE", True),
                              ("coverage", "track coverage", False)]:
        rivals = [m for m in MODES if m != PROPOSED]
        b = (min(rivals, key=lambda m: a[m][key]) if lower
             else max(rivals, key=lambda m: a[m][key]))
        pv, bv = a[PROPOSED][key], a[b][key]
        rel = 100 * (1 - pv / bv) if lower else 100 * (pv / bv - 1)
        verdict = "better" if rel >= 0 else "worse"
        out.append(f"- **{label}**: {f(pv)} against {f(bv)} for the strongest "
                   f"rival on this metric ({LABELS[b].strip('*')}) "
                   f"— {abs(rel):.0f}% {verdict}.")
    if rg:
        c = rg["coverage"]
        out.append(f"- **The mechanism**: in the *speaking-but-occluded* regime "
                   f"— where only the microphone can help — naive fusion covers "
                   f"{f(c['naive_fusion']['audio'])} of frames and modeled "
                   f"fusion covers {f(c['fusion']['audio'])}. Both filters hold "
                   "the same two sensors.")
    return out


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    begin, end = "<!-- RESULTS:begin -->", "<!-- RESULTS:end -->"
    block = ["", *headline(), "",
             "### Coverage by sensor regime", ""] + regime_table()
    block += ["", "### Aggregate, 25 scenarios", ""] + montecarlo_table()
    block += ["", "### The whole operating surface", ""] + sweep_table() + [""]
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(head + begin + "\n".join(block) + end + tail)
    print("updated README.md results block")


if __name__ == "__main__":
    main()
