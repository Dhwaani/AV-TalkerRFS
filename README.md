# AV-TalkerRFS

**The microphone misses silence. The camera misses occlusion.** These are dual,
physically-modeled failure modes, and this repository puts both inside one
random-finite-set recursion — fusion as *sensor modeling*, not as embedding
concatenation.

The claim in one line: a filter that knows **when** each sensor is blind
localises talkers more accurately, and counts them more reliably, than a filter
holding the *same two sensors* with constant detection probabilities.


---

## Why this is not just "use both sensors"

Give a textbook multi-target tracker both streams and set each sensor's
detection probability to its correct long-run *average* — the fix a
practitioner reaches for. It still loses the talker who is speaking from behind
someone else, because a constant `p_D` cannot express *the camera is blind right
now, so this missed detection means nothing*.

The proposed filter carries three latent scalars per component:

| | |
|---|---|
| `r` | probability this is a real talker at all, not a clutter birth |
| `π` | probability they are **speaking** right now |
| `v` | probability they are **visible** right now |

and factors both detection probabilities through them:

```
p_D_audio = r · π · p_aud
p_D_video = r · v · p_vid · 1{in FOV}
p_S       = r · p_S_talker + (1 − r) · p_S_spurious
```

`π` is driven by the ITU-T P.59 talkspurt/pause chain and updated by audio
evidence; `v` by an occlusion chain and video evidence; `r` by both. A silent
talker is *undetectable to the array by definition* rather than improbably
undetected — so the array's missed detections stop draining the track, and the
camera carries it. Occlusion is the mirror image. A talker dark to both is
carried by the two priors, which is exactly as long as such a track deserves to
live.

Both latent updates use the **single-target Bernoulli recursion**, shared across
every branch of a component — never read off the mixture weights. Reading them
off the weights closes a feedback loop with the PHD's cardinality overshoot; that
cost a week in the predecessor project and arrives here pre-learned, with the
regression test that holds it.

---

## Results

<!-- RESULTS:begin -->
- **OSPA**: 3.65 against 4.55 for the strongest rival on this metric (Video only (occlusion-aware)) — 20% better.
- **head-count RMSE**: 0.68 against 0.81 for the strongest rival on this metric (Naive fusion (constant p_D)) — 16% better.
- **track coverage**: 0.74 against 0.76 for the strongest rival on this metric (Naive fusion (constant p_D)) — 3% worse.
- **The mechanism**: in the *speaking-but-occluded* regime — where only the microphone can help — naive fusion covers 0.69 of frames and modeled fusion covers 0.79. Both filters hold the same two sensors.

### Coverage by sensor regime

| Variant | speaking + visible<br/>(26%) | speaking, **occluded**<br/>(12%) | silent, visible<br/>(41%) | silent + occluded<br/>(21%) |
|---|---|---|---|---|
| Audio only (pause-aware) | 0.86 | **0.88** | 0.07 | 0.07 |
| Video only (occlusion-aware) | 0.94 | 0.09 | 0.93 | 0.07 |
| Naive fusion (constant p_D) | **1.00** | 0.69 | **0.98** | **0.13** |
| **Modeled fusion (proposed)** | 0.95 | 0.79 | 0.94 | 0.09 |

_Track coverage within each sensor regime, 15 scenarios._

### Aggregate, 25 scenarios

| Variant | OSPA (deg) | count RMSE | count bias | coverage | hold (s) |
|---|---|---|---|---|---|
| Audio only (pause-aware) | 7.69 ± 0.32 | 1.18 ± 0.03 | -0.76 ± 0.04 | 0.38 ± 0.01 | 0.62 ± 0.03 |
| Video only (occlusion-aware) | 4.55 ± 0.23 | 0.88 ± 0.03 | -0.51 ± 0.03 | 0.63 ± 0.02 | 0.96 ± 0.03 |
| Naive fusion (constant p_D) | 4.67 ± 0.19 | 0.81 ± 0.03 | **-0.04 ± 0.03** | **0.76 ± 0.01** | **2.19 ± 0.16** |
| **Modeled fusion (proposed)** | **3.65 ± 0.19** | **0.68 ± 0.02** | -0.24 ± 0.02 | 0.74 ± 0.01 | 0.90 ± 0.03 |

_25 independent 40-second scenarios, three talkers, identical measurements for every variant. ± is the standard error over scenarios. Extraction thresholds calibrated per variant on six held-out seeds (audio_only=0.15, video_only=0.25, naive_fusion=0.15, fusion=0.2)._

### The whole operating surface

| Activity | Visibility | Naive OSPA | Modeled OSPA | Gain |
|---|---|---|---|---|
| 0.65 | 0.85 | 4.53° | **2.22°** | +51% |
| 0.39 | 0.85 | 3.46° | **2.06°** | +40% |
| 0.22 | 0.85 | 3.34° | **2.63°** | +21% |
| 0.65 | 0.67 | 4.81° | **2.78°** | +42% |
| 0.39 | 0.67 | 4.50° | **3.43°** | +24% |
| 0.22 | 0.67 | 5.06° | **4.60°** | +9% |
| 0.65 | 0.44 | 5.00° | **3.48°** | +30% |
| 0.39 | 0.44 | 5.59° | **4.72°** | +16% |
| 0.22 | 0.44 | 6.75° | **6.33°** | +6% |

_8 scenarios per cell, thresholds re-calibrated per cell._
<!-- RESULTS:end -->

### How to read this

**The modeled filter wins accuracy, not continuity.** It leads on OSPA (~22%
over naive fusion) and on head-count RMSE (~16%), and it leads across every cell
of the operating surface. It does **not** lead on track coverage (0.74 vs 0.76)
or on unbroken hold time (0.90 s vs 2.19 s). Naive fusion keeps tracks alive
longer and covers marginally more frames; it simply puts them in worse places
and counts them more noisily. If your downstream system cares about continuity
above accuracy, that trade is not obviously in the proposed filter's favour.

**A correction worth reading, because it nearly became the headline.** With
extraction thresholds pinned to a too-high grid floor, naive fusion covered only
0.25 of the speaking-but-occluded regime against the modeled filter's 0.76, and
"naive fusion loses three out of four occluded talkers" looked like the result.
Extending the threshold grid downward — so every variant's optimum sits in the
interior rather than on a boundary — moved naive fusion to 0.69 against 0.79.
The mechanism advantage in its own regime is about **ten points, not fifty**.
The earlier number was a calibration artifact, and the whole result would have
been overstated by a factor of five had the boundary not been checked.

**Nobody does well in `dark`.** No sensor can see a talker who is both silent
and occluded, and no amount of modeling invents information that is not there.
Naive fusion is actually *ahead* in that regime (0.13 vs 0.09) for the same
reason it holds longer everywhere: it is more reluctant to drop anything.

---

## What is in here

```
avrfs/
  world.py       talkers with two independent ways of going dark
  sensors.py     array front end (blind to silence) + camera (blind to occlusion)
  filters.py     one iterated-corrector GM-PHD host, four sensor configurations
  activity.py    ITU-T P.59 talkspurt/pause chain          [shared with TalkerRFS]
  gm.py          wrap-aware Gaussian-mixture machinery      [shared with TalkerRFS]
  conformal.py   calibrated prediction sets                 [shared with TalkerRFS]
  metrics.py     OSPA, cardinality error, coverage, hold time
experiments/
  exp1_montecarlo.py       the aggregate comparison
  exp2_regimes.py          coverage per sensor regime — the mechanism
  exp3_complementarity.py  the whole (activity × visibility) surface
tests/            32 tests, including regressions inherited with the shared code
scripts/          smoke.py (CI end-to-end), make_tables.py (README generator)
```

## Running it

```bash
pip install -e ".[dev]"
make test      # 32 tests, seconds
make all       # calibration + all three experiments + tables (~20 min)
```

```python
from avrfs import WorldConfig, evaluate, make_filter, make_world, sense_audio, sense_video

w  = make_world(WorldConfig(duration=40.0, n_talkers=3), seed=0)
au = sense_audio(w, seed=0)
vi = sense_video(w, seed=0)

out = make_filter("fusion").run(au, vi)
print(evaluate(w, out).as_row())
```

The filter takes two lists of per-frame azimuth arrays and nothing else, so any
DOA front end and any face detector can be substituted for the simulated ones.

---

## Scope & known limitations

* **Everything is simulated, at the measurement level only.** There is no
  rendered audio and no rendered video here. The predecessor project found that
  its measurement-level margin *shrank to a tie* when run end-to-end from
  rendered audio, so treat these numbers as an upper bound until they survive
  [AVA-AVD](https://arxiv.org/abs/2111.14448) or the
  [Ego4D AV benchmark](https://github.com/EGO4D/audio-visual). That is the
  single most valuable next step and it is not done.
* **The occlusion statistics are invented.** ITU-T P.59 gives the speech chain
  real numbers; nothing equivalent exists for how long a walking person stays
  behind another. The defaults were chosen by one stated criterion — both
  sensors must carry a meaningful share of frames — fixed before results were
  examined, and the entire surface around them is swept and reported.
* **The prediction in `exp3` was wrong and is left in the file.** I expected
  modeling to matter most when both sensors were starved; it matters most when
  both are reliable. The reasoning and the correction are recorded rather than
  quietly deleted.
* **Prior art.** [Checka, Wilson, Siracusa & Darrell (ICASSP
  2004)](https://groups.csail.mit.edu/vision/vip/papers/checka_icassp04.pdf)
  already carried a per-person speech-activity bit with a two-state chain and a
  likelihood conditioned on it, in an audio-visual particle filter. What is
  claimed here is narrower and specific: **both** blind spots modeled
  symmetrically (they modeled speech activity, not occlusion), an RFS recursion
  rather than a fixed-cardinality joint particle filter, and chains grounded in
  published statistics where such statistics exist. The prior-art search was
  keyword-based, not systematic — assume there is more.
* **No labels.** The PHD family carries no track identity, so continuity is
  measured by coverage and hold time rather than identity switches. A labelled
  multi-Bernoulli host would give identity directly and would likely suit this
  problem better; `r` already plays much of the role a Bernoulli existence
  probability plays there.
* **No learned baseline.** The comparison is against classical fusion, not
  against a trained audio-visual tracker. The interesting claim against those is
  about *calibration*, not raw accuracy, and testing it needs real data — see
  the first note.

## Related

Built on the shared machinery of [TalkerRFS](https://github.com:Dhwaani/AV-TalkerRFS), which does the audio-only
half of this problem. Fixes flow between the two repositories, and the regression tests travel with the shared modules.

## License

MIT.
