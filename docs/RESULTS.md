# Results

All numbers below are produced by the pipeline and stored under `results/metrics/` (JSON) and
`results/tables/` (CSV); figures are under `results/figures/`. The master synthesis figure
(`results/figures/master_synthesis.png`) places every estimate on one axis by level of analysis.

## 1. Station totals are confounded and, once corrected, null

| Method | Estimand | Estimate | Inference |
|---|---|---|---|
| Naive off-corridor synthetic control | Newcastle total | +17.4% | placebo *p* = 0.10; **confounded** |
| Within-corridor synthetic control | Newcastle vs ECML | +12% | *p* = 0.54; not significant |
| Generalised SC (interactive FE) | Newcastle | +16% | descriptive |
| Event-study DiD (Lumo stops, avg) | n/a | +3.7% | parallel trends ok; small |
| Deep counterfactual (conformal) | Newcastle | +5.0% | [−6, +18]; n.s. |
| Callaway–Sant'Anna | overall ATT | +16% | RI *p* = 0.15; n.s. |

A placebo-in-space test shows non-Lumo East Coast through-stations rose comparably after 2021
(York +18%, Doncaster +25%, Darlington +15%): the corridor recovered faster *irrespective of Lumo*.
A placebo-in-time test (a fake 2018 launch) returns +0.8% at Newcastle, confirming no spurious
pre-trend at the main treated station. Station totals do not identify a creation effect, by design.

## 2. No cannibalisation of the incumbent

| Operator | Recovery ratio (2024-25 / 2018-19) |
|---|---|
| **LNER (ECML incumbent)** | **1.254** (record volumes) |
| LNER + Lumo | 1.320 |
| CrossCountry / GWR / Avanti / TransPennine | 0.92 – 0.96 (below pre-pandemic) |
| National total | 0.994 |

Lumo's ≈1.4 million annual journeys were added on top of a surging incumbent, not subtracted from a
falling one. A Bayesian structural time-series model tempers the magnitude (LNER −4.8%, 95% CI
[−19, +12], not significant once LNER's own trend is controlled), which is why the clean resolution is
the OD level.

## 3. The market grew: significant creation at the OD level

- **London–Edinburgh flow: +60%**; Newcastle–London: +20%; off-corridor comparators ≈ 0% (flat).
- **Placebo-in-space:** among 391 comparable London flows, Edinburgh ranks **3rd** (RI *p* = 0.008,
  99.5th percentile); the median London flow recovered to only 0.81 (still 19% below pre-pandemic).
  Among long-distance flows only, Edinburgh ranks **1st**.
- **Corridor clustering:** the East Coast corridor out-recovered other long-distance routes (median
  1.21 vs 0.92; permutation **p < 10⁻⁴**; 7 of the top-10 long-distance flows are ECML).
- **OD-flow event-study DiD:** **+79.5%** (*p* = 0.002), flat pre-trend (2018 coefficient +0.7%), with
  the **launch-year coefficient +77%**: the effect onsets exactly when Lumo entered.
- **Causal forest (DML):** corridor effect **+22%**, 95% CI **[11.5, 36.0]** (excludes zero).

The one commuter Lumo stop (Stevenage) saw its London market fall (−11%), consistent with Lumo's
cheap-advance, leisure-oriented model.

## 4. Mechanism: air → rail modal shift

| Corridor | Δ Rail (k) | Δ Air (k) | Rail share pre→post | Air abstraction | Lumo? |
|---|---|---|---|---|---|
| **Edinburgh** (ECML) | +832 | −544 | 29% → 43% | ≈ 65% of growth | yes |
| Newcastle (ECML) | +256 | −13 | 73% → 77% | ≈ 5% (mostly induced) | yes |
| **Glasgow** (WCML) | +31 | −346 | 23% → 27% | rail did **not** capture | no (placebo) |

Air fell on both Scottish corridors; rail captured it only on the East Coast, where capacity grew,
isolating a rail-supply mechanism from a generic, exogenous air decline.

## 5. Carbon

Long-distance rail is roughly 12× cleaner per passenger than flying this corridor (Edinburgh: 13.8 vs
165.0 kg CO₂e). Combining the measured air-abstraction share with the ODM growth and propagating
uncertainty by Monte Carlo, the London–Edinburgh + Newcastle growth saves an estimated **84–129 kt
CO₂e per year** (90% CI; central ≈ 108 kt). The result is robust to the modal split: the growth cuts
carbon unless more than ~90% of the new journeys are purely induced.

## 6. Robustness

- **Multiplicity:** five of six headline tests survive Benjamini–Hochberg FDR control; the three
  decisive ones (corridor clustering, OD event-study DiD, Edinburgh placebo) survive the stricter
  Holm–Bonferroni family-wise control.
- **Sensitivity:** Oster δ = 1.5 and an E-value of 1.94 lean robust; the Cinelli–Hazlett robustness
  value (≈0.12) shows linear covariate adjustment alone is not bulletproof, which is why
  identification rests on the design (Glasgow placebo, flat pre-trend, corridor clustering, event-study
  onset).
- **Confounders:** the incumbent's punctuality *worsened* after entry (PPM 86.4% → 81.6%), so growth
  occurred despite a service headwind; long-distance fares tracked their long-run trend.
- **Validation:** Network Rail physical gate counts correlate with the modelled ORR usage at *r* = 0.93
  across 18 managed stations.
- **External validity:** a pre-pandemic Grand Central placebo (no COVID confound) gives Sunderland +73%
  and Bradford +49% against comparable stations, suggestive, directionally consistent replication.
