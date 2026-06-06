# Open-Access Rail Entry: Creation or Cannibalisation?

A reproducible causal-inference study of whether a no-subsidy "open-access" rail
operator entering a corridor already served by a franchised incumbent **grows the
market** (creation) or **diverts traffic from the incumbent** (cannibalisation),
and of how much of any growth comes off competing modes such as air.

The case study is **Lumo**, which began low-fare London (King's Cross)–Edinburgh
services on the East Coast Main Line (ECML) on 25 October 2021. The analysis uses
only open and public data, and every figure and number is reproducible end-to-end
from the scripts in this repository.

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)

---

## Research question

When an open-access operator enters, post-entry corridor ridership can change for
three mutually exclusive reasons:

1. **Abstraction from the incumbent** — share moves between rail operators; total
   corridor rail demand is unchanged (the regulator's "cannibalisation" concern).
2. **Modal shift** — travellers move to rail from air or car (the climate dividend).
3. **Induced demand** — genuinely new trips.

The regulator's test pits (1) against (2)+(3); the climate case rests on (2). An
operator-agnostic origin–destination matrix nets out (1) by construction, so any
growth it records is (2)+(3); pairing it with air-route data separates the two.

## Headline findings

- **Station totals are confounded.** A naive off-corridor synthetic control gives the
  Lumo stop Newcastle about **+17%**, but a placebo-in-space test shows the whole East
  Coast corridor recovered comparably after 2021 *irrespective of Lumo*. Confound-robust
  station-total estimates (within-corridor synthetic control, an event-study DiD, and two
  GPU-trained deep counterfactuals) are statistically indistinguishable from zero.
- **The market grew (creation).** The total London–Edinburgh flow grew **+60%** against
  flat off-corridor comparators. Among 391 comparable London flows, Edinburgh's recovery
  ranks **3rd** (randomisation inference *p* = 0.008); an OD-flow event-study DiD gives
  **+79.5%** (*p* = 0.002) with the effect onset at the launch year; distance-matched
  corridor clustering gives *p* < 10⁻⁴; a double-machine-learning causal forest gives
  **+22%** (95% CI [11.5, 36.0]).
- **No cannibalisation.** The incumbent LNER reached record volumes (recovery ratio 1.25
  vs ≤1.0 for every comparable long-distance operator); Lumo's journeys sit on top of a
  surging incumbent, not subtracted from a falling one.
- **Mostly modal shift from air.** On London–Edinburgh, rail's share of the air+rail
  market rose from **29% to 43%**; roughly **two-thirds** of the rail growth equals the
  absolute fall in air traffic. A Glasgow/West-Coast placebo, where an equivalent air
  decline was *not* captured by rail, isolates the rail-supply mechanism.
- **Carbon benefit.** Quantified with per-pair emissions, the growth saves an estimated
  **84–129 kt CO₂e per year** (90% CI), and is carbon-beneficial unless almost all of the
  new journeys are purely induced.

Full numbers and inference are in [`docs/RESULTS.md`](docs/RESULTS.md); honest caveats are
in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Data

All sources are open or public, released under the UK Open Government Licence. Raw files
are **not** committed (they total several GB); see [`data/README.md`](data/README.md) and
[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for canonical URLs, access dates, and the
expected on-disk layout.

| Dataset | Role |
|---|---|
| ORR *Estimates of Station Usage* (Table 1415/1410) | Station-total panel spine |
| ORR Origin–Destination Matrix (ODM), 7 financial years 2018–19 → 2024–25 | Market (OD-pair) test; event study |
| ORR passenger journeys by operator (Table 1223) | Incumbent vs peers; no-cannibalisation test |
| CAA Table 12.2 domestic air route analysis | Air → rail modal-shift decomposition |
| Green Travel per-pair emissions | Carbon / welfare quantification |
| Network Rail Daily Concourse Footfall | External validation of modelled usage |
| NaPTAN rail-node coordinates | Distance-to-London covariate |

## Methods

The design is deliberately multi-method, with the unit of analysis moved from stations to
origin–destination flows, where the policy question is properly posed. Estimators
(full descriptions in [`docs/METHODS.md`](docs/METHODS.md)):

- **Station-total counterfactuals** — convex synthetic control, within-corridor synthetic
  control, augmented/ridge synthetic control, generalised synthetic control with
  interactive fixed effects, a within-corridor event-study difference-in-differences, and a
  Callaway–Sant'Anna staggered-adoption estimator.
- **Operator level** — a difference-in-ratios of the incumbent against peer operators, and a
  Bayesian structural time-series (CausalImpact) counterfactual.
- **Market (OD) level** — the ODM recovery test, placebo-in-space randomisation inference, a
  distance-matched corridor-clustering permutation test, an OD-flow panel event-study DiD
  with randomisation inference, and a double-machine-learning causal forest.
- **Deep counterfactuals** — an attention-over-donors ensemble and a domain-adversarial,
  treatment-invariant network, both with split-conformal prediction intervals.
- **Inference and robustness** — randomisation/permutation inference throughout, family-wise
  (Holm) and false-discovery-rate (Benjamini–Hochberg) multiplicity correction, and a formal
  sensitivity analysis for unobserved confounding (Oster's δ, the VanderWeele–Ding E-value,
  and the Cinelli–Hazlett robustness value).

## Repository layout

```
.
├── src/                 # all analysis code (importable package)
│   ├── ingest/          # downloaders for the open data sources
│   ├── clean/           # parsers: raw releases -> tidy tables
│   ├── features/        # build the unified analysis panel + unit selection
│   ├── models/          # estimators
│   │   ├── classical/   #   synthetic control, DiD, CS-DiD, BSTS, ...
│   │   ├── causal_forest/#  DML causal forest + heterogeneity
│   │   └── deep/         #  attention ensemble, domain-adversarial CRN, GRU
│   ├── evaluate/        # robustness, sensitivity, multiplicity, triangulation
│   └── viz/             # figures
├── configs/             # YAML config (data URLs, panel structure) — no magic numbers in code
├── tests/               # pytest suite (imports + estimator validation on synthetic data)
├── results/             # figures/, metrics/ (JSON), tables/ (CSV) — committed outputs
├── data/                # raw/ interim/ processed/ (git-ignored; see data/README.md)
├── docs/                # METHODS, RESULTS, LIMITATIONS, DATA_SOURCES
├── run_pipeline.py      # idempotent end-to-end orchestrator (guard-skips stages w/o data)
├── Dockerfile, environment.yml, requirements*.txt, Makefile
└── .github/workflows/   # CI: lint + tests
```

## Getting started

```bash
# 1. Environment (Python 3.12)
python -m venv .venv && . .venv/Scripts/activate     # Windows
# source .venv/bin/activate                          # macOS / Linux
pip install -r requirements.txt

# 2. Obtain the open data (see data/README.md for sources and layout).
#    Some sources auto-download; the Rail Data Marketplace files need a free account.

# 3. Run the pipeline (idempotent; each stage guard-skips if its inputs are absent)
python run_pipeline.py

# 4. Tests (no data required — estimators are validated on synthetic ground truth)
pytest
```

`run_pipeline.py` is safe to re-run: completed stages are skipped, and any stage whose input
data is not present returns cleanly rather than failing. Outputs are written to `results/`.

## Results

Generated figures, metric JSONs, and tables are committed under [`results/`](results/) so the
findings can be inspected without re-running the full pipeline. A master synthesis figure
(`results/figures/master_synthesis.png`) places every treatment-effect estimate on one axis
grouped by level of analysis.

## Licensing

- **Code** in this repository is released under the [MIT License](LICENSE).
- **Data** are the property of their providers (ORR, CAA, Network Rail, DfT) and are used
  under the UK Open Government Licence v3.0; this repository does not redistribute the raw
  data — only the code that processes it and the derived results.
- The University of Sheffield coat of arms, where used in related materials, is reproduced
  under CC BY-SA 4.0 and is not covered by this repository's licence.

## Author

**Nijat Badalov** — MSc Advanced Control and Systems Engineering, School of Electrical and
Electronic Engineering, Faculty of Engineering, The University of Sheffield.
Contact: nijat.badalov.04@gmail.com

## Citation

If you use this code or its results, please cite it (see [`CITATION.cff`](CITATION.cff)):

> Badalov, N. (2026). *Open-Access Rail Entry: Creation or Cannibalisation? A causal-inference
> study of Lumo on the East Coast Main Line* [Software]. https://github.com/<your-account>/open-access-rail-causal
