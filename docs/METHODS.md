# Methods

The identification problem has three features that make naive designs unreliable, and the design
addresses each one.

1. **Few treated units.** Only four stations are directly served, so cluster-robust inference is
   untrustworthy. We use randomisation / permutation inference throughout rather than asymptotic
   standard errors.
2. **A corridor-wide recovery confound.** The East Coast Main Line recovered from the pandemic faster
   than other lines *irrespective of Lumo*, so off-corridor controls are invalid for station totals.
   We diagnose this with placebo-in-space and placebo-in-time tests.
3. **Multi-market hubs.** Station totals at hubs such as Edinburgh (whose dominant flow is Glasgow,
   not London) dilute any London-specific effect. The resolution is to analyse origin–destination
   flows: the ODM counts journeys irrespective of operator, so growth on the London–Edinburgh flow is
   net new demand, not share moved between rail operators.

**Estimand.** Entry is not randomly assigned, so the causal interpretation is design-based, resting on
four mutually reinforcing arguments rather than an instrument: a flat pre-trend in the OD event study;
the effect's onset coinciding with entry; placebo-in-space across all comparable London flows; and a
Glasgow/West-Coast placebo where an equivalent air decline did not translate into rail growth absent a
capacity expansion. At the OD level the estimand is the effect of open-access-driven corridor entry on
the total market, not Lumo's isolated marginal share (the operator-agnostic ODM cannot separate the two).

## Estimators

### Station-total counterfactuals
- **Synthetic control** (Abadie–Diamond–Hainmueller). Outcome is log(entries + exits) over the
  LENNON era (≥2004); donors are balanced, non-grouped stations screened of Elizabeth-line and Avanti
  contamination; convex weights over the nearest donors by pre-period trajectory; placebo-in-space
  inference.
- **Within-corridor synthetic control**: restricts donors to the East Coast corridor, which removes
  the corridor-wide recovery confound.
- **Augmented / ridge synthetic control** (Ben-Michael–Feller–Rothstein) and **generalised synthetic
  control with interactive fixed effects** (Xu).
- **Within-corridor event-study difference-in-differences** and a **Callaway–Sant'Anna**
  staggered-adoption estimator robust to the negative weighting of two-way fixed effects under
  heterogeneous timing.

### Operator level
- **Difference-in-ratios** of the incumbent (LNER) against peer long-distance operators.
- **Bayesian structural time-series** (CausalImpact) counterfactual for the incumbent.

### Market (OD) level, the decisive tier
- **ODM recovery test**: post/pre ratio of total London↔city journeys, operator-agnostic.
- **Placebo-in-space randomisation inference**: computes the recovery of every comparable London flow
  (pre-volume ≥ 150k) and ranks the treated flows.
- **Distance-matched corridor-clustering permutation test**: compares the East Coast corridor against
  other long-distance flows at matched distance.
- **OD-flow panel event-study DiD**: two-way (flow + year) fixed effects via Frisch–Waugh demeaning,
  with a joint event-study fit and large-draw randomisation inference. The partial launch year is
  excluded from the binary pre-vs-post contrast and reported as its own coefficient (the onset).
- **Double-machine-learning causal forest**: a cross-fitted T-learner for the conditional treatment
  surface and a Robinson partialling-out DML estimator for the average effect, adjusting for distance
  and pre-volume. Read as a complement (selection-on-observables), not a substitute for the design.

### Deep counterfactuals (uncertainty-quantified)
- An **attention-over-donors ensemble** predicting each station's no-entry trajectory, trained
  self-supervised across donors, with split-conformal intervals.
- A **domain-adversarial, treatment-invariant network** (a CRN-lite): an encoder made
  treatment-invariant by a gradient-reversal domain-adversarial head, with an outcome head trained on
  control donors and split-conformal calibration. (A recurrent/GRU variant is also implemented.)

### Inference and robustness
- **Randomisation / permutation inference** throughout.
- **Multiplicity**: family-wise (Holm) and false-discovery-rate (Benjamini–Hochberg) correction of
  the headline tests.
- **Sensitivity to unobserved confounding**: Oster's δ, the VanderWeele–Ding E-value, and the
  Cinelli–Hazlett robustness value.
- **Confounder checks**: incumbent punctuality and fares around entry.
- **External validation**: Network Rail physical gate counts against the modelled ORR usage, and a
  pre-pandemic Grand Central placebo with no COVID confound.

Each estimator is validated against ground truth: a known injected effect is recovered to machine
precision, and the DML estimator recovers a known treatment effect under simulated confounding.
