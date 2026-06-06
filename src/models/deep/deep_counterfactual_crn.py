"""
CRN-lite: treatment-invariant deep counterfactual (domain-adversarial, Tier-3 architecture B).

THINK -> RESEARCH -> CODE
  WHY: the brief asks for a Counterfactual Recurrent Network (Bica et al. 2020) — its core idea is a
        representation Phi(history) that is (a) predictive of the untreated outcome AND (b) BALANCED /
        TREATMENT-INVARIANT, so an outcome model fit on controls transfers to treated units without
        selection bias. We implement that core (a domain-adversarial GRADIENT-REVERSAL head enforcing
        invariance) on the FAST MLP-ensemble encoder from `deep_counterfactual.py` — sidestepping the
        recurrent-throughput problem that made the plain GRU impractical (D-018), so it actually trains
        and saturates the GPU.
  DESIGN:
    - encoder:  pre-period sequence -> Phi  (ensemble MLP)
    - outcome:  Phi -> full untreated trajectory; supervised on CONTROL donors only (train split)
    - adversary: GRL(Phi) -> treatment label (donor=0 / ECML-corridor=1); BCE. Gradient reversal makes
                 the encoder hide treatment from Phi -> balanced representation.
    - loss = outcome_MSE(donors) + lambda(t) * adversary_BCE(all)   [lambda ramped, DANN schedule]
    - counterfactual(treated) = outcome(Phi_treated); effect = observed_post - predicted_post
    - SPLIT-CONFORMAL: hold out 20% of donors; their forecast residuals give a distribution-free band.
  HONESTY: like the attention model, the clean-donor pool inherits the corridor confound, so the
        MAGNITUDE is a method/UQ demonstration (RQ3), not the clean causal effect. The new content is
        the treatment-invariance mechanism + a 3rd architecture that agrees with the attention model.

Run:  python -m src.models.deep.deep_counterfactual_crn
"""

from __future__ import annotations

import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.models.deep.deep_counterfactual import EnsembleLinear, GPUMonitor
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, ensure_dirs

LOG = get_logger("models.deep_counterfactual_crn", log_file="logs/models.log")

ENSEMBLE, HIDDEN, EPOCHS, LR, SEED = 48, 384, 2000, 2e-3, 20211025
LAMBDA_MAX = 0.2  # adversarial strength ceiling — gentle (a strong GRL arms-races to divergence)
GRAD_CLIP = 1.0  # stabilise the adversarial min-max game
CALIB_FRAC = 0.20  # held-out donor fraction for split-conformal


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def _grl(x, lambd):
    return _GradReverse.apply(x, lambd)


class CRNLite(nn.Module):
    """Ensemble encoder + untreated-outcome head + gradient-reversal treatment adversary."""

    def __init__(self, E, t_pre, T, hidden):
        super().__init__()
        self.enc = nn.Sequential(
            EnsembleLinear(E, t_pre, hidden), nn.GELU(), EnsembleLinear(E, hidden, hidden), nn.GELU()
        )
        self.outcome = nn.Sequential(EnsembleLinear(E, hidden, hidden), nn.GELU(), EnsembleLinear(E, hidden, T))
        self.adv = nn.Sequential(EnsembleLinear(E, hidden, hidden), nn.GELU(), EnsembleLinear(E, hidden, 1))

    def forward(self, seq_pre, lambd=0.0):
        phi = self.enc(seq_pre)  # [E,M,H]
        y = self.outcome(phi)  # [E,M,T]
        t = self.adv(_grl(phi, lambd)).squeeze(-1)  # [E,M]
        return y, t, phi


def main() -> None:
    ensure_dirs()
    if not torch.cuda.is_available():
        LOG.error("CUDA not available — aborting CRN-lite GPU run.")
        return
    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    pre = yarr < treat
    t_pre, T = int(pre.sum()), len(years)

    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    LOG.info("GPU: %s | torch %s | CRN-lite", torch.cuda.get_device_name(0), torch.__version__)

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")

    def matrix(crs_list):
        sub = panel.filter(pl.col("crs").is_in(crs_list) & pl.col("year_start").is_in(years)).select("crs", "year_start", "value")
        wide = sub.pivot(values="value", index="crs", on="year_start").with_columns(pl.col("crs").cast(pl.Enum(crs_list)).alias("_o")).sort("_o")
        return np.log(np.clip(wide.select([str(y) for y in years]).to_numpy(), 1.0, None))

    donor_crs = units.filter((pl.col("role") == "donor_clean") & pl.col("balanced"))["crs"].to_list()
    corridor_crs = units.filter(pl.col("ecml_corridor") & pl.col("balanced"))["crs"].to_list()
    Yd, Yc, Yt = matrix(donor_crs), matrix(corridor_crs), matrix(served)
    N = Yd.shape[0]

    # standardise per series on donor stats
    mu, sd = Yd.mean(1, keepdims=True), Yd.std(1, keepdims=True) + 1e-6
    Ydn = (Yd - mu) / sd
    cmu, csd = Yc.mean(1, keepdims=True), Yc.std(1, keepdims=True) + 1e-6
    Ycn = (Yc - cmu) / csd

    # split-conformal: hold out donors for calibration
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_cal = max(20, int(CALIB_FRAC * N))
    cal_idx, tr_idx = perm[:n_cal], perm[n_cal:]
    LOG.info("donors=%d (train=%d, calib=%d) | corridor=%d | treated=%d | years=%d (pre=%d)", N, len(tr_idx), n_cal, len(corridor_crs), len(served), T, t_pre)

    E = ENSEMBLE
    # domains: train-donors (label 0) + corridor (label 1) drive the adversary
    dom_pre = np.vstack([Ydn[tr_idx][:, pre], Ycn[:, pre]])
    dom_lab = np.concatenate([np.zeros(len(tr_idx)), np.ones(len(corridor_crs))])
    dom_pre_E = torch.tensor(dom_pre, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, t_pre).contiguous()
    dom_lab_E = torch.tensor(dom_lab, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1).contiguous()
    # outcome supervised on TRAIN donors only (untreated full trajectory)
    tr_pre_E = torch.tensor(Ydn[tr_idx][:, pre], dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, t_pre).contiguous()
    tr_full_E = torch.tensor(Ydn[tr_idx], dtype=torch.float32, device=dev).unsqueeze(0).expand(E, -1, T).contiguous()

    model = CRNLite(E, t_pre, T, HIDDEN).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("CRN-lite params: %.2fM (E=%d, H=%d)", n_params / 1e6, E, HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    mon = GPUMonitor()
    mon.start()
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        p = ep / EPOCHS
        lambd = LAMBDA_MAX * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)  # DANN schedule 0 -> LAMBDA_MAX
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_tr, _, _ = model(tr_pre_E, 0.0)
            outcome_loss = ((y_tr - tr_full_E) ** 2).mean()
            _, t_dom, _ = model(dom_pre_E, lambd)
            adv_loss = bce(t_dom, dom_lab_E)
            loss = outcome_loss + adv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # tame the min-max game
        opt.step()
        if ep % 500 == 0 or ep == EPOCHS - 1:
            LOG.info("  epoch %4d  outcome-MSE=%.5f  adv-BCE=%.4f  lambda=%.2f", ep, outcome_loss.item(), adv_loss.item(), lambd)
    torch.cuda.synchronize()
    train_s = time.time() - t0
    mon.stop()
    gpu = mon.summary()
    LOG.info("CRN-lite training %.1fs (%.1f ep/s) | GPU %s", train_s, EPOCHS / train_s, json.dumps(gpu))

    # ---- counterfactual for treated ----
    model.eval()
    qmu, qsd = Yt.mean(1, keepdims=True), Yt.std(1, keepdims=True) + 1e-6
    qpre_E = torch.tensor(((Yt - qmu) / qsd)[:, pre], dtype=torch.float32, device=dev).unsqueeze(0).expand(E, len(served), t_pre).contiguous()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_t, _, _ = model(qpre_E, 0.0)
    pred_t = y_t.float().cpu().numpy() * qsd[None] + qmu[None]  # [E,4,T] log
    cf_mean = pred_t.mean(0)
    cf_lo, cf_hi = np.percentile(pred_t, [5, 95], axis=0)

    # split-conformal on held-out calib donors (full-trajectory abs resid)
    cal_pre_E = torch.tensor(Ydn[cal_idx][:, pre], dtype=torch.float32, device=dev).unsqueeze(0).expand(E, n_cal, t_pre).contiguous()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        y_cal, _, _ = model(cal_pre_E, 0.0)
    y_cal = y_cal.float().mean(0).cpu().numpy() * sd[cal_idx] + mu[cal_idx]  # [n_cal,T] log
    resid = np.abs(y_cal - Yd[cal_idx]).ravel()
    q90 = float(np.quantile(resid, 0.9))

    post = ~pre
    results = {
        "architecture": "CRN-lite (domain-adversarial treatment-invariant representation)",
        "gpu": gpu,
        "train_seconds": round(train_s, 1),
        "epochs": EPOCHS,
        "params_millions": round(n_params / 1e6, 2),
        "lambda_max": LAMBDA_MAX,
        "conformal_q90_log": q90,
        "n_calib_donors": int(n_cal),
        "effects": {},
    }
    for i, crs in enumerate(served):
        obs, cfp = Yt[i, post], cf_mean[i, post]
        eff = float(np.exp((obs - cfp).mean()) - 1)
        lo = float(np.exp((obs - (cfp + q90)).mean()) - 1)
        hi = float(np.exp((obs - (cfp - q90)).mean()) - 1)
        results["effects"][crs] = {"crn_effect_pct": round(100 * eff, 1), "conformal_lo_pct": round(100 * lo, 1), "conformal_hi_pct": round(100 * hi, 1)}
        LOG.info("  %s CRN-lite effect = %+.1f%%  [conformal90 %+.1f, %+.1f]", crs, 100 * eff, 100 * lo, 100 * hi)

    # agreement with the attention model (RQ3)
    att = METRICS / "m4_deep_counterfactual.json"
    if att.exists():
        a = json.loads(att.read_text(encoding="utf-8")).get("effects", {})
        agree = {c: {"crn": results["effects"][c]["crn_effect_pct"], "attention": a.get(c, {}).get("deep_effect_pct")} for c in served if c in a}
        results["agreement_with_attention_model_pct"] = agree
        LOG.info("RQ3 two-architecture agreement (CRN vs attention): %s", json.dumps(agree))

    (METRICS / "m4_deep_crn.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(served, years, Yt, cf_mean, cf_lo, cf_hi, treat, results)
    LOG.info("M4 CRN-lite complete. GPU util mean=%.1f%% max=%.0f%%", gpu.get("gpu_util_mean", 0), gpu.get("gpu_util_max", 0))


def _plot(served, years, Yt, cf_mean, cf_lo, cf_hi, treat, results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for crs, ax in zip(served, axes.ravel()):
        i = served.index(crs)
        ax.plot(years, np.exp(Yt[i]) / 1e6, "o-", color="#d62728", lw=2, label="observed")
        ax.plot(years, np.exp(cf_mean[i]) / 1e6, "s--", color="#2ca02c", lw=1.8, label="CRN-lite counterfactual")
        ax.fill_between(years, np.exp(cf_lo[i]) / 1e6, np.exp(cf_hi[i]) / 1e6, color="#2ca02c", alpha=0.16, label="ensemble 5-95%")
        ax.axvline(treat, color="grey", ls=":", lw=1.3)
        e = results["effects"][crs]
        ax.set_title(f"{crs}: CRN-lite {e['crn_effect_pct']:+.1f}% [{e['conformal_lo_pct']:+.0f},{e['conformal_hi_pct']:+.0f}]", fontsize=10)
        ax.set_ylabel("entries+exits (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    g = results["gpu"]
    fig.suptitle(f"CRN-lite (domain-adversarial treatment-invariant) — GPU mean {g.get('gpu_util_mean','?')}% / max {g.get('gpu_util_max','?')}%", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "m4_deep_crn.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
