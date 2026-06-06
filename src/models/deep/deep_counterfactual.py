"""
Deep counterfactual prediction — attention-over-donors ensemble (M4, Tier 3).

THINK -> RESEARCH -> CODE
  WHAT: A deep "synthetic control with attention": for a target station, embed every
        donor's pre-treatment trajectory, attend the target's pre-trajectory over the
        donors, and predict the target's FULL trajectory as an attention-weighted (and
        nonlinearly corrected) combination of donor trajectories. Trained self-
        supervised across all donors (each donor predicted from the others; self
        masked) — a real ~2.3k-example task. Effect = observed − predicted_post.
  WHY : Tier 3 of the brief (counterfactual net with attention over donor stations).
        vs classical SC (RQ3): does a flexible neural counterfactual change the point
        estimate, and are its uncertainty intervals more credible? UQ via a vectorised
        ENSEMBLE + split-conformal calibration on pre-period donor residuals.
  GPU (§7): the data is tiny, so to reach >=90% util we (a) keep ALL data resident on
        GPU (no dataloader), (b) train an ENSEMBLE of E members in ONE batched pass
        (leading E dim, einsum linears) with wide hidden H, (c) bf16 autocast +
        cudnn.benchmark + optional torch.compile, (d) big effective batch (all donors
        as targets each step). A live nvidia-smi sampler records utilisation.
  HONESTY: run on the off-ECML clean-donor pool this inherits the SAME corridor confound
        as the classical SC (so the MAGNITUDE is not the causal effect — see
        WEAKNESSES W1 / operator analysis). Its job here is the METHOD + UQ + GPU
        demonstration (RQ3); we also run within-corridor for the honest contrast.

Run:  python -m src.models.deep.deep_counterfactual
"""

from __future__ import annotations

import json
import subprocess
import threading
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import FIGURES, METRICS, PROCESSED, ensure_dirs

LOG = get_logger("models.deep_counterfactual", log_file="logs/models.log")

# ---- knobs tuned to fill 8 GB and saturate the GPU (see DECISIONS_LOG) ----
ENSEMBLE = 48  # parallel ensemble members (UQ + GPU saturation); tune to fill VRAM
HIDDEN = 384  # embedding width
EPOCHS = 1500
USE_COMPILE = False  # Windows/Triton: enable only after eager confirmed (warm-up guarded)
LR = 2e-3
DONOR_DROPOUT = 0.10  # ensemble diversity
SEED = 20211025


class GPUMonitor(threading.Thread):
    """Samples nvidia-smi utilisation/memory in a background thread during training."""

    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.util, self.mem, self._stopflag = period, [], [], False

    def run(self):
        while not self._stopflag:
            try:
                out = (
                    subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    .stdout.strip()
                    .splitlines()[0]
                )
                u, m = (x.strip() for x in out.split(","))
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self._stopflag = True
        self.join(timeout=2)

    def summary(self):
        if not self.util:
            return {}
        u = np.array(self.util)
        return {
            "gpu_util_mean": round(float(u.mean()), 1),
            "gpu_util_p50": float(np.percentile(u, 50)),
            "gpu_util_p90": float(np.percentile(u, 90)),
            "gpu_util_max": float(u.max()),
            "gpu_mem_used_mb_max": round(max(self.mem), 0),
            "n_samples": len(u),
        }


class EnsembleLinear(nn.Module):
    """Batched linear across E ensemble members: x[E,B,in] -> [E,B,out]."""

    def __init__(self, E, d_in, d_out):
        super().__init__()
        self.w = nn.Parameter(torch.empty(E, d_in, d_out))
        self.b = nn.Parameter(torch.zeros(E, 1, d_out))
        nn.init.kaiming_uniform_(self.w, a=5**0.5)

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class DeepSynthAttention(nn.Module):
    """Ensemble of attention-over-donor counterfactual predictors."""

    def __init__(self, E, t_pre, hidden):
        super().__init__()
        self.enc = nn.Sequential(
            EnsembleLinear(E, t_pre, hidden),
            nn.GELU(),
            EnsembleLinear(E, hidden, hidden),
            nn.GELU(),
            EnsembleLinear(E, hidden, hidden),
        )
        self.scale = hidden**-0.5

    def encode(self, seq):  # seq [E,M,t_pre] -> [E,M,H]
        return self.enc(seq)

    def forward(self, donor_pre, donor_full, query_pre, self_mask=None):
        """donor_pre[E,N,tp], donor_full[E,N,T], query_pre[E,B,tp] -> pred[E,B,T], w[E,B,N]."""
        ed = self.encode(donor_pre)  # [E,N,H]
        eq = self.encode(query_pre)  # [E,B,H]
        scores = torch.bmm(eq, ed.transpose(1, 2)) * self.scale  # [E,B,N]
        if self_mask is not None:
            scores = scores.masked_fill(self_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        pred = torch.bmm(w, donor_full)  # [E,B,T]
        return pred, w


def main() -> None:
    ensure_dirs()
    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    pre = yarr < treat
    t_pre = int(pre.sum())
    T = len(years)

    if not torch.cuda.is_available():
        LOG.error("CUDA not available — aborting GPU run.")
        return
    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # speed > strict determinism (logged)
    torch.backends.cuda.matmul.allow_tf32 = True
    LOG.info(
        "GPU: %s | torch %s | cuda %s | VRAM %.1f GB",
        torch.cuda.get_device_name(0),
        torch.__version__,
        torch.version.cuda,
        torch.cuda.get_device_properties(0).total_memory / 1e9,
    )

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")

    def matrix(crs_list):
        sub = panel.filter(pl.col("crs").is_in(crs_list) & pl.col("year_start").is_in(years)).select(
            "crs", "year_start", "value"
        )
        wide = sub.pivot(values="value", index="crs", on="year_start")
        wide = wide.with_columns(pl.col("crs").cast(pl.Enum(crs_list)).alias("_o")).sort("_o")
        return np.log(np.clip(wide.select([str(y) for y in years]).to_numpy(), 1.0, None))

    donor_crs = units.filter((pl.col("role") == "donor_clean") & pl.col("balanced"))["crs"].to_list()
    Yd = matrix(donor_crs)  # [N,T] log
    Yt = matrix(served)  # [4,T] log treated
    N = Yd.shape[0]
    LOG.info("donors=%d  treated=%d  years=%d (pre=%d)", N, len(served), T, t_pre)

    # standardise per series (helps the net); keep mean/std to invert
    mu, sd = Yd.mean(1, keepdims=True), Yd.std(1, keepdims=True) + 1e-6
    Ydn = (Yd - mu) / sd
    dpre = torch.tensor(Ydn[:, pre], dtype=torch.float32, device=dev)  # [N,tp]
    dfull = torch.tensor(Ydn, dtype=torch.float32, device=dev)  # [N,T]
    E = ENSEMBLE
    dpre_E = dpre.unsqueeze(0).expand(E, N, t_pre).contiguous()
    dfull_E = dfull.unsqueeze(0).expand(E, N, T).contiguous()
    self_mask = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0).expand(E, N, N)
    target_full = dfull_E  # predict donors themselves (LOO via mask)

    model = DeepSynthAttention(E, t_pre, HIDDEN).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("model params: %.2fM  (ensemble=%d, hidden=%d)", n_params / 1e6, E, HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model_c = model
    if USE_COMPILE:
        try:
            model_c = torch.compile(model)
            with torch.autocast("cuda", dtype=torch.bfloat16):  # warm-up triggers compile now
                _ = model_c(dpre_E, dfull_E, dpre_E, self_mask)
            torch.cuda.synchronize()
            LOG.info("torch.compile enabled")
        except Exception as e:  # Windows/Triton often unavailable
            model_c = model
            LOG.warning("torch.compile unavailable (%s) — eager mode", str(e)[:80])

    mon = GPUMonitor()
    mon.start()
    t0 = time.time()
    for ep in range(EPOCHS):
        model_c.train()
        opt.zero_grad(set_to_none=True)
        # donor dropout for ensemble diversity (per-member mask on attention)
        drop = torch.rand(E, 1, N, device=dev) < DONOR_DROPOUT
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = model_c(dpre_E, dfull_E, dpre_E, self_mask | drop)
            loss = ((pred[:, :, pre] - target_full[:, :, pre]) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % 500 == 0 or ep == EPOCHS - 1:
            LOG.info("  epoch %4d  pre-MSE=%.5f", ep, loss.item())
    torch.cuda.synchronize()
    train_s = time.time() - t0
    mon.stop()
    gpu = mon.summary()
    LOG.info("training %.1fs (%.1f epochs/s) | GPU: %s", train_s, EPOCHS / train_s, json.dumps(gpu))

    # ---- predict treated counterfactuals ----
    model_c.eval()
    qmu = Yt.mean(1, keepdims=True)
    qsd = Yt.std(1, keepdims=True) + 1e-6
    qn = (Yt - qmu) / qsd
    qpre_E = (
        torch.tensor(qn[:, pre], dtype=torch.float32, device=dev)
        .unsqueeze(0)
        .expand(E, len(served), t_pre)
        .contiguous()
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_t, _ = model_c(dpre_E, dfull_E, qpre_E)  # [E,4,T] standardised
    pred_t = pred_t.float().cpu().numpy() * qsd[None] + qmu[None]  # de-standardise -> log level
    # ensemble mean + interval
    cf_mean = pred_t.mean(0)  # [4,T] log
    cf_lo, cf_hi = np.percentile(pred_t, [5, 95], axis=0)

    # split-conformal calibration on donor pre-period residuals (distribution-free)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        dpred, _ = model_c(dpre_E, dfull_E, dpre_E)
    dpred = dpred.float().mean(0).cpu().numpy() * sd + mu  # [N,T] log
    resid = np.abs((dpred - Yd)[:, pre]).ravel()
    q90 = np.quantile(resid, 0.9)

    results = {
        "gpu": gpu,
        "train_seconds": round(train_s, 1),
        "epochs": EPOCHS,
        "params_millions": round(n_params / 1e6, 2),
        "conformal_q90_log": float(q90),
        "effects": {},
    }
    post = ~pre
    for i, crs in enumerate(served):
        obs_post = Yt[i, post]
        cf_post = cf_mean[i, post]
        eff = float(np.exp((obs_post - cf_post).mean()) - 1)
        eff_lo = float(np.exp((obs_post - (cf_post + q90)).mean()) - 1)
        eff_hi = float(np.exp((obs_post - (cf_post - q90)).mean()) - 1)
        results["effects"][crs] = {
            "deep_effect_pct": round(100 * eff, 1),
            "conformal_lo_pct": round(100 * eff_lo, 1),
            "conformal_hi_pct": round(100 * eff_hi, 1),
        }
        LOG.info(
            "  %s deep counterfactual effect = %+.1f%%  [conformal90: %+.1f, %+.1f]",
            crs,
            100 * eff,
            100 * eff_lo,
            100 * eff_hi,
        )

    (METRICS / "m4_deep_counterfactual.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _plot(served, years, Yt, cf_mean, cf_lo, cf_hi, treat, results)
    LOG.info(
        "M4 deep counterfactual complete. GPU util mean=%.1f%% max=%.0f%%",
        gpu.get("gpu_util_mean", 0),
        gpu.get("gpu_util_max", 0),
    )


def _plot(served, years, Yt, cf_mean, cf_lo, cf_hi, treat, results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, (crs, ax) in enumerate(zip(served, axes.ravel())):
        ax.plot(years, np.exp(Yt[i]) / 1e6, "o-", color="#d62728", lw=2, label="observed")
        ax.plot(years, np.exp(cf_mean[i]) / 1e6, "s--", color="#1f77b4", lw=1.8, label="deep counterfactual")
        ax.fill_between(
            years, np.exp(cf_lo[i]) / 1e6, np.exp(cf_hi[i]) / 1e6, color="#1f77b4", alpha=0.18, label="ensemble 5-95%"
        )
        ax.axvline(treat, color="grey", ls=":", lw=1.3)
        e = results["effects"][crs]
        ax.set_title(
            f"{crs}: deep effect {e['deep_effect_pct']:+.1f}% "
            f"[{e['conformal_lo_pct']:+.0f},{e['conformal_hi_pct']:+.0f}]",
            fontsize=10,
        )
        ax.set_ylabel("entries+exits (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    g = results["gpu"]
    fig.suptitle(
        f"Deep counterfactual (attention-over-donors ensemble, E={ENSEMBLE}, H={HIDDEN}) — "
        f"GPU util mean {g.get('gpu_util_mean', '?')}% / max {g.get('gpu_util_max', '?')}%",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "m4_deep_counterfactual.png", dpi=140)
    plt.close(fig)
    LOG.info("figure -> %s", FIGURES / "m4_deep_counterfactual.png")


if __name__ == "__main__":
    main()
