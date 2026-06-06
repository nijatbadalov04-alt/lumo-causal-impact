"""
Second deep architecture: Counterfactual GRU with donor attention (M4, Tier 3 arch A).

THINK -> RESEARCH -> CODE
  WHAT: Same counterfactual task and self-supervised donor training as the attention
        model, but the donor/target PRE-trajectories are encoded by a (vectorised-
        ensemble) GRU instead of an MLP — i.e. the brief's "counterfactual LSTM/RNN with
        attention over donor stations" (architecture A). Benchmark vs the attention
        encoder (architecture C) => RQ3: do different deep architectures agree?
  WHY : §6 Tier 3 names three architectures; Vasquez wants >=2 benchmarked. Reuses the
        GPU infra (EnsembleLinear, GPUMonitor) so it also saturates the GPU (§7). The GRU
        time-loop adds compute => keeps the GPU busy on tiny data.
  HONEST: like the attention model this runs on the off-ECML pool (inherits the corridor
        caveat); its value here is the architecture comparison + GPU + UQ, not a new
        causal magnitude.

Run:  python -m src.models.deep.deep_counterfactual_gru
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from src.models.deep.deep_counterfactual import DeepSynthAttention, EnsembleLinear, GPUMonitor
from src.utils.config import load_config
from src.utils.logging_setup import get_logger
from src.utils.paths import METRICS, PROCESSED, ensure_dirs

LOG = get_logger("models.deep_gru", log_file="logs/models.log")

ENSEMBLE = 32  # smaller than the attention model: the recurrent time-loop is costlier/step
HIDDEN = 256
EPOCHS = 80  # recurrent impl is launch-overhead-bound; short run for the benchmark (GPU still 100%)
LR = 5e-3
SEED = 20211025


class EnsembleGRUEncoder(nn.Module):
    """Vectorised-ensemble GRU over a scalar sequence: [E,M,T] -> [E,M,H]."""

    def __init__(self, E: int, H: int):
        super().__init__()
        self.H = H
        self.Wz = EnsembleLinear(E, 1 + H, H)
        self.Wr = EnsembleLinear(E, 1 + H, H)
        self.Wn = EnsembleLinear(E, 1 + H, H)

    def forward(self, seq):  # seq [E, M, T]
        E_, M, T = seq.shape
        h = torch.zeros(E_, M, self.H, device=seq.device, dtype=seq.dtype)
        for t in range(T):
            x = seq[:, :, t : t + 1]
            xh = torch.cat([x, h], dim=-1)
            z = torch.sigmoid(self.Wz(xh))
            r = torch.sigmoid(self.Wr(xh))
            n = torch.tanh(self.Wn(torch.cat([x, r * h], dim=-1)))
            h = (1 - z) * n + z * h
        return h


class DeepSynthGRU(DeepSynthAttention):
    """Attention-over-donors counterfactual, but with a GRU pre-trajectory encoder."""

    def __init__(self, E, t_pre, hidden):
        super().__init__(E, t_pre, hidden)
        self.gru = EnsembleGRUEncoder(E, hidden)

    def encode(self, seq):
        return self.gru(seq)


def _matrix(panel, crs_list, years):
    sub = panel.filter(pl.col("crs").is_in(crs_list) & pl.col("year_start").is_in(years)).select(
        "crs", "year_start", "value"
    )
    wide = sub.pivot(values="value", index="crs", on="year_start")
    wide = wide.with_columns(pl.col("crs").cast(pl.Enum(crs_list)).alias("_o")).sort("_o")
    return np.log(np.clip(wide.select([str(y) for y in years]).to_numpy(), 1.0, None))


def main() -> None:
    ensure_dirs()
    if not torch.cuda.is_available():
        LOG.error("CUDA unavailable — aborting.")
        return
    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    cfg = load_config("base")
    treat = int(cfg["treatments"]["lumo"]["treat_year_start"])
    served = cfg["treatments"]["lumo"]["served_crs"]
    years = list(range(int(cfg["panel"]["lennon_era_min"]), int(cfg["panel"]["year_max"]) + 1))
    yarr = np.array(years)
    pre = yarr < treat
    t_pre, T = int(pre.sum()), len(years)

    panel = pl.read_parquet(PROCESSED / "panel.parquet")
    units = pl.read_parquet(PROCESSED / "units.parquet")
    donor_crs = units.filter((pl.col("role") == "donor_clean") & pl.col("balanced"))["crs"].to_list()
    Yd, Yt = _matrix(panel, donor_crs, years), _matrix(panel, served, years)
    N = Yd.shape[0]

    mu, sd = Yd.mean(1, keepdims=True), Yd.std(1, keepdims=True) + 1e-6
    Ydn = (Yd - mu) / sd
    E = ENSEMBLE
    dpre = torch.tensor(Ydn[:, pre], dtype=torch.float32, device=dev).unsqueeze(0).expand(E, N, t_pre).contiguous()
    dfull = torch.tensor(Ydn, dtype=torch.float32, device=dev).unsqueeze(0).expand(E, N, T).contiguous()
    self_mask = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0).expand(E, N, N)

    model = DeepSynthGRU(E, t_pre, HIDDEN).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    LOG.info("GRU model: %.2fM params (E=%d, H=%d) on %s", n_params / 1e6, E, HIDDEN, torch.cuda.get_device_name(0))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    mon = GPUMonitor()
    mon.start()
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = model(dpre, dfull, dpre, self_mask)
            loss = ((pred[:, :, pre] - dfull[:, :, pre]) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % 250 == 0 or ep == EPOCHS - 1:
            LOG.info("  epoch %4d pre-MSE=%.5f", ep, loss.item())
    torch.cuda.synchronize()
    train_s = time.time() - t0
    mon.stop()
    gpu = mon.summary()
    LOG.info(
        "GRU training %.1fs | GPU util mean=%.1f%% max=%.0f%%",
        train_s,
        gpu.get("gpu_util_mean", 0),
        gpu.get("gpu_util_max", 0),
    )

    # predict treated counterfactuals
    model.eval()
    qmu, qsd = Yt.mean(1, keepdims=True), Yt.std(1, keepdims=True) + 1e-6
    qpre = torch.tensor(((Yt - qmu) / qsd)[:, pre], dtype=torch.float32, device=dev)
    qpre = qpre.unsqueeze(0).expand(E, len(served), t_pre).contiguous()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pt, _ = model(dpre, dfull, qpre)
    pt = pt.float().cpu().numpy() * qsd[None] + qmu[None]
    cf = pt.mean(0)
    post = ~pre

    # compare to the attention model's saved effects
    attn = {}
    fp = METRICS / "m4_deep_counterfactual.json"
    if fp.exists():
        attn = json.loads(fp.read_text(encoding="utf-8")).get("effects", {})

    results = {"gpu": gpu, "train_seconds": round(train_s, 1), "architecture": "GRU+attention", "effects": {}}
    LOG.info("GRU vs attention counterfactual effects:")
    for i, crs in enumerate(served):
        eff = float(np.exp((Yt[i, post] - cf[i, post]).mean()) - 1)
        a = attn.get(crs, {}).get("deep_effect_pct")
        results["effects"][crs] = {"gru_effect_pct": round(100 * eff, 1), "attention_effect_pct": a}
        LOG.info("  %s GRU=%+.1f%%  attention=%s%%", crs, 100 * eff, a)
    (METRICS / "m4_deep_gru.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOG.info("GRU architecture complete. GPU mean=%.1f%%", gpu.get("gpu_util_mean", 0))


if __name__ == "__main__":
    main()
