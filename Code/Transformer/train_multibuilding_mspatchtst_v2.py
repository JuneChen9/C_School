# -*- coding: utf-8 -*-
"""
Multi-building Multi-scale PatchTST v2: GraphMix + Safe Baseline Fusion (Gated Residual) + Total-Load Consistency
for Industrial Park Electric Power Load Forecasting (AfterWash).

Key features
- Joint forecasting for multiple buildings (multi-task): predict all buildings together
- Spatial fusion: learnable adjacency (GraphMix), optional correlation init from train split
- Multi-scale Patch Embedding: multiple (patch_len, stride) scales in parallel
- Expert knowledge fusion (low-risk): seasonal naive baseline + residual learning
- Metrics: MAE/RMSE/MAPE/sMAPE/MASE/R2/NRMSE + total-load metrics + baseline comparison
- Visualization: prediction curves, error-by-hour/dow, corr heatmap, adjacency heatmap, saliency heatmap
- Training: AMP, grad accumulation, auto micro-batch probe, resume, (optional) torch.compile

Data layout (cleaned):
  data_root/<resolution>/<building>/train.csv|parquet, val..., test...
  resolution in {1_hour, 30_minutes, 5_minutes}
  building in {Commercial, Office, Public, Residential}

Outputs:
  out_root/MultiBuildingMSPatchTST/<run_tag>/<resolution>/<building_tag>/...
    best.pt
    checkpoint_last.pt
    metrics.json
    preds/*.npz
    plots/*.png
    explain/*.npz
    explain/plots/*.png
    corr/*.json + corr/heatmap.png

Notes / risks:
- If your cleaning involved heavy interpolation/seasonal filling (esp. Public), deep models may "learn the fill rule".
  Always compare against seasonal naive baseline in metrics.json.
"""

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", os.environ.get("PYTORCH_ALLOC_CONF", ""))

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as ckpt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def safe_json_load(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def safe_json_write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cudaerrormemoryallocation" in msg) or isinstance(e, torch.cuda.OutOfMemoryError)


def cuda_cleanup():
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def print_env_banner():
    print("PY:", sys.executable)
    print("TORCH:", torch.__version__)
    try:
        built = torch.backends.cuda.is_built()
    except Exception:
        built = False
    print("CUDA built:", built)
    print("CUDA available:", torch.cuda.is_available(), torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def fail_fast_cuda_if_needed(requested_device: str):
    dev = (requested_device or "auto").lower()
    wants_cuda = dev.startswith("cuda")
    if wants_cuda and (not torch.cuda.is_available()):
        print_env_banner()
        print("\n[ERROR] You requested --device cuda, but torch.cuda.is_available() == False in this environment.\n")
        raise SystemExit(2)


class NullScaler:
    def scale(self, loss):
        return loss
    def unscale_(self, optimizer):
        return None
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        return None


def make_grad_scaler(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.GradScaler(enabled=True)
    return NullScaler()


def autocast_ctx(device: torch.device, use_amp: bool):
    dev_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.autocast(device_type=dev_type, dtype=torch.float16, enabled=use_amp)


# -----------------------------
# Time features
# -----------------------------
def make_time_features(dt_index: pd.DatetimeIndex) -> np.ndarray:
    if not isinstance(dt_index, pd.DatetimeIndex):
        dt_index = pd.to_datetime(dt_index)

    hour = dt_index.hour.values.astype(np.float32)
    dow = dt_index.dayofweek.values.astype(np.float32)
    doy = dt_index.dayofyear.values.astype(np.float32)
    month = dt_index.month.values.astype(np.float32)
    is_weekend = (dow >= 5).astype(np.float32)

    hour_rad = 2 * np.pi * (hour / 24.0)
    dow_rad = 2 * np.pi * (dow / 7.0)
    doy_rad = 2 * np.pi * (doy / 366.0)
    mon_rad = 2 * np.pi * (month / 12.0)

    feats = np.stack([
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(dow_rad),  np.cos(dow_rad),
        np.sin(doy_rad),  np.cos(doy_rad),
        np.sin(mon_rad),  np.cos(mon_rad),
        is_weekend
    ], axis=1).astype(np.float32)
    return feats  # [N, 9]


# -----------------------------
# Window configs
# -----------------------------
@dataclass(frozen=True)
class WindowConfig:
    input_len: int
    pred_len: int


DEFAULT_WINDOWS = {
    "1_hour": WindowConfig(input_len=24 * 7, pred_len=24),
    "30_minutes": WindowConfig(input_len=48 * 7, pred_len=48),
    "5_minutes": WindowConfig(input_len=288 * 7, pred_len=288),
}

# steps per day (for seasonal baseline + MASE scaling)
STEPS_PER_DAY = {
    "1_hour": 24,
    "30_minutes": 48,
    "5_minutes": 288,
}

# Multi-scale patch defaults (two scales per resolution)
DEFAULT_PATCH_SCALES = {
    "1_hour": "12:6,24:12",
    "30_minutes": "24:12,48:24",
    "5_minutes": "32:16,64:32",
}


def parse_patch_scales(s: str) -> List[Tuple[int, int]]:
    scales = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        pl = int(a)
        st = int(b)
        if pl <= 0 or st <= 0:
            raise ValueError(f"Invalid patch scale: {part}")
        scales.append((pl, st))
    if not scales:
        raise ValueError("patch_scales parsed empty.")
    return scales


# -----------------------------
# Data loading (multi-building)
# -----------------------------
def _load_one(series_dir: Path, split: str) -> pd.DataFrame:
    p_parq = series_dir / f"{split}.parquet"
    p_csv = series_dir / f"{split}.csv"
    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"Missing {split}.parquet/csv under {series_dir}")

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"]).set_index("Time")
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "Time"

    if "Power_kW" not in df.columns:
        cand = None
        for c in df.columns:
            if "power" in c.lower():
                cand = c
                break
        if cand is None:
            raise ValueError(f"No Power column in {split} under {series_dir}: columns={list(df.columns)}")
        df = df.rename(columns={cand: "Power_kW"})

    df = df.sort_index()
    return df[["Power_kW"]]


def load_multibuilding_splits(
    data_root: Path,
    resolution: str,
    buildings: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def load_split(split: str) -> pd.DataFrame:
        dfs = []
        for b in buildings:
            d = _load_one(data_root / resolution / b, split=split).rename(columns={"Power_kW": b})
            dfs.append(d)
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.join(d, how="outer")
        merged = merged.sort_index()

        # Safety: drop any rows that still contain NaN (should be rare after AfterWash)
        n0 = len(merged)
        nan_rows = merged.isna().any(axis=1)
        if nan_rows.any():
            merged = merged.loc[~nan_rows].copy()
            print(f"[WARN] {resolution}/{split} dropped rows with NaN after join: {int(nan_rows.sum())} / {n0}")
        return merged

    return load_split("train"), load_split("val"), load_split("test")


# -----------------------------
# Scaling (per-building standardization)
# -----------------------------
@dataclass
class MultiScaler:
    mean: np.ndarray  # [N]
    scale: np.ndarray # [N]

    @classmethod
    def fit_from_df(cls, df: pd.DataFrame) -> "MultiScaler":
        arr = df.values.astype(np.float32)  # [T, N]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.maximum(std, 1e-6)
        return cls(mean=mean.astype(np.float32), scale=std.astype(np.float32))

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.scale

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.scale + self.mean


# -----------------------------
# Dataset (multi-building windows)
# -----------------------------
class SlidingWindowMultiBuilding(Dataset):
    """
    x: [input_len, C] where C = N(values for buildings) + F(time feats)
    y: [pred_len, N]
    """
    def __init__(
        self,
        y_scaled: np.ndarray,     # [T, N]
        time_feats: np.ndarray,   # [T, F]
        dt: np.ndarray,           # [T]
        input_len: int,
        pred_len: int,
    ):
        assert y_scaled.ndim == 2
        assert time_feats.ndim == 2
        assert len(y_scaled) == len(time_feats) == len(dt)

        self.y = y_scaled.astype(np.float32)
        self.tf = time_feats.astype(np.float32)
        self.dt = dt.astype("datetime64[ns]")

        self.in_len = int(input_len)
        self.out_len = int(pred_len)
        self.seq_len = self.in_len + self.out_len
        self.max_start = len(self.y) - self.seq_len
        if self.max_start <= 0:
            raise ValueError(f"Series too short: T={len(self.y)}, in={self.in_len}, out={self.out_len}")

        self.n_buildings = self.y.shape[1]
        self.n_tf = self.tf.shape[1]

    def __len__(self):
        return self.max_start + 1

    def __getitem__(self, idx: int):
        s = int(idx)
        e = s + self.seq_len
        past_y = self.y[s:s + self.in_len, :]          # [in, N]
        past_tf = self.tf[s:s + self.in_len, :]        # [in, F]
        x = np.concatenate([past_y, past_tf], axis=1)  # [in, N+F]
        y = self.y[s + self.in_len:e, :]               # [out, N]
        return torch.from_numpy(x), torch.from_numpy(y)

    def past_times(self, idx: int) -> np.ndarray:
        s = int(idx)
        return self.dt[s:s + self.in_len]

    def future_times(self, idx: int) -> np.ndarray:
        s = int(idx) + self.in_len
        return self.dt[s:s + self.out_len]


def make_loader(ds: Dataset, batch: int, shuffle: bool, num_workers: int, pin: bool, drop_last: bool) -> DataLoader:
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=pin, drop_last=drop_last)
    if num_workers and num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


# -----------------------------
# Baseline (expert knowledge): seasonal naive
# -----------------------------
def seasonal_naive_from_past(
    past: torch.Tensor,   # [B, in_len, N] scaled
    pred_len: int,
    mode: str,
    steps_per_day: int,
) -> torch.Tensor:
    """
    Returns baseline future in scaled space: [B, pred_len, N]
    mode:
      - none: zeros
      - last: repeat last value
      - daily: use last day segment
      - weekly: use same time last week (requires input_len >= 7*steps_per_day)
    """
    B, in_len, N = past.shape
    if mode == "none":
        return torch.zeros((B, pred_len, N), device=past.device, dtype=past.dtype)

    if mode == "last":
        last = past[:, -1:, :].repeat(1, pred_len, 1)
        return last

    period = int(steps_per_day)
    if period <= 0:
        return torch.zeros((B, pred_len, N), device=past.device, dtype=past.dtype)

    if mode == "daily":
        seg = past[:, -period:, :]  # [B, period, N]
        if pred_len <= period:
            return seg[:, :pred_len, :]
        reps = int(math.ceil(pred_len / period))
        return seg.repeat(1, reps, 1)[:, :pred_len, :]

    if mode == "weekly":
        need = 7 * period
        if in_len < need:
            # fall back to daily
            seg = past[:, -period:, :]
            reps = int(math.ceil(pred_len / period))
            return seg.repeat(1, reps, 1)[:, :pred_len, :]
        seg = past[:, -need:-need + period, :]  # last week's same day slice
        if pred_len <= period:
            return seg[:, :pred_len, :]
        reps = int(math.ceil(pred_len / period))
        return seg.repeat(1, reps, 1)[:, :pred_len, :]

    # default fallback
    return torch.zeros((B, pred_len, N), device=past.device, dtype=past.dtype)


# -----------------------------
# Model blocks
# -----------------------------
class SDPASelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = int(n_heads)
        self.hd = d_model // n_heads
        self.drop = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)  # [B,h,T,hd]
        k = k.view(B, T, self.n_heads, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.hd).transpose(1, 2)

        drop_p = self.drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=drop_p, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SDPASelfAttention(d_model, n_heads, dropout)
        self.dp1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.ln1(x)))
        x = x + self.dp2(self.ff(self.ln2(x)))
        return x


class GraphMix(nn.Module):
    """
    Learnable adjacency for N buildings. Produces mixed signals: X_mix = X @ A^T
    A is row-softmax(logits), so each node chooses where to borrow info.
    """
    def __init__(self, n_nodes: int, init: str = "identity", init_mat: Optional[np.ndarray] = None):
        super().__init__()
        self.n = int(n_nodes)
        logits = torch.zeros((self.n, self.n), dtype=torch.float32)

        if init == "identity":
            logits.fill_(-2.0)
            logits += torch.eye(self.n) * 4.0
        elif init == "random":
            logits.normal_(mean=0.0, std=0.2)
        elif init == "corr" and init_mat is not None:
            # init_mat: correlation matrix, we map to logits
            m = np.array(init_mat, dtype=np.float32)
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
            # keep sign but compress
            m = np.clip(m, -0.99, 0.99)
            # bias self-loop
            np.fill_diagonal(m, 1.0)
            logits = torch.from_numpy(2.0 * m)  # mild
        else:
            logits.fill_(-2.0)
            logits += torch.eye(self.n) * 4.0

        self.logits = nn.Parameter(logits)

    def adj(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)  # [N,N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N]
        A = self.adj()  # [N,N]
        # (B,T,N) @ (N,N)^T -> (B,T,N)
        return torch.matmul(x, A.transpose(0, 1))


class MultiScalePatchEmbed(nn.Module):
    """
    Parallel Conv1d patch embeddings at multiple scales.
    Input: [B, L, C]
    Output: tokens [B, T_total, D]
    """
    def __init__(self, in_channels: int, d_model: int, input_len: int, scales: List[Tuple[int, int]]):
        super().__init__()
        self.in_channels = int(in_channels)
        self.d_model = int(d_model)
        self.input_len = int(input_len)
        self.scales = [(int(pl), int(st)) for pl, st in scales]

        self.embeds = nn.ModuleList()
        self.pos = nn.ParameterList()
        self.scale_emb = nn.ParameterList()
        self.n_patches = []

        for (pl, st) in self.scales:
            if self.input_len < pl:
                raise ValueError(f"input_len({self.input_len}) < patch_len({pl})")
            n_p = 1 + (self.input_len - pl) // st
            if n_p <= 0:
                raise ValueError("n_patches <= 0, check patch_len/stride")
            self.n_patches.append(n_p)

            conv = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.d_model,
                kernel_size=pl,
                stride=st,
                bias=True,
            )
            self.embeds.append(conv)

            p = nn.Parameter(torch.zeros(1, n_p, self.d_model))
            nn.init.trunc_normal_(p, std=0.02)
            self.pos.append(p)

            se = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(se, std=0.02)
            self.scale_emb.append(se)

        self.total_tokens = int(sum(self.n_patches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape
        if L != self.input_len:
            raise RuntimeError(f"Expected input_len={self.input_len}, got {L}")
        x = x.transpose(1, 2)  # [B,C,L]
        toks = []
        for i, conv in enumerate(self.embeds):
            t = conv(x)                 # [B, D, n_p]
            t = t.transpose(1, 2)       # [B, n_p, D]
            t = t + self.pos[i] + self.scale_emb[i]
            toks.append(t)
        return torch.cat(toks, dim=1)   # [B, T_total, D]


class CrossAttentionDecoder(nn.Module):
    """
    Learnable queries for each future step; attends over encoder tokens.
    Output: [B, pred_len, D]
    """
    def __init__(self, d_model: int, pred_len: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.pred_len = int(pred_len)
        self.drop = float(dropout)
        self.q = nn.Parameter(torch.zeros(1, self.pred_len, self.d_model))
        nn.init.trunc_normal_(self.q, std=0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.ln = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        B, T, D = tokens.shape
        q = self.q.expand(B, -1, -1)           # [B,pred,D]
        q = self.q_proj(q)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)

        # SDPA expects [B, h, T, hd] usually; but it also supports [B, T, D] ? Safer: do single-head manually.
        # We'll do single-head attention in (B, pred, D) x (B, T, D)
        scale = 1.0 / math.sqrt(D)
        attn = torch.matmul(q, k.transpose(1, 2)) * scale  # [B,pred,T]
        attn = torch.softmax(attn, dim=-1)
        attn = self.dp(attn)
        out = torch.matmul(attn, v)                       # [B,pred,D]
        out = self.out(out)
        out = self.ln(out)
        return out



class MultiBuildingMSPatchTST(nn.Module):
    """
    Input x: [B, in_len, N+F] where first N channels are building values (scaled), rest are time features.

    Output yhat (scaled):
      - if quantiles enabled: [B, pred_len, N, Q]
      - else: [B, pred_len, N, 1]

    Fusion modes (safe baseline fusion):
      - baseline_only: output seasonal-naive baseline only (no learning)
      - residual:      y = base + residual (classic residual learning, can degrade baseline)
      - gated_residual:y = base + g * residual, where g in (0, gate_max]
                       (reduces risk of destroying a strong baseline)

    Notes
    - Base and residual are both in *scaled* space.
    - Gating is global parameters (per building or per building-horizon), intentionally simple and stable.
    """

    def __init__(
        self,
        input_len: int,
        pred_len: int,
        n_buildings: int,
        n_time_feats: int,
        patch_scales: List[Tuple[int, int]],
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        baseline_mode: str,
        steps_per_day: int,
        adj_init: str = "identity",
        adj_init_mat: Optional[np.ndarray] = None,
        grad_checkpoint: bool = False,
        quantiles: Optional[List[float]] = None,
        fusion: str = "gated_residual",
        gate_mode: str = "building",
        gate_init: float = -2.0,
        gate_max: float = 0.7,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.N = int(n_buildings)
        self.Ft = int(n_time_feats)
        self.baseline_mode = str(baseline_mode)
        self.steps_per_day = int(steps_per_day)
        self.grad_checkpoint = bool(grad_checkpoint)

        self.fusion = str(fusion)
        if self.fusion not in {"baseline_only", "residual", "gated_residual"}:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")

        self.gate_mode = str(gate_mode)
        if self.gate_mode not in {"building", "building_horizon"}:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        self.gate_max = float(gate_max)
        if not (0.0 < self.gate_max <= 1.0):
            raise ValueError("gate_max must be in (0, 1]")

        qs = quantiles or []
        self.quantiles = [float(q) for q in qs if q is not None]
        self.Q = max(1, len(self.quantiles))  # if empty -> point forecast (Q=1)

        # GraphMix over buildings
        self.graph = GraphMix(self.N, init=adj_init, init_mat=adj_init_mat)

        # channels into patch embed: values(N) + mixed(N) + time_feats(Ft)
        in_channels = self.N + self.N + self.Ft
        self.patch = MultiScalePatchEmbed(
            in_channels=in_channels,
            d_model=int(d_model),
            input_len=self.input_len,
            scales=patch_scales,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(int(d_model), int(n_heads), int(d_ff), float(dropout))
            for _ in range(int(n_layers))
        ])
        self.enc_ln = nn.LayerNorm(int(d_model))

        self.dec = CrossAttentionDecoder(int(d_model), self.pred_len, float(dropout))
        self.head = nn.Linear(int(d_model), self.N * self.Q, bias=True)

        # Gate (optional)
        if self.fusion == "gated_residual":
            if self.gate_mode == "building":
                self.gate_logits = nn.Parameter(torch.full((self.N,), float(gate_init), dtype=torch.float32))
            else:
                self.gate_logits = nn.Parameter(torch.full((self.pred_len, self.N), float(gate_init), dtype=torch.float32))
        else:
            self.register_parameter("gate_logits", None)

    def _gate_tensor(self) -> Optional[torch.Tensor]:
        """Returns g shaped [1, pred, N, 1] (broadcastable) or None."""
        if self.gate_logits is None:
            return None
        g = torch.sigmoid(self.gate_logits) * self.gate_max
        if self.gate_mode == "building":
            return g.view(1, 1, self.N, 1)
        return g.view(1, self.pred_len, self.N, 1)

    def forward_with_aux(self, x: torch.Tensor):
        return self._forward(x, return_aux=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._forward(x, return_aux=False)
        return out

    def _forward(self, x: torch.Tensor, return_aux: bool = False):
        # x: [B, in_len, N+Ft]
        B, L, C = x.shape
        if L != self.input_len:
            raise RuntimeError(f"Expected input_len={self.input_len}, got {L}")
        if C != (self.N + self.Ft):
            raise RuntimeError(f"Expected channels={self.N + self.Ft}, got {C}")

        past_val = x[:, :, :self.N]     # [B,in,N]
        past_tf = x[:, :, self.N:]      # [B,in,Ft]

        mixed = self.graph(past_val)    # [B,in,N]
        enc_in = torch.cat([past_val, mixed, past_tf], dim=-1)  # [B,in,2N+Ft]

        tokens = self.patch(enc_in)     # [B,T,D]
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                tokens = ckpt(lambda t: blk(t), tokens)
            else:
                tokens = blk(tokens)
        tokens = self.enc_ln(tokens)

        dec = self.dec(tokens)          # [B,pred,D]
        residual = self.head(dec)       # [B,pred,N*Q]
        residual = residual.view(B, self.pred_len, self.N, self.Q)  # residual logits in scaled space

        base = seasonal_naive_from_past(
            past=past_val,
            pred_len=self.pred_len,
            mode=self.baseline_mode,
            steps_per_day=self.steps_per_day,
        )  # [B,pred,N] scaled
        base = base.unsqueeze(-1)  # [B,pred,N,1]
        base_q = base.expand(B, self.pred_len, self.N, self.Q)

        g = self._gate_tensor()  # [1, pred|1, N, 1] or None

        if self.fusion == "baseline_only":
            out = base_q
        elif self.fusion == "residual":
            out = base_q + residual
        elif self.fusion == "gated_residual":
            out = base_q + (g * residual)
        else:
            raise RuntimeError("unreachable")

        if not return_aux:
            return out, None

        aux = {
            "base": base_q,
            "residual": residual,
            "gate": (g if g is not None else torch.zeros((1, 1, self.N, 1), device=x.device, dtype=x.dtype)),
        }
        return out, aux

    @torch.no_grad()
    def get_adj(self) -> np.ndarray:
        return self.graph.adj().detach().cpu().numpy()


# -----------------------------
# Metrics
# -----------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    rm = rmse(y_true, y_pred)
    rng = float(np.max(y_true) - np.min(y_true))
    return float(rm / max(rng, eps))


def pinball_loss(y: torch.Tensor, yhat: torch.Tensor, quantiles: List[float]) -> torch.Tensor:
    """
    y: [B,pred,N]
    yhat: [B,pred,N,Q]
    """
    assert yhat.shape[-1] == len(quantiles)
    qs = torch.tensor(quantiles, device=y.device, dtype=y.dtype).view(1, 1, 1, -1)  # [1,1,1,Q]
    e = y.unsqueeze(-1) - yhat
    loss = torch.maximum(qs * e, (qs - 1.0) * e)
    return loss.mean()


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale_denom: float, eps: float = 1e-6) -> float:
    return float(mae(y_true, y_pred) / max(scale_denom, eps))


def compute_mase_denom_from_train(train_arr: np.ndarray, m: int) -> np.ndarray:
    """
    train_arr: [T,N] original scale
    m: seasonal period (steps per day)
    return denom per building: [N]
    """
    T, N = train_arr.shape
    if T <= m:
        dif = np.abs(np.diff(train_arr, axis=0))
        return dif.mean(axis=0) + 1e-6
    dif = np.abs(train_arr[m:, :] - train_arr[:-m, :])
    return dif.mean(axis=0) + 1e-6


# -----------------------------
# Evaluation / export
# -----------------------------
@torch.no_grad()
def predict_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    quantiles: Optional[List[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true_scaled: [K, pred, N]
      y_pred_scaled: [K, pred, N] (median if quantiles else point)
    """
    model.eval()
    ys, ps = [], []
    has_q = bool(quantiles)

    q_list = quantiles or []
    median_idx = 0
    if has_q:
        # choose closest to 0.5
        median_idx = int(np.argmin([abs(q - 0.5) for q in q_list]))

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_ctx(device, use_amp):
            out = model(xb)  # [B,pred,N,Q] or [B,pred,N,1]
        if has_q:
            pred = out[:, :, :, median_idx]
        else:
            pred = out[:, :, :, 0]
        ys.append(yb.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0) if ys else np.zeros((0, 1, 1), dtype=np.float32)
    y_pred = np.concatenate(ps, axis=0) if ps else np.zeros((0, 1, 1), dtype=np.float32)
    return y_true, y_pred


def compute_metrics_bundle(
    y_true: np.ndarray,  # [K,pred,N] original scale
    y_pred: np.ndarray,  # [K,pred,N] original scale
    mase_den: np.ndarray,  # [N]
    building_names: List[str],
) -> Dict:
    K, P, N = y_true.shape
    out = {"per_building": {}, "avg": {}, "total": {}}

    # per building
    for i, b in enumerate(building_names):
        yt = y_true[:, :, i].reshape(-1)
        yp = y_pred[:, :, i].reshape(-1)
        out["per_building"][b] = {
            "MAE": mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAPE": mape(yt, yp),
            "sMAPE": smape(yt, yp),
            "MASE": mase(yt, yp, float(mase_den[i])),
            "R2": r2_score(yt, yp),
            "NRMSE": nrmse(yt, yp),
        }

    # average over buildings (simple mean of per-building)
    keys = list(out["per_building"][building_names[0]].keys())
    for k in keys:
        out["avg"][k] = float(np.mean([out["per_building"][b][k] for b in building_names]))

    # total load (sum across buildings)
    yt_tot = y_true.sum(axis=2).reshape(-1)
    yp_tot = y_pred.sum(axis=2).reshape(-1)
    out["total"] = {
        "MAE": mae(yt_tot, yp_tot),
        "RMSE": rmse(yt_tot, yp_tot),
        "MAPE": mape(yt_tot, yp_tot),
        "sMAPE": smape(yt_tot, yp_tot),
        "R2": r2_score(yt_tot, yp_tot),
        "NRMSE": nrmse(yt_tot, yp_tot),
    }
    return out


def clip_nonneg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0)


def choose_indices(n: int, k: int, strategy: str, seed: int) -> np.ndarray:
    k = max(1, min(int(k), int(n)))
    if strategy == "first":
        return np.arange(k, dtype=int)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=k, replace=False))
    # uniform
    if k == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n - 1, num=k, dtype=int)


@torch.no_grad()
def export_predictions_and_plots(
    model: nn.Module,
    ds: SlidingWindowMultiBuilding,
    scaler: MultiScaler,
    device: torch.device,
    use_amp: bool,
    out_dir: Path,
    split_name: str,
    building_names: List[str],
    export_k: int,
    plot_k: int,
    export_strategy: str,
    seed: int,
    infer_batch: int,
    quantiles: Optional[List[float]],
    clip_to_nonneg: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds)
    idxs = choose_indices(n, export_k, export_strategy, seed)

    in_len = ds.in_len
    out_len = ds.out_len
    N = ds.n_buildings

    past_time = np.empty((len(idxs), in_len), dtype="datetime64[ns]")
    fut_time = np.empty((len(idxs), out_len), dtype="datetime64[ns]")
    past_true = np.empty((len(idxs), in_len, N), dtype=np.float32)
    fut_true = np.empty((len(idxs), out_len, N), dtype=np.float32)
    fut_pred = np.empty((len(idxs), out_len, N), dtype=np.float32)

    has_q = bool(quantiles)
    q_list = quantiles or []
    median_idx = 0
    if has_q:
        median_idx = int(np.argmin([abs(q - 0.5) for q in q_list]))

    model.eval()
    ptr = 0
    while ptr < len(idxs):
        batch_ids = idxs[ptr:ptr + max(1, infer_batch)]
        xs = []
        ys = []
        for i in batch_ids:
            x, y = ds[int(i)]
            xs.append(x)
            ys.append(y)
        xb = torch.stack(xs, dim=0).to(device, non_blocking=True)

        with autocast_ctx(device, use_amp):
            out = model(xb)  # [B,pred,N,Q]
        if has_q:
            pb = out[:, :, :, median_idx]
        else:
            pb = out[:, :, :, 0]
        pb = pb.detach().cpu().numpy().astype(np.float32)  # scaled

        for j, i in enumerate(batch_ids):
            i = int(i)
            row = ptr + j
            pt = ds.past_times(i)
            ft = ds.future_times(i)

            x_np = xs[j].numpy().astype(np.float32)  # [in, N+F]
            past_scaled = x_np[:, :N]
            fut_scaled = ys[j].numpy().astype(np.float32)

            past_time[row, :] = pt
            fut_time[row, :] = ft

            past_org = scaler.inverse(past_scaled)
            fut_org = scaler.inverse(fut_scaled)
            pred_org = scaler.inverse(pb[j])

            if clip_to_nonneg:
                past_org = clip_nonneg(past_org)
                fut_org = clip_nonneg(fut_org)
                pred_org = clip_nonneg(pred_org)

            past_true[row, :, :] = past_org
            fut_true[row, :, :] = fut_org
            fut_pred[row, :, :] = pred_org

        ptr += len(batch_ids)

    npz_path = out_dir / f"{split_name}_samples.npz"
    np.savez_compressed(
        npz_path,
        past_time=past_time,
        future_time=fut_time,
        past_true=past_true,
        future_true=fut_true,
        future_pred=fut_pred,
        indices=idxs.astype(np.int32),
        buildings=np.array(building_names, dtype=object),
    )

    # plots (per building + total)
    kplot = max(0, min(int(plot_k), len(idxs)))
    for s in range(kplot):
        t_p = past_time[s]
        t_f = fut_time[s]
        y_p = past_true[s]   # [in,N]
        y_t = fut_true[s]    # [out,N]
        y_h = fut_pred[s]    # [out,N]

        for bi, bname in enumerate(building_names):
            plt.figure(figsize=(12, 4))
            plt.plot(t_p.astype("datetime64[ns]"), y_p[:, bi], label="past_true")
            plt.plot(t_f.astype("datetime64[ns]"), y_t[:, bi], label="future_true")
            plt.plot(t_f.astype("datetime64[ns]"), y_h[:, bi], label="future_pred")
            plt.title(f"{split_name} sample {s:03d} | {bname}")
            plt.xticks(rotation=30, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"{split_name}_sample_{s:03d}_{bname}.png", dpi=180)
            plt.close()

        # total
        plt.figure(figsize=(12, 4))
        plt.plot(t_p.astype("datetime64[ns]"), y_p.sum(axis=1), label="past_true_total")
        plt.plot(t_f.astype("datetime64[ns]"), y_t.sum(axis=1), label="future_true_total")
        plt.plot(t_f.astype("datetime64[ns]"), y_h.sum(axis=1), label="future_pred_total")
        plt.title(f"{split_name} sample {s:03d} | TOTAL")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{split_name}_sample_{s:03d}_TOTAL.png", dpi=180)
        plt.close()

    return {
        "npz": str(npz_path),
        "plots_dir": str(plots_dir),
        "export_k": int(export_k),
        "plot_k": int(plot_k),
        "strategy": str(export_strategy),
    }


# -----------------------------
# Explainability (adjacency + saliency)
# -----------------------------
def plot_heatmap(mat: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, out_png: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=30, ha="right")
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()


@torch.no_grad()
def compute_error_profile_by_time(
    df: pd.DataFrame,
    y_true: np.ndarray,  # [K,pred,N] original
    y_pred: np.ndarray,  # [K,pred,N] original
    steps_per_day: int,
    building_names: List[str],
) -> Dict:
    """
    Provides aggregated MAE by hour-of-day and day-of-week, using timestamps from df index.
    This uses the fact that windows are contiguous; we approximate by taking the corresponding future timestamps from df.
    For robust profiling, it's enough.
    """
    # Build all future timestamps for windows (first element of each horizon)
    # We'll approximate using rolling; avoid huge memory.
    # Result: profile on first-step forecast error (t+1) and full-horizon mean error.
    out = {"by_hour": {}, "by_dow": {}}

    # We can only profile using actual timestamps; easiest: use df index and reconstruct windows positions.
    # We'll profile first-step error at horizon=0 for each window.
    K, P, N = y_true.shape
    idx0 = np.arange(K)
    # For window k, first future time is at position k+input_len
    # But we don't have input_len here; caller must ensure df is same split and K computed from dataset len.
    # We'll pass this function only for plotting after export, using dataset times instead (below).
    return out


def explain_saliency(
    model: MultiBuildingMSPatchTST,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    building_names: List[str],
    out_dir: Path,
    steps: int = 4,
):
    """
    Gradient saliency of sum of median predictions wrt input building values.
    Saves:
      explain/saliency.npz  (value_saliency: [in_len,N], timefeat_saliency: [in_len,F])
      explain/plots/*.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    in_len = model.input_len
    N = model.N
    Ftf = model.Ft

    val_sal = torch.zeros((in_len, N), device=device)
    tf_sal = torch.zeros((in_len, Ftf), device=device)

    n_batches = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        xb.requires_grad_(True)

        with autocast_ctx(device, use_amp):
            out = model(xb)  # [B,pred,N,Q]
            # choose median if quantiles else Q=1
            if model.Q > 1 and model.quantiles:
                mid = int(np.argmin([abs(q - 0.5) for q in model.quantiles]))
                pred = out[:, :, :, mid]
            else:
                pred = out[:, :, :, 0]
            # sum prediction energy
            loss = pred.sum()

        # backprop
        model.zero_grad(set_to_none=True)
        if xb.grad is not None:
            xb.grad.zero_()
        loss.backward()

        g = xb.grad.detach()  # [B,in,N+F]
        g_val = g[:, :, :N].abs().mean(dim=0)  # [in,N]
        g_tf = g[:, :, N:].abs().mean(dim=0)   # [in,F]
        val_sal += g_val
        tf_sal += g_tf
        n_batches += 1

        if n_batches >= int(steps):
            break

    if n_batches > 0:
        val_sal = val_sal / n_batches
        tf_sal = tf_sal / n_batches

    val_np = val_sal.detach().cpu().numpy()
    tf_np = tf_sal.detach().cpu().numpy()

    np.savez_compressed(out_dir / "saliency.npz", value_saliency=val_np, timefeat_saliency=tf_np)

    # heatmaps
    plot_heatmap(val_np.T, xlabels=[str(i) for i in range(in_len)], ylabels=building_names,
                 title="Saliency | buildings x time", out_png=plots_dir / "saliency_buildings.png")
    plot_heatmap(tf_np.T, xlabels=[str(i) for i in range(in_len)], ylabels=[f"tf{i}" for i in range(Ftf)],
                 title="Saliency | timefeats x time", out_png=plots_dir / "saliency_timefeats.png")


def explain_adjacency(
    model: MultiBuildingMSPatchTST,
    building_names: List[str],
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    A = model.get_adj()  # [N,N]
    np.savez_compressed(out_dir / "adjacency.npz", adjacency=A, buildings=np.array(building_names, dtype=object))
    plot_heatmap(A, xlabels=building_names, ylabels=building_names,
                 title="GraphMix adjacency (row-softmax)", out_png=out_dir / "adjacency.png")


# -----------------------------
# Correlation analysis (support spatiotemporal rationale)
# -----------------------------
def corr_analysis(train_df: pd.DataFrame, building_names: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pear = train_df[building_names].corr(method="pearson").values.astype(np.float32)
    spear = train_df[building_names].corr(method="spearman").values.astype(np.float32)
    safe_json_write(out_dir / "corr.json", {
        "buildings": building_names,
        "pearson": pear.tolist(),
        "spearman": spear.tolist(),
        "timestamp": now_str(),
    })
    plot_heatmap(pear, building_names, building_names, "Pearson correlation (train)", out_dir / "pearson.png")
    plot_heatmap(spear, building_names, building_names, "Spearman correlation (train)", out_dir / "spearman.png")
    return pear, spear


# -----------------------------
# Auto micro-batch probing
# -----------------------------
def try_micro_batch(ds: Dataset, model: nn.Module, device: torch.device, micro_bs: int, use_amp: bool,
                    quantiles: Optional[List[float]], huber_beta: float, total_loss_weight: float,
                    mean_t: torch.Tensor, scale_t: torch.Tensor) -> bool:
    loader = make_loader(ds, micro_bs, shuffle=True, num_workers=0, pin=(device.type == "cuda"), drop_last=True)
    xb, yb = next(iter(loader))
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    model.train()
    model.zero_grad(set_to_none=True)

    scaler = make_grad_scaler(use_amp)

    try:
        with autocast_ctx(device, use_amp):
            out = model(xb)  # [B,pred,N,Q]
            if quantiles:
                loss_main = pinball_loss(yb, out, quantiles)
                # median for total-load loss
                median_idx = int(min(range(len(quantiles)), key=lambda i: abs(quantiles[i]-0.5)))
                pred_point = out[:, :, :, median_idx]
            else:
                pred_point = out[:, :, :, 0]
                loss_main = F.smooth_l1_loss(pred_point, yb, beta=float(huber_beta))

            loss = loss_main
            if float(total_loss_weight) > 0:
                y_org = yb.float() * scale_t + mean_t
                p_org = pred_point.float() * scale_t + mean_t
                yt = y_org.sum(dim=2)
                yp = p_org.sum(dim=2)
                loss_total = F.smooth_l1_loss(yp, yt, beta=float(huber_beta))
                loss = loss + float(total_loss_weight) * loss_total


        scaler.scale(loss).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return True

    except BaseException as e:
        if is_oom(e):
            cuda_cleanup()
            return False
        raise


def find_micro_batch(ds: Dataset, model: nn.Module, device: torch.device,
                     min_bs: int, max_bs: int, use_amp: bool,
                     quantiles: Optional[List[float]], huber_beta: float, total_loss_weight: float,
                     mean_t: torch.Tensor, scale_t: torch.Tensor) -> int:
    min_bs = max(1, int(min_bs))
    max_bs = max(min_bs, int(max_bs))

    best = None
    bs = min_bs
    while bs <= max_bs:
        ok = try_micro_batch(ds, model, device, bs, use_amp, quantiles, huber_beta, total_loss_weight, mean_t, scale_t)
        if ok:
            best = bs
            bs = bs * 2 if bs < 32 else bs + 16
        else:
            break
    return best if best is not None else min_bs


# -----------------------------
# Training
# -----------------------------

@dataclass
class TrainConfig:
    epochs: int = 30
    micro_batch: int = 32
    effective_batch: int = 64

    # Optim (new defaults: more stable for strong-baseline series)
    lr: float = 3e-4
    weight_decay: float = 1e-3

    # Model regularization (new defaults)
    dropout: float = 0.2

    # Base model size (auto-tuned per resolution by model_cfg_for_resolution)
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 384

    # Loss
    huber_beta: float = 1.0
    total_loss_weight: float = 0.1
    residual_reg: float = 1e-3

    # Safe fusion
    fusion: str = "gated_residual"          # baseline_only | residual | gated_residual
    gate_mode: str = "building"            # building | building_horizon
    gate_init: float = -2.0                 # sigmoid(init) ~ 0.12
    gate_max: float = 0.7                   # g in (0, gate_max]
    gate_reg: float = 1e-2                  # penalize large gates to prevent baseline destruction

    # Train misc
    grad_clip: float = 1.0
    patience: int = 8
    num_workers: int = 2
    save_every: int = 1
    grad_checkpoint: bool = False

    # Heuristic switches
    auto_res_tune: bool = True
    auto_quantiles: bool = True


def _pick_n_heads(d_model: int, preferred: int) -> int:
    preferred = int(preferred)
    d_model = int(d_model)
    cands = [preferred, 8, 6, 4, 3, 2, 1]
    for h in cands:
        if h > 0 and d_model % h == 0:
            return int(h)
    return 1


def model_cfg_for_resolution(cfg: TrainConfig, resolution: str) -> Dict[str, int]:
    """Return (d_model, n_heads, n_layers, d_ff) with stable heuristics per resolution."""
    dm = int(cfg.d_model)
    nl = int(cfg.n_layers)
    df = int(cfg.d_ff)

    if getattr(cfg, "auto_res_tune", True):
        if resolution in {"1_hour", "30_minutes"}:
            # Strong seasonal structure + strong baseline -> smaller model, more stable
            dm = min(dm, 192)
            nl = min(nl, 4)
            df = min(df, 384)
        elif resolution == "5_minutes":
            # Noisier / higher frequency -> keep enough capacity
            dm = max(dm, 256)
            nl = max(nl, 6)
            df = max(df, 512)

    nh = _pick_n_heads(dm, int(cfg.n_heads))
    return {"d_model": dm, "n_heads": nh, "n_layers": nl, "d_ff": df}


def save_checkpoint(path: Path, payload: Dict):
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)


def compute_run_tag(args: argparse.Namespace, micro_batch: int, effective_batch: int) -> str:
    if getattr(args, "run_tag", ""):
        return str(args.run_tag)

    cfg = {
        "model": "MultiBuildingMSPatchTST",
        "resolutions": args.resolutions,
        "buildings": args.buildings,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "d_ff": args.d_ff,
        "grad_clip": args.grad_clip,
        "patience": args.patience,
        "seed": args.seed,
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "patch_scales": args.patch_scales,
        "micro_batch": micro_batch,
        "effective_batch": effective_batch,
        "auto_batch": args.auto_batch,
        "min_batch": args.min_batch,
        "max_batch": args.max_batch,
        "grad_checkpoint": args.grad_checkpoint,
        "compile": args.compile,
        "baseline_mode": args.baseline_mode,
        "adj_init": args.adj_init,
        "quantiles": args.quantiles,
        "fusion": args.fusion,
        "gate_mode": args.gate_mode,
        "gate_init": args.gate_init,
        "gate_max": args.gate_max,
        "gate_reg": args.gate_reg,
        "residual_reg": args.residual_reg,
        "total_loss_weight": args.total_loss_weight,
        "huber_beta": args.huber_beta,
        "auto_res_tune": args.auto_res_tune,
        "auto_quantiles": args.auto_quantiles,
        "export_preds": args.export_preds,
        "export_k": args.export_k,
        "export_strategy": args.export_strategy,
        "explain": args.explain,
        "clip_nonneg": args.clip_nonneg,
    }
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:8]
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{ts}_{h}"


def train_one_resolution(
    data_root: Path,
    out_dir: Path,
    resolution: str,
    buildings: List[str],
    wc: WindowConfig,
    patch_scales: List[Tuple[int, int]],
    cfg: TrainConfig,
    device: torch.device,
    auto_batch: bool,
    min_batch: int,
    max_batch: int,
    resume_training: bool,
    force_retrain: bool,
    compile_model: bool,
    baseline_mode: str,
    adj_init: str,
    quantiles: Optional[List[float]],
    export_preds: bool,
    export_splits: List[str],
    export_k: int,
    plot_k: int,
    export_strategy: str,
    explain: bool,
    clip_to_nonneg: bool,
    seed: int,
) -> Dict:
    building_tag = "Joint_" + "-".join(buildings)
    run_dir = out_dir / resolution / building_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "checkpoint_last.pt"
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists() and (not force_retrain):
        print(f"[SKIP-DONE] {resolution}/{building_tag} (metrics.json exists)")
        return safe_json_load(metrics_path, default={})

    use_amp = (device.type == "cuda")

    # Load split data
    train_df, val_df, test_df = load_multibuilding_splits(data_root, resolution, buildings)
    tf_train = make_time_features(train_df.index)
    tf_val = make_time_features(val_df.index)
    tf_test = make_time_features(test_df.index)

    dt_train = train_df.index.values.astype("datetime64[ns]")
    dt_val = val_df.index.values.astype("datetime64[ns]")
    dt_test = test_df.index.values.astype("datetime64[ns]")

    # Correlation analysis (train)
    corr_dir = run_dir / "corr"
    pear, spear = corr_analysis(train_df, buildings, corr_dir)

    # Scaler fit on train only (per building)
    scaler = MultiScaler.fit_from_df(train_df[buildings])
    # tensors for inverse transform inside training loss (total-load consistency)
    mean_t = torch.tensor(scaler.mean, device=device, dtype=torch.float32).view(1, 1, len(buildings))
    scale_t = torch.tensor(scaler.scale, device=device, dtype=torch.float32).view(1, 1, len(buildings))
    y_train = scaler.transform(train_df[buildings].values.astype(np.float32))
    y_val = scaler.transform(val_df[buildings].values.astype(np.float32))
    y_test = scaler.transform(test_df[buildings].values.astype(np.float32))

    ds_train = SlidingWindowMultiBuilding(y_train, tf_train, dt_train, wc.input_len, wc.pred_len)
    ds_val = SlidingWindowMultiBuilding(y_val, tf_val, dt_val, wc.input_len, wc.pred_len)
    ds_test = SlidingWindowMultiBuilding(y_test, tf_test, dt_test, wc.input_len, wc.pred_len)

    # MASE denom computed from train in original scale
    mase_den = compute_mase_denom_from_train(train_df[buildings].values.astype(np.float32), STEPS_PER_DAY[resolution])

    mc = model_cfg_for_resolution(cfg, resolution)

    # Adjacency init matrix: use Pearson on train (original)
    adj_init_mat = pear if adj_init == "corr" else None

    model = MultiBuildingMSPatchTST(
        input_len=wc.input_len,
        pred_len=wc.pred_len,
        n_buildings=len(buildings),
        n_time_feats=tf_train.shape[1],
        patch_scales=patch_scales,
        d_model=int(mc["d_model"]),
        n_heads=int(mc["n_heads"]),
        n_layers=int(mc["n_layers"]),
        d_ff=int(mc["d_ff"]),
        dropout=float(cfg.dropout),
        baseline_mode=str(baseline_mode),
        steps_per_day=int(STEPS_PER_DAY[resolution]),
        adj_init=str(adj_init),
        adj_init_mat=adj_init_mat,
        grad_checkpoint=bool(cfg.grad_checkpoint),
        quantiles=quantiles,
        fusion=str(cfg.fusion),
        gate_mode=str(cfg.gate_mode),
        gate_init=float(cfg.gate_init),
        gate_max=float(cfg.gate_max),
    ).to(device)

    if compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
            print(f"[COMPILE] enabled for {resolution}/{building_tag}")
        except Exception as e:
            print(f"[COMPILE] failed, continue without compile: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 10))
    grad_scaler = make_grad_scaler(use_amp)

    best_val = float("inf")
    patience_left = cfg.patience
    start_epoch = 1
    micro_bs = int(cfg.micro_batch)
    effective_bs = max(1, int(cfg.effective_batch))

    # Resume
    if resume_training and last_path.exists():
        ck = load_checkpoint(last_path, device)
        same = (
            ck.get("model") == "MultiBuildingMSPatchTST"
            and ck.get("resolution") == resolution
            and ck.get("buildings") == buildings
            and ck.get("window") == asdict(wc)
            and ck.get("patch_scales") == patch_scales
            and ck.get("model_cfg") == mc
            and ck.get("baseline_mode") == baseline_mode
            and ck.get("adj_init") == adj_init
            and ck.get("quantiles") == (quantiles or [])
            and ck.get("fusion") == str(cfg.fusion)
            and ck.get("gate_mode") == str(cfg.gate_mode)
            and float(ck.get("gate_max", -1.0)) == float(cfg.gate_max)
            and ck.get("baseline_mode") == baseline_mode
            and ck.get("adj_init") == adj_init
        )
        if same:
            model.load_state_dict(ck["model_state"])
            optim.load_state_dict(ck["optim_state"])
            scheduler.load_state_dict(ck["sched_state"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            best_val = float(ck.get("best_val", best_val))
            patience_left = int(ck.get("patience_left", patience_left))
            micro_bs = int(ck.get("micro_batch", micro_bs))
            print(f"[RESUME] {resolution}/{building_tag} epoch={start_epoch} best_val={best_val:.6f} micro_bs={micro_bs}")
        else:
            print(f"[RESUME-SKIP] {resolution}/{building_tag} checkpoint incompatible -> retrain from scratch")

    # Auto micro-batch
    if auto_batch and device.type == "cuda" and start_epoch <= 1:
        print(f"[AUTO-BATCH] {resolution}/{building_tag} search micro_bs in [{min_batch},{max_batch}] ...")
        cuda_cleanup()
        micro_bs = find_micro_batch(ds_train, model, device, min_batch, max_batch, use_amp, quantiles, cfg.huber_beta, cfg.total_loss_weight, mean_t, scale_t)
        print(f"[AUTO-BATCH] selected micro_bs={micro_bs}")
        cuda_cleanup()

    micro_bs = max(1, int(micro_bs))
    accum_steps = max(1, int(math.ceil(effective_bs / micro_bs)))
    eval_bs = min(max_batch, max(1, micro_bs * 2))
    pin = (device.type == "cuda")

    dl_train = make_loader(ds_train, micro_bs, True, cfg.num_workers, pin, True)
    dl_val = make_loader(ds_val, eval_bs, False, cfg.num_workers, pin, False)
    dl_test = make_loader(ds_test, eval_bs, False, cfg.num_workers, pin, False)

    print(f"[CFG] {resolution}/{building_tag} | in={wc.input_len} out={wc.pred_len} | patch_scales={patch_scales} total_tokens={model.patch.total_tokens} | micro_bs={micro_bs} effective={effective_bs} accum={accum_steps} eval_bs={eval_bs} | baseline={baseline_mode} fusion={cfg.fusion} gate_mode={cfg.gate_mode} gate_max={cfg.gate_max} adj_init={adj_init} quantiles={(quantiles or [])}")

    # Train loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        t0 = time.time()
        micro_steps = 0
        losses = []

        pbar = tqdm(dl_train, desc=f"[{resolution}/{building_tag}] epoch {epoch}/{cfg.epochs}", leave=False)
        for step, (xb, yb) in enumerate(pbar, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with autocast_ctx(device, use_amp):
                need_aux = (float(cfg.gate_reg) > 0) or (float(cfg.residual_reg) > 0) or (float(cfg.total_loss_weight) > 0)
                if need_aux:
                    out, aux = model.forward_with_aux(xb)
                else:
                    out = model(xb)
                    aux = None

                if quantiles:
                    loss_main = pinball_loss(yb, out, quantiles)
                    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
                    pred_point = out[:, :, :, median_idx]
                else:
                    pred_point = out[:, :, :, 0]
                    loss_main = F.smooth_l1_loss(pred_point, yb, beta=float(cfg.huber_beta))

                loss_total = 0.0
                if float(cfg.total_loss_weight) > 0:
                    y_org = yb.float() * scale_t + mean_t
                    p_org = pred_point.float() * scale_t + mean_t
                    yt = y_org.sum(dim=2)
                    yp = p_org.sum(dim=2)
                    loss_total = F.smooth_l1_loss(yp, yt, beta=float(cfg.huber_beta))

                loss_gate = 0.0
                if aux is not None and float(cfg.gate_reg) > 0 and cfg.fusion == 'gated_residual':
                    # aux['gate'] shape [1, pred|1, N, 1], broadcastable
                    loss_gate = aux['gate'].mean() * float(cfg.gate_reg)

                loss_res = 0.0
                if aux is not None and float(cfg.residual_reg) > 0:
                    loss_res = aux['residual'].abs().mean() * float(cfg.residual_reg)

                loss = (loss_main + float(cfg.total_loss_weight) * loss_total + loss_gate + loss_res) / accum_steps

            grad_scaler.scale(loss).backward()

            if step % accum_steps == 0:
                if cfg.grad_clip and cfg.grad_clip > 0:
                    grad_scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_scaler.step(optim)
                grad_scaler.update()
                optim.zero_grad(set_to_none=True)

            micro_steps += 1
            losses.append(float(loss.item()) * accum_steps)
            pbar.set_postfix(loss=float(np.mean(losses)))

        if (len(dl_train) % accum_steps) != 0:
            if cfg.grad_clip and cfg.grad_clip > 0:
                grad_scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            grad_scaler.step(optim)
            grad_scaler.update()
            optim.zero_grad(set_to_none=True)

        scheduler.step()

        dt = max(1e-6, time.time() - t0)
        micro_it_s = micro_steps / dt
        samples_s = micro_it_s * micro_bs
        updates_s = micro_it_s / accum_steps
        print(f"[THROUGHPUT] {resolution}/{building_tag} epoch={epoch} micro_it/s={micro_it_s:.2f} "
              f"samples/s={samples_s:.2f} updates/s={updates_s:.3f}")

        # val loss (scaled)
        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast_ctx(device, use_amp):
                    out = model(xb)
                    if quantiles:
                        vloss_main = pinball_loss(yb, out, quantiles)
                        median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))
                        pred_point = out[:, :, :, median_idx]
                    else:
                        pred_point = out[:, :, :, 0]
                        vloss_main = F.smooth_l1_loss(pred_point, yb, beta=float(cfg.huber_beta))

                    vloss = vloss_main
                    if float(cfg.total_loss_weight) > 0:
                        y_org = yb.float() * scale_t + mean_t
                        p_org = pred_point.float() * scale_t + mean_t
                        yt = y_org.sum(dim=2)
                        yp = p_org.sum(dim=2)
                        vloss_total = F.smooth_l1_loss(yp, yt, beta=float(cfg.huber_beta))
                        vloss = vloss + float(cfg.total_loss_weight) * vloss_total
                vls.append(float(vloss.item()))
        val_loss = float(np.mean(vls)) if vls else float("inf")

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            patience_left = cfg.patience
            torch.save({
                "model": "MultiBuildingMSPatchTST",
                "model_state": model.state_dict(),
                "resolution": resolution,
                "buildings": buildings,
                "window": asdict(wc),
                "patch_scales": patch_scales,
                "model_cfg": mc,
                "train_cfg": asdict(cfg),
                "micro_batch": micro_bs,
                "effective_batch": effective_bs,
                "accum_steps": accum_steps,
                "baseline_mode": baseline_mode,
                "adj_init": adj_init,
                "quantiles": (quantiles or []),
                "fusion": str(cfg.fusion),
                "gate_mode": str(cfg.gate_mode),
                "gate_init": float(cfg.gate_init),
                "gate_max": float(cfg.gate_max),
                "gate_reg": float(cfg.gate_reg),
                "residual_reg": float(cfg.residual_reg),
                "total_loss_weight": float(cfg.total_loss_weight),
                "huber_beta": float(cfg.huber_beta),
                "scaler_mean": scaler.mean.tolist(),
                "scaler_scale": scaler.scale.tolist(),
                "best_val_loss_scaled": best_val,
                "timestamp": now_str(),
            }, best_path)
        else:
            patience_left -= 1

        if (epoch % max(1, cfg.save_every) == 0) or patience_left <= 0 or epoch == cfg.epochs:
            save_checkpoint(last_path, {
                "model": "MultiBuildingMSPatchTST",
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "sched_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "patience_left": patience_left,
                "resolution": resolution,
                "buildings": buildings,
                "window": asdict(wc),
                "patch_scales": patch_scales,
                "model_cfg": mc,
                "train_cfg": asdict(cfg),
                "micro_batch": micro_bs,
                "effective_batch": effective_bs,
                "accum_steps": accum_steps,
                "baseline_mode": baseline_mode,
                "adj_init": adj_init,
                "quantiles": (quantiles or []),
                "fusion": str(cfg.fusion),
                "gate_mode": str(cfg.gate_mode),
                "gate_init": float(cfg.gate_init),
                "gate_max": float(cfg.gate_max),
                "gate_reg": float(cfg.gate_reg),
                "residual_reg": float(cfg.residual_reg),
                "total_loss_weight": float(cfg.total_loss_weight),
                "huber_beta": float(cfg.huber_beta),
                "timestamp": now_str(),
            })

        if patience_left <= 0:
            break

    # Load best for evaluation
    ck_best = torch.load(best_path, map_location=device)
    model.load_state_dict(ck_best["model_state"])
    model.eval()

    # Predict (scaled)
    yv_s, pv_s = predict_on_loader(model, dl_val, device, use_amp, quantiles)
    yt_s, pt_s = predict_on_loader(model, dl_test, device, use_amp, quantiles)

    # Inverse
    yv = scaler.inverse(yv_s)
    pv = scaler.inverse(pv_s)
    yt = scaler.inverse(yt_s)
    pt = scaler.inverse(pt_s)

    if clip_to_nonneg:
        yv = clip_nonneg(yv)
        pv = clip_nonneg(pv)
        yt = clip_nonneg(yt)
        pt = clip_nonneg(pt)

    # Metrics
    val_metrics = compute_metrics_bundle(yv, pv, mase_den, buildings)
    test_metrics = compute_metrics_bundle(yt, pt, mase_den, buildings)

    # Baseline metrics (seasonal naive only, in original scale)
    # We compute baseline using model's baseline_mode and the same windows from dataset (scaled -> inverse)
    def baseline_metrics_on_ds(ds: SlidingWindowMultiBuilding) -> Dict:
        # build baseline in scaled, inverse, compare
        # do it in chunks to avoid memory spike
        loader = make_loader(ds, batch=min(eval_bs, 256), shuffle=False, num_workers=0, pin=pin, drop_last=False)
        ys, bs = [], []
        for xb, yb in loader:
            past_val = xb[:, :, :len(buildings)].to(device)
            base = seasonal_naive_from_past(past_val, wc.pred_len, baseline_mode, STEPS_PER_DAY[resolution])
            ys.append(yb.numpy())
            bs.append(base.detach().cpu().numpy())
        y_s = np.concatenate(ys, axis=0)
        b_s = np.concatenate(bs, axis=0)
        y = scaler.inverse(y_s)
        b = scaler.inverse(b_s)
        if clip_to_nonneg:
            y = clip_nonneg(y)
            b = clip_nonneg(b)
        return compute_metrics_bundle(y, b, mase_den, buildings)

    baseline_val = baseline_metrics_on_ds(ds_val)
    baseline_test = baseline_metrics_on_ds(ds_test)

    # Exports
    exports = {}
    if export_preds:
        pred_dir = run_dir / "preds"
        infer_bs = min(128, max(1, eval_bs))
        splits = export_splits
        if "both" in splits:
            splits = ["val", "test"]
        if "val" in splits:
            exports["val"] = export_predictions_and_plots(
                model=model, ds=ds_val, scaler=scaler, device=device, use_amp=use_amp,
                out_dir=pred_dir, split_name="val", building_names=buildings,
                export_k=export_k, plot_k=plot_k, export_strategy=export_strategy,
                seed=seed, infer_batch=infer_bs, quantiles=quantiles, clip_to_nonneg=clip_to_nonneg,
            )
        if "test" in splits:
            exports["test"] = export_predictions_and_plots(
                model=model, ds=ds_test, scaler=scaler, device=device, use_amp=use_amp,
                out_dir=pred_dir, split_name="test", building_names=buildings,
                export_k=export_k, plot_k=plot_k, export_strategy=export_strategy,
                seed=seed, infer_batch=infer_bs, quantiles=quantiles, clip_to_nonneg=clip_to_nonneg,
            )

    # Explainability
    explain_out = {}
    if explain:
        exp_dir = run_dir / "explain"
        explain_adjacency(model, buildings, exp_dir)
        # saliency on val loader (small steps)
        dl_exp = make_loader(ds_val, batch=min(16, micro_bs), shuffle=True, num_workers=0, pin=pin, drop_last=True)
        explain_saliency(model, dl_exp, device, use_amp, buildings, exp_dir, steps=4)
        explain_out = {"dir": str(exp_dir)}

    result = {
        "model": "MultiBuildingMSPatchTST",
        "resolution": resolution,
        "buildings": buildings,
        "series_dir": str(data_root / resolution),
        "window": asdict(wc),
        "patch_scales": patch_scales,
        "model_cfg": mc,
        "train_cfg": asdict(cfg),
        "best_val_loss_scaled": float(best_val),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_mode": baseline_mode,
        "fusion": str(cfg.fusion),
        "gate_mode": str(cfg.gate_mode),
        "gate_max": float(cfg.gate_max),
        "gate_reg": float(cfg.gate_reg),
        "residual_reg": float(cfg.residual_reg),
        "total_loss_weight": float(cfg.total_loss_weight),
        "huber_beta": float(cfg.huber_beta),
        "baseline_val_metrics": baseline_val,
        "baseline_test_metrics": baseline_test,
        "micro_batch": int(micro_bs),
        "effective_batch": int(effective_bs),
        "accum_steps": int(accum_steps),
        "clip_nonneg": bool(clip_to_nonneg),
        "exports": exports,
        "explain": explain_out,
        "timestamp": now_str(),
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default=r"C:\Users\13470\Desktop\C_school\AfterWash")
    ap.add_argument("--out_root", type=str, default=r"C:\Users\13470\Desktop\C_school\TransformerRuns")

    ap.add_argument("--resolutions", type=str, default="1_hour,30_minutes,5_minutes")
    ap.add_argument("--buildings", type=str, default="Commercial,Office,Public,Residential")

    ap.add_argument("--epochs", type=int, default=30)

    ap.add_argument("--micro_batch", type=int, default=None)
    ap.add_argument("--effective_batch", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None, help="(compat) alias of micro_batch")

    ap.add_argument("--input_len", type=int, default=0)
    ap.add_argument("--pred_len", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=512)

    # v2: safe baseline fusion + total-load consistency
    ap.add_argument("--fusion", type=str, default="gated_residual", choices=["baseline_only", "residual", "gated_residual"])
    ap.add_argument("--gate_mode", type=str, default="building", choices=["building", "building_horizon"])
    ap.add_argument("--gate_init", type=float, default=-2.0)
    ap.add_argument("--gate_max", type=float, default=0.7)
    ap.add_argument("--gate_reg", type=float, default=1e-2)
    ap.add_argument("--residual_reg", type=float, default=1e-3)
    ap.add_argument("--total_loss_weight", type=float, default=0.1)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    ap.add_argument("--auto_res_tune", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--auto_quantiles", action=argparse.BooleanOptionalAction, default=True,
                    help="If quantiles is empty, auto-use 0.1,0.5,0.9 for 5_minutes")

    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--auto_batch", action="store_true")
    ap.add_argument("--min_batch", type=int, default=1)
    ap.add_argument("--max_batch", type=int, default=512)

    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true")

    ap.add_argument("--resume_training", action="store_true")
    ap.add_argument("--resume", action="store_true", help="(compat) alias of --resume_training")
    ap.add_argument("--force_retrain", action="store_true")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")

    ap.add_argument("--patch_scales", type=str, default="", help="e.g. 12:6,24:12 ; if empty uses defaults per resolution")

    ap.add_argument("--baseline_mode", type=str, default="weekly", choices=["none", "last", "daily", "weekly"])
    ap.add_argument("--adj_init", type=str, default="corr", choices=["identity", "random", "corr"])

    ap.add_argument("--quantiles", type=str, default="", help="Optional quantiles, e.g. 0.1,0.5,0.9 ; empty => point forecast")

    ap.add_argument("--run_tag", type=str, default="")
    ap.add_argument(
        "--separate_runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, save each run under out_root/MultiBuildingMSPatchTST/<run_tag>/..."
    )

    ap.add_argument("--export_preds", action="store_true")
    ap.add_argument("--export_splits", type=str, default="test", help="val,test,both (comma-separated)")
    ap.add_argument("--export_k", type=int, default=64)
    ap.add_argument("--plot_k", type=int, default=12)
    ap.add_argument("--export_strategy", type=str, default="uniform", choices=["uniform", "random", "first"])

    ap.add_argument("--explain", action="store_true", help="export adjacency + saliency")
    ap.add_argument("--clip_nonneg", action="store_true", help="clip inverse-scaled predictions to >=0 for metrics/exports")

    args = ap.parse_args()

    fail_fast_cuda_if_needed(args.device)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print_env_banner()

    if args.micro_batch is None:
        micro_batch = args.batch_size if args.batch_size is not None else 32
    else:
        micro_batch = args.micro_batch

    if args.effective_batch is None:
        effective_batch = int(micro_batch)
    else:
        effective_batch = args.effective_batch

    quantiles_user = []
    if str(args.quantiles).strip():
        quantiles_user = [float(x.strip()) for x in str(args.quantiles).split(",") if x.strip()]
        # basic sanity
        for q in quantiles_user:
            if not (0.0 < q < 1.0):
                raise ValueError(f"Invalid quantile: {q}. Must be in (0,1).")

    resolutions = [s.strip() for s in args.resolutions.split(",") if s.strip()]
    buildings = [s.strip() for s in args.buildings.split(",") if s.strip()]

    run_tag = compute_run_tag(args, int(micro_batch), int(effective_batch))

    base_out = Path(args.out_root) / "MultiBuildingMSPatchTST"
    out_root = (base_out / run_tag) if args.separate_runs else base_out
    out_root.mkdir(parents=True, exist_ok=True)

    safe_json_write(out_root / "run_meta.json", {
        "run_tag": run_tag,
        "separate_runs": bool(args.separate_runs),
        "timestamp": now_str(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "args": vars(args),
        "resolved": {
            "device": str(device),
            "micro_batch": int(micro_batch),
            "effective_batch": int(effective_batch),
            "quantiles": quantiles_user,
        },
    })

    cfg = TrainConfig(
        epochs=args.epochs,
        micro_batch=int(micro_batch),
        effective_batch=int(effective_batch),
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        huber_beta=float(args.huber_beta),
        total_loss_weight=float(args.total_loss_weight),
        residual_reg=float(args.residual_reg),
        fusion=str(args.fusion),
        gate_mode=str(args.gate_mode),
        gate_init=float(args.gate_init),
        gate_max=float(args.gate_max),
        gate_reg=float(args.gate_reg),
        grad_clip=args.grad_clip,
        patience=args.patience,
        num_workers=args.num_workers,
        save_every=max(1, args.save_every),
        grad_checkpoint=bool(args.grad_checkpoint),
        auto_res_tune=bool(args.auto_res_tune),
        auto_quantiles=bool(args.auto_quantiles),
    )

    resume_training = bool(args.resume_training or args.resume)
    export_splits = [s.strip() for s in args.export_splits.split(",") if s.strip()]
    if "both" in export_splits:
        export_splits = ["val", "test"]

    data_root = Path(args.data_root)

    progress_path = out_root / "progress.json"
    summary_path = out_root / "summary.json"
    progress = safe_json_load(progress_path, default={})
    summary: List[Dict] = safe_json_load(summary_path, default=[])

    for res in resolutions:
        if res not in DEFAULT_WINDOWS:
            raise ValueError(f"Unknown resolution: {res}")

        # window override
        if args.input_len > 0 and args.pred_len > 0:
            wc = WindowConfig(input_len=int(args.input_len), pred_len=int(args.pred_len))
        else:
            wc = DEFAULT_WINDOWS[res]

        # patch scales override
        if str(args.patch_scales).strip():
            patch_scales = parse_patch_scales(args.patch_scales)
        else:
            patch_scales = parse_patch_scales(DEFAULT_PATCH_SCALES[res])

        # quantiles (auto for 5_minutes if enabled and user did not specify)
        if quantiles_user:
            quantiles_used = list(quantiles_user)
        elif cfg.auto_quantiles and res == '5_minutes':
            quantiles_used = [0.1, 0.5, 0.9]
        else:
            quantiles_used = []

        key = f"{res}/Joint({'-'.join(buildings)})"
        progress[key] = {"status": "running", "start_time": now_str()}
        safe_json_write(progress_path, progress)

        print(f"\n[RUN] {key} | device={device.type} | run_tag={run_tag}")

        try:
            result = train_one_resolution(
                data_root=data_root,
                out_dir=out_root,
                resolution=res,
                buildings=buildings,
                wc=wc,
                patch_scales=patch_scales,
                cfg=cfg,
                device=device,
                auto_batch=bool(args.auto_batch),
                min_batch=args.min_batch,
                max_batch=args.max_batch,
                resume_training=resume_training,
                force_retrain=bool(args.force_retrain),
                compile_model=bool(args.compile),
                baseline_mode=str(args.baseline_mode),
                adj_init=str(args.adj_init),
                quantiles=quantiles_used if quantiles_used else None,
                export_preds=bool(args.export_preds),
                export_splits=export_splits,
                export_k=int(args.export_k),
                plot_k=int(args.plot_k),
                export_strategy=str(args.export_strategy),
                explain=bool(args.explain),
                clip_to_nonneg=bool(args.clip_nonneg),
                seed=int(args.seed),
            )

            summary.append(result)
            safe_json_write(summary_path, summary)
            progress[key] = {"status": "done", "end_time": now_str(), "metrics_path": str(Path(result.get("series_dir", "")))}
            safe_json_write(progress_path, progress)

            # concise display
            avg_test = result["test_metrics"]["avg"]
            base_avg = result["baseline_test_metrics"]["avg"]
            print(f"[OK] {key} avg_test(MAE/RMSE/MASE)={avg_test['MAE']:.3f}/{avg_test['RMSE']:.3f}/{avg_test['MASE']:.3f} | "
                  f"baseline={base_avg['MAE']:.3f}/{base_avg['RMSE']:.3f}/{base_avg['MASE']:.3f}")

        except Exception as e:
            cuda_cleanup()
            progress[key] = {"status": "failed", "end_time": now_str(), "error": str(e)}
            safe_json_write(progress_path, progress)
            print(f"[FAIL] {key}: {e}")

    print(f"\n[DONE] run_tag: {run_tag}")
    print(f"[DONE] out_root: {out_root}")
    print(f"[DONE] summary: {summary_path}")
    print(f"[DONE] progress: {progress_path}")


if __name__ == "__main__":
    main()
