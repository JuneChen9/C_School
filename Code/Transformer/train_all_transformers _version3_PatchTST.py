# -*- coding: utf-8 -*-
"""
PatchTST training for ElectricPowerLoadData (AfterWash).

Features:
1) Fail-fast if user requests CUDA but current torch is CPU-only.
2) Separate outputs for different model structures / hyperparams via run_tag + config hash:
   out_root/<model_name>/<run_tag>/<resolution>/<building>/...
3) Export prediction sequences (y_true/y_pred + timestamps) to NPZ and plot PNG curves:
   <run_dir>/preds/{val_samples.npz,test_samples.npz} and <run_dir>/preds/plots/*.png

Compatibility:
- --batch_size (alias of --micro_batch)
- --input_len/--pred_len override window
- --denoise_prob accepted but ignored
- --resume alias of --resume_training
"""

import os

# New env var name (PyTorch recommends PYTORCH_ALLOC_CONF)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")
# Keep old name as fallback for older PyTorch
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

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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
        print("\n[ERROR] You requested --device cuda, but this Python environment has CPU-only PyTorch.")
        print("Reason: torch.cuda.is_available() == False -> 'Torch not compiled with CUDA enabled' is inevitable.\n")
        print("Fix checklist:")
        print("  1) Make sure you're using your venv python, not system python:")
        print(r"     C:\Users\13470\Desktop\C_school\.venv\Scripts\python.exe -c ""import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)""")
        print("  2) If it still shows '+cpu' / False / None, reinstall CUDA-enabled torch in that venv (from pytorch.org).")
        print("  3) If you're on Python 3.13 and CUDA wheels aren't available, use Python 3.12/3.11 to create the venv.\n")
        raise SystemExit(2)


# -----------------------------
# AMP Scaler compatibility
# -----------------------------
class NullScaler:
    """A no-op GradScaler replacement (for CPU or when AMP disabled)."""
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def make_grad_scaler(use_amp: bool):
    # PyTorch 2.9.x: torch.amp.GradScaler does NOT accept device_type kwarg.
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

DEFAULT_PATCH = {
    "1_hour":      (12, 6),
    "30_minutes":  (24, 12),
    "5_minutes":   (32, 16),
}


# -----------------------------
# Data loading
# -----------------------------
def load_split_files(series_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _load(name: str) -> pd.DataFrame:
        p_parq = series_dir / f"{name}.parquet"
        p_csv = series_dir / f"{name}.csv"
        if p_parq.exists():
            df = pd.read_parquet(p_parq)
        elif p_csv.exists():
            df = pd.read_csv(p_csv)
        else:
            raise FileNotFoundError(f"Missing {name}.parquet/csv under {series_dir}")

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
                raise ValueError(f"No Power column in {name} under {series_dir}: columns={list(df.columns)}")
            df = df.rename(columns={cand: "Power_kW"})

        df = df.sort_index()
        return df[["Power_kW"]]

    return _load("train"), _load("val"), _load("test")


class SlidingWindowDataset(Dataset):
    """
    x_past: [input_len, C]  C = 1(value) + F(time_feats)
    y_future: [pred_len]
    Also stores dt array so we can export timestamps for visualization.
    """
    def __init__(self, values_scaled_1d: np.ndarray, time_feats: np.ndarray,
                 dt: np.ndarray, input_len: int, pred_len: int):
        assert values_scaled_1d.ndim == 1
        assert time_feats.ndim == 2 and len(time_feats) == len(values_scaled_1d)
        assert len(dt) == len(values_scaled_1d)

        self.y = values_scaled_1d.astype(np.float32)
        self.tf = time_feats.astype(np.float32)
        self.dt = dt.astype("datetime64[ns]")

        self.in_len = int(input_len)
        self.out_len = int(pred_len)
        self.seq_len = self.in_len + self.out_len
        self.max_start = len(self.y) - self.seq_len
        if self.max_start <= 0:
            raise ValueError(f"Series too short: N={len(self.y)}, in={self.in_len}, out={self.out_len}")

    def __len__(self):
        return self.max_start + 1

    def __getitem__(self, idx: int):
        s = idx
        e = idx + self.seq_len
        past_y = self.y[s:s + self.in_len]
        past_tf = self.tf[s:s + self.in_len, :]
        x = np.concatenate([past_y[:, None], past_tf], axis=1).astype(np.float32)
        y = self.y[s + self.in_len:e].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    def past_times(self, idx: int) -> np.ndarray:
        s = idx
        return self.dt[s:s + self.in_len]

    def future_times(self, idx: int) -> np.ndarray:
        s = idx + self.in_len
        return self.dt[s:s + self.out_len]

    def past_values_scaled(self, idx: int) -> np.ndarray:
        s = idx
        return self.y[s:s + self.in_len]

    def future_values_scaled(self, idx: int) -> np.ndarray:
        s = idx + self.in_len
        return self.y[s:s + self.out_len]


def make_loader(ds: Dataset, batch: int, shuffle: bool, num_workers: int, pin: bool, drop_last: bool) -> DataLoader:
    kw = dict(batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=pin, drop_last=drop_last)
    if num_workers and num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


# -----------------------------
# Model: PatchTST
# -----------------------------
class SDPASelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.hd = d_model // n_heads
        self.drop = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.hd).transpose(1, 2)
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


class PatchTST(nn.Module):
    def __init__(
        self,
        input_len: int,
        pred_len: int,
        in_channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.grad_checkpoint = bool(grad_checkpoint)

        if self.input_len < self.patch_len:
            raise ValueError(f"input_len({self.input_len}) < patch_len({self.patch_len})")

        self.n_patches = 1 + (self.input_len - self.patch_len) // self.stride
        if self.n_patches <= 0:
            raise ValueError("n_patches <= 0, check patch_len/stride")

        self.patch_embed = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=d_model,
            kernel_size=self.patch_len,
            stride=self.stride,
            bias=True
        )

        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * d_model, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        if L != self.input_len:
            raise RuntimeError(f"Expected input_len={self.input_len}, got {L}")

        x = x.transpose(1, 2)
        tok = self.patch_embed(x)
        tok = tok.transpose(1, 2)
        tok = tok + self.pos

        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                tok = ckpt(lambda t: blk(t), tok)
            else:
                tok = blk(tok)

        tok = self.out_ln(tok)
        y = self.head(tok)
        return y


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, scaler: StandardScaler, device: torch.device, use_amp: bool) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with autocast_ctx(device, use_amp):
            pred = model(xb)

        pred_np = pred.detach().cpu().numpy().reshape(-1, 1)
        true_np = yb.detach().cpu().numpy().reshape(-1, 1)

        pred_inv = scaler.inverse_transform(pred_np).reshape(-1)
        true_inv = scaler.inverse_transform(true_np).reshape(-1)
        ys.append(true_inv)
        ps.append(pred_inv)

    y_true = np.concatenate(ys, axis=0) if ys else np.array([])
    y_pred = np.concatenate(ps, axis=0) if ps else np.array([])
    if y_true.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred)}


# -----------------------------
# Export sequences + plots
# -----------------------------
def choose_indices(n: int, k: int, strategy: str, seed: int) -> np.ndarray:
    k = max(1, min(int(k), int(n)))
    if strategy == "first":
        return np.arange(k, dtype=int)
    if strategy == "random":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=k, replace=False))
    # default: uniform
    if k == 1:
        return np.array([0], dtype=int)
    return np.linspace(0, n - 1, num=k, dtype=int)


def inv_scale_1d(scaler: StandardScaler, arr_1d: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(arr_1d.reshape(-1, 1)).reshape(-1)


@torch.no_grad()
def export_samples_and_plots(
    model: nn.Module,
    ds: SlidingWindowDataset,
    scaler: StandardScaler,
    device: torch.device,
    use_amp: bool,
    out_dir: Path,
    split_name: str,
    export_k: int,
    plot_k: int,
    export_strategy: str,
    seed: int,
    infer_batch: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds)
    idxs = choose_indices(n, export_k, export_strategy, seed)

    in_len = ds.in_len
    out_len = ds.out_len

    past_time = np.empty((len(idxs), in_len), dtype="datetime64[ns]")
    fut_time = np.empty((len(idxs), out_len), dtype="datetime64[ns]")
    past_true = np.empty((len(idxs), in_len), dtype=np.float32)
    fut_true = np.empty((len(idxs), out_len), dtype=np.float32)
    fut_pred = np.empty((len(idxs), out_len), dtype=np.float32)

    model.eval()

    # batched inference over chosen indices
    ptr = 0
    while ptr < len(idxs):
        batch_ids = idxs[ptr:ptr + max(1, infer_batch)]
        xs = []
        for i in batch_ids:
            x, _ = ds[int(i)]
            xs.append(x)
        xb = torch.stack(xs, dim=0).to(device, non_blocking=True)

        with autocast_ctx(device, use_amp):
            pb = model(xb)  # [B, out_len]
        pb = pb.detach().cpu().numpy().astype(np.float32)  # scaled preds

        for j, i in enumerate(batch_ids):
            i = int(i)
            row = ptr + j
            pt = ds.past_times(i)
            ft = ds.future_times(i)
            py = ds.past_values_scaled(i)
            fy = ds.future_values_scaled(i)

            past_time[row, :] = pt
            fut_time[row, :] = ft

            past_true[row, :] = inv_scale_1d(scaler, py).astype(np.float32)
            fut_true[row, :] = inv_scale_1d(scaler, fy).astype(np.float32)
            fut_pred[row, :] = inv_scale_1d(scaler, pb[j]).astype(np.float32)

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
    )

    # plot first plot_k samples
    kplot = max(0, min(int(plot_k), len(idxs)))
    for s in range(kplot):
        t_p = past_time[s]
        t_f = fut_time[s]
        y_p = past_true[s]
        y_t = fut_true[s]
        y_p2 = fut_pred[s]

        plt.figure(figsize=(12, 4))
        plt.plot(t_p.astype("datetime64[ns]"), y_p)     # past true
        plt.plot(t_f.astype("datetime64[ns]"), y_t)     # future true
        plt.plot(t_f.astype("datetime64[ns]"), y_p2)    # future pred
        plt.title(f"{split_name} sample {s:03d} (idx={int(idxs[s])})")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{split_name}_sample_{s:03d}.png", dpi=180)
        plt.close()

    return {
        "npz": str(npz_path),
        "plots_dir": str(plots_dir),
        "export_k": int(export_k),
        "plot_k": int(plot_k),
        "strategy": export_strategy,
    }


# -----------------------------
# Auto micro-batch probing
# -----------------------------
def try_micro_batch(ds: Dataset, model: nn.Module, device: torch.device, micro_bs: int, use_amp: bool) -> bool:
    loader = make_loader(ds, micro_bs, shuffle=True, num_workers=0, pin=(device.type == "cuda"), drop_last=True)
    xb, yb = next(iter(loader))
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    model.train()
    model.zero_grad(set_to_none=True)

    scaler = make_grad_scaler(use_amp)

    try:
        with autocast_ctx(device, use_amp):
            pred = model(xb)
            loss = F.smooth_l1_loss(pred, yb, beta=1.0)

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
                     min_bs: int, max_bs: int, use_amp: bool) -> int:
    min_bs = max(1, int(min_bs))
    max_bs = max(min_bs, int(max_bs))

    best = None
    bs = min_bs
    while bs <= max_bs:
        ok = try_micro_batch(ds, model, device, bs, use_amp)
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
    micro_batch: int = 64
    effective_batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 512
    grad_clip: float = 1.0
    patience: int = 6
    num_workers: int = 2
    save_every: int = 1
    grad_checkpoint: bool = False


def model_cfg_for_resolution(cfg: TrainConfig, resolution: str) -> Dict[str, int]:
    if resolution == "5_minutes":
        return {"d_model": max(cfg.d_model, 256), "n_heads": 8, "n_layers": max(6, cfg.n_layers), "d_ff": max(512, cfg.d_ff)}
    if resolution == "30_minutes":
        return {"d_model": min(cfg.d_model, 256), "n_heads": 8, "n_layers": min(cfg.n_layers, 6), "d_ff": min(cfg.d_ff, 512)}
    return {"d_model": min(cfg.d_model, 256), "n_heads": 8, "n_layers": min(cfg.n_layers, 6), "d_ff": min(cfg.d_ff, 512)}


def save_checkpoint(path: Path, payload: Dict):
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)


def train_one_series(
    series_dir: Path,
    out_dir: Path,
    resolution: str,
    building: str,
    wc: WindowConfig,
    patch_len: int,
    stride: int,
    cfg: TrainConfig,
    device: torch.device,
    auto_batch: bool,
    min_batch: int,
    max_batch: int,
    resume_training: bool,
    compile_model: bool,
    model_name: str,
    export_preds: bool,
    export_splits: List[str],
    export_k: int,
    plot_k: int,
    export_strategy: str,
    seed: int,
) -> Dict:
    if model_name != "PatchTST":
        # 你说 1h/30min 用原模型：如果你的原模型脚本不是这个文件，这里会报错。
        # 如果你把原模型也融合进同一个脚本，请在这里按 model_name 分支构建相应模型。
        raise ValueError(f"Only PatchTST is implemented in this script, got model_name={model_name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "checkpoint_last.pt"
    metrics_path = out_dir / "metrics.json"

    use_amp = (device.type == "cuda")

    train_df, val_df, test_df = load_split_files(series_dir)

    scaler = StandardScaler()
    scaler.fit(train_df["Power_kW"].values.reshape(-1, 1).astype(np.float32))

    def prep(df: pd.DataFrame):
        y = df["Power_kW"].values.astype(np.float32).reshape(-1, 1)
        y_s = scaler.transform(y).reshape(-1).astype(np.float32)
        tf = make_time_features(df.index)
        dt = df.index.values.astype("datetime64[ns]")
        return y_s, tf, dt

    train_y, train_tf, train_dt = prep(train_df)
    val_y, val_tf, val_dt = prep(val_df)
    test_y, test_tf, test_dt = prep(test_df)

    ds_train = SlidingWindowDataset(train_y, train_tf, train_dt, wc.input_len, wc.pred_len)
    ds_val = SlidingWindowDataset(val_y, val_tf, val_dt, wc.input_len, wc.pred_len)
    ds_test = SlidingWindowDataset(test_y, test_tf, test_dt, wc.input_len, wc.pred_len)

    in_channels = 1 + train_tf.shape[1]
    mc = model_cfg_for_resolution(cfg, resolution)

    model = PatchTST(
        input_len=wc.input_len,
        pred_len=wc.pred_len,
        in_channels=in_channels,
        patch_len=patch_len,
        stride=stride,
        d_model=int(mc["d_model"]),
        n_heads=int(mc["n_heads"]),
        n_layers=int(mc["n_layers"]),
        d_ff=int(mc["d_ff"]),
        dropout=float(cfg.dropout),
        grad_checkpoint=cfg.grad_checkpoint,
    ).to(device)

    if compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
            print(f"[COMPILE] enabled for {resolution}/{building}")
        except Exception as e:
            print(f"[COMPILE] failed, continue without compile: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 10))

    grad_scaler = make_grad_scaler(use_amp)

    best_val = float("inf")
    patience_left = cfg.patience
    start_epoch = 1
    micro_bs = int(cfg.micro_batch)

    # Resume
    if resume_training and last_path.exists():
        ck = load_checkpoint(last_path, device)
        same = (
            ck.get("model_name") == "PatchTST"
            and ck.get("resolution") == resolution
            and ck.get("building") == building
            and ck.get("window") == asdict(wc)
            and ck.get("patch") == {"patch_len": patch_len, "stride": stride}
            and ck.get("model_cfg") == mc
        )
        if same:
            model.load_state_dict(ck["model_state"])
            optim.load_state_dict(ck["optim_state"])
            scheduler.load_state_dict(ck["sched_state"])
            start_epoch = int(ck.get("epoch", 0)) + 1
            best_val = float(ck.get("best_val", best_val))
            patience_left = int(ck.get("patience_left", patience_left))
            micro_bs = int(ck.get("micro_batch", micro_bs))
            print(f"[RESUME] {resolution}/{building} epoch={start_epoch} best_val={best_val:.6f} micro_bs={micro_bs}")
        else:
            print(f"[RESUME-SKIP] {resolution}/{building} checkpoint incompatible -> retrain from scratch")

    # Auto micro-batch
    if auto_batch and device.type == "cuda" and start_epoch <= 1:
        print(f"[AUTO-BATCH] {resolution}/{building} search micro_bs in [{min_batch},{max_batch}] ...")
        cuda_cleanup()
        micro_bs = find_micro_batch(ds_train, model, device, min_batch, max_batch, use_amp)
        print(f"[AUTO-BATCH] selected micro_bs={micro_bs}")
        cuda_cleanup()

    effective_bs = max(1, int(cfg.effective_batch))
    micro_bs = max(1, int(micro_bs))
    accum_steps = max(1, int(math.ceil(effective_bs / micro_bs)))

    eval_bs = min(max_batch, max(1, micro_bs * 2))
    pin = (device.type == "cuda")

    dl_train = make_loader(ds_train, micro_bs, True, cfg.num_workers, pin, True)
    dl_val = make_loader(ds_val, eval_bs, False, cfg.num_workers, pin, False)
    dl_test = make_loader(ds_test, eval_bs, False, cfg.num_workers, pin, False)

    print(f"[CFG] patch_len={patch_len} stride={stride} n_patches={model.n_patches} | micro_bs={micro_bs} effective={effective_bs} accum={accum_steps} eval_bs={eval_bs} grad_ckpt={cfg.grad_checkpoint}")

    # Train loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        optim.zero_grad(set_to_none=True)
        t0 = time.time()
        micro_steps = 0
        losses = []

        pbar = tqdm(dl_train, desc=f"[{resolution}/{building}] epoch {epoch}/{cfg.epochs}", leave=False)
        for step, (xb, yb) in enumerate(pbar, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with autocast_ctx(device, use_amp):
                pred = model(xb)
                loss = F.smooth_l1_loss(pred, yb, beta=1.0) / accum_steps

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
        print(f"[THROUGHPUT] {resolution}/{building} epoch={epoch} micro_it/s={micro_it_s:.2f} samples/s={samples_s:.2f} updates/s={updates_s:.3f}")

        # val loss (scaled domain)
        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast_ctx(device, use_amp):
                    pred = model(xb)
                    vloss = F.smooth_l1_loss(pred, yb, beta=1.0)
                vls.append(float(vloss.item()))
        val_loss = float(np.mean(vls)) if vls else float("inf")

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            patience_left = cfg.patience
            torch.save({
                "model_name": "PatchTST",
                "model_state": model.state_dict(),
                "resolution": resolution,
                "building": building,
                "window": asdict(wc),
                "patch": {"patch_len": patch_len, "stride": stride},
                "model_cfg": mc,
                "train_cfg": asdict(cfg),
                "micro_batch": micro_bs,
                "effective_batch": effective_bs,
                "accum_steps": accum_steps,
                "best_val_loss_scaled": best_val,
                "timestamp": now_str(),
            }, best_path)
        else:
            patience_left -= 1

        if (epoch % max(1, cfg.save_every) == 0) or patience_left <= 0 or epoch == cfg.epochs:
            save_checkpoint(last_path, {
                "model_name": "PatchTST",
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "sched_state": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "patience_left": patience_left,
                "resolution": resolution,
                "building": building,
                "window": asdict(wc),
                "patch": {"patch_len": patch_len, "stride": stride},
                "model_cfg": mc,
                "train_cfg": asdict(cfg),
                "micro_batch": micro_bs,
                "effective_batch": effective_bs,
                "accum_steps": accum_steps,
                "timestamp": now_str(),
            })

        if patience_left <= 0:
            break

    # Load best for evaluation + export
    ck_best = torch.load(best_path, map_location=device)
    model.load_state_dict(ck_best["model_state"])

    val_metrics = evaluate(model, dl_val, scaler, device, use_amp)
    test_metrics = evaluate(model, dl_test, scaler, device, use_amp)

    exports = {}
    if export_preds:
        pred_dir = out_dir / "preds"
        infer_bs = min(256, max(1, eval_bs))
        if "val" in export_splits:
            exports["val"] = export_samples_and_plots(
                model=model, ds=ds_val, scaler=scaler, device=device, use_amp=use_amp,
                out_dir=pred_dir, split_name="val",
                export_k=export_k, plot_k=plot_k,
                export_strategy=export_strategy, seed=seed,
                infer_batch=infer_bs,
            )
        if "test" in export_splits:
            exports["test"] = export_samples_and_plots(
                model=model, ds=ds_test, scaler=scaler, device=device, use_amp=use_amp,
                out_dir=pred_dir, split_name="test",
                export_k=export_k, plot_k=plot_k,
                export_strategy=export_strategy, seed=seed,
                infer_batch=infer_bs,
            )

    result = {
        "model": "PatchTST",
        "resolution": resolution,
        "building": building,
        "series_dir": str(series_dir),
        "window": asdict(wc),
        "patch": {"patch_len": patch_len, "stride": stride, "n_patches": model.n_patches},
        "model_cfg": mc,
        "train_cfg": asdict(cfg),
        "best_val_loss_scaled": best_val,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "micro_batch": micro_bs,
        "effective_batch": effective_bs,
        "accum_steps": accum_steps,
        "timestamp": now_str(),
        "exports": exports,
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# -----------------------------
# Main
# -----------------------------
def compute_run_tag(args: argparse.Namespace, micro_batch: int, effective_batch: int) -> str:
    if getattr(args, "run_tag", ""):
        return str(args.run_tag)

    cfg = {
        "model_name": args.model_name,
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
        "patch_len": args.patch_len,
        "stride": args.stride,
        "micro_batch": micro_batch,
        "effective_batch": effective_batch,
        "auto_batch": args.auto_batch,
        "min_batch": args.min_batch,
        "max_batch": args.max_batch,
        "grad_checkpoint": args.grad_checkpoint,
        "compile": args.compile,
        "export_preds": args.export_preds,
        "export_k": args.export_k,
        "export_strategy": args.export_strategy,
    }
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")
    h = hashlib.sha1(raw).hexdigest()[:8]
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{ts}_{h}"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default=r"C:\Users\13470\Desktop\C_school\AfterWash")
    ap.add_argument("--out_root", type=str, default=r"C:\Users\13470\Desktop\C_school\TransformerRuns")

    ap.add_argument("--model_name", type=str, default="PatchTST")
    ap.add_argument("--resolutions", type=str, default="1_hour,30_minutes,5_minutes")
    ap.add_argument("--buildings", type=str, default="Commercial,Office,Public,Residential")

    ap.add_argument("--epochs", type=int, default=30)

    ap.add_argument("--micro_batch", type=int, default=None, help="micro-batch size. If omitted, uses batch_size or default 64.")
    ap.add_argument("--effective_batch", type=int, default=None, help="effective batch via grad accumulation.")

    ap.add_argument("--batch_size", type=int, default=None, help="(compat) old name for micro_batch")
    ap.add_argument("--input_len", type=int, default=0, help="override default input_len per resolution")
    ap.add_argument("--pred_len", type=int, default=0, help="override default pred_len per resolution")
    ap.add_argument("--denoise_prob", type=float, default=0.0, help="(compat) accepted but ignored")

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=512)

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

    ap.add_argument("--patch_len", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0)

    ap.add_argument("--run_tag", type=str, default="", help="Optional: name this run folder manually.")
    ap.add_argument(
        "--separate_runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, save each run under <out_root>/<model>/<run_tag>/... to avoid overwriting."
    )

    # Export prediction sequences
    ap.add_argument("--export_preds", action="store_true", help="export y_true/y_pred + timestamps to NPZ and plot PNG curves")
    ap.add_argument("--export_splits", type=str, default="test", help="val/test/both (comma-separated)")
    ap.add_argument("--export_k", type=int, default=64, help="how many windows to export per split")
    ap.add_argument("--plot_k", type=int, default=12, help="how many exported windows to plot as PNG")
    ap.add_argument("--export_strategy", type=str, default="uniform", choices=["uniform", "random", "first"],
                    help="how to choose windows for export")

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
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

    print_env_banner()

    if args.micro_batch is None:
        micro_batch = args.batch_size if args.batch_size is not None else 64
    else:
        micro_batch = args.micro_batch

    if args.effective_batch is None:
        effective_batch = micro_batch
    else:
        effective_batch = args.effective_batch

    run_tag = compute_run_tag(args, int(micro_batch), int(effective_batch))
    base_out = Path(args.out_root) / str(args.model_name)
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
        },
    })

    data_root = Path(args.data_root)

    progress_path = out_root / "progress.json"
    summary_path = out_root / "summary.json"

    progress = safe_json_load(progress_path, default={})
    summary: List[Dict] = safe_json_load(summary_path, default=[])

    resolutions = [s.strip() for s in args.resolutions.split(",") if s.strip()]
    buildings = [s.strip() for s in args.buildings.split(",") if s.strip()]

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
        grad_clip=args.grad_clip,
        patience=args.patience,
        num_workers=args.num_workers,
        save_every=max(1, args.save_every),
        grad_checkpoint=bool(args.grad_checkpoint),
    )

    resume_training = bool(args.resume_training or args.resume)
    export_splits = [s.strip() for s in args.export_splits.split(",") if s.strip()]
    if "both" in export_splits:
        export_splits = ["val", "test"]

    for res in resolutions:
        if res not in DEFAULT_WINDOWS:
            raise ValueError(f"Unknown resolution: {res}")

        if args.input_len > 0 and args.pred_len > 0:
            wc = WindowConfig(input_len=int(args.input_len), pred_len=int(args.pred_len))
        else:
            wc = DEFAULT_WINDOWS[res]

        if args.patch_len > 0 and args.stride > 0:
            patch_len, stride = int(args.patch_len), int(args.stride)
        else:
            patch_len, stride = DEFAULT_PATCH[res]

        for bld in buildings:
            series_dir = data_root / res / bld
            run_dir = out_root / res / bld
            key = f"{res}/{bld}"

            if not series_dir.exists():
                print(f"[SKIP] missing data dir: {series_dir}")
                continue

            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists() and (not args.force_retrain):
                print(f"[SKIP-DONE] {key} (metrics.json exists)")
                continue

            progress[key] = {"status": "running", "start_time": now_str(), "series_dir": str(series_dir), "run_dir": str(run_dir)}
            safe_json_write(progress_path, progress)

            print(f"\n[RUN] {key} | patch_len={patch_len} stride={stride} | device={device.type} | model={args.model_name} | run_tag={run_tag}")

            try:
                result = train_one_series(
                    series_dir=series_dir,
                    out_dir=run_dir,
                    resolution=res,
                    building=bld,
                    wc=wc,
                    patch_len=patch_len,
                    stride=stride,
                    cfg=cfg,
                    device=device,
                    auto_batch=bool(args.auto_batch),
                    min_batch=args.min_batch,
                    max_batch=args.max_batch,
                    resume_training=resume_training,
                    compile_model=bool(args.compile),
                    model_name=str(args.model_name),
                    export_preds=bool(args.export_preds),
                    export_splits=export_splits,
                    export_k=int(args.export_k),
                    plot_k=int(args.plot_k),
                    export_strategy=str(args.export_strategy),
                    seed=int(args.seed),
                )
                summary.append(result)
                safe_json_write(summary_path, summary)
                progress[key] = {"status": "done", "end_time": now_str(), "metrics_path": str(run_dir / "metrics.json")}
                safe_json_write(progress_path, progress)
                print(f"[OK] {key} test={result['test_metrics']} micro_bs={result['micro_batch']} accum={result['accum_steps']}")

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
