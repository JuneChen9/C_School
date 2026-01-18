# -*- coding: utf-8 -*-
"""
Train Transformers for power load forecasting on cleaned dataset with RESUME + AUTO-BATCH support.

Data root example:
C:\\Users\\13470\\Desktop\\C_school\\AfterWash\\
  1_hour\\Commercial\\train.csv / val.csv / test.csv (or .parquet)
  1_hour\\Office\\...
  30_minutes\\...
  5_minutes\\...

Output:
C:\\Users\\13470\\Desktop\\C_school\\TransformerRuns\\
  progress.json
  summary.json
  <resolution>\\<building>\\
     best.pt
     checkpoint_last.pt
     metrics.json

Resume:
- metrics.json exists -> skip (unless --force_retrain)
- checkpoint_last.pt exists -> resume training
- best.pt exists but metrics.json missing -> evaluate and write metrics.json (no retrain)

New:
- --auto_batch: automatically find max feasible train batch per series on current GPU.
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -----------------------------
# Utils: safe json IO (atomic)
# -----------------------------
def _safe_json_load(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _safe_json_write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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

    return feats


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class WindowConfig:
    input_len: int
    pred_len: int


DEFAULT_WINDOWS = {
    "1_hour": WindowConfig(input_len=24 * 7, pred_len=24),
    "30_minutes": WindowConfig(input_len=48 * 7, pred_len=48),
    "5_minutes": WindowConfig(input_len=288 * 7, pred_len=288),
}


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,         # [N] scaled
        time_feats: np.ndarray,     # [N, F]
        input_len: int,
        pred_len: int,
        is_train: bool,
        denoise_prob: float = 0.0,
    ):
        assert values.ndim == 1
        assert time_feats.ndim == 2 and len(time_feats) == len(values)

        self.values = values.astype(np.float32)
        self.time_feats = time_feats.astype(np.float32)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.seq_len = self.input_len + self.pred_len
        self.is_train = bool(is_train)
        self.denoise_prob = float(denoise_prob)

        self.max_start = len(values) - self.seq_len
        if self.max_start <= 0:
            raise ValueError(
                f"Series too short: N={len(values)}, input_len={input_len}, pred_len={pred_len}"
            )

    def __len__(self):
        return self.max_start + 1

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.seq_len

        y_all = self.values[start:end]         # [L]
        tf_all = self.time_feats[start:end, :] # [L, F]

        past = y_all[: self.input_len].copy()
        future_true = y_all[self.input_len:].copy()

        future_placeholder = np.zeros((self.pred_len,), dtype=np.float32)

        future_flag = np.concatenate([
            np.zeros((self.input_len,), dtype=np.float32),
            np.ones((self.pred_len,), dtype=np.float32),
        ], axis=0)

        masked_flag = np.zeros((self.seq_len,), dtype=np.float32)
        if self.is_train and self.denoise_prob > 0:
            mask = (np.random.rand(self.input_len) < self.denoise_prob)
            past[mask] = 0.0
            masked_flag[: self.input_len] = mask.astype(np.float32)

        value_channel = np.concatenate([past, future_placeholder], axis=0)  # [L]

        x = np.concatenate([
            value_channel[:, None],      # 1
            tf_all,                      # F
            future_flag[:, None],        # 1
            masked_flag[:, None],        # 1
        ], axis=1).astype(np.float32)    # [L, 1+F+2]

        return torch.from_numpy(x), torch.from_numpy(future_true)


# -----------------------------
# Model
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:L, :].unsqueeze(0)


def generate_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


class LoadTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 10000,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = self.input_proj(x)
        h = self.pos(h)
        mask = generate_causal_mask(L, x.device)
        h = self.encoder(h, mask=mask)
        h = self.out_norm(h)
        y = self.head(h).squeeze(-1)  # [B, L]
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


# -----------------------------
# Training / Eval
# -----------------------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    d_model: int = 128
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 256
    denoise_prob: float = 0.10
    grad_clip: float = 1.0
    patience: int = 6
    num_workers: int = 2
    save_every: int = 1


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
            if df.index.name != "Time":
                try:
                    df.index = pd.to_datetime(df.index)
                    df.index.name = "Time"
                except Exception:
                    pass

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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler: StandardScaler,
    pred_len: int,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    all_true, all_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(xb)
            pred = out[:, -pred_len:]

        pred_np = pred.detach().cpu().numpy().reshape(-1, 1)
        true_np = yb.detach().cpu().numpy().reshape(-1, 1)

        pred_inv = scaler.inverse_transform(pred_np).reshape(-1)
        true_inv = scaler.inverse_transform(true_np).reshape(-1)

        all_true.append(true_inv)
        all_pred.append(pred_inv)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def _build_model_cfg_for_resolution(cfg: TrainConfig, resolution: str) -> Dict[str, int]:
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    n_layers = cfg.n_layers
    d_ff = cfg.d_ff

    if resolution == "5_minutes":
        d_model = max(d_model, 192)
        if d_model % 8 == 0:
            n_heads = 8
        n_layers = min(n_layers, 4)
        d_ff = max(d_ff, 384)

    return {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff}


def _dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if num_workers and num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(dataset, **kw)


def _try_train_step_for_batch(
    train_set: Dataset,
    model: nn.Module,
    window_cfg: WindowConfig,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> bool:
    """
    Try a single forward+backward step to test if batch_size fits GPU memory.
    Returns True if OK, False if OOM.
    """
    loader = _dataloader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # probe uses 0 to reduce overhead
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    it = iter(loader)
    xb, yb = next(it)
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    model.train()
    model.zero_grad(set_to_none=True)

    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(xb)
            pred = out[:, -window_cfg.pred_len:]
            loss = F.smooth_l1_loss(pred, yb, beta=1.0)

        if use_amp:
            gs = torch.amp.GradScaler("cuda", enabled=True)
            gs.scale(loss).backward()
        else:
            loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        return True

    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False
        raise


def find_max_batch_size(
    train_set: Dataset,
    model: nn.Module,
    window_cfg: WindowConfig,
    device: torch.device,
    start_bs: int,
    min_bs: int,
    max_bs: int,
    use_amp: bool,
) -> int:
    """
    Exponential growth + binary search to find max feasible batch size.
    """
    start_bs = int(max(min_bs, start_bs))
    max_bs = int(max(max_bs, start_bs))
    min_bs = int(max(1, min_bs))

    # quick fail-safe
    if not _try_train_step_for_batch(train_set, model, window_cfg, device, start_bs, use_amp):
        # shrink down to min_bs
        bs = start_bs
        while bs > min_bs:
            bs = max(min_bs, bs // 2)
            ok = _try_train_step_for_batch(train_set, model, window_cfg, device, bs, use_amp)
            if ok:
                return bs
        return min_bs

    last_ok = start_bs
    bs = start_bs

    # grow
    while True:
        nxt = bs * 2
        if nxt > max_bs:
            break
        ok = _try_train_step_for_batch(train_set, model, window_cfg, device, nxt, use_amp)
        if ok:
            last_ok = nxt
            bs = nxt
        else:
            break

    # binary search between last_ok+1 and upper
    low = last_ok + 1
    high = min(max_bs, bs * 2 - 1)
    if low > high:
        return last_ok

    best = last_ok
    while low <= high:
        mid = (low + high) // 2
        ok = _try_train_step_for_batch(train_set, model, window_cfg, device, mid, use_amp)
        if ok:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    return best


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler,
    scaler: StandardScaler,
    epoch: int,
    best_val: float,
    patience_left: int,
    resolution: str,
    building: str,
    window_cfg: WindowConfig,
    train_cfg: TrainConfig,
    in_dim: int,
    model_cfg: Dict[str, int],
    train_batch_size: int,
):
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "patience_left": patience_left,
        "resolution": resolution,
        "building": building,
        "window": asdict(window_cfg),
        "train_cfg": asdict(train_cfg),
        "in_dim": in_dim,
        "model_cfg": model_cfg,
        "train_batch_size": int(train_batch_size),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "timestamp": _now_str(),
    }
    torch.save(ckpt, path)


def _load_checkpoint(path: Path, device: torch.device):
    return torch.load(path, map_location=device)


def _write_metrics(
    save_dir: Path,
    resolution: str,
    building: str,
    series_dir: Path,
    window_cfg: WindowConfig,
    best_val: float,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    train_batch_size: int,
    extra_note: str = ""
) -> Dict:
    result = {
        "resolution": resolution,
        "building": building,
        "series_dir": str(series_dir),
        "window": asdict(window_cfg),
        "best_val_loss_scaled": best_val,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_batch_size": int(train_batch_size),
        "timestamp": _now_str(),
        "notes": {
            "warning": "Commercial/Public 可能反映插值规律；建议对照季节 naive baseline，避免假信心。",
            "extra": extra_note
        }
    }
    (save_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def train_one_series(
    series_dir: Path,
    save_dir: Path,
    resolution: str,
    building: str,
    window_cfg: WindowConfig,
    cfg: TrainConfig,
    device: torch.device,
    resume_training: bool,
    auto_batch: bool,
    min_batch: int,
    max_batch: int,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_dir / "metrics.json"
    best_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "checkpoint_last.pt"

    use_amp = (device.type == "cuda")

    # If best exists but metrics missing: evaluate and write metrics
    def _eval_from_best_and_write():
        train_df, val_df, test_df = load_split_files(series_dir)
        scaler = StandardScaler()
        scaler.fit(train_df["Power_kW"].values.reshape(-1, 1).astype(np.float32))

        def _prep(df: pd.DataFrame):
            vals = df["Power_kW"].values.astype(np.float32).reshape(-1, 1)
            vals_scaled = scaler.transform(vals).reshape(-1).astype(np.float32)
            tf = make_time_features(df.index)
            return vals_scaled, tf

        val_y, val_tf = _prep(val_df)
        test_y, test_tf = _prep(test_df)

        val_set = SlidingWindowDataset(val_y, val_tf, window_cfg.input_len, window_cfg.pred_len, is_train=False)
        test_set = SlidingWindowDataset(test_y, test_tf, window_cfg.input_len, window_cfg.pred_len, is_train=False)

        ckpt = torch.load(best_path, map_location=device)
        in_dim = int(ckpt.get("in_dim", 1 + val_tf.shape[1] + 2))
        mc = ckpt.get("model_cfg") or _build_model_cfg_for_resolution(cfg, resolution)

        model = LoadTransformer(
            in_dim=in_dim,
            d_model=int(mc["d_model"]),
            n_heads=int(mc["n_heads"]),
            n_layers=int(mc["n_layers"]),
            d_ff=int(mc["d_ff"]),
            dropout=float(cfg.dropout),
            max_len=window_cfg.input_len + window_cfg.pred_len + 8,
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        train_bs = int(ckpt.get("train_batch_size", cfg.batch_size))
        eval_bs = min(max_batch, max(1, train_bs * 2))

        val_loader = _dataloader(val_set, eval_bs, False, cfg.num_workers, (device.type == "cuda"), False)
        test_loader = _dataloader(test_set, eval_bs, False, cfg.num_workers, (device.type == "cuda"), False)

        val_metrics = evaluate(model, val_loader, scaler, window_cfg.pred_len, device, use_amp)
        test_metrics = evaluate(model, test_loader, scaler, window_cfg.pred_len, device, use_amp)

        best_val = float(ckpt.get("best_val_loss_scaled", ckpt.get("best_val", float("nan"))))
        if not np.isfinite(best_val):
            best_val = float("nan")

        return _write_metrics(
            save_dir, resolution, building, series_dir, window_cfg,
            best_val, val_metrics, test_metrics, train_batch_size=train_bs,
            extra_note="metrics.json 缺失但 best.pt 存在：已自动补写 metrics（未重复训练）。"
        )

    if best_path.exists() and (not metrics_path.exists()):
        return _eval_from_best_and_write()

    # Load data
    train_df, val_df, test_df = load_split_files(series_dir)

    # scaler fitted on train only
    scaler = StandardScaler()
    scaler.fit(train_df["Power_kW"].values.reshape(-1, 1).astype(np.float32))

    def _prep(df: pd.DataFrame):
        vals = df["Power_kW"].values.astype(np.float32).reshape(-1, 1)
        vals_scaled = scaler.transform(vals).reshape(-1).astype(np.float32)
        tf = make_time_features(df.index)
        return vals_scaled, tf

    train_y, train_tf = _prep(train_df)
    val_y, val_tf = _prep(val_df)
    test_y, test_tf = _prep(test_df)

    train_set = SlidingWindowDataset(train_y, train_tf, window_cfg.input_len, window_cfg.pred_len,
                                     is_train=True, denoise_prob=cfg.denoise_prob)
    val_set = SlidingWindowDataset(val_y, val_tf, window_cfg.input_len, window_cfg.pred_len,
                                   is_train=False, denoise_prob=0.0)
    test_set = SlidingWindowDataset(test_y, test_tf, window_cfg.input_len, window_cfg.pred_len,
                                    is_train=False, denoise_prob=0.0)

    in_dim = 1 + train_tf.shape[1] + 2
    model_cfg = _build_model_cfg_for_resolution(cfg, resolution)

    model = LoadTransformer(
        in_dim=in_dim,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(cfg.dropout),
        max_len=window_cfg.input_len + window_cfg.pred_len + 8,
    ).to(device)

    # Determine train batch size
    train_bs = int(cfg.batch_size)

    # Resume: if checkpoint exists, keep its batch size unless user wants auto-batch
    best_val = float("inf")
    patience_left = cfg.patience
    start_epoch = 1

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 10))
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if resume_training and last_ckpt_path.exists():
        ckpt = _load_checkpoint(last_ckpt_path, device)
        same_window = (ckpt.get("window", {}) == asdict(window_cfg))
        same_in_dim = (int(ckpt.get("in_dim", in_dim)) == int(in_dim))
        same_model_cfg = (ckpt.get("model_cfg", {}) == model_cfg)

        if same_window and same_in_dim and same_model_cfg:
            model.load_state_dict(ckpt["model_state"])
            optim.load_state_dict(ckpt["optim_state"])
            if ckpt.get("sched_state") is not None:
                scheduler.load_state_dict(ckpt["sched_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))
            patience_left = int(ckpt.get("patience_left", patience_left))
            train_bs = int(ckpt.get("train_batch_size", train_bs))
            print(f"[RESUME] {resolution}/{building} epoch={start_epoch} best_val={best_val:.6f} patience_left={patience_left} batch={train_bs}")
        else:
            print(f"[RESUME-SKIP] {resolution}/{building} checkpoint 配置不匹配，改为从头训练。")

    # Auto-batch (only if CUDA and requested)
    if auto_batch and device.type == "cuda":
        # Important: if resuming training mid-way, changing batch changes optimization dynamics.
        # Here we only auto-tune if starting from epoch 1 OR you explicitly want it.
        if start_epoch <= 1:
            print(f"[AUTO-BATCH] probing {resolution}/{building} start_bs={train_bs} range=[{min_batch},{max_batch}] ...")
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            best_bs = find_max_batch_size(
                train_set=train_set,
                model=model,
                window_cfg=window_cfg,
                device=device,
                start_bs=train_bs,
                min_bs=min_batch,
                max_bs=max_batch,
                use_amp=use_amp,
            )
            train_bs = int(best_bs)
            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"[AUTO-BATCH] selected batch={train_bs} (probe peak alloc ~ {peak:.0f} MiB)")
        else:
            print(f"[AUTO-BATCH] skipped because resuming at epoch {start_epoch} (保持 batch={train_bs} 避免训练动态突变)")

    # Eval batch: larger is usually safe (no backward)
    eval_bs = min(max_batch, max(1, train_bs * 2))

    train_loader = _dataloader(train_set, train_bs, True, cfg.num_workers, (device.type == "cuda"), True)
    val_loader = _dataloader(val_set, eval_bs, False, cfg.num_workers, (device.type == "cuda"), False)
    test_loader = _dataloader(test_set, eval_bs, False, cfg.num_workers, (device.type == "cuda"), False)

    # Training loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"[{resolution}/{building}] epoch {epoch}/{cfg.epochs} bs={train_bs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(xb)
                pred = out[:, -window_cfg.pred_len:]
                loss = F.smooth_l1_loss(pred, yb, beta=1.0)

            grad_scaler.scale(loss).backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                grad_scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            grad_scaler.step(optim)
            grad_scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        scheduler.step()

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = model(xb)
                    pred = out[:, -window_cfg.pred_len:]
                    vloss = F.smooth_l1_loss(pred, yb, beta=1.0)
                val_losses.append(vloss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        improved = (val_loss < best_val - 1e-6)
        if improved:
            best_val = val_loss
            patience_left = cfg.patience
            torch.save({
                "model_state": model.state_dict(),
                "resolution": resolution,
                "building": building,
                "window": asdict(window_cfg),
                "train_cfg": asdict(cfg),
                "in_dim": in_dim,
                "model_cfg": model_cfg,
                "train_batch_size": int(train_bs),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "best_val_loss_scaled": best_val,
                "timestamp": _now_str(),
            }, best_path)
        else:
            patience_left -= 1

        # Save checkpoint_last
        need_save = (cfg.save_every >= 1 and (epoch % cfg.save_every == 0))
        if need_save or patience_left <= 0 or epoch == cfg.epochs:
            _save_checkpoint(
                last_ckpt_path, model, optim, scheduler, scaler,
                epoch=epoch, best_val=best_val, patience_left=patience_left,
                resolution=resolution, building=building,
                window_cfg=window_cfg, train_cfg=cfg,
                in_dim=in_dim, model_cfg=model_cfg,
                train_batch_size=train_bs,
            )

        if patience_left <= 0:
            break

    if not best_path.exists():
        raise RuntimeError(f"best.pt 未生成：{save_dir}")

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    val_metrics = evaluate(model, val_loader, scaler, window_cfg.pred_len, device, use_amp)
    test_metrics = evaluate(model, test_loader, scaler, window_cfg.pred_len, device, use_amp)

    return _write_metrics(
        save_dir, resolution, building, series_dir, window_cfg,
        best_val=best_val, val_metrics=val_metrics, test_metrics=test_metrics,
        train_batch_size=train_bs
    )


# -----------------------------
# Orchestrator
# -----------------------------
def is_series_done(save_dir: Path) -> bool:
    return (save_dir / "metrics.json").exists()


def load_metrics_if_exists(save_dir: Path) -> Optional[Dict]:
    p = save_dir / "metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"C:\Users\13470\Desktop\C_school\AfterWash")
    ap.add_argument("--out_root", type=str, default=r"C:\Users\13470\Desktop\C_school\TransformerRuns")

    ap.add_argument("--resolutions", type=str, default="1_hour,30_minutes,5_minutes")
    ap.add_argument("--buildings", type=str, default="Commercial,Office,Public,Residential")

    ap.add_argument("--input_len", type=int, default=0)
    ap.add_argument("--pred_len", type=int, default=0)

    # train config
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--denoise_prob", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=1)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")

    # resume controls
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_retrain", action="store_true")
    ap.add_argument("--resume_training", action="store_true")

    # auto-batch controls
    ap.add_argument("--auto_batch", action="store_true", help="自动探测最大 batch（推荐在 CUDA 上开启）")
    ap.add_argument("--min_batch", type=int, default=4)
    ap.add_argument("--max_batch", type=int, default=512)

    args = ap.parse_args()

    # default: resume+resume_training on
    if not args.resume:
        args.resume = True
    if not args.resume_training:
        args.resume_training = True

    # default: auto_batch on when cuda
    if args.auto_batch is False and args.device in ("auto", "cuda"):
        # 保守：不强制打开；你想要就显式传 --auto_batch
        pass

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Debug prints (验证解释器/torch/cuda 一致性)
    print("PY:", sys.executable)
    print("TORCH:", torch.__version__)
    print("CUDA:", torch.cuda.is_available(), torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    progress_path = out_root / "progress.json"
    summary_path = out_root / "summary.json"

    progress = _safe_json_load(progress_path, default={})
    all_done_metrics: List[Dict] = []

    resolutions = [s.strip() for s in args.resolutions.split(",") if s.strip()]
    buildings = [s.strip() for s in args.buildings.split(",") if s.strip()]

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        denoise_prob=args.denoise_prob,
        grad_clip=args.grad_clip,
        patience=args.patience,
        num_workers=args.num_workers,
        save_every=max(1, args.save_every),
    )

    # collect existing metrics into summary first
    for res in resolutions:
        if res not in DEFAULT_WINDOWS:
            raise ValueError(f"Unknown resolution: {res}. Expected one of {list(DEFAULT_WINDOWS.keys())}")
        for bld in buildings:
            save_dir = out_root / res / bld
            m = load_metrics_if_exists(save_dir)
            if m is not None:
                all_done_metrics.append(m)

    _safe_json_write(summary_path, all_done_metrics)
    _safe_json_write(progress_path, progress)

    for res in resolutions:
        wc = DEFAULT_WINDOWS[res]
        if args.input_len and args.input_len > 0:
            wc = WindowConfig(input_len=args.input_len, pred_len=wc.pred_len)
        if args.pred_len and args.pred_len > 0:
            wc = WindowConfig(input_len=wc.input_len, pred_len=args.pred_len)

        for bld in buildings:
            key = f"{res}/{bld}"
            series_dir = data_root / res / bld
            save_dir = out_root / res / bld

            if not series_dir.exists():
                print(f"[SKIP] missing data dir: {series_dir}")
                continue

            if args.resume and (not args.force_retrain) and is_series_done(save_dir):
                print(f"[SKIP-DONE] {key} (metrics.json exists)")
                continue

            progress[key] = {
                "status": "running",
                "start_time": _now_str(),
                "series_dir": str(series_dir),
                "save_dir": str(save_dir),
            }
            _safe_json_write(progress_path, progress)

            print(f"\n[RUN] {key} | data={series_dir} | save={save_dir} | window={wc} | device={device.type}")

            try:
                resu = train_one_series(
                    series_dir=series_dir,
                    save_dir=save_dir,
                    resolution=res,
                    building=bld,
                    window_cfg=wc,
                    cfg=cfg,
                    device=device,
                    resume_training=args.resume_training,
                    auto_batch=args.auto_batch,
                    min_batch=args.min_batch,
                    max_batch=args.max_batch,
                )

                progress[key] = {
                    "status": "done",
                    "end_time": _now_str(),
                    "metrics_path": str(save_dir / "metrics.json"),
                }
                _safe_json_write(progress_path, progress)

                all_done_metrics.append(resu)
                _safe_json_write(summary_path, all_done_metrics)

                print(f"[OK] {key} test={resu['test_metrics']} batch={resu['train_batch_size']}")

            except Exception as e:
                progress[key] = {
                    "status": "failed",
                    "end_time": _now_str(),
                    "error": str(e),
                }
                _safe_json_write(progress_path, progress)
                print(f"[FAIL] {key}: {e}")

    print(f"\n[DONE] summary: {summary_path}")
    print(f"[DONE] progress: {progress_path}")


if __name__ == "__main__":
    main()
