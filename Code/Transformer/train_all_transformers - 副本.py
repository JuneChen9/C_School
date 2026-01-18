# -*- coding: utf-8 -*-
"""
Train Transformers for power load forecasting on cleaned dataset with RESUME support.

Data root example:
C:\\Users\\13470\\Desktop\\C_school\\AfterWash\\
  1_hour\\Commercial\\train.csv / val.csv / test.csv (or .parquet)
  1_hour\\Office\\...
  30_minutes\\...
  5_minutes\\...

Output:
C:\\Users\\13470\\Desktop\\C_school\\TransformerRuns\\
  progress.json                # 全局进度（每个序列 done/running/failed）
  summary.json                 # 汇总所有 done 的 metrics
  <resolution>\\<building>\\
     best.pt
     checkpoint_last.pt        # 断点续训用
     metrics.json

Resume rules (默认开启):
- 若 metrics.json 存在 -> 认为该序列已完成，直接跳过（除非 --force_retrain）
- 若 metrics.json 不存在，但 checkpoint_last.pt 存在 -> 从断点继续训练
- 若只有 best.pt 存在 -> 会直接重新评估并写出 metrics.json（不再重复训练）

Important note:
Your data has been imputed to zero-missing. For Commercial/Public, raw missing is non-trivial,
so metrics may partly reflect imputation rules. This code uses denoising masking to reduce that risk.
"""

import argparse
import json
import math
import os
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
    # 不强行 deterministic（会更慢）
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
    os.replace(tmp, p)  # atomic on same filesystem


def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


# -----------------------------
# Time features
# -----------------------------
def make_time_features(dt_index: pd.DatetimeIndex) -> np.ndarray:
    """
    Periodic time features using sin/cos.
    Features:
      - hour-of-day sin/cos
      - day-of-week sin/cos
      - day-of-year sin/cos
      - month-of-year sin/cos
      - is_weekend (0/1)
    """
    if not isinstance(dt_index, pd.DatetimeIndex):
        dt_index = pd.to_datetime(dt_index)

    hour = dt_index.hour.values.astype(np.float32)
    dow = dt_index.dayofweek.values.astype(np.float32)  # 0..6
    doy = dt_index.dayofyear.values.astype(np.float32)  # 1..366
    month = dt_index.month.values.astype(np.float32)    # 1..12
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
    "1_hour": WindowConfig(input_len=24 * 7, pred_len=24),          # 7d -> 1d
    "30_minutes": WindowConfig(input_len=48 * 7, pred_len=48),      # 7d -> 1d
    "5_minutes": WindowConfig(input_len=288 * 7, pred_len=288),     # 7d -> 1d
}


class SlidingWindowDataset(Dataset):
    """
    Sequence length L = input_len + pred_len.
    - Past input_len: value channel is observed past
    - Future pred_len: value channel set to 0 placeholder
    - Time features provided for all steps
    - future_flag indicates future positions
    - masked_flag indicates denoising-masked past positions (train only)
    """

    def __init__(
        self,
        values: np.ndarray,         # [N], scaled
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

        y_all = self.values[start:end]             # [L]
        tf_all = self.time_feats[start:end, :]     # [L, F]

        past = y_all[: self.input_len].copy()
        future_true = y_all[self.input_len:].copy()

        future_placeholder = np.zeros((self.pred_len,), dtype=np.float32)

        future_flag = np.concatenate([
            np.zeros((self.input_len,), dtype=np.float32),
            np.ones((self.pred_len,), dtype=np.float32),
        ], axis=0)  # [L]

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
# Model: Causal Transformer Encoder
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
    # True means masked
    return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)


class LoadTransformer(nn.Module):
    """
    QKV 采用标准多头注意力线性映射：
      Q = X W_Q, K = X W_K, V = X W_V
    由 nn.TransformerEncoderLayer 内部实现，选择 Pre-LN (norm_first=True) 稳定训练。
    """

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
    num_workers: int = 0
    save_every: int = 1  # 每多少个 epoch 落盘 checkpoint_last.pt（>=1）


def load_split_files(series_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prefer parquet, fallback to csv.
    Each file must have Time index and Power_kW column.
    """
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
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    all_true, all_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        out = model(xb)                  # [B, L]
        pred = out[:, -pred_len:]        # [B, pred_len]

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
    """
    适度保守：5min 序列更长，默认用更宽但不更深的模型，避免训练不稳定。
    """
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


def _maybe_autotune_batch(cfg: TrainConfig, resolution: str, device: torch.device) -> int:
    """
    你昨晚跑不完，很可能是 5min 序列 + 超长窗口 + batch 太大导致极慢/爆内存。
    这里给一个保守自动降 batch 的策略（不改你传参的其它内容）。
    """
    bs = cfg.batch_size
    if resolution == "5_minutes":
        # CPU 上更保守
        if device.type != "cuda":
            return min(bs, 8)
        return min(bs, 16)
    return bs


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
    resume_training: bool = True,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = save_dir / "metrics.json"
    best_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "checkpoint_last.pt"

    # 如果 best.pt 存在但 metrics.json 不存在：只做评估 + 写 metrics（不再浪费训练）
    # 这能覆盖“训练完成但重启前没来得及写 summary”的场景。
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

        bs = _maybe_autotune_batch(cfg, resolution, device)
        val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=cfg.num_workers,
                                pin_memory=(device.type == "cuda"), drop_last=False)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=(device.type == "cuda"), drop_last=False)

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

        val_metrics = evaluate(model, val_loader, scaler, window_cfg.pred_len, device)
        test_metrics = evaluate(model, test_loader, scaler, window_cfg.pred_len, device)

        best_val = float(ckpt.get("best_val_loss_scaled", ckpt.get("best_val", float("nan"))))
        if not np.isfinite(best_val):
            best_val = float("nan")

        return _write_metrics(
            save_dir, resolution, building, series_dir, window_cfg, best_val,
            val_metrics, test_metrics,
            extra_note="metrics.json 缺失但 best.pt 存在：已自动补写 metrics（未重复训练）。"
        )

    if best_path.exists() and (not metrics_path.exists()):
        return _eval_from_best_and_write()

    # 正常训练流程
    train_df, val_df, test_df = load_split_files(series_dir)

    # scaler 只用 train 拟合
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

    bs = _maybe_autotune_batch(cfg, resolution, device)

    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=(device.type == "cuda"), drop_last=False)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=(device.type == "cuda"), drop_last=False)

    in_dim = 1 + train_tf.shape[1] + 2

    # model cfg
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

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(cfg.epochs, 10))

    # AMP（仅 CUDA）
    use_amp = (device.type == "cuda")
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    patience_left = cfg.patience
    start_epoch = 1

    # 断点续训：加载 checkpoint_last.pt
    if resume_training and last_ckpt_path.exists():
        ckpt = _load_checkpoint(last_ckpt_path, device)
        # 简单一致性检查（不一致就不加载）
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
            print(f"[RESUME] {resolution}/{building} 从 epoch={start_epoch} 继续（best_val={best_val:.6f}, patience_left={patience_left}）")
        else:
            print(f"[RESUME-SKIP] {resolution}/{building} checkpoint 配置不匹配，改为从头训练。")

    # training loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"[{resolution}/{building}] epoch {epoch}/{cfg.epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(xb)                          # [B, L]
                pred = out[:, -window_cfg.pred_len:]     # [B, pred_len]
                loss = F.smooth_l1_loss(pred, yb, beta=1.0)  # Huber

            grad_scaler.scale(loss).backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                grad_scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            grad_scaler.step(optim)
            grad_scaler.update()

            losses.append(loss.item())
            pbar.set_postfix(loss=float(np.mean(losses)))

        scheduler.step()

        # val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                pred = out[:, -window_cfg.pred_len:]
                vloss = F.smooth_l1_loss(pred, yb, beta=1.0)
                val_losses.append(vloss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        # early stop / save best
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
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "best_val_loss_scaled": best_val,
                "timestamp": _now_str(),
            }, best_path)
        else:
            patience_left -= 1

        # checkpoint_last.pt：用于断点续训（每 save_every 个 epoch 写一次，或最后一次/早停前强制写）
        need_save = (cfg.save_every >= 1 and (epoch % cfg.save_every == 0))
        if need_save or patience_left <= 0 or epoch == cfg.epochs:
            _save_checkpoint(
                last_ckpt_path, model, optim, scheduler, scaler,
                epoch=epoch, best_val=best_val, patience_left=patience_left,
                resolution=resolution, building=building,
                window_cfg=window_cfg, train_cfg=cfg, in_dim=in_dim, model_cfg=model_cfg
            )

        if patience_left <= 0:
            break

    # load best and evaluate
    if not best_path.exists():
        raise RuntimeError(f"训练异常：best.pt 未生成：{save_dir}")

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    val_metrics = evaluate(model, val_loader, scaler, window_cfg.pred_len, device)
    test_metrics = evaluate(model, test_loader, scaler, window_cfg.pred_len, device)

    return _write_metrics(
        save_dir, resolution, building, series_dir, window_cfg,
        best_val=best_val, val_metrics=val_metrics, test_metrics=test_metrics
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
    ap.add_argument("--data_root", type=str, default=r"C:\Users\13470\Desktop\C_school\AfterWash",
                    help="AfterWash 根目录")
    ap.add_argument("--out_root", type=str, default=r"C:\Users\13470\Desktop\C_school\TransformerRuns",
                    help="模型输出目录")

    ap.add_argument("--resolutions", type=str, default="1_hour,30_minutes,5_minutes",
                    help="要训练的分辨率，用逗号分隔")
    ap.add_argument("--buildings", type=str, default="Commercial,Office,Public,Residential",
                    help="要训练的建筑类型，用逗号分隔")

    ap.add_argument("--input_len", type=int, default=0, help="覆盖默认 input_len（0=默认）")
    ap.add_argument("--pred_len", type=int, default=0, help="覆盖默认 pred_len（0=默认）")

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
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=1, help="每多少 epoch 保存一次 checkpoint_last.pt（>=1）")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")

    # resume controls
    ap.add_argument("--resume", action="store_true", help="启用断点续跑（默认建议开启）")
    ap.add_argument("--force_retrain", action="store_true", help="无视 metrics.json，强制重训")
    ap.add_argument("--resume_training", action="store_true", help="若 checkpoint_last.pt 存在，从中续训（建议开启）")

    args = ap.parse_args()

    # 默认行为：resume + resume_training 都开启（用户不传参数也能续跑）
    if not args.resume:
        args.resume = True
    if not args.resume_training:
        args.resume_training = True

    set_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    progress_path = out_root / "progress.json"
    summary_path = out_root / "summary.json"

    progress = _safe_json_load(progress_path, default={})
    all_done_metrics: List[Dict] = []

    resolutions = [s.strip() for s in args.resolutions.split(",") if s.strip()]
    buildings = [s.strip() for s in args.buildings.split(",") if s.strip()]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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

    # 先把已有的 metrics 收集进 summary（防止你中途中断导致 summary 丢）
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

            # resume: done -> skip
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
                )

                progress[key] = {
                    "status": "done",
                    "end_time": _now_str(),
                    "metrics_path": str(save_dir / "metrics.json"),
                }
                _safe_json_write(progress_path, progress)

                # 增量更新 summary（每跑完一个序列就落盘）
                all_done_metrics.append(resu)
                _safe_json_write(summary_path, all_done_metrics)

                print(f"[OK] {key} test={resu['test_metrics']}")

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
