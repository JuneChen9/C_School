# -*- coding: utf-8 -*-
"""
Clean + audit + split industrial park load dataset for Transformer training.

Expected extracted folder structure (your description):
电力负荷数据/
  2016|2017|2018/
    1_hour|30_minutes|5_minutes/
      2016_1hour_Commercial|... etc
        20160101_1hour_Commercial.xlsx ...

Each leaf xlsx: columns: Time, Power (kW)

Outputs:
output_root/
  1_hour/
    Commercial/
      train.parquet, val.parquet, test.parquet (+ csv)
      stats.json
    Office/...
  30_minutes/...
  5_minutes/...
  audit_report.json
  audit_report.csv

Note:
- Chronological split only.
- Outliers -> set NaN -> impute.
- Imputation uses:
  (1) time reindex to full grid
  (2) short gaps: time interpolation
  (3) long gaps: seasonal median by (weekday, time-of-day)
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
BUILDING_TYPES = ["Commercial", "Office", "Public", "Residential"]

# mapping from folder name to pandas frequency
RESOLUTION_FREQ = {
    "1_hour": "1H",
    "30_minutes": "30T",
    "5_minutes": "5T",
}

# some datasets use 5min or 30min naming; normalize
RESOLUTION_ALIASES = {
    "1hour": "1_hour",
    "1_hour": "1_hour",
    "30min": "30_minutes",
    "30_minutes": "30_minutes",
    "5min": "5_minutes",
    "5_minutes": "5_minutes",
}

# Train/Val/Test split ratios (chronological)
SPLIT = (0.70, 0.15, 0.15)

# Outlier detection parameters (robust rolling)
OUTLIER_WINDOW_STEPS = 48  # for 1H ~ 2 days; for 30min ~ 1 day; for 5min ~ 4 hours -> adjusted later
OUTLIER_Z = 8.0            # robust threshold; higher = less aggressive

# Imputation thresholds
SHORT_GAP_MAX_STEPS = 3    # <=3 steps interpolate; longer use seasonal fill
SEASONAL_MIN_HISTORY_DAYS = 7  # if too little history, fallback to ffill/bfill

# Minimal viability: if missing rate > this, you should reconsider training
MAX_REASONABLE_MISSING_RATE = 0.30


# -----------------------------
# Helpers
# -----------------------------
def _run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def try_extract_rar(rar_path: Path, extract_to: Path) -> Path:
    """
    Try to extract .rar using system tools.
    Prefer 7z if available, else unrar.
    If neither exists, raise.
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    # 1) try 7z
    code, out = _run(["bash", "-lc", "command -v 7z >/dev/null 2>&1; echo $?"])
    if out.strip() == "0":
        rc, log = _run(["bash", "-lc", f"7z x -y -o{shlex_quote(str(extract_to))} {shlex_quote(str(rar_path))}"])
        if rc == 0:
            return extract_to
        raise RuntimeError(f"7z 解压失败，日志：\n{log}")

    # 2) try unrar
    code, out = _run(["bash", "-lc", "command -v unrar >/dev/null 2>&1; echo $?"])
    if out.strip() == "0":
        rc, log = _run(["bash", "-lc", f"unrar x -o+ {shlex_quote(str(rar_path))} {shlex_quote(str(extract_to))}"])
        if rc == 0:
            return extract_to
        raise RuntimeError(f"unrar 解压失败，日志：\n{log}")

    raise RuntimeError(
        "未检测到可用的 rar 解压工具（7z/unrar）。\n"
        "请先手动把 ElectricPowerLoadData.rar 解压成目录，然后把 --input 指向解压后的根目录（包含“电力负荷数据”那个文件夹）。"
    )


def shlex_quote(s: str) -> str:
    # minimal safe quote for bash -lc
    return "'" + s.replace("'", "'\"'\"'") + "'"


def normalize_resolution_from_path(p: Path) -> Optional[str]:
    s = str(p)
    for key in ["1_hour", "30_minutes", "5_minutes", "1hour", "30min", "5min"]:
        if key in s:
            return RESOLUTION_ALIASES.get(key, key)
    return None


def infer_building_type(p: Path) -> Optional[str]:
    # from filename or parent folder
    s = p.name
    for b in BUILDING_TYPES:
        if b.lower() in s.lower():
            return b
    for part in p.parts[::-1]:
        for b in BUILDING_TYPES:
            if b.lower() in part.lower():
                return b
    return None


def read_leaf_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """
    Read a single daily file.
    Returns DataFrame with columns: Time (datetime64), Power (float)
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    # normalize column names
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # identify time/power columns robustly
    time_col = None
    power_col = None
    for c in df.columns:
        if c.lower() == "time":
            time_col = c
        if "power" in c.lower():
            power_col = c
    if time_col is None or power_col is None:
        raise ValueError(f"列名不符合预期：{xlsx_path}，columns={list(df.columns)}")

    out = df[[time_col, power_col]].copy()
    out.columns = ["Time", "Power_kW"]

    # parse time
    out["Time"] = pd.to_datetime(out["Time"], errors="coerce")

    # parse power: coerce non-numeric
    out["Power_kW"] = pd.to_numeric(out["Power_kW"], errors="coerce")

    # drop rows with invalid time
    out = out.dropna(subset=["Time"])

    return out


def robust_rolling_outlier_mask(x: pd.Series, window: int, z: float) -> pd.Series:
    """
    Robust rolling outlier detection using rolling median and MAD.
    Mark True where outlier.
    """
    # rolling median
    med = x.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()
    # rolling MAD
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, center=True, min_periods=max(5, window // 5)).median()

    # avoid divide by zero
    mad = mad.replace(0, np.nan)
    robust_z = abs_dev / (1.4826 * mad)

    return robust_z > z


def seasonal_fill(series: pd.Series, freq: str) -> pd.Series:
    """
    Seasonal fill by (weekday, time-of-day) median.
    Works for fixed freq grid.
    """
    if series.index.inferred_type not in ("datetime64", "datetime"):
        raise ValueError("series index must be datetime")

    # Need enough history
    days = (series.index.max() - series.index.min()).days
    if days < SEASONAL_MIN_HISTORY_DAYS:
        return series.ffill().bfill()

    idx = series.index
    key = pd.MultiIndex.from_arrays(
        [idx.weekday, idx.time],
        names=["weekday", "time"]
    )
    med = series.groupby(key).transform("median")

    filled = series.copy()
    filled = filled.fillna(med)
    filled = filled.ffill().bfill()
    return filled


@dataclass
class AuditRow:
    resolution: str
    building: str
    n_points: int
    start: str
    end: str
    missing_rate_raw: float
    missing_rate_after: float
    outliers_marked: int
    empty_files: int
    total_files: int
    note: str


def build_series_from_folder(root: Path) -> Tuple[Dict[Tuple[str, str], pd.DataFrame], List[AuditRow]]:
    """
    Traverse root, read all xlsx, aggregate into continuous series for each (resolution, building).
    Returns:
      data_map[(resolution, building)] = df with index Time, column Power_kW
      audit_rows: list of audit info
    """
    xlsx_files = sorted(root.rglob("*.xlsx"))
    if not xlsx_files:
        raise RuntimeError(f"在目录下未找到任何 .xlsx：{root}")

    # group by (resolution, building)
    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for f in xlsx_files:
        res = normalize_resolution_from_path(f)
        bld = infer_building_type(f)
        if res is None or bld is None:
            continue
        grouped.setdefault((res, bld), []).append(f)

    data_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    audit_rows: List[AuditRow] = []

    for (res, bld), files in grouped.items():
        freq = RESOLUTION_FREQ.get(res)
        if freq is None:
            continue

        total_files = len(files)
        empty_files = 0
        parts = []
        raw_missing_points = 0
        raw_total_points = 0

        for f in files:
            try:
                df = read_leaf_xlsx(f)
            except Exception as e:
                # treat unreadable as empty
                empty_files += 1
                continue

            raw_total_points += len(df)
            raw_missing_points += int(df["Power_kW"].isna().sum())

            if df["Power_kW"].notna().sum() == 0:
                empty_files += 1

            parts.append(df)

        if not parts:
            audit_rows.append(AuditRow(
                resolution=res, building=bld, n_points=0, start="", end="",
                missing_rate_raw=1.0, missing_rate_after=1.0,
                outliers_marked=0, empty_files=empty_files, total_files=total_files,
                note="无可读取数据（全部文件读取失败或为空）"
            ))
            continue

        full = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["Time"]).sort_values("Time")
        full = full.set_index("Time")

        # Reindex to full grid
        start, end = full.index.min(), full.index.max()
        full_index = pd.date_range(start=start, end=end, freq=freq)
        full = full.reindex(full_index)
        raw_missing_rate = float(full["Power_kW"].isna().mean())

        # Outlier detection window: scale by freq
        # make window roughly ~2 days for each resolution
        if freq == "1H":
            window = 48
        elif freq == "30T":
            window = 96
        elif freq == "5T":
            window = 24 * 12  # 24h * 12 per hour = 288
        else:
            window = OUTLIER_WINDOW_STEPS

        # mark outliers (ignore NaN)
        x = full["Power_kW"]
        mask_outlier = robust_rolling_outlier_mask(x, window=window, z=OUTLIER_Z)
        outliers_marked = int(mask_outlier.fillna(False).sum())
        full.loc[mask_outlier, "Power_kW"] = np.nan

        # Impute:
        # 1) short gaps interpolate by time
        s = full["Power_kW"]

        # find gap lengths (consecutive NaN runs)
        is_na = s.isna().to_numpy()
        gap_len = np.zeros_like(is_na, dtype=int)
        run = 0
        for i in range(len(is_na)):
            if is_na[i]:
                run += 1
            else:
                run = 0
            gap_len[i] = run
        # forward pass gives ending lengths; we want per position, so do a second pass
        # compute run lengths for each NaN segment
        seg_lengths = np.zeros_like(is_na, dtype=int)
        i = 0
        while i < len(is_na):
            if not is_na[i]:
                i += 1
                continue
            j = i
            while j < len(is_na) and is_na[j]:
                j += 1
            seg_lengths[i:j] = (j - i)
            i = j

        short_mask = (is_na) & (seg_lengths <= SHORT_GAP_MAX_STEPS)
        s_interp = s.copy()
        s_interp.loc[short_mask] = np.nan  # keep NaN, interpolation will fill them
        s_interp = s_interp.interpolate(method="time", limit=SHORT_GAP_MAX_STEPS)

        # 2) remaining NaNs -> seasonal fill
        s_filled = seasonal_fill(s_interp, freq=freq)

        full["Power_kW"] = s_filled
        missing_rate_after = float(full["Power_kW"].isna().mean())

        note = ""
        if raw_missing_rate > MAX_REASONABLE_MISSING_RATE:
            note = f"警告：原始缺失率过高({raw_missing_rate:.2%})，模型训练风险极大；优先检查源数据是否真的有值。"
        elif empty_files > 0:
            note = f"提示：检测到空/不可读文件 {empty_files}/{total_files}，已通过拼接后插值/季节性填补。"

        audit_rows.append(AuditRow(
            resolution=res, building=bld,
            n_points=int(len(full)),
            start=str(start), end=str(end),
            missing_rate_raw=raw_missing_rate,
            missing_rate_after=missing_rate_after,
            outliers_marked=outliers_marked,
            empty_files=empty_files,
            total_files=total_files,
            note=note
        ))

        data_map[(res, bld)] = full

    return data_map, audit_rows


def chronological_split(df: pd.DataFrame, split=SPLIT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by time order.
    """
    assert abs(sum(split) - 1.0) < 1e-9
    n = len(df)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    # remainder to test
    n_test = n - n_train - n_val

    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train:n_train + n_val].copy()
    test = df.iloc[n_train + n_val:].copy()
    assert len(test) == n_test
    return train, val, test


def save_outputs(
    out_root: Path,
    data_map: Dict[Tuple[str, str], pd.DataFrame],
    audit_rows: List[AuditRow],
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    # audit report
    audit_df = pd.DataFrame([asdict(r) for r in audit_rows])
    audit_df = audit_df.sort_values(["resolution", "building"])
    audit_df.to_csv(out_root / "audit_report.csv", index=False, encoding="utf-8-sig")
    (out_root / "audit_report.json").write_text(json.dumps(audit_df.to_dict(orient="records"), ensure_ascii=False, indent=2), encoding="utf-8")

    # per series outputs
    for (res, bld), df in data_map.items():
        series_dir = out_root / res / bld
        series_dir.mkdir(parents=True, exist_ok=True)

        # basic stats
        s = df["Power_kW"].astype(float)
        stats = {
            "resolution": res,
            "building": bld,
            "n_points": int(len(df)),
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s)),
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
        }

        train, val, test = chronological_split(df)

        # save
        for name, part in [("train", train), ("val", val), ("test", test)]:
            part_out = part.copy()
            part_out.index.name = "Time"
            part_out.to_parquet(series_dir / f"{name}.parquet", index=True)
            part_out.to_csv(series_dir / f"{name}.csv", index=True, encoding="utf-8-sig")

        (series_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def find_dataset_root(input_path: Path) -> Path:
    """
    If input is already a folder containing 电力负荷数据 -> return that folder or its child.
    """
    if input_path.is_dir():
        # if user points directly to 电力负荷数据
        if (input_path / "2016").exists() or (input_path / "2017").exists() or (input_path / "2018").exists():
            return input_path
        # else search for 电力负荷数据 folder
        for cand in input_path.rglob("电力负荷数据"):
            if cand.is_dir():
                return cand
        # fallback: use input_path
        return input_path
    else:
        raise RuntimeError("input_path must be a directory (建议先解压 rar 成目录).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True,
                    help="解压后的数据根目录（推荐，包含“电力负荷数据”文件夹），或一个更高层目录也行。")
    ap.add_argument("--output", type=str, required=True, help="输出目录")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise RuntimeError(f"输入路径不存在：{input_path}")

    dataset_root = find_dataset_root(input_path)
    print(f"[INFO] dataset_root = {dataset_root}")

    data_map, audit_rows = build_series_from_folder(dataset_root)

    if not data_map:
        raise RuntimeError("未构建出任何可用序列：可能是路径结构不匹配，或文件名/文件夹名与描述不一致。请先查看 audit_report.csv。")

    save_outputs(out_root, data_map, audit_rows)

    # final hard warning if too missing
    bad = [r for r in audit_rows if r.n_points > 0 and r.missing_rate_raw > MAX_REASONABLE_MISSING_RATE]
    if bad:
        print("\n[WARNING] 以下序列原始缺失率过高，训练 Transformer 风险很大：")
        for r in bad:
            print(f"  - {r.resolution}/{r.building}: missing_rate_raw={r.missing_rate_raw:.2%}, empty_files={r.empty_files}/{r.total_files}")
        print("建议：优先确认源 xlsx 的 Power 列是否真的有数值；如果大量为空，问题不在“清洗”，而在“数据本身不可用”。")

    print(f"\n[DONE] 输出已写入：{out_root}")
    print("请先打开 audit_report.csv 看每个序列的缺失率、空文件比例、异常值标记数量，再决定训练策略。")


if __name__ == "__main__":
    main()
