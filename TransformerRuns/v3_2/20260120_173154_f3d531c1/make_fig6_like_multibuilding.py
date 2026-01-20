# -*- coding: utf-8 -*-
"""
Generate Fig.6-like 2x2 performance analysis figures for MultiBuilding runs.

It uses EXISTING artifacts in your run directory:
- metrics.json (for stable test metrics across resolutions)
- preds/**/test_samples.npz (for distribution / horizon-profile / seasonal profile)

NPZ format is produced by train_multibuilding_mspatchtst_v3.py:
  past_time, future_time, past_true, future_true, future_pred, indices, buildings
(see saving code in your trainer):contentReference[oaicite:2]{index=2}

NOTE (important):
If export_k is small (e.g., 64), violin/seasonal curves will be noisy.
For paper-grade smoothness, export more windows (>= 512, ideally 2048+).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STEP_MINUTES = {"1_hour": 60, "30_minutes": 30, "5_minutes": 5}
DEFAULT_RESOLUTIONS = ["1_hour", "30_minutes", "5_minutes"]
DEFAULT_BUILDINGS = ["Commercial", "Office", "Public", "Residential"]


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first(globs):
    for p in globs:
        hits = list(p)
        if hits:
            # prefer shortest path (usually the most direct match)
            hits = sorted(hits, key=lambda x: len(str(x)))
            return hits[0]
    return None


def find_metrics_json(run_root: Path, resolution: str) -> Path | None:
    # Common layouts:
    #  run_root/<resolution>/Joint_.../metrics.json
    #  run_root/<resolution>/**/metrics.json
    return find_first([
        run_root.glob(f"{resolution}/Joint_*/metrics.json"),
        run_root.glob(f"{resolution}/**/metrics.json"),
    ])


def find_test_npz(run_root: Path, resolution: str) -> Path | None:
    # Common layouts:
    #  run_root/<resolution>/Joint_.../preds/test/test_samples.npz   (v3 latest)
    #  run_root/<resolution>/Joint_.../preds/test_samples.npz        (older)
    return find_first([
        run_root.glob(f"{resolution}/Joint_*/preds/test/test_samples.npz"),
        run_root.glob(f"{resolution}/Joint_*/preds/test_samples.npz"),
        run_root.glob(f"{resolution}/**/test_samples.npz"),
    ])


def load_test_samples(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    # expected keys:
    # past_time: [K,in]
    # future_time: [K,out]
    # past_true: [K,in,N]
    # future_true: [K,out,N]
    # future_pred: [K,out,N]
    out = {k: d[k] for k in d.files}
    # buildings could be stored as object array
    if "buildings" in out:
        out["buildings"] = [str(x) for x in out["buildings"].tolist()]
    return out


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = np.maximum(np.abs(y_true), eps)
    return 100.0 * np.abs((y_true - y_pred) / denom)


def compute_window_mape(samples: dict):
    """
    Return:
      win_mape_bn: [K,N]  each window's mean MAPE over horizon (per building)
      hor_mape_pn: [P,N]  horizon-step mean MAPE over windows (per building)
      overall_win: [K]    each window's mean MAPE over horizon+buildings
      month: [K]          month index (1..12) from the first future timestamp
      buildings: list[str]
    """
    y_t = samples["future_true"].astype(np.float64)   # [K,P,N]
    y_p = samples["future_pred"].astype(np.float64)   # [K,P,N]
    K, P, N = y_t.shape

    buildings = samples.get("buildings", None)
    if not buildings:
        buildings = [f"B{i}" for i in range(N)]

    e = mape_percent(y_t, y_p)  # [K,P,N]
    win_mape_bn = e.mean(axis=1)       # [K,N]
    hor_mape_pn = e.mean(axis=0)       # [P,N]
    overall_win = e.mean(axis=(1, 2))  # [K]

    ft = samples["future_time"]
    # ft: [K,P] datetime64[ns]
    ft0 = pd.to_datetime(ft[:, 0])
    month = ft0.month.values.astype(int)

    return win_mape_bn, hor_mape_pn, overall_win, month, buildings


def panel_a_violin(ax, win_mape_bn: np.ndarray, buildings: list[str], title: str):
    data = [win_mape_bn[:, i] for i in range(win_mape_bn.shape[1])]
    vp = ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(np.arange(1, len(buildings) + 1))
    ax.set_xticklabels(buildings, rotation=0)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)


def panel_b_horizon(ax, hor_mape_pn: np.ndarray, buildings: list[str], resolution: str, title: str):
    P, N = hor_mape_pn.shape
    step_min = STEP_MINUTES.get(resolution, 60)
    x = (np.arange(P) + 1) * (step_min / 60.0)  # hours ahead

    for i in range(N):
        ax.plot(x, hor_mape_pn[:, i], marker="o", linewidth=1.5, markersize=3, label=buildings[i])

    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)


def panel_c_resolution_bar(ax, res_to_test_mape: dict, buildings_order: list[str], title: str):
    """
    res_to_test_mape: {resolution: {building: MAPE (0..1)}} from metrics.json
    """
    resolutions = [r for r in DEFAULT_RESOLUTIONS if r in res_to_test_mape]
    if not resolutions:
        ax.text(0.5, 0.5, "No metrics.json found for resolution comparison", ha="center", va="center")
        ax.set_axis_off()
        return

    x = np.arange(len(resolutions))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(buildings_order))

    for j, b in enumerate(buildings_order):
        ys = []
        for r in resolutions:
            v = res_to_test_mape[r].get(b, np.nan)
            ys.append(100.0 * v if np.isfinite(v) else np.nan)
        ax.bar(x + offsets[j], ys, width=width, label=b)

    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_ylabel("Test MAPE (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)


def panel_d_monthly(ax, overall_win: np.ndarray, month: np.ndarray, title: str):
    df = pd.DataFrame({"mape": overall_win, "month": month})
    g = df.groupby("month")["mape"]
    mean = g.mean()
    std = g.std().fillna(0.0)
    cnt = g.count().astype(float)

    xs = np.arange(1, 13)
    ys = np.array([mean.get(m, np.nan) for m in xs], dtype=float)
    ss = np.array([std.get(m, 0.0) for m in xs], dtype=float)
    nn = np.array([cnt.get(m, 0.0) for m in xs], dtype=float)

    # 95% CI (if n>1); else just std band
    ci = np.zeros_like(ss)
    mask = nn > 1
    ci[mask] = 1.96 * ss[mask] / np.sqrt(nn[mask])

    ax.plot(xs, ys, marker="o", linewidth=1.8)
    ax.fill_between(xs, ys - ci, ys + ci, alpha=0.2)

    ax.set_xticks(xs)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def make_fig6_like_for_resolution(run_root: Path, resolution: str, out_dir: Path, buildings_order: list[str]):
    metrics_path = find_metrics_json(run_root, resolution)
    npz_path = find_test_npz(run_root, resolution)

    if npz_path is None:
        raise FileNotFoundError(f"[{resolution}] test_samples.npz not found under {run_root}")

    samples = load_test_samples(npz_path)
    win_mape_bn, hor_mape_pn, overall_win, month, buildings = compute_window_mape(samples)

    # reorder buildings to stable order (if possible)
    idx = list(range(len(buildings)))
    if buildings_order:
        name_to_i = {n: i for i, n in enumerate(buildings)}
        idx = [name_to_i[n] for n in buildings_order if n in name_to_i]
        # append any remaining
        idx += [i for i in range(len(buildings)) if i not in idx]
    buildings = [buildings[i] for i in idx]
    win_mape_bn = win_mape_bn[:, idx]
    hor_mape_pn = hor_mape_pn[:, idx]

    # resolution title info
    pred_len = samples["future_true"].shape[1]
    step_min = STEP_MINUTES.get(resolution, 60)
    horizon_hours = pred_len * step_min / 60.0

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    panel_a_violin(axA, win_mape_bn, buildings, "(A) Error Distribution by Building")
    panel_b_horizon(axB, hor_mape_pn, buildings, resolution, f"(B) Error vs Horizon (up to {horizon_hours:.1f}h)")
    axC.set_axis_off()  # filled by cross-resolution figure in a separate function
    panel_d_monthly(axD, overall_win, month, "(D) Seasonal Performance Variation (by month)")

    fig.suptitle(f"Building-specific Performance Analysis ({resolution})", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"Fig6_like_{resolution}.png"
    out_pdf = out_dir / f"Fig6_like_{resolution}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    # return for later use (optional)
    return {
        "resolution": resolution,
        "npz": str(npz_path),
        "metrics": str(metrics_path) if metrics_path else None,
        "out_png": str(out_png),
        "out_pdf": str(out_pdf),
        "export_k": int(samples["future_true"].shape[0]),
    }


def make_cross_resolution_figure(run_root: Path, out_dir: Path, buildings_order: list[str]):
    # read metrics.json for each resolution (stable, full-test numbers)
    res_to_test_mape = {}
    for r in DEFAULT_RESOLUTIONS:
        mp = find_metrics_json(run_root, r)
        if not mp:
            continue
        mj = read_json(mp)
        tm = (mj.get("test_metrics", {}) or {}).get("per_building", {}) or {}
        res_to_test_mape[r] = {b: float(tm.get(b, {}).get("MAPE", np.nan)) for b in buildings_order}

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    panel_c_resolution_bar(ax, res_to_test_mape, buildings_order, "Performance Across Resolutions (Test MAPE)")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "Fig_resolution_comparison.png"
    out_pdf = out_dir / "Fig_resolution_comparison.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    return {"out_png": str(out_png), "out_pdf": str(out_pdf)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True,
                    help=r'Run root, e.g. C:\Users\13470\Desktop\C_school\TransformerRuns\MultiBuildingMSPatchTST\20260120_173154_f3d531c1')
    ap.add_argument("--out_dir", type=str, default="paper_figs_custom",
                    help="Where to save generated figures (relative to run_root if not absolute).")
    ap.add_argument("--resolutions", type=str, default=",".join(DEFAULT_RESOLUTIONS),
                    help="Comma separated list, e.g. 1_hour,30_minutes,5_minutes")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = run_root / out_dir

    resolutions = [x.strip() for x in args.resolutions.split(",") if x.strip()]

    results = []
    for r in resolutions:
        try:
            info = make_fig6_like_for_resolution(run_root, r, out_dir, DEFAULT_BUILDINGS)
            results.append(info)
            print(f"[OK] {r}: {info['out_png']} (export_k={info['export_k']})")
        except Exception as e:
            print(f"[FAIL] {r}: {e}")

    try:
        info = make_cross_resolution_figure(run_root, out_dir, DEFAULT_BUILDINGS)
        print(f"[OK] cross-resolution: {info['out_png']}")
    except Exception as e:
        print(f"[FAIL] cross-resolution: {e}")

    # write a ready-to-paste caption snippet
    cap = out_dir / "caption_Fig6_like.txt"
    cap.write_text(
        "Figure X. Building-specific performance analysis. "
        "(A) Error distribution by building (violin: per-window mean MAPE). "
        "(B) Forecast error versus horizon (mean MAPE at each step). "
        "(C) Performance across temporal resolutions (test MAPE from metrics.json). "
        "(D) Seasonal performance variation (monthly mean MAPE with 95% CI).\n",
        encoding="utf-8",
    )
    print(f"[OK] caption: {cap}")


if __name__ == "__main__":
    main()
