# -*- coding: utf-8 -*-
"""
Radar charts for MultiBuildingMSPatchTST runs.

What it does:
- Read metrics.json under run_root for given resolutions (1_hour / 30_minutes / 5_minutes)
- Use TEST metrics (test_metrics -> per_building)
- Build radar plots comparing buildings across multiple metrics
- Normalize each metric to [0,1] across compared series
- Flip direction for error metrics so that "larger is better" on the radar

Outputs:
- radar_dual_<resA>_vs_<resB>.png/pdf
- radar_single_<res>.png/pdf

Usage (PowerShell one-liner):
python .\make_radar_multibuilding.py --run_root "C:\...\20260120_173154_f3d531c1" --resA 1_hour --resB 5_minutes --out_dir "paper_figs_custom"

Notes:
- If some metrics are missing in metrics.json, they will be skipped automatically.
- Radar is qualitative; always keep the numeric table (metrics.json) as the primary evidence.
"""

import argparse
import json
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RESOLUTIONS = ["1_hour", "30_minutes", "5_minutes"]
DEFAULT_BUILDINGS = ["Commercial", "Office", "Public", "Residential"]

# Metrics you *prefer* to show (will be filtered by availability in your metrics.json)
PREFERRED_METRICS = ["MAE", "RMSE", "MAPE", "SMAPE", "R2"]

# Metrics where smaller is better (will be inverted after normalization)
ERROR_METRICS = {"MAE", "RMSE", "MAPE", "SMAPE", "WAPE", "MSE"}


def read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_first(paths):
    paths = list(paths)
    if not paths:
        return None
    paths = sorted(paths, key=lambda x: len(str(x)))
    return paths[0]


def find_metrics_json(run_root: Path, resolution: str) -> Path | None:
    # Try common layouts
    cand = []
    cand += list(run_root.glob(f"{resolution}/Joint_*/metrics.json"))
    cand += list(run_root.glob(f"{resolution}/**/metrics.json"))
    return find_first(cand)


def extract_test_per_building(metrics_json: dict) -> dict:
    """
    Return dict:
      {building: {metric_name: value(float)}}
    Expected structure:
      metrics_json["test_metrics"]["per_building"][building][metric] = float
    """
    tm = metrics_json.get("test_metrics", {}) or {}
    pb = tm.get("per_building", {}) or {}
    out = {}
    for b, md in pb.items():
        if not isinstance(md, dict):
            continue
        out[b] = {}
        for k, v in md.items():
            try:
                out[b][str(k)] = float(v)
            except Exception:
                # skip non-numeric
                pass
    return out


def choose_metrics(per_building: dict, preferred: list[str]) -> list[str]:
    # Keep only metrics that exist for at least one building
    available = set()
    for b, md in per_building.items():
        available |= set(md.keys())
    chosen = [m for m in preferred if m in available]
    # If none found, fall back to whatever exists (stable order)
    if not chosen:
        chosen = sorted(list(available))
    return chosen


def normalize_series(series_to_vals: dict, metric_names: list[str]) -> dict:
    """
    series_to_vals: {series_name: {metric: raw_value}}
    Return normalized and direction-aligned:
      {series_name: [v0..vM-1] in [0,1], bigger is better}
    For ERROR_METRICS: normalize then invert (1 - x)
    For beneficial metrics (e.g., R2): normalize directly
    """
    # gather per-metric arrays
    norm = {s: [] for s in series_to_vals.keys()}
    eps = 1e-12

    for m in metric_names:
        raw = []
        keys = []
        for s, md in series_to_vals.items():
            if m in md and md[m] is not None and np.isfinite(md[m]):
                raw.append(float(md[m]))
            else:
                raw.append(np.nan)
            keys.append(s)

        raw_arr = np.array(raw, dtype=float)

        # If all NaN, set to 0.5 (should not happen if metric chosen properly)
        if np.all(np.isnan(raw_arr)):
            for s in keys:
                norm[s].append(0.5)
            continue

        # min-max ignoring NaN
        mn = np.nanmin(raw_arr)
        mx = np.nanmax(raw_arr)

        if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < eps:
            # Degenerate: all equal -> 0.5
            z = np.full_like(raw_arr, 0.5)
        else:
            z = (raw_arr - mn) / (mx - mn)

        # NaN -> 0.0 (conservative; missing metric should not look good)
        z = np.where(np.isnan(z), 0.0, z)

        # Flip direction for errors so larger is better
        if m in ERROR_METRICS:
            z = 1.0 - z

        for s, v in zip(keys, z.tolist()):
            norm[s].append(float(v))

    return norm


def radar_plot(ax, categories, series_norm: dict, title: str):
    """
    categories: list[str] axis labels
    series_norm: {name: [values]} in [0,1]
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], fontsize=8)
    ax.grid(True, alpha=0.3)

    for name, vals in series_norm.items():
        v = vals + vals[:1]
        ax.plot(angles, v, linewidth=1.8, label=name)
        ax.fill(angles, v, alpha=0.08)

    ax.set_title(title, y=1.08, fontsize=12)


def build_series_for_resolution(run_root: Path, resolution: str, buildings_order: list[str]):
    mp = find_metrics_json(run_root, resolution)
    if mp is None:
        raise FileNotFoundError(f"[{resolution}] metrics.json not found under: {run_root}")

    mj = read_json(mp)
    pb = extract_test_per_building(mj)

    # reorder / filter buildings
    if buildings_order:
        series_names = [b for b in buildings_order if b in pb]
        # include any extras
        series_names += [b for b in pb.keys() if b not in series_names]
    else:
        series_names = list(pb.keys())

    series_to_vals = {b: pb[b] for b in series_names}
    metrics = choose_metrics(series_to_vals, PREFERRED_METRICS)
    return series_to_vals, metrics, mp


def save_dual_radar(run_root: Path, resA: str, resB: str, out_dir: Path, buildings_order: list[str]):
    seriesA, metricsA, mpA = build_series_for_resolution(run_root, resA, buildings_order)
    seriesB, metricsB, mpB = build_series_for_resolution(run_root, resB, buildings_order)

    # Use intersection of metrics to make the two radars comparable
    metrics = [m for m in PREFERRED_METRICS if (m in metricsA and m in metricsB)]
    if not metrics:
        # fallback: union, but warn by title
        metrics = sorted(list(set(metricsA) | set(metricsB)))

    normA = normalize_series(seriesA, metrics)
    normB = normalize_series(seriesB, metrics)

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax2 = fig.add_subplot(1, 2, 2, polar=True)

    radar_plot(ax1, metrics, normA, f"{resA} Performance Comparison (Test)")
    radar_plot(ax2, metrics, normB, f"{resB} Performance Comparison (Test)")

    # Shared legend (right side)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=9)

    fig.tight_layout(rect=[0.10, 0.02, 1.0, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"radar_dual_{resA}_vs_{resB}.png"
    out_pdf = out_dir / f"radar_dual_{resA}_vs_{resB}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    # also dump the raw numbers used
    raw_dump = out_dir / f"radar_dual_{resA}_vs_{resB}_raw.json"
    raw_dump.write_text(json.dumps({
        "resA": resA, "metrics_json_A": str(mpA), "raw_A": seriesA,
        "resB": resB, "metrics_json_B": str(mpB), "raw_B": seriesB,
        "metrics_used": metrics,
        "normalization": "min-max per metric across compared series; error-metrics inverted so larger-is-better"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_png, out_pdf, raw_dump


def save_single_radar(run_root: Path, resolution: str, out_dir: Path, buildings_order: list[str]):
    series, metrics, mp = build_series_for_resolution(run_root, resolution, buildings_order)
    norm = normalize_series(series, metrics)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(1, 1, 1, polar=True)
    radar_plot(ax, metrics, norm, f"{resolution} Performance Comparison (Test)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, fontsize=9)

    fig.tight_layout(rect=[0.08, 0.02, 1.0, 0.98])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"radar_single_{resolution}.png"
    out_pdf = out_dir / f"radar_single_{resolution}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    raw_dump = out_dir / f"radar_single_{resolution}_raw.json"
    raw_dump.write_text(json.dumps({
        "resolution": resolution,
        "metrics_json": str(mp),
        "raw": series,
        "metrics_used": metrics,
        "normalization": "min-max per metric across compared series; error-metrics inverted so larger-is-better"
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_png, out_pdf, raw_dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=str, required=True,
                    help=r'Run root, e.g. C:\Users\...\TransformerRuns\MultiBuildingMSPatchTST\20260120_173154_f3d531c1')
    ap.add_argument("--out_dir", type=str, default="paper_figs_custom",
                    help="Output folder name (relative to run_root if not absolute).")
    ap.add_argument("--resA", type=str, default="1_hour", choices=DEFAULT_RESOLUTIONS,
                    help="Left radar resolution.")
    ap.add_argument("--resB", type=str, default="5_minutes", choices=DEFAULT_RESOLUTIONS,
                    help="Right radar resolution.")
    ap.add_argument("--single", type=str, default="",
                    help="If set, also output a single radar for this resolution (e.g., 30_minutes).")
    ap.add_argument("--buildings", type=str, default=",".join(DEFAULT_BUILDINGS),
                    help="Comma-separated building order for legend.")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = run_root / out_dir

    buildings_order = [x.strip() for x in args.buildings.split(",") if x.strip()]

    try:
        png, pdf, raw = save_dual_radar(run_root, args.resA, args.resB, out_dir, buildings_order)
        print(f"[OK] dual radar: {png}")
        print(f"[OK] dual radar pdf: {pdf}")
        print(f"[OK] raw dump: {raw}")
    except Exception as e:
        print(f"[FAIL] dual radar: {e}")

    if args.single:
        try:
            png, pdf, raw = save_single_radar(run_root, args.single, out_dir, buildings_order)
            print(f"[OK] single radar: {png}")
            print(f"[OK] single radar pdf: {pdf}")
            print(f"[OK] raw dump: {raw}")
        except Exception as e:
            print(f"[FAIL] single radar: {e}")


if __name__ == "__main__":
    main()
