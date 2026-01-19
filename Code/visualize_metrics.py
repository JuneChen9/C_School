import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_get(d: Dict[str, Any], keys: List[str], default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def scan_metrics(runs_root: Path) -> pd.DataFrame:
    rows = []
    for mp in runs_root.rglob("metrics.json"):
        m = read_json(mp)
        if not m:
            continue
        resolution = m.get("resolution")
        building = m.get("building")
        model = m.get("model", "Unknown")
        if not resolution or not building:
            continue

        run_dir = mp.parent
        # Try infer run_tag: .../<model_name>/<run_tag>/<resolution>/<building>/metrics.json
        run_tag = ""
        try:
            parts = run_dir.parts
            if len(parts) >= 4 and parts[-2] == building and parts[-3] == resolution:
                run_tag = parts[-4]
        except Exception:
            run_tag = ""

        rows.append({
            "model": model,
            "run_tag": run_tag,
            "resolution": resolution,
            "building": building,
            "val_mae": safe_get(m, ["val_metrics", "MAE"]),
            "val_rmse": safe_get(m, ["val_metrics", "RMSE"]),
            "val_mape": safe_get(m, ["val_metrics", "MAPE"]),
            "test_mae": safe_get(m, ["test_metrics", "MAE"]),
            "test_rmse": safe_get(m, ["test_metrics", "RMSE"]),
            "test_mape": safe_get(m, ["test_metrics", "MAPE"]),
            "run_dir": str(run_dir),
            "metrics_path": str(mp),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["val_mae", "val_rmse", "val_mape", "test_mae", "test_rmse", "test_mape"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_selected(final_root: Path) -> pd.DataFrame:
    rows = []
    for mp in final_root.rglob("metrics.json"):
        m = read_json(mp)
        if not m:
            continue
        resolution = m.get("resolution")
        building = m.get("building")
        model = m.get("model", "Unknown")
        if not resolution or not building:
            continue

        # in FinalSelected/<res>/<bld>/metrics.json
        rows.append({
            "resolution": resolution,
            "building": building,
            "selected_model": model,
            "selected_metrics_path": str(mp),
            "selected_dir": str(mp.parent),
        })
    return pd.DataFrame(rows)


def ensure_out(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def plot_overview_heatmap(best_df: pd.DataFrame, metric: str, out_path: Path, title: str):
    # pivot: resolution x building
    pv = best_df.pivot(index="resolution", columns="building", values=metric)
    plt.figure(figsize=(10, 3 + 0.4 * len(pv.index)))
    # simple heatmap without seaborn
    data = pv.values.astype(float)
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.yticks(range(len(pv.index)), pv.index.tolist())
    plt.xticks(range(len(pv.columns)), pv.columns.tolist(), rotation=30, ha="right")
    plt.title(title)
    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_best_bar(best_df: pd.DataFrame, metric: str, out_path: Path, title: str):
    # bar: 12 tasks
    x = [f"{r}/{b}" for r, b in zip(best_df["resolution"], best_df["building"])]
    y = best_df[metric].astype(float).values
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(x)), y)
    plt.xticks(range(len(x)), x, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_model_compare(df_all: pd.DataFrame, selected_df: pd.DataFrame, metric: str, out_path: Path):
    """
    For each task, show scatter of runs colored by model, highlight selected run if it exists in all runs table.
    """
    # build lookup of selected (resolution, building) -> selected_dir or selected_metrics_path
    sel_keys = set(zip(selected_df["resolution"], selected_df["building"]))
    tasks = sorted(list(set(zip(df_all["resolution"], df_all["building"]))))

    n = len(tasks)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(14, 3.6 * rows))

    for idx, (res, bld) in enumerate(tasks, start=1):
        ax = plt.subplot(rows, cols, idx)
        sub = df_all[(df_all["resolution"] == res) & (df_all["building"] == bld)].copy()
        sub = sub.dropna(subset=[metric])
        if sub.empty:
            ax.set_title(f"{res}/{bld} (no data)")
            ax.axis("off")
            continue

        # x = run index sorted by metric
        sub = sub.sort_values(metric, ascending=True).reset_index(drop=True)
        x = np.arange(len(sub))
        y = sub[metric].values

        # color by model
        models = sub["model"].astype(str).values
        uniq = sorted(set(models))
        # map model -> marker
        markers = ["o", "s", "^", "D", "x", "*"]
        for mi, mname in enumerate(uniq):
            mask = (models == mname)
            ax.scatter(x[mask], y[mask], marker=markers[mi % len(markers)], label=mname, s=30)

        # highlight selected best within this task (if we can match by minimum metric among all)
        # Since FinalSelected copies metrics.json, simplest: mark the minimum y as "best in all runs"
        ax.scatter([0], [y[0]], s=120, facecolors='none', edgecolors='k', linewidths=2, label="best(all runs)")

        ax.set_title(f"{res}/{bld}")
        ax.set_xlabel("rank (lower is better)")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.2)

        if idx == 1:
            ax.legend(loc="best", fontsize=8)

    plt.suptitle(f"Per-task run ranking by {metric} (markers = model; circle outline = best among scanned runs)", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default=r"C:\Users\13470\Desktop\C_school\TransformerRuns")
    ap.add_argument("--final_root", type=str, default=r"C:\Users\13470\Desktop\C_school\FinalSelected")
    ap.add_argument("--out_dir", type=str, default=r"C:\Users\13470\Desktop\C_school\VizOut")
    ap.add_argument("--metric", type=str, default="val_mae",
                    help="val_mae/val_rmse/val_mape/test_mae/test_rmse/test_mape")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    final_root = Path(args.final_root)
    out_dir = Path(args.out_dir)
    ensure_out(out_dir)

    df_all = scan_metrics(runs_root)
    if df_all.empty:
        print(f"[ERROR] No metrics.json found under {runs_root}")
        return

    sel = load_selected(final_root)
    # Create "best_df" by taking minimal metric in df_all per task (this should match your select_best_runs result)
    metric = args.metric
    sub = df_all.dropna(subset=[metric]).copy()
    sub = sub.sort_values(["resolution", "building", metric], ascending=True)
    best_df = sub.groupby(["resolution", "building"], as_index=False).first()

    # Save tables
    df_all.to_csv(out_dir / "all_runs_metrics.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(out_dir / "best_by_task.csv", index=False, encoding="utf-8-sig")

    # Plots
    plot_best_bar(
        best_df, metric,
        out_dir / f"best_{metric}_bar.png",
        title=f"Best per task by {metric} (lower is better)"
    )
    plot_overview_heatmap(
        best_df, metric,
        out_dir / f"best_{metric}_heatmap.png",
        title=f"Best per task heatmap: {metric} (lower is better)"
    )
    plot_model_compare(
        df_all, sel, metric,
        out_dir / f"run_ranking_{metric}_per_task.png"
    )

    print(f"[OK] Saved CSV + figures to: {out_dir}")
    print(f"[OK] all_runs: {len(df_all)} | best_tasks: {len(best_df)} | metric: {metric}")


if __name__ == "__main__":
    main()

