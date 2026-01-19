# select_best_runs.py
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
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

        model = m.get("model", "Unknown")
        resolution = m.get("resolution")
        building = m.get("building")
        if not resolution or not building:
            # Skip malformed
            continue

        val_mae = safe_get(m, ["val_metrics", "MAE"])
        val_rmse = safe_get(m, ["val_metrics", "RMSE"])
        val_mape = safe_get(m, ["val_metrics", "MAPE"])
        test_mae = safe_get(m, ["test_metrics", "MAE"])
        test_rmse = safe_get(m, ["test_metrics", "RMSE"])
        test_mape = safe_get(m, ["test_metrics", "MAPE"])

        # run_dir is the folder containing metrics.json
        run_dir = mp.parent

        # infer run_tag: .../<ModelName>/<run_tag>/<resolution>/<building>/metrics.json
        # If structure doesn't match, keep empty
        parts = run_dir.parts
        run_tag = ""
        try:
            # find "...TransformerRuns/<something>/<maybe_run_tag>/<resolution>/<building>"
            # resolution and building are last two folders
            if len(parts) >= 3 and parts[-2] == building and parts[-3] == resolution:
                # candidate tag is parts[-4]
                if len(parts) >= 4:
                    run_tag = parts[-4]
        except Exception:
            run_tag = ""

        rows.append({
            "model": model,
            "run_tag": run_tag,
            "resolution": resolution,
            "building": building,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_mape": val_mape,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_mape": test_mape,
            "run_dir": str(run_dir),
            "metrics_path": str(mp),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # make numeric
    for c in ["val_mae", "val_rmse", "val_mape", "test_mae", "test_rmse", "test_mape"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pick_best(df: pd.DataFrame, key_metric: str) -> pd.DataFrame:
    """
    Pick best run per (resolution, building) by smallest key_metric.
    """
    if df.empty:
        return df

    if key_metric not in df.columns:
        raise ValueError(f"Unknown key_metric: {key_metric}. Available: {list(df.columns)}")

    sub = df.dropna(subset=[key_metric]).copy()
    if sub.empty:
        raise ValueError(f"No valid numbers for metric '{key_metric}' in scanned metrics.json")

    sub = sub.sort_values([ "resolution", "building", key_metric], ascending=[True, True, True])
    best = sub.groupby(["resolution", "building"], as_index=False).first()
    return best


def copy_selected(best_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, r in best_df.iterrows():
        res = r["resolution"]
        bld = r["building"]
        run_dir = Path(r["run_dir"])

        target = out_dir / res / bld
        target.mkdir(parents=True, exist_ok=True)

        # copy best.pt if exists, else fallback to checkpoint_last.pt
        src_best = run_dir / "best.pt"
        src_last = run_dir / "checkpoint_last.pt"
        if src_best.exists():
            shutil.copy2(src_best, target / "best.pt")
        elif src_last.exists():
            shutil.copy2(src_last, target / "checkpoint_last.pt")

        # copy metrics.json
        src_metrics = run_dir / "metrics.json"
        if src_metrics.exists():
            shutil.copy2(src_metrics, target / "metrics.json")

        # copy run_meta.json if present (run root is usually 2 levels up)
        # run_dir: .../<run_tag>/<res>/<bld>
        run_meta = run_dir.parent.parent / "run_meta.json"
        if run_meta.exists():
            shutil.copy2(run_meta, target / "run_meta.json")

        # also write a tiny pointer file
        (target / "SOURCE.txt").write_text(
            f"model={r['model']}\nrun_tag={r['run_tag']}\nrun_dir={run_dir}\n",
            encoding="utf-8"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default=r"C:\Users\13470\Desktop\C_school\TransformerRuns")
    ap.add_argument("--out_dir", type=str, default=r"C:\Users\13470\Desktop\C_school\FinalSelected")
    ap.add_argument("--metric", type=str, default="val_mae",
                    help="val_mae/val_rmse/val_mape/test_mae/test_rmse/test_mape (smaller is better)")
    ap.add_argument("--save_csv", action="store_true", help="save leaderboard.csv and best.csv")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)

    df = scan_metrics(runs_root)
    if df.empty:
        print(f"[ERROR] No metrics.json found under {runs_root}")
        return

    best = pick_best(df, args.metric)

    if args.save_csv:
        out_dir.mkdir(parents=True, exist_ok=True)
        df.sort_values(["resolution", "building", args.metric], ascending=True).to_csv(out_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
        best.to_csv(out_dir / "best.csv", index=False, encoding="utf-8-sig")

    copy_selected(best, out_dir)

    print(f"[OK] Scanned runs: {len(df)}")
    print(f"[OK] Selected best per (resolution, building): {len(best)}")
    print(f"[OK] Saved to: {out_dir}")
    print(f"[OK] Metric used: {args.metric}")


if __name__ == "__main__":
    main()
