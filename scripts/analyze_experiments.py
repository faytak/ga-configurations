#!/usr/bin/env python3
"""
Unified analyzer for experiment logs (all datasets/models).

Features:
- Auto-detects and reads metrics from TensorBoard event files or CSV logs
- Works for any experiment run created by train.py (naming: "{dataset}_{model}_nsm{num}_nst{steps}_bs{bs}")
- Aggregates across all Lightning "version_*" subruns under each run directory
- Computes statistics over full metric series and final values per version
- Saves a tidy CSV with one row per (run, metric) including stats
"""

import os
import re
import glob
import argparse
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except Exception:
    EventAccumulator = None  # type: ignore
    TENSORBOARD_AVAILABLE = False


def list_run_dirs(log_dir: str) -> List[str]:
    """Return all run directory names directly under log_dir."""
    if not os.path.isdir(log_dir):
        return []
    entries = []
    for name in os.listdir(log_dir):
        full = os.path.join(log_dir, name)
        if os.path.isdir(full):
            entries.append(name)
    return sorted(entries)


def parse_run_name(run_name: str) -> Dict[str, Any]:
    """
    Parse a run directory name into components if it matches the common pattern:
      "{dataset}_{model}_nsm{num_samples}_nst{max_steps}_bs{batch_size}"

    Returns a dict with optional keys: dataset, model, num_samples, max_steps, batch_size.
    If parsing fails, returns an empty dict.
    """
    # Match trailing hyperparameters regardless of underscores in model
    m = re.match(r"^(?P<prefix>.+?)_nsm(?P<num_samples>\d+)_nst(?P<max_steps>\d+)_bs(?P<batch_size>\d+)$", run_name)
    dataset_options = ["convex_hull", "nbody", "on_regression", "top_tagging"]
    model_options = ["convex_hull_glgmlp", "nbody_glgenn_gnn", "on_glg", "o5_glgmlp", "lorentz_cggnn"]
    
    if not m:
        return {}
    prefix = m.group("prefix")
    dataset = None
    model = None
    if "_" in prefix:
        # dataset, model = prefix.split("_", 1)
        for dataset_option in dataset_options:
            if dataset_option in run_name:
                dataset = dataset_option
        for model_option in model_options:
            if model_option in run_name:
                model = model_option
    else:
        dataset = prefix
        model = None
    return {
        "dataset": dataset,
        "model": model,
        "num_samples": int(m.group("num_samples")),
        "max_steps": int(m.group("max_steps")),
        "batch_size": int(m.group("batch_size")),
    }


def list_version_dirs(run_path: str) -> List[str]:
    """List Lightning version directories under a run path."""
    return sorted(glob.glob(os.path.join(run_path, "version_*")))


def extract_metrics_from_tensorboard_version(version_dir: str) -> Dict[str, List[float]]:
    """Extract all scalar metrics from a TensorBoard version directory."""
    if not TENSORBOARD_AVAILABLE:
        return {}
    if not os.path.isdir(version_dir):
        return {}
    try:
        # EventAccumulator can take a directory and aggregate all event files within
        ea = EventAccumulator(version_dir)
        ea.Reload()
        metrics: Dict[str, List[float]] = {}
        scalar_tags = ea.Tags().get("scalars", [])
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            values = [event.value for event in scalar_events]
            if values:
                metrics[tag] = values
        return metrics
    except Exception:
        return {}


def extract_metrics_from_csv_version(version_dir: str) -> Dict[str, List[float]]:
    """Extract metrics from a CSVLogger metrics.csv file in a version directory."""
    csv_path = os.path.join(version_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        metrics: Dict[str, List[float]] = {}
        for column in df.columns:
            if column in ("epoch", "step"):
                continue
            series = df[column]
            if np.issubdtype(series.dtype, np.number):
                values = series.dropna().astype(float).tolist()
                if values:
                    metrics[column] = values
        return metrics
    except Exception:
        return {}


def merge_metrics(into: Dict[str, List[float]], add: Dict[str, List[float]]):
    for key, vals in add.items():
        if not vals:
            continue
        if key not in into:
            into[key] = []
        into[key].extend(vals)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "count": 0}
    arr = np.array(values, dtype=float)
    count = int(arr.size)
    if count <= 1:
        std_val = 0.0
    else:
        std_val = float(np.std(arr, ddof=1))
    return {
        "mean": float(np.mean(arr)),
        "std": std_val,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": count,
    }


def analyze_experiments(
    log_dir: str = "lightning_logs",
    output_file: Optional[str] = None,
    source: str = "auto",  # auto | tb | csv
    include_regex: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
):
    print(f"Analyzing experiments in: {log_dir}")
    run_names = list_run_dirs(log_dir)
    if include_regex:
        try:
            rx = re.compile(include_regex)
            run_names = [r for r in run_names if rx.search(r)]
        except re.error as e:
            print(f"Warning: invalid include_regex '{include_regex}': {e}")

    if not run_names:
        print("No run directories found.")
        return

    results: List[Dict[str, Any]] = []

    for run_name in run_names:
        params = parse_run_name(run_name)
        if dataset_filter and params.get("dataset") != dataset_filter:
            continue
        if model_filter and params.get("model") != model_filter:
            continue


        run_path = os.path.join(log_dir, run_name)
        versions = list_version_dirs(run_path)
        if not versions:
            # Allow runs with metrics directly under run_path (rare)
            versions = [run_path]

        print(f"- {run_name}: {len(versions)} version(s)")

        # Aggregate across versions
        all_metrics: Dict[str, List[float]] = {}
        final_values_by_metric: Dict[str, List[float]] = {}

        for version_dir in versions:
            metrics_this: Dict[str, List[float]] = {}

            tried_tb = tried_csv = False
            if source in ("auto", "tb"):
                tried_tb = True
                metrics_this = extract_metrics_from_tensorboard_version(version_dir)
            if (not metrics_this) and source in ("auto", "csv"):
                tried_csv = True
                metrics_this = extract_metrics_from_csv_version(version_dir)

            if not metrics_this:
                # If both tried and failed, skip this version silently
                continue

            # Merge full series
            merge_metrics(all_metrics, metrics_this)

            # Capture final values per version
            for metric_name, values in metrics_this.items():
                if not values:
                    continue
                if metric_name not in final_values_by_metric:
                    final_values_by_metric[metric_name] = []
                final_values_by_metric[metric_name].append(values[-1])

        if not all_metrics:
            print(f"  Warning: no metrics found for {run_name}")
            continue

        # Compute stats per metric
        for metric_name, values in all_metrics.items():
            if not values:
                continue
            stats_all = calculate_statistics(values)
            finals = final_values_by_metric.get(metric_name, [])
            print(f"- {metric_name}: {finals}")
            stats_final = calculate_statistics(finals)

            record: Dict[str, Any] = {
                "run_name": run_name,
                "metric": metric_name,
                "all_values_mean": stats_all["mean"],
                "all_values_std": stats_all["std"],
                "all_values_min": stats_all["min"],
                "all_values_max": stats_all["max"],
                "all_values_count": stats_all["count"],
                "final_values_mean": stats_final["mean"],
                "final_values_std": stats_final["std"],
                "final_values_min": stats_final["min"],
                "final_values_max": stats_final["max"],
                "final_values_count": stats_final["count"],
                "num_versions": len(versions),
            }
            # Attach parsed params if available
            record.update({
                "dataset": params.get("dataset"),
                "model": params.get("model"),
                "num_samples": params.get("num_samples"),
                "max_steps": params.get("max_steps"),
                "batch_size": params.get("batch_size"),
            })
            results.append(record)

    if not results:
        print("No results to report.")
        return

    df = pd.DataFrame(results)

    # Optional save
    # if output_file:
    #     out_dir = os.path.dirname(output_file)
    #     if out_dir and not os.path.exists(out_dir):
    #         os.makedirs(out_dir, exist_ok=True)
    #     df.to_csv(output_file, index=False)
    #     print(f"Saved results to: {output_file}")

    # Print brief summary of key metrics if present
    key_metrics = [
        "train/loss",
        "val/loss",
        "test/loss",
        "val/accuracy",
        "test/accuracy",
    ]
    present = [m for m in key_metrics if m in set(df["metric"]) ]
    if present:
        print("\nSummary of final values (mean ± std):")
        for run_name in sorted(df["run_name"].unique()):
            sub = df[df["run_name"] == run_name]
            # Prefer known key metrics
            lines = []
            for m in present:
                rows = sub[sub["metric"] == m]
                if not rows.empty:
                    r = rows.iloc[0]
                    lines.append(f"  {m}: {r['final_values_mean']:.6f} ± {r['final_values_std']:.6f}")
            if lines:
                print(f"{run_name}")
                for ln in lines:
                    print(ln)


def main():
    parser = argparse.ArgumentParser(description="Unified experiment analyzer (TensorBoard/CSV)")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Directory with run subfolders")
    parser.add_argument("--output", type=str, default="src/experiments_summary.csv", help="Output CSV path")
    parser.add_argument("--source", type=str, default="auto", choices=["auto", "tb", "csv"], help="Data source preference")
    parser.add_argument("--include_regex", type=str, default=None, help="Only analyze run names matching this regex")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset name")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")

    args = parser.parse_args()

    analyze_experiments(
        log_dir=args.log_dir,
        output_file=args.output,
        source=args.source,
        include_regex=args.include_regex,
        dataset_filter=args.dataset,
        model_filter=args.model,
    )


if __name__ == "__main__":
    main()


