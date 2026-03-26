#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def sanitize_filename(name: str) -> str:
    """Convert metric names to safe filenames."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def parse_log(log_path: Path):
    """
    Parse JSONL log file where each line is a JSON dict.
    Returns:
      epochs: list[float]
      metric_series: dict[str, list[float]]
    """
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                print(f"[warn] skipping non-JSON line {i}")

    if not rows:
        raise ValueError(f"No valid JSON rows found in {log_path}")

    # Keep rows that have an epoch value and at least one key.
    rows = [r for r in rows if "epoch" in r]
    if not rows:
        raise ValueError("No rows with 'epoch' found.")

    # Collect all keys that look like losses (case-insensitive).
    loss_keys = sorted(
        {
            k
            for r in rows
            for k in r.keys()
            if "loss" in k.lower() and isinstance(r.get(k, None), (int, float))
        }
    )
    if not loss_keys:
        raise ValueError("No loss-like keys found (keys containing 'loss').")

    epochs = [float(r["epoch"]) for r in rows]
    metric_series = {}
    for k in loss_keys:
        ys = []
        for r in rows:
            v = r.get(k, None)
            ys.append(float(v) if isinstance(v, (int, float)) else float("nan"))
        metric_series[k] = ys

    return epochs, metric_series


def plot_individual_losses(epochs, metric_series, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, ys in metric_series.items():
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, ys, linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(key)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out_file = out_dir / f"{sanitize_filename(key)}.png"
        plt.savefig(out_file, dpi=160)
        plt.close()


def plot_all_losses(epochs, metric_series, out_file: Path):
    plt.figure(figsize=(12, 7))
    for key, ys in metric_series.items():
        plt.plot(epochs, ys, linewidth=1.3, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.title("All losses")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot all loss metrics from JSONL training logs."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to log file (JSON per line).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for individual loss plots. "
        "Default: <log_dir>/loss_plots",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Disable combined plot with all losses together.",
    )

    args = parser.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_dir = args.out_dir or (log_path.parent / "loss_plots")
    epochs, metric_series = parse_log(log_path)

    plot_individual_losses(epochs, metric_series, out_dir)
    print(f"[ok] Saved {len(metric_series)} individual plots to: {out_dir}")

    if not args.no_combined:
        combined = out_dir / "all_losses.png"
        plot_all_losses(epochs, metric_series, combined)
        print(f"[ok] Saved combined plot: {combined}")


if __name__ == "__main__":
    main()