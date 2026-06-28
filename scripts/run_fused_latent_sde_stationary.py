#!/usr/bin/env python
"""Fused latent-SDE + stationary-clutter rule evaluation across corruptions.

    flag_abnormal = (latent_sde total_nll > train_p99)  OR  stationary_rule
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from adsb_latent_sde.config import ensure_dir, load_config
from adsb_latent_sde import corruption as C
from adsb_latent_sde.inference import load_checkpoint, score_sequences
from adsb_latent_sde.kinematic_rules import apply_stationary_rule, load_thresholds
from adsb_latent_sde.reporting import print_summary_table, save_dataframe
from adsb_latent_sde.utils import get_device


CASES = {
    "clean": lambda x: x,
    "speed_scaled_1.5": lambda x: C.speed_scale(x, 1.5),
    "speed_scaled_2.0": lambda x: C.speed_scale(x, 2.0),
    "random_walk_velocity": lambda x: C.random_walk_velocity(x),
    "sudden_turn_90": lambda x: C.sudden_turn_90(x),
    "position_jump": lambda x: C.position_jump(x, 2.0),
    "stationary_clutter": lambda x: C.stationary_clutter(x),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent_sde_default.yaml")
    parser.add_argument("--checkpoint", default="outputs/latent_sde/latent_sde.pt")
    parser.add_argument("--score-name", default="total_nll")
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    out_dir = ensure_dir(config["output_dir"])
    model, _ = load_checkpoint(args.checkpoint, device)

    thr_df = pd.read_csv(out_dir / "latent_sde_thresholds.csv")
    row = thr_df[(thr_df["score_name"] == args.score_name) & (thr_df["quantile"] == args.quantile)]
    if row.empty:
        sys.exit(f"No threshold for {args.score_name} q={args.quantile}; run scoring first.")
    sde_threshold = float(row["threshold"].iloc[0])

    stat_thresholds = load_thresholds(out_dir / "stationary_thresholds.json")

    X_test = np.load(Path(config["data_dir"]) / config["test_file"], mmap_mode="r")
    n = min(args.max_samples, len(X_test))
    rng = np.random.default_rng(config["seed"])
    idx = np.sort(rng.choice(len(X_test), n, replace=False))
    base = np.asarray(X_test[idx], dtype=np.float32)
    bs, k = config["batch_size"], config["num_score_samples"]
    print(f"SDE threshold {args.score_name} p{int(args.quantile*100)} = {sde_threshold:.4f}; n={n}")

    rows = []
    for name, fn in CASES.items():
        print(f"Evaluating {name}...")
        xc = fn(base)
        sde_flag = score_sequences(model, xc, bs, device, num_samples=k,
                                   seed=config["seed"])[args.score_name].values > sde_threshold
        stat_flag = apply_stationary_rule(xc, stat_thresholds)["stationary_flag"].values
        fused = sde_flag | stat_flag
        rows.append({
            "case": name, "score_name": args.score_name, "quantile": args.quantile,
            "sde_detection_rate": float(sde_flag.mean()),
            "stationary_detection_rate": float(stat_flag.mean()),
            "fused_detection_rate": float(fused.mean()),
            "stationary_adds_pp": float(fused.mean() - sde_flag.mean()),
        })

    summary = pd.DataFrame(rows)
    out_path = out_dir / "fused_latent_sde_stationary_summary.csv"
    save_dataframe(summary, out_path)
    print(f"\n{'='*72}\nFUSED LATENT-SDE + STATIONARY RULE  |  {args.score_name}  p{int(args.quantile*100)}\n{'='*72}")
    disp = summary[["case", "sde_detection_rate", "stationary_detection_rate",
                    "fused_detection_rate", "stationary_adds_pp"]].copy()
    disp.columns = ["case", f"SDE p{int(args.quantile*100)}", "stationary", "fused", "+pp"]
    print_summary_table(disp)
    print(f"{'='*72}\nSaved to {out_path}")


if __name__ == "__main__":
    main()
