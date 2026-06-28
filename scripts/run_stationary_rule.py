#!/usr/bin/env python
"""Calibrate and apply the stationary-clutter kinematic rule (data-driven,
model-independent). Closes the latent-SDE blind spot on near-stationary
trajectories, exactly as in C4."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adsb_latent_sde.config import ensure_dir, load_config
from adsb_latent_sde.kinematic_rules import (
    apply_stationary_rule,
    calibrate_stationary_thresholds,
    save_thresholds,
)
from adsb_latent_sde.reporting import print_summary_table, save_dataframe


def flag_rate(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent_sde_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(config["data_dir"])
    output_dir = ensure_dir(config["output_dir"])

    print("Loading data...")
    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")

    print(f"Calibrating stationary thresholds on {len(X_train)} train sequences...")
    thresholds = calibrate_stationary_thresholds(np.asarray(X_train, dtype=np.float32))
    save_thresholds(thresholds, output_dir / "stationary_thresholds.json")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")

    print("\nApplying to train/test...")
    train_feats = apply_stationary_rule(np.asarray(X_train, dtype=np.float32), thresholds)
    test_feats = apply_stationary_rule(np.asarray(X_test, dtype=np.float32), thresholds)
    save_dataframe(train_feats, output_dir / "train_stationary_features.csv")
    save_dataframe(test_feats, output_dir / "test_stationary_features.csv")

    summary = pd.DataFrame([
        {"split": s,
         "stationary_flag_rate": flag_rate(f, "stationary_flag"),
         "low_mean_speed_rate": flag_rate(f, "low_mean_speed"),
         "low_displacement_rate": flag_rate(f, "low_displacement"),
         "low_path_length_rate": flag_rate(f, "low_path_length"),
         "low_step_displacement_rate": flag_rate(f, "low_step_displacement")}
        for s, f in (("train", train_feats), ("test", test_feats))
    ])
    save_dataframe(summary, output_dir / "stationary_rule_summary.csv")
    print("\nStationary rule summary:")
    print_summary_table(summary)
    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()
