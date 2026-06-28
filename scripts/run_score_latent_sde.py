"""Score train/test sequences with the latent SDE; write anomaly-score
thresholds (negative-ELBO quantiles) and test false-flag rates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adsb_latent_sde.config import load_config, set_seed, ensure_dir
from adsb_latent_sde.inference import load_checkpoint, score_sequences
from adsb_latent_sde.reporting import save_dataframe, print_summary_table
from adsb_latent_sde.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent_sde_default.yaml")
    parser.add_argument("--checkpoint", default="outputs/latent_sde/latent_sde.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    data_dir = Path(config["data_dir"])
    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")
    bs = config["batch_size"]
    k = config["num_score_samples"]

    print(f"Scoring train ({len(X_train)}) ...")
    train_scores = score_sequences(model, X_train, bs, device, num_samples=k, seed=config["seed"])
    print(f"Scoring test  ({len(X_test)}) ...")
    test_scores = score_sequences(model, X_test, bs, device, num_samples=k, seed=config["seed"])

    out_dir = ensure_dir(config["output_dir"])
    save_dataframe(train_scores, out_dir / "train_latent_sde_scores.csv")
    save_dataframe(test_scores, out_dir / "test_latent_sde_scores.csv")

    rows = []
    for col in config["score_columns"]:
        for q in config["threshold_quantiles"]:
            thr = float(np.quantile(train_scores[col].values, q))
            test_rate = float((test_scores[col].values > thr).mean())
            rows.append({"score_name": col, "quantile": q, "threshold": thr,
                         "test_false_flag_rate": test_rate})
    thresholds = pd.DataFrame(rows)
    save_dataframe(thresholds, out_dir / "latent_sde_thresholds.csv")
    print_summary_table(thresholds)
    print(f"\nSaved scores + thresholds to {out_dir}")


if __name__ == "__main__":
    main()
