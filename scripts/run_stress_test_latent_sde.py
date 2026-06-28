"""Latent-SDE anomaly-detection stress test across synthetic corruptions.

A sequence is flagged if its score exceeds the train-derived threshold at the
chosen quantile. Reports detection rate per corruption type."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adsb_latent_sde.config import load_config, set_seed, ensure_dir
from adsb_latent_sde.inference import load_checkpoint, score_sequences
from adsb_latent_sde.reporting import save_dataframe, print_summary_table
from adsb_latent_sde.utils import get_device
from adsb_latent_sde import corruption as C


CORRUPTIONS = {
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
    set_seed(config["seed"])
    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)
    out_dir = ensure_dir(config["output_dir"])

    thr_df = pd.read_csv(out_dir / "latent_sde_thresholds.csv")
    sel = thr_df[(thr_df["score_name"] == args.score_name) & (thr_df["quantile"] == args.quantile)]
    if sel.empty:
        raise SystemExit(f"No threshold for {args.score_name} q={args.quantile}; run scoring first.")
    threshold = float(sel["threshold"].iloc[0])
    print(f"Threshold {args.score_name} p{int(args.quantile*100)} = {threshold:.4f}")

    X_test = np.load(Path(config["data_dir"]) / config["test_file"], mmap_mode="r")
    n = min(args.max_samples, len(X_test))
    rng = np.random.default_rng(config["seed"])
    idx = np.sort(rng.choice(len(X_test), n, replace=False))
    base = np.asarray(X_test[idx], dtype=np.float32)
    print(f"Using {n} test sequences")

    bs, k = config["batch_size"], config["num_score_samples"]
    rows = []
    for name, fn in CORRUPTIONS.items():
        print(f"Evaluating {name} ...")
        xc = fn(base)
        scores = score_sequences(model, xc, bs, device, num_samples=k, seed=config["seed"])
        det = float((scores[args.score_name].values > threshold).mean())
        rows.append({
            "case": name, "score_name": args.score_name, "quantile": args.quantile,
            "detect_rate": det,
            "mean_score": float(scores[args.score_name].mean()),
            "p95_score": float(np.quantile(scores[args.score_name].values, 0.95)),
        })

    summary = pd.DataFrame(rows)
    out_path = out_dir / "latent_sde_stress_summary.csv"
    # merge across (score_name, quantile) like C4 so multiple runs accumulate
    if out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[~((existing["score_name"] == args.score_name) &
                              (existing["quantile"] == args.quantile))]
        summary = pd.concat([existing, summary], ignore_index=True)
    save_dataframe(summary, out_path)
    print()
    print_summary_table(summary[(summary["score_name"] == args.score_name) &
                                (summary["quantile"] == args.quantile)])
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
