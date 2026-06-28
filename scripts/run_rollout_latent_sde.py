"""Generate trajectories from the prior latent SDE and save samples + a plot.

Demonstrates that C5 is a genuine generative SDE: integrating the prior
dz = h(t,z) dt + g(t,z) dW with a real solver yields sampled trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adsb_latent_sde.config import load_config, set_seed, ensure_dir
from adsb_latent_sde.inference import load_checkpoint
from adsb_latent_sde.rollout import sample_prior_trajectories
from adsb_latent_sde.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent_sde_default.yaml")
    parser.add_argument("--checkpoint", default="outputs/latent_sde/latent_sde.pt")
    parser.add_argument("--n-samples", type=int, default=12)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()
    model, _ = load_checkpoint(args.checkpoint, device)

    samples = sample_prior_trajectories(model, args.n_samples, device)  # (N, T, 4)
    out_dir = ensure_dir(config["output_dir"])
    np.save(out_dir / "prior_rollouts.npy", samples)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(samples.shape[0]):
        ax.plot(samples[i, :, 0], samples[i, :, 1], alpha=0.7, lw=1.0)
        ax.scatter(samples[i, 0, 0], samples[i, 0, 1], s=12, c="k", zorder=3)
    ax.set_xlabel("E (normalised)"); ax.set_ylabel("N (normalised)")
    ax.set_title("Prior latent-SDE trajectory samples")
    fig.tight_layout()
    fig.savefig(out_dir / "prior_rollouts.png", dpi=120)
    print(f"Saved {out_dir/'prior_rollouts.npy'} and prior_rollouts.png  shape={samples.shape}")


if __name__ == "__main__":
    main()
