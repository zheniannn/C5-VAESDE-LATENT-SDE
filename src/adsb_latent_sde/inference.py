from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .model import LatentSDEModel, build_model


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[LatentSDEModel, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


@torch.no_grad()
def score_sequences(
    model: LatentSDEModel,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    num_samples: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Per-sequence anomaly scores from the latent SDE.

    The main score `total_nll` is the negative ELBO (Monte-Carlo averaged over
    `num_samples` latent draws): high = the trajectory is poorly explained by
    the learned stochastic dynamics. Also returns the reconstruction term
    `recon_nll` (= -E[log p(x|z)]) and the KL `kl`.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    n = len(X)
    neg_elbo = np.zeros(n, dtype=np.float64)
    recon_nll = np.zeros(n, dtype=np.float64)
    kl_arr = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        xb = np.asarray(X[start:start + batch_size], dtype=np.float32)
        x = torch.from_numpy(xb).to(device)
        B = x.size(0)
        acc_elbo = torch.zeros(B, device=device)
        acc_ll = torch.zeros(B, device=device)
        acc_kl = torch.zeros(B, device=device)
        for _ in range(num_samples):
            eps = torch.randn(B, model.latent_dim, device=device, generator=gen)
            out = model(x, eps=eps)
            acc_elbo += out["elbo"]
            acc_ll += out["ll"]
            acc_kl += out["kl"]
        acc_elbo /= num_samples
        acc_ll /= num_samples
        acc_kl /= num_samples
        neg_elbo[start:start + B] = (-acc_elbo).cpu().numpy()
        recon_nll[start:start + B] = (-acc_ll).cpu().numpy()
        kl_arr[start:start + B] = acc_kl.cpu().numpy()

    return pd.DataFrame({"total_nll": neg_elbo, "recon_nll": recon_nll, "kl": kl_arr})
