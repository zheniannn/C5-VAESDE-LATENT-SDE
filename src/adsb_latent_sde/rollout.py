from __future__ import annotations

import numpy as np
import torch

from .model import LatentSDEModel


@torch.no_grad()
def sample_prior_trajectories(
    model: LatentSDEModel,
    n_samples: int,
    device: torch.device,
    steps: int | None = None,
) -> np.ndarray:
    """Draw trajectories from the *prior* latent SDE and decode to observations.

    Demonstrates that C5 is generative: integrating dz = h(t,z) dt + g(t,z) dW
    with a real solver produces sampled trajectories. Returns (n_samples, T, 4).
    """
    model.eval()
    if steps is None:
        ts = None
    else:
        ts = torch.linspace(0.0, 1.0, steps, device=device)
    obs = model.sample_prior(n_samples, device, ts=ts)
    return obs.cpu().numpy()
