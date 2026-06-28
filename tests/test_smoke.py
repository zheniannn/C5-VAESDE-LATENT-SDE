"""Smoke tests for the latent SDE — no real data required."""

import numpy as np
import torch

from adsb_latent_sde.model import build_model
from adsb_latent_sde.dataset import SequenceDataset
from adsb_latent_sde.inference import score_sequences
from adsb_latent_sde.rollout import sample_prior_trajectories
from adsb_latent_sde import corruption as C


CONFIG = {
    "n_features": 4, "sequence_length": 30,
    "latent_dim": 3, "context_dim": 8,
    "enc_hidden_dim": 16, "drift_hidden_dim": 16, "diffusion_hidden_dim": 16,
    "min_diffusion": 0.05, "sde_method": "euler", "sde_substeps": 1,
}


def _fake(n=8):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 30, 4)).astype(np.float32)


def test_build_and_forward_shapes():
    model = build_model(CONFIG)
    x = torch.from_numpy(_fake(5))
    out = model(x)
    assert out["elbo"].shape == (5,)
    assert out["ll"].shape == (5,)
    assert out["kl"].shape == (5,)
    assert out["x_mean"].shape == (5, 30, 4)


def test_elbo_is_finite_and_differentiable():
    model = build_model(CONFIG)
    x = torch.from_numpy(_fake(4))
    out = model(x)
    loss = -(out["ll"].mean() - out["kl"].mean())
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_kl_is_nonnegative():
    model = build_model(CONFIG)
    x = torch.from_numpy(_fake(16))
    out = model(x)
    # KL = KL(z0) + Girsanov path term; both are >= 0 in expectation.
    assert out["kl0"].min() >= -1e-4
    assert out["kl"].mean() >= -1e-3


def test_diffusion_positive_and_dt_consistency():
    model = build_model(CONFIG)
    # dt must scale as 1/((T-1)*substeps): genuine sub-stepping, valid dt->0 limit.
    assert abs(model.dt - 1.0 / ((30 - 1) * 1)) < 1e-9
    t = torch.tensor(0.3)
    y = torch.randn(7, CONFIG["latent_dim"])
    g = model.sde.g(t, y)
    assert (g >= CONFIG["min_diffusion"] - 1e-6).all()
    assert g.shape == (7, CONFIG["latent_dim"])


def test_score_sequences_columns():
    model = build_model(CONFIG)
    df = score_sequences(model, _fake(10), batch_size=4, device=torch.device("cpu"), num_samples=2)
    assert list(df.columns) == ["total_nll", "recon_nll", "kl"]
    assert len(df) == 10
    assert np.isfinite(df.values).all()


def test_prior_rollout_shape():
    model = build_model(CONFIG)
    samples = sample_prior_trajectories(model, 6, torch.device("cpu"))
    assert samples.shape == (6, 30, 4)
    assert np.isfinite(samples).all()


def test_dataset_returns_full_window():
    ds = SequenceDataset(_fake(12))
    assert len(ds) == 12
    assert ds[0].shape == (30, 4)


def test_corruptions_preserve_shape():
    x = _fake(5)
    for fn in (lambda a: C.speed_scale(a, 1.5), C.random_walk_velocity,
               C.sudden_turn_90, lambda a: C.position_jump(a, 2.0), C.stationary_clutter):
        out = fn(x)
        assert out.shape == x.shape
        assert np.isfinite(out).all()
