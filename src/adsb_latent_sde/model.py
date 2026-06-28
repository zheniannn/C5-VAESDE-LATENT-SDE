"""Amortised latent neural SDE for ADS-B trajectory modelling.

This is a *genuine* stochastic differential equation model, unlike C4's
discrete Gaussian transition model. The latent state z_t evolves in
continuous time

    dz = f_phi(t, z, context) dt + g_theta(t, z) dW_t        (approx. posterior)
    dz = h_theta(t, z)        dt + g_theta(t, z) dW_t        (prior)

integrated by a real SDE solver (torchsde) with Brownian motion and
sqrt(dt) noise. Observations are decoded from the latent path, and the model
is trained by maximising the evidence lower bound (ELBO). The KL between the
approximate-posterior SDE and the prior SDE is computed by the Girsanov change
of measure (the `logqp` path term), exactly as in Li et al. 2020,
"Scalable Gradients for Stochastic Differential Equations".

Properties that make this a correct SDE (and that C4 lacked):
  * drift and diffusion are continuous-time functions of (t, z);
  * a real solver integrates with multiple sub-steps per observation interval;
  * the Brownian increment scales as sqrt(dt) -> valid dt -> 0 limit;
  * the diffusion g is shared between prior and posterior, so the KL is the
    well-defined Girsanov term, not an ad-hoc penalty.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torchsde


def _stable_division(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    b = torch.where(b.abs().detach() > eps, b, torch.full_like(b, eps) * b.sign() + eps)
    return a / b


class Encoder(nn.Module):
    """Amortised recognition network: x -> q(z0|x) and a context vector.

    Runs a GRU backwards over the observed window (as in latent ODE/SDE work)
    so that the initial-state posterior is informed by the whole trajectory.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, context_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.to_qz0 = nn.Linear(hidden_dim, 2 * latent_dim)
        self.to_context = nn.Linear(hidden_dim, context_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, input_dim)
        x_rev = torch.flip(x, dims=[1])
        _, h = self.gru(x_rev)          # h: (1, B, hidden_dim)
        h = h[-1]                        # (B, hidden_dim)
        qz0_mean, qz0_logvar = self.to_qz0(h).chunk(2, dim=-1)
        context = self.to_context(h)
        return qz0_mean, qz0_logvar, context


class LatentSDE(torchsde.SDEIto):
    """The latent SDE: posterior drift f, prior drift h, shared diffusion g.

    The approximate-posterior drift is conditioned on a per-sequence context
    vector, set as `self._ctx` before each `sdeint` call (the standard
    amortisation trick for batched latent SDEs).
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        drift_hidden_dim: int,
        diffusion_hidden_dim: int,
        min_diffusion: float = 0.05,
    ) -> None:
        super().__init__(noise_type="diagonal")
        self.latent_dim = latent_dim
        self.min_diffusion = min_diffusion

        # time is encoded as [sin(2*pi*t), cos(2*pi*t)] -> 2 features
        self.prior_drift = nn.Sequential(
            nn.Linear(latent_dim + 2, drift_hidden_dim), nn.Tanh(),
            nn.Linear(drift_hidden_dim, drift_hidden_dim), nn.Tanh(),
            nn.Linear(drift_hidden_dim, latent_dim),
        )
        self.post_drift = nn.Sequential(
            nn.Linear(latent_dim + 2 + context_dim, drift_hidden_dim), nn.Tanh(),
            nn.Linear(drift_hidden_dim, drift_hidden_dim), nn.Tanh(),
            nn.Linear(drift_hidden_dim, latent_dim),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + 2, diffusion_hidden_dim), nn.Tanh(),
            nn.Linear(diffusion_hidden_dim, latent_dim),
        )

        # Glow-style small init on the last drift layers for stable early training.
        for net in (self.prior_drift, self.post_drift):
            net[-1].weight.data.mul_(0.1)
            net[-1].bias.data.zero_()

        # Prior over the initial latent state p(z0) = N(0, I).
        self.register_buffer("pz0_mean", torch.zeros(1, latent_dim))
        self.register_buffer("pz0_logvar", torch.zeros(1, latent_dim))

        self._ctx: torch.Tensor | None = None  # set per forward()

    # --- time feature helper -------------------------------------------------
    def _tfeat(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # t is a 0-dim tensor from the solver; broadcast to (B, 2).
        two_pi_t = 2.0 * math.pi * t
        tf = torch.stack([torch.sin(two_pi_t), torch.cos(two_pi_t)]).view(1, 2)
        return tf.expand(y.size(0), 2)

    # --- drift / diffusion ---------------------------------------------------
    def f(self, t, y):  # approximate posterior drift
        tf = self._tfeat(t, y)
        inp = torch.cat([y, tf, self._ctx], dim=-1)
        return self.post_drift(inp)

    def h(self, t, y):  # prior drift
        tf = self._tfeat(t, y)
        return self.prior_drift(torch.cat([y, tf], dim=-1))

    def g(self, t, y):  # shared diffusion (diagonal, strictly positive)
        tf = self._tfeat(t, y)
        raw = self.diffusion_net(torch.cat([y, tf], dim=-1))
        return self.min_diffusion + nn.functional.softplus(raw)

    # --- augmented dynamics carrying the KL (logqp) accumulator --------------
    def f_aug(self, t, y):
        z = y[:, : self.latent_dim]
        f, g, h = self.f(t, z), self.g(t, z), self.h(t, z)
        u = _stable_division(f - h, g)                       # Girsanov integrand
        f_logqp = 0.5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def g_aug(self, t, y):
        z = y[:, : self.latent_dim]
        g = self.g(t, z)
        return torch.cat([g, torch.zeros(y.size(0), 1, device=y.device)], dim=1)


class LatentSDEModel(nn.Module):
    """Full VAE-SDE: encoder + latent SDE + decoder, trained on the ELBO."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.input_dim = config.get("n_features", 4)
        self.latent_dim = config["latent_dim"]
        self.seq_len = config.get("sequence_length", 30)
        self.method = config.get("sde_method", "euler")
        substeps = config.get("sde_substeps", 2)
        # dt small enough for several solver steps per observation interval.
        self.dt = 1.0 / ((self.seq_len - 1) * max(1, substeps))

        self.encoder = Encoder(
            self.input_dim, config["enc_hidden_dim"], self.latent_dim, config["context_dim"]
        )
        self.sde = LatentSDE(
            latent_dim=self.latent_dim,
            context_dim=config["context_dim"],
            drift_hidden_dim=config["drift_hidden_dim"],
            diffusion_hidden_dim=config["diffusion_hidden_dim"],
            min_diffusion=config.get("min_diffusion", 0.05),
        )
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)
        # observation noise (homoscedastic, per feature, learned)
        self.obs_logvar = nn.Parameter(torch.zeros(self.input_dim))

        # integration grid over [0, 1] hitting the 30 observation times.
        self.register_buffer("ts", torch.linspace(0.0, 1.0, self.seq_len))

    def _decode_ll(self, zs_btl: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Gaussian reconstruction log-likelihood per sequence. (B,)"""
        x_mean = self.decoder(zs_btl)                       # (B, T, input_dim)
        obs_var = torch.exp(self.obs_logvar)
        ll = -0.5 * (((x - x_mean) ** 2) / obs_var + self.obs_logvar + math.log(2 * math.pi))
        return ll.sum(dim=[1, 2]), x_mean

    def forward(self, x: torch.Tensor, eps: torch.Tensor | None = None) -> dict:
        """Single-sample ELBO estimate for a batch x: (B, T, input_dim)."""
        B = x.size(0)
        qz0_mean, qz0_logvar, ctx = self.encoder(x)
        if eps is None:
            eps = torch.randn_like(qz0_mean)
        z0 = qz0_mean + eps * torch.exp(0.5 * qz0_logvar)

        # KL at t=0 between q(z0|x) and p(z0)=N(0,I).
        qz0_var, pz0_var = torch.exp(qz0_logvar), torch.exp(self.sde.pz0_logvar)
        kl0 = 0.5 * (
            (qz0_var + (qz0_mean - self.sde.pz0_mean) ** 2) / pz0_var
            - 1.0
            + self.sde.pz0_logvar
            - qz0_logvar
        ).sum(dim=1)                                        # (B,)

        # Integrate the augmented latent SDE; last aug-channel accumulates KL(path).
        self.sde._ctx = ctx
        aug_z0 = torch.cat([z0, torch.zeros(B, 1, device=x.device)], dim=1)
        aug_zs = torchsde.sdeint(
            self.sde, aug_z0, self.ts,
            method=self.method, dt=self.dt,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )                                                   # (T, B, latent+1)
        zs = aug_zs[:, :, : self.latent_dim].permute(1, 0, 2)   # (B, T, latent)
        logqp_path = aug_zs[-1, :, self.latent_dim]              # (B,)

        ll, x_mean = self._decode_ll(zs, x)
        kl = kl0 + logqp_path                                # (B,)
        elbo = ll - kl                                       # (B,)
        return {
            "elbo": elbo, "ll": ll, "kl": kl,
            "kl0": kl0, "kl_path": logqp_path,
            "x_mean": x_mean,
        }

    @torch.no_grad()
    def sample_prior(self, batch_size: int, device: torch.device, ts: torch.Tensor | None = None) -> torch.Tensor:
        """Generate trajectories from the *prior* SDE (drift h) -> decoded obs."""
        ts = self.ts if ts is None else ts
        z0 = self.sde.pz0_mean + torch.randn(batch_size, self.latent_dim, device=device) * torch.exp(
            0.5 * self.sde.pz0_logvar
        )
        zs = torchsde.sdeint(self.sde, z0, ts, method="euler", dt=self.dt, names={"drift": "h", "diffusion": "g"})
        zs = zs.permute(1, 0, 2)                             # (B, T, latent)
        return self.decoder(zs)                              # (B, T, input_dim)


def build_model(config: dict) -> LatentSDEModel:
    return LatentSDEModel(config)


def save_checkpoint(model: LatentSDEModel, config: dict, history: list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "history": history}, path)
