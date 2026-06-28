# C5-VAESDE-LATENT-SDE

Amortised **latent neural SDE** (a true VAE-SDE) for ADS-B trajectory anomaly detection.

---

## Why this exists

C4 is named "SDE" but is actually a discrete-time Gaussian transition model: its
"drift/diffusion" reparameterisation algebraically cancels (`dt` disappears),
it has no continuous-time limit, and no SDE solver is ever invoked. C5 is the
**genuine** stochastic differential equation the project name ("VAESDE" = VAE-SDE)
always implied — and, being a variational latent-variable model, it is also a
**correct VAE**.

| | C4 (`ProbabilisticMotionLSTM`) | **C5 (`LatentSDEModel`)** |
|---|---|---|
| Latent variable + KL (VAE) | ❌ none | ✅ latent path `z_t`, KL via Girsanov |
| Continuous-time drift/diffusion `f(t,z), g(t,z)` | ❌ (heads on an LSTM) | ✅ neural nets of `(t, z)` |
| Real SDE solver, Brownian motion, √dt noise | ❌ | ✅ `torchsde.sdeint` |
| Valid `dt → 0` limit | ❌ (`drift, diff → ∞`) | ✅ Euler–Maruyama sub-steps |
| `dt` actually used | ❌ cancels out | ✅ `dt = 1/((T-1)·substeps)` |

---

## Model

An amortised latent SDE in the sense of Li et al. 2020, *"Scalable Gradients for
Stochastic Differential Equations"*.

```
Encoder (GRU, backward over window)  x_{0:T}  ->  q(z0|x) = N(mu0, sig0),  context c

Latent SDE on z_t in continuous time [0,1]:
    posterior:  dz = f_phi(t, z, c) dt + g_theta(t, z) dW_t
    prior:      dz = h_theta(t, z)    dt + g_theta(t, z) dW_t     (g shared)

Decoder:  x_hat_t = Dec(z_t),   p(x_t | z_t) = N(x_hat_t, diag(obs_var))

Objective (ELBO):
    log p(x|z-path)  -  KL                       maximise
    KL = KL(q(z0)||p(z0))  +  E[ 0.5 ∫ |(f - h)/g|^2 dt ]   (Girsanov path KL)
```

The diffusion `g` is **shared** between prior and posterior, so the KL between
the two SDEs is the well-defined Girsanov change-of-measure term — carried as an
augmented state channel (`logqp`) integrated alongside `z` by the solver. KL is
linearly annealed over the first few epochs to avoid posterior collapse.

**Anomaly score** = negative ELBO per sequence (`total_nll`): a trajectory the
learned stochastic dynamics explains poorly gets a high score. `recon_nll`
(reconstruction term) and `kl` are also reported.

---

## Relationship to C1–C4

| Repo | Model | Anomaly signal |
|------|-------|----------------|
| C1 | preprocessing | normalised ENU windows `(N,30,4)` |
| C2 | β-VAE (whole-window) | reconstruction MSE + kinematic flags |
| C3 | deterministic LSTM | next-step MSE |
| C4 | probabilistic LSTM ("SDE-style") | Gaussian transition NLL |
| **C5** | **latent neural SDE (VAE-SDE)** | **negative ELBO of a continuous-time SDE** |

C5 consumes the same C1 arrays — no dependency on C2–C4.

---

## Quick start

```bash
pip install -e .
pytest                              # 8 smoke tests, no data needed

cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_train.npy data/
cp ../C1-VAESDE-ADSB-PREPROCESSING/data/X_test.npy  data/

python scripts/run_train_latent_sde.py        --config configs/latent_sde_default.yaml
python scripts/run_score_latent_sde.py        --config configs/latent_sde_default.yaml
python scripts/run_stress_test_latent_sde.py  --config configs/latent_sde_default.yaml --quantile 0.99
python scripts/run_rollout_latent_sde.py      --config configs/latent_sde_default.yaml
```

`debug_mode: true` (default) trains on a 50k subset — latent-SDE training is
solver-bound and slower than C3/C4. Set `debug_mode: false` for the full 1.41M
set. Scoring and stress tests always use the full arrays.

---

## Scripts

| Script | Output |
|---|---|
| `run_train_latent_sde.py` | `outputs/latent_sde/latent_sde.pt`, `history.csv` |
| `run_score_latent_sde.py` | `train/test_latent_sde_scores.csv`, `latent_sde_thresholds.csv` |
| `run_stress_test_latent_sde.py` | `latent_sde_stress_summary.csv` (merges across quantiles) |
| `run_rollout_latent_sde.py` | `prior_rollouts.npy`, `prior_rollouts.png` |

---

## Key configuration (`configs/latent_sde_default.yaml`)

| Key | Meaning |
|---|---|
| `latent_dim` | dimension of the latent SDE state |
| `sde_method` | solver (`euler` = Euler–Maruyama, Itô diagonal noise) |
| `sde_substeps` | solver sub-steps per observation interval → `dt = 1/((T-1)·substeps)` |
| `min_diffusion` | floor on `g` (keeps the Girsanov division well-posed) |
| `kl_anneal_epochs` | linear KL warm-up to prevent posterior collapse |
| `num_score_samples` | Monte-Carlo latent draws when computing the ELBO score |

---

## Notes / limitations

- Like C4, the SDE score is weak on **stationary clutter** (a near-constant
  trajectory is highly probable under any smooth dynamics). To match the C4
  detector, fuse C5's `total_nll` with the kinematic stationary lower-tail rule
  (`flag = total_nll > train_p99 OR stationary_rule`).
- Training cost is dominated by the SDE solve; `sde_substeps` and `latent_dim`
  trade fidelity against speed.
- Euler–Maruyama is first-order; `sde_method` can be raised (e.g. `srk`) for a
  higher-order solver at extra cost.
