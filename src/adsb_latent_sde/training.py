from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import LatentSDEModel


def _kl_weight(epoch: int, anneal_epochs: int) -> float:
    """Linear KL warm-up from 0 -> 1 over the first `anneal_epochs` epochs.

    Annealing the KL avoids posterior collapse early in training (a standard
    VAE/latent-SDE trick)."""
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / anneal_epochs)


def run_epoch(
    model: LatentSDEModel,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    desc: str = "",
) -> dict:
    train = optimizer is not None
    model.train(train)
    tot_neg_elbo = tot_ll = tot_kl = 0.0
    n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            out = model(x)
            ll = out["ll"].mean()
            kl = out["kl"].mean()
            loss = -(ll - kl_weight * kl)               # weighted negative ELBO
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            bs = x.size(0)
            tot_neg_elbo += float((-(ll - kl)).item()) * bs   # report true (unweighted) -ELBO
            tot_ll += float(ll.item()) * bs
            tot_kl += float(kl.item()) * bs
            n += bs
    return {"neg_elbo": tot_neg_elbo / n, "ll": tot_ll / n, "kl": tot_kl / n}


def fit_model(
    model: LatentSDEModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    kl_anneal_epochs: int = 3,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict] = []
    print(f"{'Epoch':>6} | {'klW':>5} | {'train -ELBO':>12} {'ll':>10} {'kl':>9} | {'test -ELBO':>11} {'ll':>10}")
    print("-" * 78)
    for epoch in range(epochs):
        kw = _kl_weight(epoch, kl_anneal_epochs)
        tr = run_epoch(model, train_loader, device, kw, optimizer, grad_clip, desc=f"train {epoch+1}")
        te = run_epoch(model, test_loader, device, 1.0, None, grad_clip, desc=f"test {epoch+1}")
        row = {
            "epoch": epoch + 1, "kl_weight": kw,
            "train_neg_elbo": tr["neg_elbo"], "train_ll": tr["ll"], "train_kl": tr["kl"],
            "test_neg_elbo": te["neg_elbo"], "test_ll": te["ll"], "test_kl": te["kl"],
        }
        history.append(row)
        print(f"{epoch+1:>6} | {kw:>5.2f} | {tr['neg_elbo']:>12.4f} {tr['ll']:>10.3f} {tr['kl']:>9.4f} "
              f"| {te['neg_elbo']:>11.4f} {te['ll']:>10.3f}")
    return history
