"""Train the latent SDE (VAE-SDE) and save a checkpoint + loss history."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from adsb_latent_sde.config import load_config, set_seed, ensure_dir
from adsb_latent_sde.dataset import SequenceDataset
from adsb_latent_sde.model import build_model, save_checkpoint
from adsb_latent_sde.training import fit_model
from adsb_latent_sde.reporting import save_dataframe
from adsb_latent_sde.utils import get_device, count_parameters, describe_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/latent_sde_default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()
    print(f"Device: {device}")

    data_dir = Path(config["data_dir"])
    X_train = np.load(data_dir / config["train_file"], mmap_mode="r")
    X_test = np.load(data_dir / config["test_file"], mmap_mode="r")
    describe_array("X_train", X_train)
    describe_array("X_test", X_test)

    if config["debug_mode"]:
        n_tr, n_te = config["debug_train_size"], config["debug_test_size"]
        print(f"DEBUG mode: {n_tr} train / {n_te} test")
    else:
        n_tr = n_te = None
        print("FULL mode")

    train_ds = SequenceDataset(np.asarray(X_train), n_tr, seed=config["seed"])
    test_ds = SequenceDataset(np.asarray(X_test), n_te, seed=config["seed"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=config["num_workers"])

    model = build_model(config)
    print(f"Model: LatentSDEModel  params={count_parameters(model):,}  "
          f"latent_dim={config['latent_dim']}  dt={model.dt:.5f}  method={config['sde_method']}")

    history = fit_model(
        model, train_loader, test_loader, device,
        epochs=config["epochs"], lr=config["learning_rate"],
        weight_decay=config["weight_decay"], grad_clip=config["gradient_clip"],
        kl_anneal_epochs=config["kl_anneal_epochs"],
    )

    out_dir = ensure_dir(config["output_dir"])
    ckpt_path = out_dir / "latent_sde.pt"
    save_checkpoint(model, config, history, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    save_dataframe(pd.DataFrame(history), out_dir / "history.csv")
    print(f"History saved to {out_dir / 'history.csv'}")


if __name__ == "__main__":
    main()
