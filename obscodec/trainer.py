"""Generic training loop with early stopping, checkpointing, and logging."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from copy import deepcopy
import json, time

from .config import CHECKPOINT_DIR, BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, EARLY_STOP_PATIENCE


def train_model(
    model,
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    device: str = "cpu",
    model_name: str = "model",
    lr: float = DEFAULT_LR,
    patience: int = EARLY_STOP_PATIENCE,
    verbose: bool = True,
) -> dict:
    """Generic training loop with early stopping.

    The model must implement `training_step(x)` and `validation_step(x)`,
    each returning (loss, ...). For VAE models, `set_epoch(epoch)` is
    called each epoch to drive KL warmup.

    Returns:
        dict with keys: model, best_epoch, best_val_loss, history.
    """
    train_t = torch.FloatTensor(train_data)
    val_t = torch.FloatTensor(val_data)

    train_loader = DataLoader(
        TensorDataset(train_t), batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(val_t), batch_size=batch_size, shuffle=False
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_recon": [], "val_recon": [],
               "train_kl": [], "val_kl": []}
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)

        model.train()
        train_loss, train_recon, train_kl = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model.training_step(x)
            loss = outputs[0]
            recon = outputs[1] if len(outputs) > 1 else loss
            kl = outputs[2] if len(outputs) > 2 else torch.tensor(float("nan"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_recon += recon.item() if isinstance(recon, torch.Tensor) else recon
            train_kl += kl.item() if isinstance(kl, torch.Tensor) else float("nan")
            n_batches += 1

        train_loss /= max(n_batches, 1)
        train_recon /= max(n_batches, 1)
        train_kl /= max(n_batches, 1)

        model.eval()
        val_loss, val_recon, val_kl = 0.0, 0.0, 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                outputs = model.validation_step(x)
                loss = outputs[0]
                recon = outputs[1] if len(outputs) > 1 else loss
                kl = outputs[2] if len(outputs) > 2 else torch.tensor(float("nan"))
                val_loss += loss.item()
                val_recon += recon.item() if isinstance(recon, torch.Tensor) else recon
                val_kl += kl.item() if isinstance(kl, torch.Tensor) else float("nan")
                n_val += 1

        val_loss /= max(n_val, 1)
        val_recon /= max(n_val, 1)
        val_kl /= max(n_val, 1)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_recon"].append(float(train_recon))
        history["val_recon"].append(float(val_recon))
        history["train_kl"].append(float(train_kl))
        history["val_kl"].append(float(val_kl))

        if verbose and (epoch % 20 == 0 or epoch == 1 or epoch == epochs):
            beta_str = ""
            if hasattr(model, "beta_current"):
                beta_str = f" β={model.beta_current:.3f}"
            print(f"  E{epoch:3d}{beta_str} | train L={train_loss:.4f} R={train_recon:.4f} "
                  f"| val L={val_loss:.4f} R={val_recon:.4f} | best={best_val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Save checkpoint
    ckpt_path = CHECKPOINT_DIR / f"{model_name}.pt"
    torch.save({"model_state": best_state, "epoch": best_epoch, "history": history}, ckpt_path)
    if verbose:
        print(f"  Saved: {ckpt_path}")

    training_time_s = time.time() - t_start

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "training_time_s": training_time_s,
    }


def train_ae_codec(model, train_data, val_data, epochs=200, batch_size=256,
                   device="cpu", model_name="ae", lr=1e-3, patience=30):
    """Specialized training for AE codecs that use forward(), not training_step()."""
    train_t = torch.FloatTensor(train_data)
    val_t = torch.FloatTensor(val_data)

    train_loader = DataLoader(TensorDataset(train_t), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_t), batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n += 1
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                x_hat, _ = model(x)
                loss = F.mse_loss(x_hat, x)
                val_loss += loss.item()
                n += 1
        val_loss /= max(n, 1)

        if epoch % 40 == 0 or epoch == 1:
            print(f"  E{epoch:3d} | train={train_loss:.4f} val={val_loss:.4f} | best={best_val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    ckpt_path = CHECKPOINT_DIR / f"{model_name}.pt"
    torch.save({"model_state": best_state}, ckpt_path)
    return {"model": model, "best_val_loss": best_val_loss}
