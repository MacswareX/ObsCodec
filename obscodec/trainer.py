"""Shared training loop used by all neural codecs."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from obscodec.config import BITS_PER_FLOAT32


def _resolve_bandwidth(model: torch.nn.Module) -> float | None:
    """Best-effort bandwidth without falling for falsy-zero gotchas."""
    bw = getattr(model, "equivalent_bandwidth", None)
    if bw is not None:
        return float(bw)
    if hasattr(model, "latent_dim"):
        return float(getattr(model, "latent_dim") * BITS_PER_FLOAT32)
    return None


def train_model(
    model: torch.nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
    model_name: str = "model",
    patience: int = 20,
) -> dict[str, Any]:
    """Train one codec with early stopping and return the best model state.

    Models are expected to implement:
    - ``forward(x) -> (x_hat, z)``
    - ``loss(x, x_hat) -> scalar tensor``
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_data)),
        batch_size=batch_size, shuffle=True,
    )
    val_tensor = torch.FloatTensor(val_data).to(device)

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    pbar = tqdm(range(epochs), desc=model_name)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            x_hat, _ = model(x_batch)
            loss = model.loss(x_batch, x_hat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        train_loss = epoch_loss / len(train_data)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            x_hat_val, _ = model(val_tensor)
            val_loss = model.loss(val_tensor, x_hat_val).item()
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "history": history,
        "equivalent_bandwidth": _resolve_bandwidth(model),
    }
