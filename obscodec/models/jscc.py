"""Joint Source-Channel Coding (JSCC) wrapper.

Wraps an existing encoder-decoder codec and inserts a differentiable channel
between encoding and decoding during both training and inference. The encoder
learns to produce channel-robust latent representations because it experiences
channel noise during the forward pass.

Supported base models: BetaVAE, VQVAE, AECodec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JSCCWrapper(nn.Module):
    """Wrap any encode/decode codec with a differentiable channel in the loop.

    Parameters
    ----------
    base_model:
        Must have ``encode(x)``, ``decode(z)``, and ideally
        ``training_step(x)`` / ``validation_step(x)`` methods.
    channel:
        A differentiable ``nn.Module`` that applies channel impairments
        (e.g. ``DiffAWGN``, ``DiffErasure``).
    mode:
        ``"post_latent"`` — apply channel to the latent z (default).
        ``"post_encode"`` — apply channel to encoder output before reparam.
    """

    def __init__(self, base_model: nn.Module, channel: nn.Module, mode: str = "post_latent"):
        super().__init__()
        self.base = base_model
        self.channel = channel
        self.mode = mode

    # ── pass-through properties ──

    @property
    def latent_dim(self):
        return self.base.latent_dim if hasattr(self.base, "latent_dim") else None

    @property
    def beta(self):
        return getattr(self.base, "beta", None)

    @property
    def beta_current(self):
        return self.base.beta_current if hasattr(self.base, "beta_current") else self.beta

    @property
    def free_bits(self):
        return getattr(self.base, "free_bits", 0)

    @property
    def obs_dim(self):
        return self.base.obs_dim if hasattr(self.base, "obs_dim") else None

    def set_epoch(self, epoch: int):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    # ── core encode / decode ──

    def encode(self, x):
        return self.base.encode(x)

    def decode(self, z):
        return self.base.decode(z)

    # ── forward with channel in the loop ──

    def forward(self, x):
        """Encode → channel → decode. Returns full forward pass."""
        if hasattr(self.base, "reparameterize"):
            # BetaVAE path
            mu, logvar = self.base.encoder(x)
            z = self.base.reparameterize(mu, logvar)
            z_noisy = self.channel(z)
            x_hat = self.base.decode(z_noisy)
            return x_hat, z_noisy, mu, logvar
        elif hasattr(self.base, "vq"):
            # VQVAE path
            z_e = self.base.encoder(x)
            z_q, vq_loss, indices = self.base.vq(z_e)
            z_noisy = self.channel(z_q)
            x_hat = self.base.decode(z_noisy)
            return x_hat, z_noisy, vq_loss, indices
        else:
            # AE / generic path
            z = self.base.encode(x)
            if isinstance(z, tuple):
                z = z[0]
            z_noisy = self.channel(z)
            x_hat = self.base.decode(z_noisy)
            return x_hat, z_noisy

    # ── training / validation steps ──

    def training_step(self, x):
        """Training step with channel noise injected."""
        self.train()
        if hasattr(self.base, "reparameterize"):
            # BetaVAE: compute KL from pre-channel latents, recon from post-channel
            x_hat, z_noisy, mu, logvar = self.forward(x)
            recon = F.mse_loss(x_hat, x)
            kl = self.base.kl_divergence(mu, logvar)
            if self.base.free_bits > 0:
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_per_dim = kl_per_dim.mean(dim=0)
                kl_active = torch.clamp(kl_per_dim - self.base.free_bits, min=0).sum()
                loss = recon + self.base.beta_current * kl_active
            else:
                loss = recon + self.base.beta_current * kl
            return loss, recon, kl
        elif hasattr(self.base, "vq"):
            # VQVAE: same structure but with vq_loss
            x_hat, z_noisy, vq_loss, indices = self.forward(x)
            recon = F.mse_loss(x_hat, x)
            loss = recon + vq_loss.mean()
            return loss, recon, float(vq_loss.mean())
        else:
            # AE / generic
            x_hat, z_noisy = self.forward(x)
            recon = F.mse_loss(x_hat, x)
            loss = recon
            return loss, recon, 0.0

    def validation_step(self, x):
        """Validation step — channel still applied, eval mode (deterministic)."""
        self.eval()
        with torch.no_grad():
            return self.training_step(x)  # same computation, no grad

    def kl_nats(self, x):
        """Compute KL in nats for a test batch."""
        self.eval()
        with torch.no_grad():
            if hasattr(self.base, "kl_nats"):
                return self.base.kl_nats(x)
            if hasattr(self.base, "reparameterize"):
                _, mu, logvar = self.base.encode(x)
                kl = self.base.kl_divergence(mu, logvar)
                return float(kl.item())
            return 0.0

    def equivalent_bandwidth(self):
        if hasattr(self.base, "equivalent_bandwidth"):
            if callable(self.base.equivalent_bandwidth):
                return self.base.equivalent_bandwidth()
            return self.base.equivalent_bandwidth
        return 0

    def get_rate_estimate(self, dataloader, device):
        if hasattr(self.base, "get_rate_estimate"):
            return self.base.get_rate_estimate(dataloader, device)
        return 0.0
