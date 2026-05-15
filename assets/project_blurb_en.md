## ObsCodec: Learned Observation Compression for Multi-Agent Systems

**One-liner**: Across 5 codec families and 93 benchmark configurations (89 trained models + 4 PCA fits), β-VAE reveals a semantic compression boundary at ~6.4 bits of effective KL rate for multi-agent observations under extreme bandwidth constraints. Posterior collapse at β≥0.5 is systematically demonstrated across all latent dimensions.

**Motivation**: Multi-robot coordination faces a communication bandwidth bottleneck. Conventional approaches either transmit raw observations uncompressed or apply fixed-bitrate quantization — neither provides adaptive semantic compression. The central question: **how far can an observation signal be compressed before task-relevant structure disappears?** This work is a pre-study for SemCom-MARL — isolating codec selection before integration into the full reinforcement learning loop.

**Approach**: 50,000 multi-agent observation frames collected from PettingZoo/MPE `simple_spread_v3`. Five codec families (PCA / AE / Digital Quantization / β-VAE / VQ-VAE) are systematically benchmarked under a shared training loop, identical data splits, and a unified early-stopping criterion.

**Key Findings** (with chart evidence):
- **Digital Quantization** is the strongest reconstruction baseline (MSE≈0.0001 at 128 nominal bits, PSNR≈40 dB) → Table 1, Fig. rate_distortion
- **β-VAE (β=0.01, LD=8)** achieves ~6.4 bits effective KL rate (256 nominal bits) — maximal semantic compression in the information-bottleneck sense → Table 2, Fig. ablation_heatmap
- **β≥0.5 triggers posterior collapse**: KL divergence drops to 10⁻⁴–10⁻⁷, MSE saturates at ~0.545 (prior variance), observed across all latent dimensions LD=2 through 32 → Table 2, Fig. kl_collapse
- **VQ-VAE codebook usage ≤14% at LD=8, reaches 100% at LD=2** — low-dimensional discrete latent spaces are more effective for this observation distribution → Table 3, Fig. vqvae_usage_heatmap
- The collapse boundary aligns with Alemi et al. (2018)'s rate-distortion theory: the β-VAE Lagrangian L = MSE + β·KL causes the encoder to degenerate to the prior N(0,I) when β dominates

**Important Caveats**: Reconstruction MSE is a proxy metric — downstream task performance (coverage ratio, collision avoidance, policy return) requires SemCom-MARL closed-loop validation. The effective KL rate is an information-theoretic estimate, not a deployable packet size; real-world transmission requires entropy coding (bits-back) or channel adaptation.

**Technical Stack**: PyTorch, PettingZoo/MPE, scikit-learn · 89 trained models + 4 PCA fits · 8 diagnostic figures · 4 cross-reference tables

**Bridging to SemCom-MARL**: The β-VAE's rate-distortion Lagrangian form (Alemi et al. 2018; Burgess et al. 2018) makes it a natural fit for information-bottleneck-driven communication scheduling — β governs the KL→rate mapping, LD sets the bandwidth ceiling, and KL divergence monitoring serves as a collapse early-warning signal. Recommended configuration: β-VAE (β≈0.01, LD=8, effective rate≈6.4 bits).

[GitHub](https://github.com/MacswareX/ObsCodec) · [Full Numerical Results](results_summary.md)
