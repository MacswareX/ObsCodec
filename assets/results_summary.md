# ObsCodec Results Summary

> Raw observation = 18 dims × 32 bits = **576 bits**. All MSE computed on the held-out test set.

## Table 1: Cross-Method Best-Config Comparison — Reconstruction Quality, Bandwidth Accounting, and Rate-Distortion Efficiency

> Each method is represented by its lowest-MSE configuration at a nominal bandwidth of ~64–256 bits. The seven columns span three measurement categories: **reconstruction fidelity** (MSE, PSNR), **bandwidth accounting** (nominal BW computed as `latent_dim × bits_per_element` for digital/PCA/AE or `latent_dim × 32` for β-VAE continuous latents; effective KL rate for β-VAE only), and **efficiency ratios** (compression ratio = 576 / BW; RD Efficiency = 1/(MSE × BW), a combined metric penalizing both high distortion and high bandwidth). Digital Quantization achieves MSE < 1e-4 at 128 bits; additional higher-bandwidth configurations exist (e.g., LD=16 B=32 → 512 bits → MSE < 1e-6) but are excluded for cross-method comparability. VQ-VAE operates in a distinct bandwidth regime (5–10 bits, see Table 3).

| Method | Best Config | MSE | PSNR (dB) | BW (bits) | Eff. Rate (b) | Comp. Ratio | RD Efficiency |
|--------|-------------|-----|-----------|-----------|---------------|-------------|---------------|
| PCA | n=8 | 0.0773 | 11.1 | 256 | — | 2× | 0.05 |
| Standard AE | LD=8 | 0.0227 | 16.4 | 256 | — | 2× | 0.17 |
| Digital Quant. | LD=16, B=8 | 0.0001 | 40.0 | 128 | — | 4× | 78.12 |
| β-VAE β=0.001 | LD=8 | 0.0284 | 15.5 | 256 | 15.8 | 2× | 0.14 |
| β-VAE β=0.01 (✦ recommended) | LD=8 | 0.0873 | 10.6 | 256 | 6.4 | 2× | 0.04 |
| β-VAE β=0.1 | LD=8 | 0.3205 | 4.9 | 256 | 0.8 | 2× | 0.01 |
| β-VAE β=1.0 (✗ R→0) | LD=8 | 0.5455 | 2.6 | 256 | 0.0 | 2× | 0.01 |
| VQ-VAE | cb256_ld2_cc0.25 | 0.1756 | 7.6 | 8 | — | 72× | 0.71 |

> **Comparative insights from Table 1:**
>
> - **Digital Quantization dominates the RD Efficiency ranking** (78.12) by combining near-lossless reconstruction (MSE=0.0001) with moderate nominal bandwidth (128 bits). It is the recommended baseline when pure observation fidelity is the sole objective.
> - **β-VAE β=0.01 (✦) is the recommended SemCom-MARL probe.** Its nominal BW is 256 bits (8 latent dims × 32-bit float), but the KL-measured effective information rate is only 6.4 bits. The ~90× gap between raw storage (576 bits) and information-theoretic content (6.4 bits) quantifies the compressibility achievable through a probabilistic bottleneck. Actual deployed compression requires entropy coding (e.g., bits-back) or learned channel adaptation on top of this estimate.
> - **β≥0.5 triggers posterior collapse:** KL → 0 nats, MSE saturates at ~0.545 (the variance of the prior N(0,I)), and effective rate → 0. The encoder outputs the prior regardless of input.
> - **VQ-VAE achieves the highest compression ratio** (72×) by operating at only 8 bits, but its MSE (0.1756) is 8× higher than β-VAE β=0.01. The choice between them depends on whether a discrete, ultra-low-bitrate channel interface is more important than reconstruction fidelity.
>
> See Fig. rate_distortion, Fig. pareto_frontier.

## Table 2: β-VAE β-Sweep at Fixed LD=8 — Reconstruction-Vs-Information Trade-Off with Regime Classification

> Latent dimension fixed at LD=8; β swept from 0.001 to 10.0 (four orders of magnitude). This table isolates the β-VAE's core operating characteristic: the Lagrangian multiplier β governs the weight of the KL regularization term `β · KL(q(z|x) || N(0,I))` relative to the reconstruction loss. As β increases, the optimizer prioritizes matching the prior over preserving input information, producing the rate-distortion trajectory captured below. LD=8 is chosen as the focal dimension because it provides sufficient capacity to observe the full β-MSE curve (LD=2 and 4 have higher MSE floors due to capacity constraints; LD=16 and 32 reproduce the same collapse boundary at higher training cost without additional insight).

| β | MSE | PSNR (dB) | BW (nominal) | Eff. Rate (bits) | KL (nats) | Regime |
|---|-----|-----------|-------------|-------------------|-----------|--------|
| 0.001 | 0.0284 | 15.5 | 256 | 15.8 | 22.8256 | ✓ effective |
| 0.01 | 0.0873 | 10.6 | 256 | 6.4 | 9.1929 | ✓ effective |
| 0.1 | 0.3205 | 4.9 | 256 | 0.8 | 1.1112 | ⚠ transition |
| 0.5 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 1.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 2.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 5.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 10.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |

> **Regime classification criteria:**
> - **✓ effective (β≤0.01):** KL > 1 nat — the encoder carries significant input-dependent information, producing a structured latent representation. β=0.001 yields the best reconstruction (MSE=0.0284, 15.8 bits effective rate); β=0.01 is the recommended SemCom-MARL operating point (MSE=0.0873, 6.4 bits effective rate).
> - **⚠ transition (β=0.1):** KL ≈ 1.1 nats (0.8 bits) and MSE ≈ 0.32 — the encoder is discarding most input information. This regime is a boundary stress-test; small changes in β produce large changes in behavior.
> - **✗ collapsed (β≥0.5):** KL < 1e-4 nats, MSE saturated at ~0.545 (the variance of N(0,I)). The encoder has degenerated to the prior — the latent channel carries zero input-dependent information. This boundary is sharp and reproducible across all tested latent dimensions (LD=2 through 32).
>
> See Fig. kl_collapse, Fig. ablation_heatmap, Fig. latent_space.

## Table 3: VQ-VAE Commitment Cost Sweep (LD=8, CB=256) — Can Commitment Cost Mitigate Codebook Underutilization?

> Codebook size CB=256 and latent dimension LD=8 are held fixed; commitment cost (cc) is swept from 0.01 to 5.0. This experiment tests a specific hypothesis: since commitment cost weights the term `||sg[z_e] - e||²` that pulls the encoder toward the nearest codebook vector, increasing cc should force higher codebook utilization. The data **refutes** this hypothesis — at LD=8, codebook usage remains ≤14% across two orders of magnitude of cc, indicating the underutilization is structural rather than a consequence of insufficient commitment regularization.

| cc | MSE | PSNR (dB) | BW (bits) | Codebook Usage | RD Efficiency | Note |
|-----|-----|-----------|-----------|----------------|---------------|------|
| 0.01 | 0.2204 | 6.6 | 8 | 8.6% | 0.57 | — underused |
| 0.05 | 0.2071 | 6.8 | 8 | 14.1% | 0.60 | — underused |
| 0.1 | 0.2177 | 6.6 | 8 | 9.0% | 0.57 | — underused |
| 0.25 | 0.2080 | 6.8 | 8 | 13.7% | 0.60 | — underused |
| 0.5 | 0.2253 | 6.5 | 8 | 7.0% | 0.55 | — underused |
| 1.0 | 0.2133 | 6.7 | 8 | 10.9% | 0.59 | — underused |
| 2.0 | 0.2083 | 6.8 | 8 | 11.3% | 0.60 | — underused |
| 5.0 | 0.2198 | 6.6 | 8 | 8.2% | 0.57 | — underused |

> **Interpretation:**
> - The insensitivity of codebook usage to cc (≤14% across the full sweep) indicates the underutilization is **structural**: a 256-entry codebook is over-provisioned for an 18-dim MPE observation with limited modality diversity. No amount of commitment cost tuning can force the encoder to use codebook entries that are not needed to cover the data manifold.
> - **Contrast with LD=2:** At LD=2, codebook usage reaches 100% and yields the best VQ-VAE point (cb256_ld2_cc0.25: MSE=0.1756, 8 bits, BW=8). Low-dimensional discretization saturates the codebook more effectively because fewer latent dimensions mean each codebook vector covers a larger fraction of the representation space. **Practical recommendation:** use LD=2 for discrete semantic channels on this data distribution; reserve LD≥8 for continuous (β-VAE) bottlenecks only.
> - Commitment cost serves as VQ-VAE's analogue to β-VAE's β — both control a regularization-vs-fidelity trade-off. However, unlike β in β-VAE (which produces a continuous rate-distortion curve), cc in VQ-VAE produces a flat response when the codebook is structurally over-provisioned.
>
> See Fig. vqvae_usage_heatmap, Fig. vqvae_commitment.

## Table 4: Fixed-Bandwidth Cross-Method MSE Comparison — Scaling Behavior at 64 / 128 / 256 Bits

> Whereas Table 1 shows each method's single best configuration, this table fixes three bandwidth breakpoints (64, 128, 256 bits) and reports the MSE of each method at those budgets. This enables direct comparison of **how reconstruction quality scales with bandwidth within each codec family**, and reveals bandwidth-invariant behavior (e.g., β-VAE β=0.1 at ~0.32 MSE across all three breakpoints). Methods without a configuration at an exact breakpoint use the nearest available configuration. VQ-VAE is excluded because its 5–10 bit operating range does not overlap with these breakpoints (see Table 3 for its full sweep; its best point is listed in Table 1).

| Method | MSE@64b | PSNR@64b | MSE@128b | PSNR@128b | MSE@256b | PSNR@256b |
|--------|---------|----------|----------|-----------|----------|-----------|
| PCA | 0.1908 | 7.2 | 0.1750 | 7.6 | 0.0773 | 11.1 |
| Standard AE | 0.1582 | 8.0 | 0.0720 | 11.4 | 0.0227 | 16.4 |
| Digital Quant. | 0.0039 | 24.1 | 0.0001 | 38.6 | 0.0001 | 38.6 |
| β-VAE β=0.001 | 0.1612 | 7.9 | 0.0841 | 10.8 | 0.0284 | 15.5 |
| β-VAE β=0.01 | 0.1803 | 7.4 | 0.1202 | 9.2 | 0.0873 | 10.6 |
| β-VAE β=0.1 | 0.3200 | 4.9 | 0.3209 | 4.9 | 0.3205 | 4.9 |

> **Scaling observations across bandwidth breakpoints:**
> - **Digital Quantization** shows the steepest improvement from 64b to 128b (MSE drops from 0.0039 to 0.0001), then saturates — MSE@128b and MSE@256b are identical because 128 bits already achieves near-lossless reconstruction at the tested quantization depth.
> - **Standard AE** scales smoothly with bandwidth: each doubling of LD roughly halves MSE (0.1582 → 0.0720 → 0.0227), consistent with the increased representational capacity.
> - **β-VAE β=0.1 exhibits bandwidth-invariant MSE** (~0.32 at all three breakpoints), confirming it operates near the collapse boundary regardless of allocated latent capacity. This is a diagnostic signature of the transition regime: adding more latent dimensions does not improve reconstruction because the KL regularizer suppresses their use.
> - **β-VAE β=0.001 and β=0.01** both improve with bandwidth, but β=0.01's MSE@256b (0.0873) is worse than β=0.001's MSE@128b (0.0841) — illustrating the rate-distortion trade-off: the lower-β model achieves better reconstruction at lower bandwidth because it uses its latent capacity more aggressively (15.8 bits vs. 6.4 bits effective rate).
>
> **Posterior collapse rate:** 100% of β-VAE configurations with β≥0.5 show KL < 0.05 nats across all tested latent dimensions (LD=2, 4, 8, 16, 32). The encoder degenerates completely to the prior N(0,I) — see Fig. kl_collapse, Fig. latent_space.
