# ObsCodec Results Summary

> Raw observation = 18 dims × 32 bits = **576 bits**. All MSE on held-out test set.

## Table 1: Method Comparison — Best Config at Comparable Bandwidth (~64–256 bits)

> Digital Quantization achieves MSE<1e-4 at 128 bits; higher-bandwidth configurations (e.g. LD=16 B=32 → 512 bits → MSE<1e-6) exist but are excluded for comparability. VQ-VAE operates at a different bandwidth regime (5–10 bits, see Table 3).

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

> **RD Efficiency** = 1/(MSE × BW), higher is better. Measures distortion reduction per bit.
> See Fig. rate_distortion, Fig. pareto_frontier.
> **Key finding**: β-VAE β=0.01 compresses the latent channel to a **6.4-bit effective information rate** as measured by KL divergence. For reference, the raw observation occupies 576 nominal bits (18 dims × 32-bit float). The factor of ~90× reflects the ratio of raw storage to information-theoretic content — actual deployed compression requires entropy coding and channel modeling.
> β≥0.5 triggers posterior collapse: KL→0, MSE→0.545 (prior variance), rate→0.

## Table 2: β-VAE LD=8 — Rate-Distortion vs β

| β | MSE | PSNR (dB) | BW (equiv) | Eff. Rate (bits) | KL (nats) | Regime |
|---|-----|-----------|------------|-------------------|-----------|--------|
| 0.001 | 0.0284 | 15.5 | 256 | 15.8 | 22.8256 | ✓ effective |
| 0.01 | 0.0873 | 10.6 | 256 | 6.4 | 9.1929 | ✓ effective |
| 0.1 | 0.3205 | 4.9 | 256 | 0.8 | 1.1112 | ⚠ transition |
| 0.5 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 1.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 2.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 5.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |
| 10.0 | 0.5455 | 2.6 | 256 | 0.0 | 0.0000 | ✗ collapsed |

> β≤0.01: effective information bottleneck regime. β≥0.5: full posterior collapse across all configurations. See Fig. kl_collapse, Fig. ablation_heatmap, Fig. latent_space.

## Table 3: VQ-VAE Commitment Cost Sweep (LD=8, CB=256)

| cc | MSE | PSNR (dB) | BW | Codebook Usage | RD Efficiency | Note |
|-----|-----|-----------|----|----------------|---------------|------|
| 0.01 | 0.2204 | 6.6 | 8 | 8.6% | 0.57 | — underused |
| 0.05 | 0.2071 | 6.8 | 8 | 14.1% | 0.60 | — underused |
| 0.1 | 0.2177 | 6.6 | 8 | 9.0% | 0.57 | — underused |
| 0.25 | 0.2080 | 6.8 | 8 | 13.7% | 0.60 | — underused |
| 0.5 | 0.2253 | 6.5 | 8 | 7.0% | 0.55 | — underused |
| 1.0 | 0.2133 | 6.7 | 8 | 10.9% | 0.59 | — underused |
| 2.0 | 0.2083 | 6.8 | 8 | 11.3% | 0.60 | — underused |
| 5.0 | 0.2198 | 6.6 | 8 | 8.2% | 0.57 | — underused |

> Commitment cost serves as VQ-VAE's analogue to β-VAE's β — controlling the trade-off between codebook adherence and reconstruction fidelity. At LD=8, codebook usage never exceeds 14%, in contrast to 100% at LD=2 — low-dimensional spaces saturate discrete codebooks more efficiently. See Fig. vqvae_usage_heatmap.

## Table 4: Pareto Ranking — Best MSE at Key Bandwidths

| Method | MSE@64b | PSNR@64b | MSE@128b | PSNR@128b | MSE@256b | PSNR@256b |
|--------|---------|----------|----------|-----------|----------|-----------|
| PCA | 0.1908 | 7.2 | 0.1750 | 7.6 | 0.0773 | 11.1 |
| Standard AE | 0.1582 | 8.0 | 0.0720 | 11.4 | 0.0227 | 16.4 |
| Digital Quant. | 0.0039 | 24.1 | 0.0001 | 38.6 | 0.0001 | 38.6 |
| β-VAE β=0.001 | 0.1612 | 7.9 | 0.0841 | 10.8 | 0.0284 | 15.5 |
| β-VAE β=0.01 | 0.1803 | 7.4 | 0.1202 | 9.2 | 0.0873 | 10.6 |
| β-VAE β=0.1 | 0.3200 | 4.9 | 0.3209 | 4.9 | 0.3205 | 4.9 |

> **Posterior collapse rate**: 100% of configurations with β≥0.5 show KL<0.05 — the encoder degenerates completely to the prior N(0,I). See Fig. kl_collapse, Fig. latent_space.
>
> **Note**: VQ-VAE is excluded from this table because it operates in the 5–10 bit bandwidth regime (see Table 3), which is not directly comparable to the 64–256 bit methods listed here. VQ-VAE's best point (cb256_ld2_cc0.25, 8 bits, MSE=0.1756) is shown in Table 1.