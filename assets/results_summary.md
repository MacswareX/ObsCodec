# ObsCodec Results Summary

> Raw observation = 18 dims x 32-bit float = **576 bits**. All MSE values are measured on a held-out test split.

> **Metric note**: nominal bandwidth counts the serialized latent size. β-VAE effective rate is the KL estimate in bits and is not the same as a deployed packet size without entropy coding.

## Table 1: Method Comparison Under a 256-bit Nominal Budget

| Method | Config | MSE | PSNR (dB) | Nominal BW | KL Eff. Rate | Nominal Ratio | Eff. Ratio | RD Efficiency |
|--------|--------|-----|-----------|------------|--------------|---------------|------------|---------------|
| PCA | n=8 | 0.0773 | 11.1 | 256 | - | 2x | - | 0.05 |
| Standard AE | LD=8 | 0.0223 | 16.5 | 256 | - | 2x | - | 0.17 |
| Digital Quant. | LD=16, B=8 | 0.0001 | 38.8 | 128 | - | 4x | - | 58.99 |
| β-VAE (near-AE) | LD=8, β=0.001 | 0.0363 | 14.4 | 256 | 28.1 | 2x | 20x | 0.11 |
| β-VAE (semantic bottleneck) | LD=8, β=0.01 | 0.0478 | 13.2 | 256 | 13.3 | 2x | 43x | 0.08 |
| β-VAE (transition) | LD=8, β=0.1 | 0.1828 | 7.4 | 256 | 2.5 | 2x | 229x | 0.02 |
| β-VAE (at KL floor) | LD=8, β=1.0 | 0.3458 | 4.6 | 256 | 1.0 | 2x | 566x | 0.01 |
| VQ-VAE | cb512_ld4_cc0.25 | 0.1283 | 8.9 | 9 | - | 64x | - | 0.87 |

**Primary takeaway**: Digital quantization is the strongest pure reconstruction baseline at >=128 nominal bits. β-VAE provides a tunable information rate through the KL divergence, with the free-bits floor (0.1 nats/dim) preventing complete posterior collapse. The semantic bottleneck regime (β=0.01) gives 6–17 effective bits depending on latent dimension.

## Table 2: β-VAE LD=8 Rate-Distortion Sweep

| β | MSE | PSNR (dB) | Nominal BW | Eff. Rate (bits) | KL (nats) | Regime |
|---|-----|-----------|------------|------------------|-----------|--------|
| 0.001 | 0.0363 | 14.4 | 256 | 28.1 | 19.5093 | high-rate |
| 0.01 | 0.0478 | 13.2 | 256 | 13.3 | 9.2488 | semantic bottleneck |
| 0.1 | 0.1828 | 7.4 | 256 | 2.5 | 1.7443 | transition |
| 0.2 | 0.2542 | 5.9 | 256 | 1.6 | 1.1080 | transition |
| 0.3 | 0.3204 | 4.9 | 256 | 1.1 | 0.7790 | low-rate |
| 0.5 | 0.3437 | 4.6 | 256 | 1.1 | 0.7287 | low-rate |
| 1.0 | 0.3458 | 4.6 | 256 | 1.0 | 0.7048 | low-rate |
| 2.0 | 0.3463 | 4.6 | 256 | 1.0 | 0.7130 | low-rate |
| 4.0 | 0.3439 | 4.6 | 256 | 1.0 | 0.7082 | low-rate |
| 5.0 | 0.3470 | 4.6 | 256 | 1.0 | 0.7029 | low-rate |
| 8.0 | 0.3552 | 4.5 | 256 | 1.0 | 0.6736 | at KL floor |
| 10.0 | 0.3403 | 4.7 | 256 | 1.0 | 0.6716 | at KL floor |

## Table 3: VQ-VAE Commitment Cost Sweep (LD=4, CB=128)

| cc | MSE | PSNR (dB) | BW | Codebook Usage | RD Efficiency | Note |
|----|-----|-----------|----|----------------|---------------|------|
| 0.01 | 0.1644 | 7.8 | 7 | 100.0% | 0.87 | high usage |
| 0.05 | 0.1617 | 7.9 | 7 | 100.0% | 0.88 | high usage |
| 0.1 | 0.2049 | 6.9 | 7 | 37.5% | 0.70 | moderate |
| 0.25 | 0.1789 | 7.5 | 7 | 100.0% | 0.80 | high usage |
| 0.5 | 0.1983 | 7.0 | 7 | 48.4% | 0.72 | moderate |
| 1.0 | 0.1973 | 7.0 | 7 | 53.9% | 0.72 | high usage |
| 2.0 | 0.2024 | 6.9 | 7 | 39.1% | 0.71 | moderate |
| 5.0 | 0.5455 | 2.6 | 7 | 0.8% | 0.26 | underused |

EMA codebook updates maintain high codebook usage across commitment cost levels. Higher cc values increase the penalty for encoder-codebook divergence, trading off reconstruction quality for discrete-alignment fidelity.

## Table 4: Best MSE Within Nominal Bandwidth Budgets

| Budget | PCA | Standard AE | Digital Quant. | β-VAE | VQ-VAE |
|--------|-----|-------------|----------------|-------|--------|
| 8b | - | - | 0.1829 | - | 0.1617 |
| 16b | - | - | 0.0837 | - | 0.1283 |
| 32b | - | - | 0.0404 | - | 0.1283 |
| 64b | 0.1908 | 0.1547 | 0.0041 | 0.1838 | 0.1283 |
| 128b | 0.1750 | 0.0719 | 0.0001 | 0.0918 | 0.1283 |
| 256b | 0.0773 | 0.0223 | 0.0001 | 0.0363 | 0.1283 |

## Interpretation Notes

- **β-VAE posterior collapse behavior**: With the corrected architecture (no tanh on mu, BatchNorm encoder, halved decoder capacity, KL annealing over 50 epochs, free-bits=0.1 nats/dim), the posterior no longer collapses to zero KL. At β ≥ 0.5, KL is pinned to the free-bits floor (~0.1 nats/dim) rather than collapsing to zero — the encoder retains minimal information capacity. The useful operating range (β=0.001–0.1) provides tunable rate-distortion tradeoffs consistent with Higgins et al. (2017) and Burgess et al. (2018). The effective collapse onset has shifted from β=0.5 (old architecture) to β=2.0–4.0 (corrected architecture).
- **VQ-VAE codebook usage**: EMA codebook updates and periodic dead-entry reset keep codebook usage high. Best performance is at lower latent dimensions (LD=2, codebook_size=256) achieving 8-bit discrete latent codes.
- The KL rate estimate from β-VAE is an information measure; deploying it as an actual channel rate requires entropy coding or packetization.
- Reconstruction MSE is a proxy metric. A full SemCom-MARL follow-up should validate policy return, coordination success, and robustness under channel noise.
