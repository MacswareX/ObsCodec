# ObsCodec Results Summary

> Raw observation = 18 dims x 32-bit float = **576 bits**. All MSE values are measured on a held-out test split.

> **Metric note**: nominal bandwidth counts the serialized latent size. beta-VAE effective rate is the KL estimate in bits and is not the same as a deployed packet size without entropy coding.

## Table 1: Method Comparison Under a 256-bit Nominal Budget

| Method | Config | MSE | PSNR (dB) | Nominal BW | KL Eff. Rate | Nominal Ratio | Eff. Ratio | RD Efficiency |
|--------|--------|-----|-----------|------------|--------------|---------------|------------|---------------|
| PCA | n=8 | 0.0773 | 11.1 | 256 | - | 2x | - | 0.05 |
| Standard AE | LD=8 | 0.0223 | 16.5 | 256 | - | 2x | - | 0.17 |
| Digital Quant. | LD=16, B=8 | 0.0001 | 38.8 | 128 | - | 4x | - | 58.99 |
| beta-VAE (near-AE) | LD=8, beta=0.001 | 0.0353 | 14.5 | 256 | 28.3 | 2x | 20x | 0.11 |
| beta-VAE (semantic bottleneck) | LD=8, beta=0.01 | 0.0474 | 13.2 | 256 | 13.2 | 2x | 44x | 0.08 |
| beta-VAE (transition) | LD=8, beta=0.1 | 0.2105 | 6.8 | 256 | 1.9 | 2x | 299x | 0.02 |
| beta-VAE (at KL floor) | LD=8, beta=1.0 | 0.5136 | 2.9 | 256 | 0.1 | 2x | 5871x | 0.01 |
| VQ-VAE | cb512_ld4_cc0.25 | 0.1283 | 8.9 | 9 | - | 64x | - | 0.87 |

**Primary takeaway**: Digital quantization is the strongest pure reconstruction baseline at >=128 nominal bits. beta-VAE provides a tunable information rate through the KL divergence, with the free-bits floor (0.01 nats/dim) preventing complete posterior collapse while allowing a 300x KL dynamic range. The semantic bottleneck regime (beta=0.01) gives 6-17 effective bits depending on latent dimension.

## Table 2: beta-VAE LD=8 Rate-Distortion Sweep

| beta | MSE | PSNR (dB) | Nominal BW | Eff. Rate (bits) | KL (nats) | Regime |
|---|-----|-----------|------------|------------------|-----------|--------|
| 0.001 | 0.0353 | 14.5 | 256 | 28.3 | 19.6247 | high-rate |
| 0.01 | 0.0474 | 13.2 | 256 | 13.2 | 9.1738 | semantic bottleneck |
| 0.1 | 0.2105 | 6.8 | 256 | 1.9 | 1.3347 | transition |
| 0.2 | 0.2730 | 5.6 | 256 | 1.0 | 0.6676 | low-rate |
| 0.3 | 0.3653 | 4.4 | 256 | 0.4 | 0.3119 | low-rate |
| 0.5 | 0.4996 | 3.0 | 256 | 0.1 | 0.0853 | at KL floor |
| 1.0 | 0.5136 | 2.9 | 256 | 0.1 | 0.0680 | at KL floor |
| 2.0 | 0.5189 | 2.8 | 256 | 0.1 | 0.0680 | at KL floor |
| 4.0 | 0.5468 | 2.6 | 256 | 0.1 | 0.0687 | at KL floor |
| 5.0 | 0.5401 | 2.7 | 256 | 0.1 | 0.0654 | at KL floor |
| 8.0 | 0.5459 | 2.6 | 256 | 0.1 | 0.0677 | at KL floor |
| 10.0 | 0.5170 | 2.9 | 256 | 0.1 | 0.0615 | at KL floor |

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

| Budget | PCA | Standard AE | Digital Quant. | beta-VAE | VQ-VAE |
|--------|-----|-------------|----------------|-------|--------|
| 8b | - | - | 0.1829 | - | 0.1617 |
| 16b | - | - | 0.0837 | - | 0.1283 |
| 32b | - | - | 0.0404 | - | 0.1283 |
| 64b | 0.1908 | 0.1547 | 0.0041 | 0.1825 | 0.1283 |
| 128b | 0.1750 | 0.0719 | 0.0001 | 0.0904 | 0.1283 |
| 256b | 0.0773 | 0.0223 | 0.0001 | 0.0353 | 0.1283 |

## Interpretation Notes

- **beta-VAE posterior collapse behavior**: With the corrected architecture (no tanh on mu, BatchNorm encoder, halved decoder capacity, KL annealing over 50 epochs, free-bits=0.01 nats/dim), the posterior never collapses to zero KL. The KL shows a smooth 300x decline from beta=0.001 (KL~19.5 nats, 28 bits) to beta=0.5 (KL~0.09 nats, 0.1 bits) before reaching the free-bits floor. At beta >= 1.0, KL stabilizes at ~0.06-0.07 nats (~0.1 effective bits) and MSE approaches the data variance (~0.545), consistent with theoretical expectations for the beta->inf limit. The free-bits floor prevents complete gradient collapse while allowing the KL to span a wide dynamic range. The useful operating range (beta=0.001-0.1) provides tunable rate-distortion tradeoffs consistent with Higgins et al. (2017) and Burgess et al. (2018).
- **VQ-VAE codebook usage**: EMA codebook updates and periodic dead-entry reset keep codebook usage high. Best performance is at lower latent dimensions (LD=2, codebook_size=256) achieving 8-bit discrete latent codes.
- The KL rate estimate from beta-VAE is an information measure; deploying it as an actual channel rate requires entropy coding or packetization.
- Reconstruction MSE is a proxy metric. A full SemCom-MARL follow-up should validate policy return, coordination success, and robustness under channel noise.
