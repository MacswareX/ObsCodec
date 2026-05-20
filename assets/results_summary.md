# ObsCodec Results Summary

> **Phase 1 & Extended Benchmark** — 263+ trained models across 7 MPE scenarios (18–90 dim, 3–15 agents)

---

## Phase 1: Single-Scenario Benchmark (simple_spread, 30-dim)

Raw observation = 30 dims × 32-bit float = **960 bits**. All MSE on held-out test split.

### Table 1: Method Comparison Under a 256-bit Nominal Budget

| Method | Config | MSE | Nominal BW | KL Eff. Rate | Nominal Ratio | Eff. Ratio |
|--------|--------|-----|------------|--------------|---------------|------------|
| PCA | n=8 | 0.0773 | 256 | - | 4x | - |
| Standard AE | LD=8 | 0.0223 | 256 | - | 4x | - |
| Digital Quant. | LD=16, B=8 | 0.0001 | 128 | - | 8x | - |
| β-VAE (near-AE) | LD=8, β=0.001 | 0.0353 | 256 | 28.3 | 4x | 34x |
| β-VAE (semantic bottleneck) | LD=8, β=0.01 | 0.0474 | 256 | 13.2 | 4x | 73x |
| β-VAE (transition) | LD=8, β=0.1 | 0.2105 | 256 | 1.9 | 4x | 505x |
| β-VAE (at KL floor) | LD=8, β=1.0 | 0.5136 | 256 | 0.1 | 4x | 9600x |
| VQ-VAE | cb512_ld4_cc0.25 | 0.1283 | 9 | - | 107x | - |

### Table 2: β-VAE LD=8 Rate-Distortion Sweep (free_bits=0.01)

| β | MSE | Nominal BW | Eff. Rate (bits) | KL (nats) | Regime |
|---|-----|------------|------------------|-----------|--------|
| 0.001 | 0.0353 | 256 | 28.3 | 19.62 | high-rate |
| 0.01 | 0.0474 | 256 | 13.2 | 9.17 | semantic bottleneck |
| 0.1 | 0.2105 | 256 | 1.9 | 1.33 | transition |
| 0.2 | 0.2730 | 256 | 1.0 | 0.67 | low-rate |
| 0.3 | 0.3653 | 256 | 0.4 | 0.31 | low-rate |
| 0.5 | 0.4996 | 256 | 0.1 | 0.09 | at KL floor |
| 1.0 | 0.5136 | 256 | 0.1 | 0.07 | at KL floor |
| 2.0 | 0.5189 | 256 | 0.1 | 0.07 | at KL floor |
| 4.0 | 0.5468 | 256 | 0.1 | 0.07 | at KL floor |
| 10.0 | 0.5170 | 256 | 0.1 | 0.06 | at KL floor |

KL spans 300× range (19.6→0.06 nats) before reaching the free-bits floor at β≥0.5.

### Table 3: Best MSE Within Nominal Bandwidth Budgets

| Budget | PCA | AE | Digital | β-VAE | VQ-VAE |
|--------|-----|----|---------|-------|--------|
| 8b | - | - | 0.1829 | - | 0.1617 |
| 16b | - | - | 0.0837 | - | 0.1283 |
| 32b | - | - | 0.0404 | - | 0.1283 |
| 64b | 0.1908 | 0.1547 | 0.0041 | 0.1825 | 0.1283 |
| 128b | 0.1750 | 0.0719 | 0.0001 | 0.0904 | 0.1283 |
| 256b | 0.0773 | 0.0223 | 0.0001 | 0.0353 | 0.1283 |

---

## Extended Benchmark: High-Dimensional Scaling & Collapse Prevention

### Table 4: FB=0.1 Universal Anti-Collapse

| Scenario | Dim | FB=0.0 Collapse Rate | FB=0.1 Collapse Rate | KL@β=2.0, FB=0.1 |
|----------|-----|:-:|:-:|:-:|
| tag_hd | 40 | 80% | **0%** | 1.55 |
| comm_hd | 60 | 100% | **0%** | 1.47 |
| spread_xhd | 90 | 50% | **0%** | 1.56 |

FB=0.1 eliminates posterior collapse across all three high-dimensional scenarios. Without free-bits (FB=0.0), collapse rates range from 50% to 100%.

### Table 5: FB Fine-Sweep on spread_xhd (β=2.0, LD=16)

| FB | MSE | KL (nats) | Regime |
|----|-----|-----------|--------|
| 0.00 | 2.521 | 0.002 | COLLAPSED |
| **0.02** | 1.753 | 0.311 | OK |
| 0.05 | 1.608 | 0.922 | OK |
| 0.07 | 1.528 | 1.161 | OK |
| 0.10 | 1.409 | 1.540 | OK |
| 0.12 | 1.353 | 1.738 | OK |
| 0.15 | 1.307 | 2.016 | OK |
| 0.17 | 1.286 | 2.196 | OK |
| 0.20 | 1.259 | 2.454 | OK |
| 0.25 | 1.252 | 2.883 | OK |

Minimum effective FB dose = **0.02** nats/dim (5× lower than 0.1, 25-100× lower than literature 0.5-2.0). Monotonic MSE improvement across the full sweep range.

### Table 6: Agent-Count Scaling (β=2.0, LD=16, FB=0.1)

| N | Dim | FB=0.0 KL | FB=0.0 Regime | FB=0.1 KL | FB=0.1 Regime | MSE Delta |
|---|-----|-----------|---------------|-----------|---------------|-----------|
| 3 | 18 | 0.001 | COLLAPSED | 1.480 | OK | -39.3% |
| 5 | 30 | 0.002 | COLLAPSED | 1.512 | OK | -35.3% |
| 7 | 42 | 0.001 | COLLAPSED | 1.534 | OK | -37.1% |
| 10 | 60 | 0.002 | COLLAPSED | 1.523 | OK | -37.4% |
| 12 | 72 | 0.001 | COLLAPSED | 1.508 | OK | -38.0% |
| 15 | 90 | 0.003 | COLLAPSED | 1.539 | OK | -37.3% |

Key finding: KL is **dimension-independent** at ~1.5 nats across 18→90 dim range. FB=0.0 collapses at every agent count. FB=0.1 produces 35-39% MSE improvement universally.

### Table 7: VQ-VAE Multi-Scenario + Channel

| Scenario (dim) | Best CB | Clean MSE | AWGN 10dB | AWGN 0dB | Rayleigh 10dB |
|----------------|---------|-----------|-----------|----------|---------------|
| simple_spread (30) | 512 | 0.658 | 0.533 | 0.853 | 0.921 |
| spread_hd (48) | 512 | 0.898 | 0.852 | 1.120 | 1.292 |
| spread_xhd (90) | 512 | 1.093 | 1.087 | 1.320 | 1.612 |

Moderate AWGN (10-20 dB) acts as **denoising regularization** — VQ-VAE MSE at AWGN 10dB is _lower_ than clean channel for simple_spread (0.533 vs 0.658). Rayleigh fading is consistently more destructive than AWGN at equivalent SNR.

### Table 8: Cross-Scenario Unified Codec (β=2.0, LD=16, FB=0.1)

| Scenario | Per-Scenario MSE | Unified MSE | Delta |
|----------|:-:|:-:|:-:|
| simple_spread (30) | 1.07 | 1.04 | -2.8% |
| spread_hd (48) | 1.14 | 1.12 | -1.8% |
| spread_xhd (90) | 1.58 | 1.50 | **-5.0%** |
| All scenarios (combined) | - | 1.29 | - |

A single BetaVAE trained on all 3 scenarios jointly matches or beats per-scenario models. Positive cross-scenario transfer is strongest on the hardest task (spread_xhd, 90-dim).

---

## Regime Classification

| Regime | KL Range | Interpretation |
|--------|----------|----------------|
| OK | KL ≥ 0.10 | Posterior carries information; codec is functional |
| LOW | 0.01 ≤ KL < 0.10 | Partial collapse; marginal information |
| COLLAPSED | KL < 0.01 | Posterior matches prior; latent carries no observation information |

## Interpretation Notes

- **Free-bits anti-collapse**: FB=0.1 universally prevents collapse across all scenarios and agent counts. The minimum effective dose (FB=0.02) is 25-100× lower than literature defaults, suggesting the collapse threshold is much lower than previously assumed.
- **KL dimension-independence**: With FB=0.1, absolute KL stays ~1.5 nats regardless of observation dimensionality (18-90 dim). The per-dimension free-bits floor means total KL depends only on how many dimensions exceed the floor — constant across scales.
- **Decoder expansion is not a solution**: Sweeping decoder hidden dimensions from 1× to 4× the encoder shows zero anti-collapse effect. The collapse bottleneck is in the rate (KL) term, not the decoder's capacity.
- **VQ-VAE channel effects**: AWGN at moderate SNR serves as implicit denoising regularization for VQ-VAE. Rayleigh fading is more destructive, especially at low SNR.
- **Cross-scenario transfer**: The unified codec's superior performance on the hardest scenario (spread_xhd) suggests shared representations benefit high-dimensional tasks via regularization from simpler scenarios.
- All β-VAE results use the corrected free-bits logic: `max(0, KL_per_dim.mean(dim=0) - free_bits).sum()` — per-dimension batch-averaged clamping, not per-sample.
- Reconstruction MSE is a proxy metric. Phase 3 scripts (7-10) instrument task-aware, JSCC, and end-to-end closed-loop evaluation — results below document the experimental designs pending execution at scale.
- All β-VAE results use the corrected free-bits logic: `max(0, KL_per_dim.mean(dim=0) - free_bits).sum()` — per-dimension batch-averaged clamping, not per-sample.

---

## Phase 3: Semantic Communication — Experimental Designs

Scripts 7-10 implement four semantic communication experiments. Results below describe the
experimental design, measured quantities, and expected outcomes. (Full execution requires GPU
time; scripts are ready to run.)

### Table 9: Differentiable Channel Benchmark (`7_diff_channel.py`)

| Train Channel | Test Condition | Codec | β | FB | Measured Metric |
|---------------|---------------|-------|---|---|-----------------|
| none (baseline) | clean | BetaVAE | 2.0 | 0.1 | MSE, KL |
| DiffAWGN 20dB | clean, matched 20dB, mismatched 10/5/0dB | JSCC-BetaVAE | 2.0 | 0.1 | MSE per test SNR |
| DiffAWGN 10dB | clean, matched 10dB, mismatched 20/5/0dB | JSCC-BetaVAE | 2.0 | 0.1 | MSE per test SNR |
| DiffAWGN 5dB | clean, matched 5dB, mismatched 20/10/0dB | JSCC-BetaVAE | 2.0 | 0.1 | MSE per test SNR |
| DiffAWGN 0dB | clean, matched 0dB, mismatched 20/10/5dB | JSCC-BetaVAE | 2.0 | 0.1 | MSE per test SNR |
| DiffErasure 10% | clean, matched 10%, mismatched 30% | JSCC-BetaVAE | 2.0 | 0.1 | MSE per test loss rate |
| DiffErasure 30% | clean, matched 30%, mismatched 10% | JSCC-BetaVAE | 2.0 | 0.1 | MSE per test loss rate |

Tests whether training with differentiable channels improves robustness to matched AND
mismatched channel conditions (a core JSCC hypothesis). The reparameterization trick enables
gradient flow through AWGN; straight-through estimator handles erasure.

### Table 10: JSCC Training Grid (`8_jscc_training.py`)

| Dimension | Values |
|-----------|--------|
| Scenarios | simple_spread (30d), spread_hd (48d), spread_xhd (90d) |
| Codecs | BetaVAE β=0.1, BetaVAE β=2.0, VQ-VAE (cb=512) |
| Free Bits | 0.0, 0.1 |
| Train Channels | clean, AWGN (20/10/0 dB), Erasure (10%, 30%) |
| Test Channels | clean, AWGN (20/10/0/-5 dB), Erasure (0/10/30%) |
| Metrics per config | MSE, KL, NMSE (normalized by data variance) |

Full factorial: 3 scenarios × 3 codecs × 2 FB × 6 train channels × 8 test channels = up to
864 data points. This is the central JSCC experiment testing whether channel-in-the-loop
training generalizes across codec families and channel types.

### Table 11: Task-Aware Loss (`9_task_aware.py`)

| Parameter | Values |
|-----------|--------|
| β | 0.01, 0.1, 1.0 |
| Task Loss Types | none (baseline), self_only (MSE on first 2 dims), weighted (0.7×self + 0.3×others) |
| Task Weights | 0.01, 0.1, 0.5, 1.0 |
| Scenarios | simple_spread, simple_tag, simple_world_comm |

**Key question**: Can task-aware loss substitute for or complement free-bits in maintaining
latent activity? Measured metrics: total MSE, self-position MSE, other-agent MSE,
coordination gap (ratio of self to others error), KL divergence, per-agent MSE breakdown.

### Table 12: End-to-End Closed-Loop (`10_end_to_end.py`)

| Condition | Compression | Channel | Description |
|-----------|-------------|---------|-------------|
| no_compression | None | Clean | Raw observations → heuristic policy (upper bound) |
| jscc_clean | JSCC-BetaVAE (β=2.0, FB=0.1) | Clean | Encode→decode→policy (compression cost only) |
| jscc_noisy | JSCC-BetaVAE (β=2.0, FB=0.1) | AWGN (configurable SNR) | Full pipeline including channel noise |

10 episodes per condition, 200-step rollouts. Metrics: final distance to targets (avg last
10 steps), early/late mean distance, path efficiency, collision count. A SpreadSimulator
with 5 agents and 5 landmarks closes the observation→encode→channel→decode→policy→action
loop.

### Phase 3 Library Additions

| Module | Purpose |
|--------|---------|
| `obscodec/channel/diff_channel.py` | Differentiable AWGN (reparameterization) and Erasure (straight-through) layers |
| `obscodec/models/jscc.py` | JSCCWrapper — composes a base codec with a differentiable channel |
| `obscodec/task_metrics.py` | Task-aware evaluation: self-position MSE, coordination error, per-agent metrics |
