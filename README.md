# ObsCodec — Semantic Codec Benchmark for Multi-Agent Coordination

ObsCodec evaluates compression codecs for multi-agent trajectory observations under the lens of **semantic communication** — where the distortion metric is task-aware and the channel is unreliable.

## Route B: High-Dimensional Scaling + Collapse Prevention — COMPLETE (100%)

This branch follows **Route B**: aggressive dimensionality scaling (up to 90-dim, 15 agents), posterior collapse prevention via free-bits, channel impairment robustness testing, and cross-scenario generalization — all as pre-adaptation for Phase 3 (joint semantic communication + MARL).

## Quickstart

```bash
pip install -e .
python scripts/1_collect_data.py --all       # Generate 7 scenarios
python scripts/2_train_baselines.py          # PCA + AE + Digital
python scripts/3_train_vae.py --phase all    # Beta-VAE pipeline (4 phases)
python scripts/4_train_vqvae.py              # VQ-VAE + channel
python scripts/5_generate_figures.py         # All figures
python scripts/6_summary_table.py            # Final report
```

**Supplementary Route B experiments** (completed):
```bash
python scripts/3b_fb_finesweep.py            # FB fine-sweep 0.02-0.25
python scripts/3c_agent_scaling.py           # Agent-count scaling N=3-15
python scripts/3d_unified_codec.py           # Cross-scenario unified codec
python scripts/4b_vqvae_multiscenario.py     # VQ-VAE multi-scenario + channel
```

## Repository Structure

```
ObsCodec/
├── obscodec/           # Core library
│   ├── models/         # PCA, AE, Digital (baselines) + beta-VAE + VQ-VAE
│   ├── channel/        # AWGN, Rayleigh fading, packet loss, adaptive allocation
│   ├── data/           # Synthetic multi-agent trajectory generators (7 scenarios)
│   ├── config.py       # Central configuration
│   ├── metrics.py      # Evaluation metrics (MSE, KL, rate, codebook usage)
│   ├── trainer.py      # Training loops
│   └── visualize.py    # Figure generation utilities
├── scripts/            # Experiment pipeline (11 scripts, numbered)
├── data/               # Generated .npy observation files
├── assets/             # Results JSONs + figures + project blurbs
└── checkpoints/        # Trained model weights (gitignored)
```

## Codec Comparison

| Codec | Latent Type | Rate Measure | Collapse Risk | Status |
|-------|------------|--------------|---------------|--------|
| PCA | continuous | LD (dim count) | N/A | Done |
| AE | continuous | LD | N/A | Done |
| Digital | discrete | bits/dim x dim | N/A | Done |
| beta-VAE | stochastic | KL nats -> bits | **SOLVED (FB=0.1)** | Done |
| VQ-VAE | discrete | log2(CB) x LD | N/A | Done |

## Key Finding: FB=0.1 Universal Anti-Collapse

Free-bits at lambda=0.1 nats/dim **universally prevents posterior collapse** across all scenarios AND all agent counts:

| Scenario | FB=0.0 Collapse | FB=0.1 Collapse | KL@beta=2.0 |
|----------|:-:|:-:|:-:|
| tag_hd (40-dim) | 80% | **0%** | 1.55 |
| comm_hd (60-dim) | 100% | **0%** | 1.47 |
| spread_xhd (90-dim) | 50% | **0%** | 1.56 |

**FB fine-sweep**: Minimum effective dose = **0.02 nats/dim** — 5x lower than 0.1, 25-100x lower than literature defaults (0.5-2.0).

## Agent-Count Scaling (N=3→15)

FB=0.1 maintains KL at ~1.5 nats across all scales (18-90 dim). FB=0.0 collapses at every N. MSE improvement 35-39% at all scales.

| N | Dim | FB=0.0 Regime | FB=0.1 Regime | MSE Delta |
|---|-----|--------------|--------------|-----------|
| 3 | 18 | COLLAPSED | OK | -39.3% |
| 5 | 30 | COLLAPSED | OK | -35.3% |
| 7 | 42 | COLLAPSED | OK | -37.1% |
| 10 | 60 | COLLAPSED | OK | -37.4% |
| 12 | 72 | COLLAPSED | OK | -38.0% |
| 15 | 90 | COLLAPSED | OK | -37.3% |

## VQ-VAE Multi-Scenario + Channel

| Scenario | Best CB | Clean MSE | AWGN 10dB | AWGN 0dB | Rayleigh 10dB |
|----------|---------|-----------|-----------|----------|----------------|
| simple_spread (30-dim) | 512 | 0.658 | 0.533 | 0.853 | 0.921 |
| spread_hd (48-dim) | 512 | 0.898 | 0.852 | 1.120 | 1.292 |
| spread_xhd (90-dim) | 512 | 1.093 | 1.087 | 1.320 | 1.612 |

## Unified Codec

A single BetaVAE trained on all 3 scenarios matches or beats per-scenario models (spread_xhd: -5.0% MSE). Positive cross-scenario transfer.

## Channel Impairments

Six channel models for robustness testing: AWGN, Rayleigh Fading (iid/block/agent-block modes), Packet Loss, Burst Packet Loss, Heterogeneous SNR, Composite (chained impairments).

## Key Research Results

1. **FB=0.1 universally prevents collapse** across all scenarios and agent counts (3→15)
2. **Minimum effective FB dose = 0.02** — 25-100x lower than literature (0.5-2.0)
3. **Decoder expansion alone has zero anti-collapse effect** — bottleneck is in the rate term
4. **Cross-scenario dynamics dominate** over raw dimensionality for collapse behavior
5. **KL is dimension-independent** at ~1.5 nats across 18-90 dim range
6. **Unified codec beats per-scenario** on hardest task (5% MSE reduction)
7. **AWGN at moderate SNR improves VQ-VAE MSE** (denoising regularization effect)

## Route B Completion: 11/11 (100%) — 263 models, 15 results JSONs, 9 figures

Phase 3 (Semantic Communication: task-aware compression + JSCC) is the next phase — see `scripts/6_summary_table.py` for the detailed plan.

## License

MIT
