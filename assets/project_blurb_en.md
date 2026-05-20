# ObsCodec Project Overview

## Research Question

How to efficiently compress multi-agent observations for coordination under bandwidth constraints and unreliable channels? Traditional codecs (PCA, AE, scalar quantization) optimize for reconstruction fidelity, but semantic communication demands a balance between "usefulness for downstream tasks" and "transmission robustness."

## Approach

This project benchmarks five codec families — PCA (linear), Standard AE (nonlinear reconstruction), Scalar Quantization (digital baseline), β-VAE (probabilistic semantic bottleneck), and VQ-VAE (discrete codebook) — under a unified framework across **7 MPE scenarios** (18-90 dim, 3-15 agents).

**The extended benchmark** focuses on high-dimensional scaling and collapse prevention: β-VAE posterior collapse at 90-dim observations is fully resolved via free-bits (FB=0.1 universal anti-collapse, minimum effective dose 0.02 nats/dim), VQ-VAE is evaluated under 6 channel impairment models across scenarios, and a unified codec demonstrates positive cross-scenario transfer.

## Key Findings

1. **FB=0.1 universally prevents posterior collapse** across all scenarios and agent counts (3→15) — collapse rate 90%→0%
2. **Minimum effective FB dose = 0.02 nats/dim** — 25-100× lower than literature values (0.5-2.0)
3. **KL is dimension-independent** at ~1.5 nats across 18-90 dim, with 35-39% MSE improvement
4. **Decoder expansion alone has zero anti-collapse effect** — the bottleneck is in the rate term, not decoder capacity
5. **AWGN at moderate SNR improves VQ-VAE MSE** — denoising regularization effect
6. **Unified codec beats per-scenario models** — 5.0% MSE reduction on hardest task
7. **Cross-scenario dynamics dominate** over raw dimensionality for collapse behavior

## Technical Stack

Python 3.10+, PyTorch 2.6.0, CUDA 12.6, RTX 3050 (8 GB)

## Scale

**263 trained models, 15 results JSONs, 17 figures, 13 datasets** (33,333 samples each), 15 experiment scripts (incl. 4 semantic communication scripts)

## Phase 3 Complete

Semantic communication experimental designs complete: differentiable channels (AWGN reparameterization, erasure straight-through), joint source-channel coding (JSCC-BetaVAE, JSCC-VQ-VAE), task-aware compression loss (self_only, weighted MSE), end-to-end closed-loop prototype (obs→encode→channel→decode→policy→task). Library additions: `channel/diff_channel.py`, `models/jscc.py`, `task_metrics.py`.
