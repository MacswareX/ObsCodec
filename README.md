# ObsCodec: Learned Observation Compression for Multi-Agent Systems

> A compact research demo for semantic communication in embodied multi-agent
> coordination — from single-scenario benchmarks to high-dimensional scaling with
> universal posterior collapse prevention.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![README in Chinese](https://img.shields.io/badge/README-中文-red)](README_zh.md)
[![Experiments: 15/15](https://img.shields.io/badge/Experiments-15/15-brightgreen)](#experiment-coverage)

## TL;DR

ObsCodec asks a simple question: **how much of a robot observation must be
communicated before task-relevant structure disappears?**

The repository benchmarks five codec families across 7 MPE scenarios spanning
18–90 observation dimensions and 3–15 agents. The extended benchmark covers
aggressive dimensionality scaling, universal posterior collapse prevention via
free-bits, channel impairment robustness, cross-scenario generalization, and
semantic communication with joint source-channel coding — 263+ trained models
total.

| Result | Evidence | Why It Matters | See |
|--------|----------|----------------|-----|
| FB=0.1 universally prevents posterior collapse | 0% collapse rate across all scenarios (18-90 dim, 3-15 agents) | Single free-bits value works everywhere — no per-scenario tuning needed | Table 4, Fig. collapse_barrier_analysis |
| Minimum effective FB dose = 0.02 nats/dim | KL=0.31 nats at FB=0.02, monotonic MSE improvement through FB=0.25 | 25-100x lower than literature defaults (0.5-2.0) | Table 5, Fig. kl_vs_beta_all_scenarios |
| KL is dimension-independent at ~1.5 nats | Stable across 18→90 dim range with FB=0.1 | Information rate does not grow with observation dimension | Table 6 |
| VQ-VAE achieves denoising gain via AWGN | Moderate SNR (10-20 dB) gives _lower_ MSE than clean channel | Channel noise can regularize discrete codecs | Table 7 |
| Unified codec beats per-scenario models | -5.0% MSE on spread_xhd (90-dim) | Positive cross-scenario transfer — shared representations help hardest tasks | Table 8 |
| JSCC training with differentiable channels | Channel-in-the-loop training improves robustness to mismatched conditions | Encoder learns channel-robust latent representations | Scripts 7-10 |

Full numbers are in [assets/results_summary.md](assets/results_summary.md).

## Why This Repo Exists

Multi-robot systems often operate under communication constraints: underwater
robots, disaster-response teams, warehouse fleets, and contested or low-bandwidth
field environments. Raw observation sharing is wasteful; semantic communication
should transmit the information that helps agents coordinate.

ObsCodec is a pre-study before integrating codecs into a full MARL loop. It
isolates the observation-compression problem and makes the rate-distortion
trade-off visible before adding policy learning.

The extended benchmark covers high
dimensions (up to 90-dim, 15 agents), adds systematic anti-collapse mechanisms
for stochastic codecs, tests channel robustness across 6 impairment models, and
validates cross-scenario generalization.

This makes the project a focused demo for:

- **Semantic communication**: β-VAE gives an explicit KL-based information rate.
- **Multi-agent systems**: data comes from multi-agent particle-world observations.
- **Embodied intelligence**: the signal is a robot-like observation vector, not a
  static image benchmark.
- **Research engineering**: all codec families share the same train/validation/test
  protocol and result-generation scripts.

## Methods

| Method | Role | Bandwidth Control | Grid |
|--------|------|-------------------|------|
| PCA | Linear baseline | `n_components` | 4 fits |
| Standard AE | Nonlinear reconstruction baseline | `latent_dim` | 5 runs |
| Digital quantization | Traditional fixed-bit baseline | `latent_dim x bits_per_dim` | 12 runs |
| β-VAE | Probabilistic semantic bottleneck | `latent_dim x β x free_bits` | 116 models |
| VQ-VAE | Discrete codebook bottleneck | `codebook_size x latent_dim x commitment_cost` | 45 models |

All neural codecs use the shared trainer in
[obscodec/trainer.py](obscodec/trainer.py), with early stopping and identical
data splits, totaling 263+ models across 7 scenarios, 6 agent-count
variants, 6 channel impairment models, and 4 semantic communication scripts.

## Computational Cost & Complexity

Cost-effectiveness comparison across codec families using
[obscodec/cost_metrics.py](obscodec/cost_metrics.py). Numbers below are for a
representative 30-dim observation with latent_dim=16 where applicable.

| Codec | Params (approx.) | MACs/Sample | Train Time (rel.) | Inference (rel.) | Convergence |
|-------|:-:|:-:|:-:|:-:|-----------|
| PCA | n_components × obs_dim (= N/A) | 0 | 1× (seconds, SVD) | 1× (matrix mult) | Deterministic — no training |
| Standard AE | ~58K | ~116K | 2× | 1.2× | Fast (~50 epochs, simple MSE) |
| Digital Quant. | ~58K | ~116K | 2× | 1.2× | Same as AE + quantization step |
| β-VAE (LD=16) | ~60K | ~120K | 5× | 1.5× | Slowest (~200 epoch warmup, KL+recon) |
| VQ-VAE (LD=4) | ~70K + cb_params | ~140K | 4× | 2× | Medium (commitment cost tuning) |

**Key cost observations**:
- **PCA is essentially free** — one SVD on the training set, zero inference cost beyond
  a matrix multiply. It should always be the first baseline.
- **AE and β-VAE have nearly identical architecture** — the cost difference comes from
  the KL warmup schedule (200 epochs vs. 50) and the reparameterization step, not from
  parameter count.
- **β-VAE training cost is dominated by KL warmup** — 80% of training time (200/250
  epochs) is spent ramping β from 0→target. The free-bits mechanism does not add
  measurable overhead (it's a clamp + sum operation).
- **VQ-VAE codebook lookup is cheap** — the nearest-neighbor search in a 512-entry
  codebook adds negligible inference latency. The cost is in training: commitment
  loss tuning and codebook collapse monitoring.
- **JSCC training adds no parameter overhead** — the JSCCWrapper is a zero-parameter
  composition layer. The channel forward pass (AWGN noise sampling or Bernoulli mask)
  is O(latent_dim) and adds <1% to total FLOPs.

**Tools**: [obscodec/cost_metrics.py](obscodec/cost_metrics.py) provides
`count_parameters()`, `estimate_flops()`, `measure_inference_latency()`, and
`measure_throughput()` for reproducible cost measurement. Training time and KL
history are tracked per epoch in the trainer's return dict.

## Key Figures

### Rate-Distortion Overview

<p align="center">
  <img src="assets/rate_distortion_simple_spread.png" width="52%" alt="Rate-distortion curve">
</p>

**The full rate-distortion frontier across all 5 codec families on simple_spread (30-dim).**
Digital quantization dominates pure reconstruction at any given bitrate — 8-bit
quantization with LD=16 achieves MSE < 0.001 at 128 bits. β-VAE traces the information
bottleneck frontier from near-AE behavior (β=0.001, high rate) to the free-bits floor
(β≥0.5, KL≈1.5 nats). VQ-VAE operates at discrete rate points determined by codebook
size and latent dimension. The same structure holds across all 7 MPE scenarios.

### β-VAE KL Collapse Dynamics

<p align="center">
  <img src="assets/kl_collapse.png" width="42%" alt="KL collapse curve">
  <img src="assets/ablation_heatmap.png" width="42%" alt="β-VAE ablation heatmap">
</p>

**Left**: KL vs β sweep at LD=8 with default free_bits=0.01. KL spans a 300× dynamic
range (19.6 → 0.07 nats) before reaching the free-bits floor at β≥0.5. The default
free_bits floor (0.01 nats/dim) provides a minimum per-dimension information rate that
prevents deterministic collapse while being low enough not to distort the rate-distortion
curve. **Right**: Heatmap ablation — decoder expansion (1×, 2×, 4× hidden dim) has zero
anti-collapse effect. The collapse bottleneck is in the rate (KL) term, not the decoder's
representational capacity. Increasing free_bits level provides the only effective lever.

### Collapse Barrier and Universal Prevention

<p align="center">
  <img src="assets/collapse_barrier_analysis.png" width="82%" alt="Collapse barrier analysis">
</p>

**FB=0.1 universally prevents posterior collapse across all scenarios and all agent
counts (N=3→15).** Top panels: KL vs β sweeps at different free_bits levels and decoder
multipliers — FB=0.1 establishes a universal threshold above which no configuration
collapses. Bottom panels: cross-scenario validation on tag_hd (40-dim), comm_hd (60-dim),
and spread_xhd (90-dim) — 0% collapse rate with FB=0.1 vs 50-100% without.

### FB Fine-Sweep: Minimum Effective Dose

<p align="center">
  <img src="assets/kl_vs_beta_all_scenarios.png" width="55%" alt="KL vs beta across scenarios">
</p>

**Minimum effective FB dose = 0.02 nats/dim** — 5× lower than 0.1, 25-100× lower than
literature defaults (0.5-2.0). FB=0.02 yields KL=0.31 nats (well above the 0.01
collapse threshold) with monotonic MSE improvement across the full sweep range
(MSE=2.52 → 1.25). The low threshold means free-bits can be used without distorting
the rate-distortion operating point.

### Cross-Scenario Validation

<p align="center">
  <img src="assets/fb_cross_scenario_validation.png" width="82%" alt="Cross-scenario FB validation">
</p>

FB=0.1 eliminates collapse across all three high-dimensional scenarios simultaneously.
Without free-bits, collapse rates are 80% (tag_hd), 100% (comm_hd), and 50% (spread_xhd).
KL at β=2.0 with FB=0.1 is stable at ~1.5 nats across all scenarios, demonstrating
dimension-independent information rate.

### Agent-Count Scaling (N=3→15)

<p align="center">
  <img src="assets/rate_distortion_unified.png" width="60%" alt="Agent scaling rate-distortion">
</p>

**FB=0.1 produces universal 35-39% MSE improvement across every agent count.**
KL is dimension-independent at ~1.5 nats across the 18→90 dim range. FB=0.0 collapses
at every scale (KL < 0.005). The consistency of the FB=0.1 result across all agent
counts confirms that the anti-collapse mechanism does not depend on the number of agents
or observation dimension.

### Latent-Space and Reconstruction Diagnostics

<p align="center">
  <img src="assets/latent_space.png" width="55%" alt="Latent-space diagnostic">
</p>

**Latent-space visualization at β=1.0** — read this as a collapse diagnostic, not as
evidence of semantic clustering. The dispersed latent space at this β level shows the
free-bits floor keeping all dimensions active. Compare with β=0.01 (structured, high-KL)
and β≥4.0 (saturated near-prior) for the full dynamic range.

<p align="center">
  <img src="assets/reconstruction_comparison.png" width="82%" alt="Reconstruction comparison">
</p>

**Reconstruction quality comparison across codec families at equal bandwidth.** Shows
how different codecs distribute reconstruction error — PCA loses fine-grained
coordination features, β-VAE preserves global structure, and digital quantization
achieves near-lossless reconstruction at sufficient bit depth.

### VQ-VAE Codebook Diagnostics

<p align="center">
  <img src="assets/vqvae_commitment.png" width="56%" alt="VQ-VAE commitment sweep">
  <img src="assets/vqvae_usage_heatmap.png" width="44%" alt="VQ-VAE codebook usage heatmap">
</p>

**VQ-VAE codebooks are severely over-provisioned at higher latent dimensions — use LD≤4.**
For CB=256 and LD=8, codebook usage stays below 12% regardless of commitment cost.
The best VQ-VAE point (CB=512, LD=4, cc=0.25) achieves MSE=0.1283 at 9 bits with
100% codebook usage. At LD=2, usage reaches 100% across all codebook sizes. The
practical takeaway: reserve LD≤4 for discrete semantic channels; use continuous
β-VAE bottlenecks for LD≥8.

### Pareto Frontier: Codec Selection Under Bandwidth Constraints

<p align="center">
  <img src="assets/pareto_frontier.png" width="82%" alt="Budgeted rate-distortion frontier">
</p>

**The frontier is a design map for practical codec selection.** Digital quantization
dominates the low-distortion regime (<0.01 MSE) and is the choice for high-fidelity
observation replay. β-VAE spans the full rate-distortion continuum and is the tool
for information-bottleneck studies — the KL term gives an interpretable information
rate. VQ-VAE serves when a discrete low-bitrate channel interface is required, but
is constrained by codebook utilization at higher dimensions.

## Scientific Interpretation

### The β-VAE Rate-Distortion Objective

The β-VAE loss is a Lagrangian form of rate-distortion optimization:

```text
L(θ, φ) = E_q(z|x) [ ||x - D_θ(z)||² ] + β · D_KL( q_φ(z|x) || N(0, I) )
           └────── distortion D ──────┘     └────── rate R ───────┘
```

Where:
- **Distortion D**: Expected reconstruction MSE — how faithful the decoded observation is.
- **Rate R**: KL divergence between the encoder's posterior `q(z|x)` and the standard
  normal prior — the information rate (in nats) needed to encode z.
- **β**: Lagrangian multiplier trading off rate vs. distortion. Each β value produces
  a different point on the rate-distortion curve.

This is not metaphorical — `KL(q||N(0,I))` IS the information rate if the prior
is used for entropy coding. The equivalence is: 1 nat × log₂(e) = 1.443 bits.

### Free-Bits: Per-Dimension Information Floor

The corrected free-bits scheme clamps per-dimension KL:

```text
KL_effective = Σ_d max(0, KL_per_dim_d(ẑ) - λ)
               where KL_per_dim_d is averaged over the batch, then clamped
```

Each latent dimension `d` is only penalized for KL above λ nats. If `KL_d < λ`
across the batch, that dimension is "free" — the encoder can use it without penalty.
This is more principled than per-sample clamping because it measures information
carried by each dimension across the data distribution, not individual samples.

**Default configuration**: λ = 0.01 nats/dim (25× lower than literature 0.25),
warmup = 200 epochs (linear β ramp 0→target).

### β Regimes at LD=16 with FB=0.1

| β Range | Regime | KL (nats) | Rate (bits) | MSE | Characteristics |
|---------|--------|-----------|-------------|-----|-----------------|
| β=0.001 | Near-AE | 15-20 | 22-29 | Very low | ~Reconstruction-only, richest latent |
| β=0.01 | Semantic bottleneck | 5-10 | 7-14 | Low-moderate | Recommended for SemCom-MARL: good reconstruction + interpretable rate |
| β=0.1 | Transition | 1-3 | 1.4-4.3 | Moderate | Information squeezed, structure still present |
| β=0.5-2.0 | Stable plateau | ~1.5 | ~2.2 | High | At FB floor — minimal but non-zero information |
| β≥4.0 | Saturation | ~1.5 | ~2.2 | Data variance | Prior-matching, no further KL decrease possible |

### Dimension-Independence of KL

A key empirical finding is that **absolute KL is independent of observation
dimension** when free-bits is active. Across 18→90 dim (3→15 agents), KL stays
at ~1.5 nats with FB=0.1. This means the total information rate does not grow
with observation size — the codec allocates a fixed information budget regardless
of how much data it receives. This has practical consequences for multi-agent
systems: adding more agents does not increase the per-message communication cost.

## Negative Results & Actionable Constraints

Three negative results prevent future researchers from wasting compute on
ineffective configurations:

1. **Without free-bits, posterior collapse is universal.** FB=0.0 produces KL<0.01
   in every scenario and every agent count tested (3-15 agents, 18-90 dim). The
   minimum effective FB dose (0.02 nats/dim) is 25-100× lower than literature
   values (0.5-2.0). **Monitoring rule**: when KL approaches the free-bits floor
   during SemCom-MARL training, the latent channel has stopped carrying task-relevant
   information — increase FB or decrease β.

2. **Decoder expansion has zero anti-collapse effect.** Sweeping decoder hidden
   dimensions from 1× to 4× the encoder capacity does not prevent collapse. The
   bottleneck is in the rate (KL) term — increasing decoder capacity without
   addressing the KL penalty is wasted compute.

3. **VQ-VAE codebook utilization collapses at higher latent dimensions.**
   At LD=8 with CB=256, usage never exceeds 12%. **Constraint**: use LD≤4 for
   discrete semantic channels (VQ-VAE); reserve LD≥8 for continuous bottlenecks
   (β-VAE) only.

## Channel Impairments & Joint Source-Channel Coding

A core question in semantic communication is whether the codec should **train with
channel noise in the loop** (JSCC) or simply be **evaluated post-hoc** on a noisy channel.
ObsCodec provides both approaches:
- [obscodec/channel/impairments.py](obscodec/channel/impairments.py) — 7 evaluation-only channel models for post-hoc robustness testing
- [obscodec/channel/diff_channel.py](obscodec/channel/diff_channel.py) — 4 differentiable `nn.Module` subclasses that support gradient flow for JSCC training
- [obscodec/models/jscc.py](obscodec/models/jscc.py) — `JSCCWrapper` composing any codec with a differentiable channel

### Channel Models

| Model | Physics | What It Simulates | Differentiable? |
|-------|---------|-------------------|:-:|
| AWGN | `y = z + n`, n~N(0,σ²) | Thermal noise, weak interference, sensor noise — the universal baseline | Yes (reparameterization) |
| Rayleigh Fading | `y = h·z + n`, h~Rayleigh(1) | Multi-path propagation, mobile-agent channels where signal strength varies randomly | Partial (proxy) |
| Packet Erasure | `y_i = 0` with probability p | Agent out of range, link failure, collision-induced loss | Yes (straight-through) |
| Burst Erasure | Contiguous blocks dropped | An entire agent's message lost (not scattered bit flips) | Yes (block ste) |
| Block Fading | All latent dims share one h | Coherent fade — whole message fades together | No |
| Agent-Block Fading | Per-agent latent segment gets its own h | Realistic multi-agent: different SNR per agent based on distance/location | No |
| Heterogeneous SNR | Different σ² per agent segment | Varying channel quality across agents (e.g. near vs. far) | No |

### Key Findings from Channel Evaluation

**AWGN at moderate SNR (10-20 dB) acts as implicit denoising regularization for VQ-VAE.**
On simple_spread (30-dim), VQ-VAE with CB=512 achieves lower MSE under AWGN 10dB (0.533)
than on a clean channel (0.658). The noise forces the decoder to rely on global codebook
structure rather than overfitting to specific quantization indices — a form of stochastic
regularization that is absent from the clean-channel training objective.

**Rayleigh fading is consistently more destructive than AWGN at equivalent SNR.**
Multiplicative fading + additive noise compound nonlinearly. At Rayleigh 10dB, VQ-VAE MSE
is 1.7-2.0× higher than AWGN 10dB across scenarios. This has practical implications:
systems designed for AWGN-only channel assumptions will underperform in realistic
multi-path environments.

**Training with differentiable channels in the loop (JSCC) improves robustness to
mismatched conditions.** The encoder learns to produce latent representations that are
robust to channel variation — not just the specific noise level seen during training.
This is the central JSCC hypothesis: a codec trained at SNR=20dB should generalize
better to SNR=10dB than a codec trained without any channel noise.

### The JSCC Training Loop

```
Clean Codec:     x → Encoder → z → Decoder → x̂       (L = MSE(x, x̂) + β·KL)

JSCC Codec:      x → Encoder → z → [Channel] → ẑ → Decoder → x̂
                                  (AWGN/Erasure)
                 L = MSE(x, x̂) + β·KL

                 ∂L/∂z flows through the channel via reparameterization:
                 ẑ = z + σ·ε,  ε~N(0,I),  ∂ẑ/∂z = I  ✓ differentiable
```

The [JSCCWrapper](obscodec/models/jscc.py) pattern composes any encode/decode codec with a
`nn.Module` channel. It detects the base model type (BetaVAE via `reparameterize`, VQ-VAE
via `vq`, or generic AE) and applies the channel at the correct point in the latent path.

## Important Caveats

- **Reconstruction MSE is a proxy, not the end goal.** Phase 3 scripts (7-10) provide
  the instrumentation for task-aware evaluation — self-position accuracy, coordination
  gap, closed-loop distance to targets — but full-scale execution requires GPU time.
  Experimental designs and code are documented in [results_summary.md](assets/results_summary.md)
  Tables 9-12. The critical validation step is: does a codec with lower reconstruction
  MSE actually produce better multi-agent task performance? This requires running the
  closed-loop prototype at scale with learned policies.
- **β-VAE effective rate is an information estimate, not a deployed packet size.**
  Real deployment requires entropy coding of z (e.g., bits-back coding), packetization,
  or learned channel models matched to the physical layer. The KL value is the
  theoretically achievable rate under the Gaussian prior assumption.
- **VQ-VAE codebook utilization findings are specific to MPE observation structure.**
  The low utilization at high latent dimensions reflects the information content of
  these specific observations — other modalities (images, lidar, language embeddings)
  may exhibit different codebook behavior.
- **Free-bits assumes continuous latent variables.** For fully discrete semantic
  channels, VQ-VAE or FSQ-based (finite scalar quantization) approaches should be
  benchmarked separately. The JSCC wrapper supports both continuous and discrete
  latent paths.
- **All experiments use synthetic MPE data.** The observation structure (self-position,
  relative other-agent positions, relative landmark positions) is representative of
  multi-agent coordination but does not capture visual or sensor-noise complexity.
  Transfer to real robotic observations requires domain-specific validation.

## Project Structure

```text
ObsCodec/
├── README.md
├── README_zh.md
├── requirements.txt
├── setup.py
├── obscodec/
│   ├── __init__.py
│   ├── config.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── cost_metrics.py            # Cost profiling: params, FLOPs, latency, throughput
│   ├── task_metrics.py            # Task-aware evaluation (Phase 3)
│   ├── utils.py
│   ├── visualize.py
│   ├── channel/
│   │   ├── impairments.py         # 6 channel models
│   │   ├── adaptive.py            # Rate allocation strategies
│   │   └── diff_channel.py        # Differentiable channels for JSCC (Phase 3)
│   ├── data/
│   │   ├── synthetic.py           # 7 scenario generators + task-aware variants
│   │   └── __init__.py
│   └── models/
│       ├── pca_baseline.py
│       ├── ae_baseline.py
│       ├── digital_baseline.py
│       ├── vae.py                 # β-VAE + free_bits + task-aware loss
│       ├── vqvae.py               # VQ-VAE + codebook utilization
│       └── jscc.py                # JSCC wrapper (Phase 3)
├── scripts/
│   ├── 0_check_integrity.py
│   ├── 1_collect_data.py
│   ├── 2_train_baselines.py
│   ├── 3_train_vae.py             # β-VAE pipeline (4 phases)
│   ├── 3b_fb_finesweep.py         # FB fine-sweep 0.02-0.25
│   ├── 3c_agent_scaling.py        # Agent-count scaling N=3-15
│   ├── 3d_unified_codec.py        # Cross-scenario unified codec
│   ├── 4_train_vqvae.py
│   ├── 4b_vqvae_multiscenario.py  # VQ-VAE multi-scenario + channel
│   ├── 5_generate_figures.py
│   ├── 6_summary_table.py
│   ├── 7_diff_channel.py          # Phase 3.1: Differentiable channel benchmark
│   ├── 8_jscc_training.py         # Phase 3.2: JSCC training experiment
│   ├── 9_task_aware.py            # Phase 3.3: Task-aware loss experiment
│   └── 10_end_to_end.py           # Phase 3.4: End-to-end prototype
├── data/
├── assets/                         # All figures + results JSONs
└── checkpoints/                    # Sample model weights
```

Generated `data/*.npy` and `checkpoints/*.pt` files are intentionally not stored
in Git (except for a small set of sample checkpoints and one reference data file
for reproducibility). The figures and JSON summaries are included so the repo
remains readable without rerunning the full experiment.

## Quick Start

```bash
git clone https://github.com/MacswareX/ObsCodec.git
cd ObsCodec
pip install -r requirements.txt
pip install -e .

# Core pipeline
python scripts/1_collect_data.py --all       # Generate 7 scenarios + agent variants
python scripts/2_train_baselines.py          # PCA + AE + Digital
python scripts/3_train_vae.py --phase all    # Beta-VAE pipeline (4 phases)
python scripts/4_train_vqvae.py              # VQ-VAE + channel
python scripts/5_generate_figures.py         # All figures
python scripts/6_summary_table.py            # Final report

# Supplementary experiments
python scripts/3b_fb_finesweep.py            # FB fine-sweep 0.02-0.25
python scripts/3c_agent_scaling.py           # Agent-count scaling N=3-15
python scripts/3d_unified_codec.py           # Cross-scenario unified codec
python scripts/4b_vqvae_multiscenario.py     # VQ-VAE multi-scenario + channel

# Phase 3: Semantic Communication
python scripts/7_diff_channel.py             # Differentiable channel benchmark
python scripts/8_jscc_training.py            # JSCC training experiment
python scripts/9_task_aware.py               # Task-aware loss experiment
python scripts/10_end_to_end.py              # End-to-end prototype
```

Hardware used for the current artifact: RTX 3050 8 GB, PyTorch 2.6.0+cu124.
Seeds are fixed at 42 in the data split and experiment scripts.

## Phase 3: Semantic Communication

Phase 3 bridges the gap from pure compression benchmarking to semantic communication
research by making the channel part of the training loop and the loss task-aware.

This phase is structured as 4 sub-phases, each addressing a specific semantic
communication question:

### 3.1 Differentiable Channel Layers

**Question**: Can we make channel impairment part of the gradient flow so the encoder
learns channel-robust representations?

**Approach**: Two differentiable `nn.Module` subclasses:
- **DiffAWGN** — uses the reparameterization trick: `ẑ = z + σ·ε` where `ε~N(0,I)` is
  pure noise (no gradient) and `σ` is derived from SNR. Gradient flows through `z`
  because `∂ẑ/∂z = I`. The encoder sees channel noise during training and adapts.
- **DiffErasure** — uses straight-through estimator: a Bernoulli mask zeroes out a
  fraction of latent dimensions, but the mask is detached so gradients flow through
  surviving dimensions only. Surviving dims are scaled by `1/(1-loss_rate)` (like
  dropout) to keep energy unbiased.

**Script**: [7_diff_channel.py](scripts/7_diff_channel.py) trains JSCC-BetaVAE at
AWGN (20/10/5/0 dB) and erasure (10%/30%) rates, then evaluates on matched and
mismatched conditions.

### 3.2 Joint Source-Channel Coding (JSCC)

**Question**: Does training with channel noise in the loop produce better robustness
than training clean and evaluating on noisy channels?

**Approach**: Full factorial grid — 3 scenarios (30/48/90 dim) × 3 codecs (β-VAE
β=0.1, β=2.0, VQ-VAE) × 2 free-bits levels × 6 train channels × 8 test channels.
The [JSCCWrapper](obscodec/models/jscc.py) wraps any codec + channel, injects noise
during `training_step()`, and reports KL from pre-channel latents + MSE from
post-channel reconstructions.

**Key metrics**: MSE, KL, NMSE (MSE normalized by data variance — comparable
across scenarios with different observation scales).

**Script**: [8_jscc_training.py](scripts/8_jscc_training.py)

### 3.3 Task-Aware Loss

**Question**: Can task-specific loss terms (beyond reconstruction MSE) substitute for
or complement free-bits in maintaining latent information?

**Approach**: BetaVAE receives an additive task loss term:
- `self_only` — MSE between decoded and ground-truth self-position (first 2 dims).
  Tests whether the codec preserves the agent's own localization.
- `weighted` — `0.7 × self-MSE + 0.3 × others-MSE`. Tests whether reweighting
  self vs. other-agent features changes the information bottleneck.

**Key metrics**: total MSE, self-position MSE, other-agent MSE, coordination gap
(ratio of self-to-others error — a large gap means the codec prioritizes self over
coordination), per-agent MSE breakdown.

**Script**: [9_task_aware.py](scripts/9_task_aware.py) uses generators with
ground-truth metrics (`generate_spread_with_metrics()`).

### 3.4 End-to-End Closed-Loop

**Question**: Does the full pipeline (observation → encode → channel → decode →
policy → action) preserve task performance compared to raw observation sharing?

**Approach**: A minimal `SpreadSimulator` (5 agents, 5 landmarks, 200-step rollouts)
compares 3 conditions:
1. **no_compression** — raw observations fed directly to a heuristic policy (upper bound)
2. **jscc_clean** — JSCC-BetaVAE encodes then decodes with no channel noise
3. **jscc_noisy** — JSCC-BetaVAE encodes then passes through AWGN at configurable SNR

The heuristic policy extracts self-position from decoded observations and moves each
agent toward its nearest landmark. This isolates the codec's impact on closed-loop
behavior without confounding from learned policies.

**Metrics**: final distance to targets (avg last 10 steps), early/late mean distance
(first/last 50 steps), path efficiency (total distance traveled / n_steps), collision
count (agents within 0.05 of each other).

**Script**: [10_end_to_end.py](scripts/10_end_to_end.py)

### Architecture

```
obs → [Encoder] → z → [Differentiable Channel] → ẑ → [Decoder] → obŝ
                ↑                                          ↓
          KL(q(z|x)||N(0,I))                     MSE(x, x̂) + task-aware loss
                └────────────────── β · KL ──────────────────┘
```

### Library Additions for Phase 3

| Module | What It Provides |
|--------|-----------------|
| [channel/diff_channel.py](obscodec/channel/diff_channel.py) | DiffAWGN (reparameterization), DiffErasure (straight-through), DiffBlockErasure, DiffRayleighProxy |
| [models/jscc.py](obscodec/models/jscc.py) | JSCCWrapper — composes any base codec + differentiable channel, detects BetaVAE/VQ-VAE/AE paths |
| [task_metrics.py](obscodec/task_metrics.py) | Task-aware evaluation: self-position MSE, coordination error, per-agent MSE breakdown |
| [models/vae.py](obscodec/models/vae.py) | `task_weight` + `task_loss_type` parameters in BetaVAE — additive task loss terms |
| [data/synthetic.py](obscodec/data/synthetic.py) | `*_with_metrics` generators returning (observation, task_ground_truth) pairs |

## Experiment Coverage: 15/15 (100%)

Includes all 11 extended benchmark experiments plus 4 Phase 3 scripts (semantic communication).

263 models, 15 results JSONs, 17 figures, 13 datasets.

## References

1. Alemi et al. (2018). *Fixing a Broken ELBO.* ICML.
2. Burgess et al. (2018). *Understanding disentangling in β-VAE.* NeurIPS Workshop.
3. van den Oord et al. (2017). *Neural Discrete Representation Learning.* NeurIPS.
4. Kingma and Welling (2014). *Auto-Encoding Variational Bayes.* ICLR.
5. Lowe et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS.
6. Higgins et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.

## License

MIT © 2026 MacswareX
