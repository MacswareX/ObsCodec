# ObsCodec: Learned Observation Compression for Multi-Agent Systems

> A compact research demo for semantic communication in embodied multi-agent
> coordination.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## TL;DR

ObsCodec asks a simple question: **how much of a robot observation must be
communicated before task-relevant structure disappears?**

The repository benchmarks five codec families on observations collected from
PettingZoo/MPE `simple_spread_v3`. The current artifact contains **93 benchmark
configurations**: 89 neural training runs plus 4 PCA fits.

| Result | Evidence | Why It Matters |
|--------|----------|----------------|
| Digital quantization is the strongest reconstruction baseline | MSE=0.0001 at 128 nominal bits | A useful upper reference for pure observation fidelity |
| β-VAE exposes a tunable semantic information rate | β=0.01, LD=8: MSE=0.0873, KL rate=6.4 bits | Useful for semantic communication because rate is measured by an information bottleneck |
| β≥0.5 causes posterior collapse | KL≈0 and MSE≈0.545 across latent dimensions | Gives a concrete failure boundary to monitor during SemCom-MARL training |
| VQ-VAE is compact but codebook-limited | Best: CB=256, LD=2, 8 bits, MSE=0.1756 | A discrete packet alternative when symbolic channel payloads are required |

Full numbers are in [assets/results_summary.md](assets/results_summary.md).

## Why This Repo Exists

Multi-robot systems often operate under communication constraints: underwater
robots, disaster-response teams, warehouse fleets, and contested or low-bandwidth
field environments. Raw observation sharing is wasteful; semantic communication
should transmit the information that helps agents coordinate.

ObsCodec is a pre-study before integrating codecs into a full MARL loop. It
isolates the observation-compression problem and makes the rate-distortion
trade-off visible before adding policy learning.

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
| β-VAE | Probabilistic semantic bottleneck | `latent_dim x β` | 40 runs |
| VQ-VAE | Discrete codebook bottleneck | `codebook_size x latent_dim x commitment_cost` | 32 unique runs |

All neural codecs use the shared trainer in
[obscodec/trainer.py](obscodec/trainer.py), with early stopping and identical
data splits.

## Key Figures

### Rate-Distortion Overview

<p align="center">
  <img src="assets/rate_distortion.png" width="82%" alt="Rate-distortion curve">
</p>

The digital baseline dominates when reconstruction MSE is the only objective and
128+ nominal bits are available. β-VAE is still central for semantic communication
because it measures an **effective information rate** through KL, letting us study
where the latent channel becomes semantically empty.

### Budgeted Frontier

<p align="center">
  <img src="assets/pareto_frontier.png" width="82%" alt="Budgeted rate-distortion frontier">
</p>

The frontier is best read as a design map: choose digital quantization for
high-fidelity observation replay, β-VAE for information-bottleneck studies, and
VQ-VAE when a discrete channel interface is more important than reconstruction
accuracy.

### β-VAE Collapse Boundary

<p align="center">
  <img src="assets/kl_collapse.png" width="60%" alt="KL collapse curve">
  <img src="assets/ablation_heatmap.png" width="70%" alt="β-VAE ablation heatmap">
</p>

At β≥0.5, KL falls below 0.05 nats across the grid and reconstruction MSE
saturates near 0.545. This is posterior collapse: the encoder stops carrying
input-dependent information and the latent channel approaches the prior.

### Latent and Reconstruction Diagnostics

<p align="center">
  <img src="assets/latent_space.png" width="55%" alt="Latent-space diagnostic">
</p>

The included β=1.0 latent-space plot should be read as a collapse diagnostic,
not as evidence of strong semantic clustering. In a full SemCom-MARL extension,
the recommended visualization is to compare β=0.01 and β≥0.5 side by side.

<p align="center">
  <img src="assets/reconstruction_comparison.png" width="82%" alt="Reconstruction comparison">
</p>

### VQ-VAE Codebook Diagnostics

<p align="center">
  <img src="assets/vqvae_commitment.png" width="56%" alt="VQ-VAE commitment sweep">
  <img src="assets/vqvae_usage_heatmap.png" width="44%" alt="VQ-VAE codebook usage heatmap">
</p>

For CB=256 and LD=8, codebook usage stays below 15%, suggesting the discrete
latent space is over-provisioned for this observation distribution. In tested
LD=2 settings, codebook usage reaches 100% and yields the best VQ-VAE point.

## Scientific Interpretation

The β-VAE objective is a Lagrangian form of rate-distortion optimization:

```text
L = E[||x - x_hat||^2] + β * KL(q(z|x) || N(0, I))
```

Observed regimes:

| β Range | Regime | KL / Rate Behavior | Use |
|---------|--------|--------------------|-----|
| β=0.001 | High-rate near-AE | high KL, low MSE | Reconstruction reference |
| β=0.01 | Semantic bottleneck | 6.4-bit effective rate, moderate MSE | Recommended probe for SemCom-MARL |
| β=0.1 | Transition | low rate, high distortion | Boundary stress test |
| β≥0.5 | Collapse | KL≈0, MSE≈0.545 | Failure mode to avoid or detect |

Important caveats:

- Reconstruction MSE is a proxy metric; downstream policy return and coordination
  success still need to be tested.
- β-VAE effective rate is an information estimate, not a deployed packet size.
  Real channel use requires entropy coding, packetization, or learned channel models.
- VQ-VAE results should be rerun before using the discrete codec as the primary
  claim, especially after changing the VQ loss or codebook schedule.

## Project Structure

```text
ObsCodec/
├── README.md
├── requirements.txt
├── setup.py
├── obscodec/
│   ├── config.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── visualize.py
│   └── models/
│       ├── ae_baseline.py
│       ├── digital_baseline.py
│       ├── pca_baseline.py
│       ├── vae.py
│       └── vqvae.py
├── scripts/
│   ├── 0_check_integrity.py
│   ├── 1_collect_data.py
│   ├── 2_train_baselines.py
│   ├── 3_train_vae.py
│   ├── 4_train_vqvae.py
│   ├── 5_generate_figures.py
│   └── 6_summary_table.py
└── assets/
    ├── *.png
    ├── *_results.json
    ├── project_blurb.md
    └── results_summary.md
```

Generated `data/*.npy` and `checkpoints/*.pt` files are intentionally not stored
in Git. The figures and JSON summaries are included so the repo remains readable
without rerunning the full experiment.

## Quick Start

```bash
git clone https://github.com/MacswareX/ObsCodec.git
cd ObsCodec
pip install -r requirements.txt
pip install -e .

python scripts/1_collect_data.py
python scripts/2_train_baselines.py
python scripts/3_train_vae.py
python scripts/4_train_vqvae.py
python scripts/5_generate_figures.py
python scripts/6_summary_table.py
```

Hardware used for the current artifact: RTX 3050 8 GB. Seeds are fixed at 42 in
the data split and experiment scripts.

## Next Research Steps

1. Insert the β-VAE codec into a MARL policy loop and evaluate return under
   bandwidth limits.
2. Replace reconstruction-only metrics with task metrics: coverage, collision
   avoidance, communication load, and robustness to channel noise.
3. Add entropy coding or learned packetization so KL effective rate becomes a
   deployable channel budget.
4. Compare continuous β-VAE latents with discrete VQ-VAE packets under the same
   downstream coordination objective.

## References

1. Alemi et al. (2018). *Fixing a Broken ELBO.* ICML.
2. Burgess et al. (2018). *Understanding disentangling in β-VAE.* NeurIPS Workshop.
3. van den Oord et al. (2017). *Neural Discrete Representation Learning.* NeurIPS.
4. Kingma and Welling (2014). *Auto-Encoding Variational Bayes.* ICLR.
5. Lowe et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS.

## License

MIT © 2026 MacswareX
