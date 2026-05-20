# ObsCodec Project Overview

## Research Question

How to efficiently compress multi-agent observations for coordination under bandwidth constraints and unreliable channels? Traditional codecs (PCA, AE, scalar quantization) optimize for reconstruction fidelity, but semantic communication demands a balance between "usefulness for downstream tasks" and "transmission robustness."

## Route B Approach

1. **High-dimensional scaling**: From 18-dim/5 agents to 90-dim/15 agents, characterizing the beta-VAE posterior collapse boundary
2. **Anti-collapse mechanisms**: Free-bits (Kingma et al. 2016) + asymmetric encoder/decoder, pushing the collapse cliff from beta~0.7 past beta=10.0
3. **Channel robustness**: AWGN, Rayleigh fading, packet loss, heterogeneous channels, adaptive rate allocation
4. **VQ-VAE codebook optimization**: Discrete latent space with codebook utilization analysis
5. **Cross-scenario generalization**: 7 scenarios (spread/tag/comm x low-dim/high-dim/extreme-dim)

## Key Results

- **FB=0.1 nats/dim** eliminates posterior collapse across all scenarios (collapse rate 90% → 0%)
- KL maintained at ~1.5 nats at beta=2.0, MSE improved 13-38%
- 6 channel impairment models + 3 adaptive rate allocation strategies
- 222 trained models, 9 figures, 11 results JSONs

## Remaining Work

- Agent-count scaling study (N=3→15)
- Cross-scenario unified codec
- Phase 3: Task-aware compression + joint source-channel coding
