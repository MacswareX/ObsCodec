# ObsCodec: Learned Observation Compression for Multi-Agent Systems

Author: MacswareX
Draft prepared: 2026-06-07

Repository: https://github.com/MacswareX/ObsCodec

## Abstract

ObsCodec is a compact benchmark for learned observation compression in multi-agent environments. The project studies how much information can be removed from low-dimensional embodied observations before reconstruction structure degrades. Using 50,000 observations from PettingZoo/MPE `simple_spread_v3`, it compares PCA, standard autoencoders, digital quantization, beta-VAE, and VQ-VAE under a shared train/validation/test protocol. The benchmark finds that digital quantization is the strongest pure reconstruction baseline, while beta-VAE provides the most interpretable information-bottleneck diagnostic through KL effective rate. VQ-VAE offers a low-bitrate discrete interface, but codebook utilization depends strongly on latent dimensionality. The result is best understood as a pre-study for semantic communication and embodied representation learning, not as a closed-loop multi-agent reinforcement learning result.

## 1. Motivation

Embodied multi-agent systems often face communication constraints. Sharing raw observations may be wasteful, especially when agents only need task-relevant state information to coordinate. Semantic communication asks whether an agent can transmit compressed representations that preserve useful structure while discarding irrelevant detail.

Before integrating a codec into a full multi-agent reinforcement learning loop, ObsCodec isolates a simpler question:

How do different codec families behave when compressing multi-agent observation vectors under controlled bandwidth constraints?

This isolation is useful because it separates representation bottleneck behavior from policy-learning instability.

## 2. Benchmark Setup

Environment:

- PettingZoo/MPE `simple_spread_v3`
- 3 agents
- continuous random actions during data collection
- 50,000 observation frames
- observation dimension: 18
- raw observation size: 18 dimensions x 32-bit float = 576 bits

Protocol:

- fixed train/validation/test split
- shared neural training loop
- early stopping
- result JSON files preserved
- figures generated from archived result files

## 3. Codec Families

### PCA

PCA provides a linear compression baseline. It is simple, deterministic, and useful as a sanity check for whether nonlinear models are needed.

### Standard Autoencoder

The standard autoencoder provides a nonlinear reconstruction baseline. Its latent dimensionality controls nominal bandwidth, but it does not provide an information-theoretic rate estimate.

### Digital Quantization

Digital quantization compresses observations through fixed-bit scalar quantization. It is a strong classical baseline for pure reconstruction and should not be omitted when evaluating learned codecs.

### beta-VAE

beta-VAE optimizes a rate-distortion-like objective:

```text
L = reconstruction_loss + beta * KL(q(z|x) || N(0, I))
```

Here, beta controls the pressure toward the prior. KL divergence can be interpreted as an effective information-rate estimate, although it is not a deployed packet size without entropy coding or channel modeling.

### VQ-VAE

VQ-VAE uses a discrete codebook, giving a natural interface for symbolic or packet-like latent communication. In this benchmark, it is most useful as a discrete-latent diagnostic rather than the strongest reconstruction method.

## 4. Metrics

Primary metrics:

- test MSE
- PSNR
- nominal bandwidth
- compression ratio
- beta-VAE KL effective rate
- VQ-VAE codebook usage
- rate-distortion efficiency

Important limitation:

Reconstruction MSE is a proxy. It does not prove that a compressed representation preserves downstream coordination ability. Full semantic communication evaluation requires policy return, coverage, collision count, communication cost, and robustness under channel noise.

## 5. Main Results

### 5.1 Digital Quantization Is The Strongest Reconstruction Baseline

Under high enough nominal bandwidth, digital quantization achieves the lowest reconstruction error. This is important because it prevents overclaiming: learned codecs are not automatically better for raw reconstruction.

Representative result:

- Digital quantization, LD=16, 8 bits per dimension
- nominal bandwidth: 128 bits
- MSE around 0.0001
- PSNR around 38.8 dB

Interpretation:

If the only objective is reconstructing low-dimensional observations, classical quantization is extremely competitive.

### 5.2 beta-VAE Gives The Clearest Semantic Bottleneck Diagnostic

The beta-VAE sweep shows how KL effective rate changes as beta increases. For LD=8:

| beta | MSE | KL nats | Effective rate bits | Regime |
|---:|---:|---:|---:|---|
| 0.001 | 0.0353 | 19.6247 | 28.3 | high-rate |
| 0.01 | 0.0474 | 9.1738 | 13.2 | semantic bottleneck |
| 0.1 | 0.2105 | 1.3347 | 1.9 | transition |
| 0.5 | 0.4996 | 0.0853 | 0.1 | at KL floor |
| 1.0 | 0.5136 | 0.0680 | 0.1 | at KL floor |

Interpretation:

The useful operating range is low beta, especially around beta=0.01 for a compact bottleneck that still preserves meaningful reconstruction structure. At beta >= 0.5, KL reaches the free-bits floor and MSE approaches the data variance, indicating effective collapse.

### 5.3 VQ-VAE Is Useful But Sensitive To Latent Dimensionality

VQ-VAE provides a discrete bottleneck, but the benchmark shows that high latent dimensionality can over-provision the codebook for this simple observation distribution.

Representative result:

- best VQ-VAE: CB=512, LD=4, commitment cost=0.25
- nominal bandwidth: 9 bits
- MSE: 0.1283
- codebook usage: 100%

Interpretation:

For this dataset, lower-dimensional discrete latents are more stable and efficient. VQ-VAE should be treated as a discrete-channel candidate, not as the overall reconstruction winner.

## 6. Negative Results

### 6.1 beta-VAE Collapse Is A Useful Diagnostic

The collapse behavior is not merely a failure. It identifies where the information bottleneck becomes too strong. In future SemCom-MARL experiments, monitoring KL can warn when the communication channel is no longer carrying useful information.

### 6.2 High-Dimensional VQ Codebooks Are Over-Provisioned

Poor codebook usage at higher latent dimensions suggests that this observation distribution does not require a large discrete latent space. This prevents wasted tuning on configurations that are structurally mismatched to the data.

## 7. Limitations

ObsCodec has four major limitations:

1. It uses reconstruction metrics rather than downstream task metrics.
2. It uses low-dimensional synthetic observations rather than real robot sensory streams.
3. It does not evaluate learned communication inside a closed-loop MARL policy.
4. KL effective rate is an information estimate, not an implemented channel rate.

These limitations are acceptable for a pre-study but must be stated clearly in any research conversation.

## 8. Research Interpretation

ObsCodec is most valuable as a representation-bottleneck diagnostic. It shows how to compare codec families before deciding what to integrate into a larger embodied system.

The main research lesson is:

Codec selection should depend on the downstream role of the latent channel.

- For high-fidelity replay, digital quantization is strong.
- For information-bottleneck analysis, beta-VAE is the clearest tool.
- For discrete communication interfaces, VQ-VAE is promising but needs careful codebook-size and latent-dimension control.

## 9. Recommended Next Step

The most valuable continuation is not a larger MPE benchmark. The higher-leverage next project is:

Audio Event Tokenizer for Embodied State Understanding

The bridge is direct:

- ObsCodec studies bottlenecks for vector observations.
- The next project studies bottlenecks for acoustic state evidence.
- Embodied audio should focus on "what happened" and "what state changed," not only sound classification or localization.

Minimum next-project evaluation:

- reconstruction quality
- event-class retention
- token usage
- robustness under noise
- short temporal prediction or state-transition inference

## 10. Conclusion

ObsCodec is a complete and useful proof-of-work artifact. Its value is not that it produces a final embodied-AI result, but that it demonstrates controlled experimentation, representation-learning implementation, rate-distortion reasoning, and honest interpretation of negative results.

The correct next action is to archive it cleanly, use it in RA outreach, and move toward embodied audio/tokenizer research.

