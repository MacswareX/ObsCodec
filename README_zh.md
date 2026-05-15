# ObsCodec：面向多智能体系统的学习型观测压缩

> 面向具身多智能体协调中语义通信的紧凑型研究原型。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![README in English](https://img.shields.io/badge/README-English-blue)](README.md)

## TL;DR

ObsCodec 用以探究一个简洁的问题：**机器人的观测信号在丢失任务相关信息之前，可以被压缩到什么程度？**

本仓库在 PettingZoo/MPE `simple_spread_v3` 环境收集的观测数据上，对五种编解码器（codec）家族进行基准测试。当前实验产物包含 **93组基准配置**：89个神经网络训练运行 + 4个PCA拟合。

| 结果 | 数据证据 | 意义 | 详见 |
|--------|----------|----------------|-----|
| 数字量化是重建精度的最强基线 | 128 bits名义带宽下MSE=0.0001 | 为纯观测保真度提供了上界参考 | 表1，图 rate_distortion |
| β-VAE揭示了可调节的语义信息率 | β=0.01, LD=8: MSE=0.0873, KL码率=6.4 bits | 对语义通信至关重要的一点：码率由信息瓶颈度量 | 表2，图 ablation_heatmap |
| β≥0.5情形下出现后验坍塌 | KL≈0，MSE≈0.545，横跨所有潜在维度 | 为SemCom-MARL训练中的坍塌监测提供了具体阈值 | 表2，图 kl_collapse |
| VQ-VAE紧凑但码本受限 | 最佳: CB=256, LD=2, 8 bits, MSE=0.1756; LD=8下码本利用率≤14% | 需要离散符号信道载荷时的替代方案 | 表3，图 vqvae_usage_heatmap |

完整数值结果见 [assets/results_summary_zh.md](assets/results_summary_zh.md)。

## 为什么做这个项目

多机器人系统经常在通信受限条件下运行：水下机器人、灾害响应编队、仓储车队以及对抗性或低带宽的野外环境。原始观测的完整共享流程效率低下——语义通信应传输有助于智能体协调的信息。

ObsCodec是将编解码器集成到完整MARL（多智能体强化学习）回路之前的前置研究。它隔离了观测压缩问题，使率失真权衡在加入策略学习之前变得可见可测。

因此，本项目作为一项引入性研究，聚焦于：

- **语义通信**：β-VAE通过KL散度提供了显式的信息率度量
- **多智能体系统**：数据来源于多智能体粒子世界观测
- **具身智能**：信号表现为类机器人的观测向量，而非静态图像基准
- **研究工程**：所有codec家族共享相同的训练/验证/测试协议和结果生成脚本

## 方法

| 方法 | 特征 | 带宽控制 | 网格检索 |
|--------|------|-------------------|------|
| PCA | 线性基线 | `n_components` | 4个拟合 |
| 标准AE | 非线性重建基线 | `latent_dim` | 5个运行 |
| 数字量化 | 传统定比特基线 | `latent_dim × bits_per_dim` | 12个运行 |
| β-VAE | 概率语义瓶颈 | `latent_dim × β` | 40个运行 |
| VQ-VAE | 离散码本瓶颈 | `codebook_size × latent_dim × commitment_cost` | 32个独立运行 |

所有神经codec使用 [obscodec/trainer.py](obscodec/trainer.py) 中的共享训练器，配合早停和相同的数据划分。

## 关键图表

### 率失真概览

<p align="center">
  <img src="assets/rate_distortion.png" width="82%" alt="率失真曲线">
</p>

**数字量化在纯重建任务上占据主导地位；β-VAE则刻画了信息瓶颈前沿。** 当重建是唯一目标且可用128以上名义bits时，数字基线达到最佳MSE。β-VAE对语义通信至关重要，因为它通过KL散度度量了**有效信息率**，这使我们得以研究潜在信道中语义空洞发生的位置。（数据来源：表1、表2。）

### 带宽预算前沿

<p align="center">
  <img src="assets/pareto_frontier.png" width="82%" alt="带宽预算率失真前沿">
</p>

**该前沿是带宽约束下codec选型的设计指引。** 数字量化适用于高保真观测回放；β-VAE是信息瓶颈研究的工具，在有效码率比原始MSE更重要的场景下使用；VQ-VAE适用于离散、低比特率信道接口优先级高于重建精度的场景。（数据来源：表1、表4。）

### β-VAE 坍塌边界

<p align="center">
  <img src="assets/kl_collapse.png" width="60%" alt="KL坍塌曲线">
  <img src="assets/ablation_heatmap.png" width="70%" alt="β-VAE消融热力图">
</p>

**后验坍塌在β≥0.5时急剧发生——一个可复现的失效边界。** 在此阈值下，KL在整个(LD, β)网格中降至0.05 nats以下，MSE重建饱和值在约0.545——即先验N(0,I)的方差。编码器停止携带与输入相关的信息；潜在信道退化为先验分布。该边界在所有测试的潜在维度（LD=2至32）上一致成立。（数据来源：表2。）

### 潜在空间与重建诊断

<p align="center">
  <img src="assets/latent_space.png" width="55%" alt="潜在空间诊断图">
</p>

附带的β=1.0潜在空间图应被解读为坍塌诊断图，而非强语义聚类的证据。在完整的SemCom-MARL扩展中，推荐的对比可视化是将β=0.01与β≥0.5并排呈现。

<p align="center">
  <img src="assets/reconstruction_comparison.png" width="82%" alt="重建对比图">
</p>

### VQ-VAE 码本诊断

<p align="center">
  <img src="assets/vqvae_commitment.png" width="56%" alt="VQ-VAE commitment cost扫描">
  <img src="assets/vqvae_usage_heatmap.png" width="44%" alt="VQ-VAE码本利用率热力图">
</p>

**VQ-VAE在高潜在维度下码本严重过度配置。** 对于CB=256和LD=8，无论commitment cost如何变化，码本利用率始终低于15%——离散潜在空间对该18维MPE观测分布过度配置。在LD=2的设置下，码本利用率达到100%，并产生了VQ-VAE的最佳点（MSE=0.1756, 8 bits）。低维离散化在该数据上既更高效也更为稳定。（数据来源：表3。）

## 理论说明

β-VAE的目标函数是率失真优化的Lagrangian形式：

```text
L = E[||x - x_hat||^2] + β * KL(q(z|x) || N(0, I))
```

Lagrangian乘子β控制了每个训练模型在率失真曲线上的落点——从近似AE的行为（β→0，高码率，低失真）到完全坍塌到先验（β≫0.5，零码率，最大失真）。观察到的区间如下：

| β范围 | 区间 | KL / 码率行为 | 用途 |
|---------|--------|--------------------|-----|
| β=0.001 | 高码率近AE区 | 高KL，低MSE | 重建参考基线 |
| β=0.01 | 语义瓶颈区 | 6.4-bit有效码率，中等MSE | 推荐的SemCom-MARL探针 |
| β=0.1 | 过渡区 | 低码率，高失真 | 边界压力测试 |
| β≥0.5 | 坍塌区 | KL≈0，MSE≈0.545 | 需要规避或检测的失效模式 |

## 负面结果及其方法论价值

本基准测试中的两项负面结果为未来的SemCom-MARL工作承载了重要的方法论价值：

1. **β≥0.5的后验坍塌在所有潜在维度上普遍存在。** 全部20个β≥0.5的配置（LD=2至32）均坍塌至KL<10⁻⁴和MSE≈0.545。这提供了一个干净、可复现的阈值：**在SemCom-MARL训练期间监测KL，当KL降至0.1 nats以下时触发干预。** 这也证实了Alemi et al. (2018)对β-VAE的率失真框架在非图像（机器人观测）数据上仍然准确预测了坍塌行为。

2. **VQ-VAE在高潜在维度下码本利用率坍塌。** LD=8时，在所有测试的commitment cost和码本大小下，码本利用率从未超过14%。这并非训练失败——它表明离散潜在空间对仅有18维的MPE观测（模态多样性有限）存在结构性过度配置。实践启示：**在该数据分布上使用LD=2进行离散语义信道编码；LD≥8仅为连续（β-VAE）瓶颈保留。**

两项结果均为**可操作的约束条件**——它们使后续研究者免于在基准测试已明确显示无效的配置上浪费算力。

## 重要限定

- 重建MSE是代理指标；下游策略回报和协调成功率仍需测试
- β-VAE的有效码率是信息论估计，并非可部署数据包大小。实际信道使用需要熵编码、数据包化或学习的信道模型
- 在将离散codec作为主要声明之前，尤其是在更改VQ损失或码本调度策略后，应重新运行VQ-VAE实验

## 项目结构

```text
ObsCodec/
├── README.md
├── README_zh.md
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
    ├── project_blurb.md（中文）
    ├── project_blurb_en.md（英文）
    ├── results_summary.md（英文）
    └── results_summary_zh.md（中文）
```

生成的`data/*.npy`和`checkpoints/*.pt`文件有意不存入Git。图表和JSON摘要已包含在仓库中，无需重新运行完整实验即可阅读。

## 快速开始

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

当前实验产物的硬件环境：RTX 3050 8 GB。数据划分和实验脚本中的随机种子固定为42。

## 下一步研究

1. 将β-VAE codec嵌入MARL策略回路，在带宽限制下评估策略回报
2. 将仅重建的指标替换为`simple_spread_v3`的任务指标：智能体到地标的距离、碰撞次数、覆盖率以及信道噪声下的通信负载
3. 加入熵编码或学习型数据包化，使KL有效码率转化为可部署的信道预算
4. 在同一协调目标下对比连续β-VAE潜在表示与离散VQ-VAE数据包

## 参考文献

1. Alemi et al. (2018). *Fixing a Broken ELBO.* ICML.
2. Burgess et al. (2018). *Understanding disentangling in β-VAE.* NeurIPS Workshop.
3. van den Oord et al. (2017). *Neural Discrete Representation Learning.* NeurIPS.
4. Kingma and Welling (2014). *Auto-Encoding Variational Bayes.* ICLR.
5. Lowe et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS.

## 许可证

MIT © 2026 MacswareX
