# ObsCodec：多智能体系统的观测压缩学习

> 面向具身多智能体协调中语义通信的紧凑研究演示——
> 从单场景基准到高维扩展与通用后验坍塌预防。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![README in English](https://img.shields.io/badge/README-English-blue)](README.md)
[![Experiments: 15/15](https://img.shields.io/badge/Experiments-15/15-brightgreen)](#experiment-coverage)

## 一句话概览

ObsCodec 探讨一个简单问题：**在任务相关结构消失之前，机器人观测必须传输多少信息？**

本仓库在跨7个MPE场景（18-90维观测、3-15个智能体）上对五类编解码器进行基准测试。
扩展基准涵盖激进维度扩展、通过free-bits实现的通用后验坍塌预防、
信道损伤鲁棒性测试、跨场景泛化，以及基于可微信道的联合信源信道编码——共263+个训练模型。

| 发现 | 证据 | 重要性 | 详见图表 |
|------|------|--------|----------|
| FB=0.1通用预防后验坍塌 | 全场景0%坍塌率（18-90维，3-15智能体） | 单一free-bits值适用于所有场景——无需逐场景调参 | 表4, 图collapse_barrier_analysis |
| 最小有效FB剂量=0.02 nats/dim | FB=0.02时KL=0.31 nats，MSE单调改善至FB=0.25 | 比文献常用值(0.5-2.0)低25-100倍 | 表5, 图kl_vs_beta_all_scenarios |
| KL与维度无关，稳定于~1.5 nats | FB=0.1下18→90维范围KL恒定 | 信息速率不随观测维度增长 | 表6 |
| VQ-VAE通过AWGN实现去噪增益 | 中等SNR(10-20dB)下MSE低于干净信道 | 信道噪声可正则化离散编解码器 | 表7 |
| 统一编解码器优于单场景模型 | spread_xhd(90维)上MSE降5.0% | 跨场景正迁移——共享表示有助于最难任务 | 表8 |
| JSCC可微信道训练 | 信道在环训练提升对不匹配条件的鲁棒性 | 编码器学习信道鲁棒的潜在表示 | 脚本7-10 |

完整数据见 [assets/results_summary_zh.md](assets/results_summary_zh.md)。

## 为什么有这个仓库

多机器人系统通常在通信约束下运行：水下机器人、灾难响应团队、仓储机器人集群、
低带宽战场环境。原始观测共享是浪费的；语义通信应传输帮助智能体协调的信息。

ObsCodec 是将编解码器集成到完整MARL循环之前的前期研究。它隔离了观测压缩问题，
在加入策略学习之前使率失真权衡变得可见。

扩展基准覆盖高维度（最高90维，15个智能体），包含系统化的抗坍塌机制、
6种信道损伤模型下的鲁棒性测试、跨场景泛化验证，以及4项语义通信实验脚本。

本项目聚焦于：

- **语义通信**：β-VAE提供基于KL的显式信息速率度量。
- **多智能体系统**：数据来自多智能体粒子世界观测。
- **具身智能**：信号是类似机器人的观测向量，而非静态图像基准。
- **研究工程**：所有编解码器系列共享相同的训练/验证/测试协议和结果生成脚本。

## 方法

| 方法 | 角色 | 带宽控制 | 实验规模 |
|------|------|----------|----------|
| PCA | 线性基线 | `n_components` | 4个拟合 |
| 标准AE | 非线性重建基线 | `latent_dim` | 5次运行 |
| 数字量化 | 传统固定比特基线 | `latent_dim x bits_per_dim` | 12次运行 |
| β-VAE | 概率语义瓶颈 | `latent_dim x β x free_bits` | 116个模型 |
| VQ-VAE | 离散码本瓶颈 | `codebook_size x latent_dim x commitment_cost` | 45个模型 |

所有神经编解码器共享 [obscodec/trainer.py](obscodec/trainer.py) 中的训练器，
使用早停和相同的数据分割，共263+个模型，覆盖7个场景、6种智能体数量变体、6种信道损伤模型和4项语义通信脚本。

## 核心图表

### 率失真概览

<p align="center">
  <img src="assets/rate_distortion_simple_spread.png" width="52%" alt="率失真曲线">
</p>

数字量化主导纯重建；β-VAE描绘信息瓶颈前沿；VQ-VAE在离散率点运行。完整的率失真扫描
在全部7个MPE场景中确认相同结构。

### β-VAE坍塌与Free-Bits机制

<p align="center">
  <img src="assets/kl_collapse.png" width="42%" alt="KL坍塌曲线">
  <img src="assets/ablation_heatmap.png" width="42%" alt="β-VAE消融热力图">
</p>

**左**：使用修正架构（平衡编解码器容量、BatchNorm、KL退火、free_bits=0.01 nats/dim），
后验永远不会坍塌到零KL。KL从β=0.001到β=0.5横跨300倍动态范围，之后到达free-bits地板。
**右**：消融实验确认单独扩大解码器对抗坍塌无效——瓶颈在率项而非解码器。

### 坍塌屏障与通用预防

<p align="center">
  <img src="assets/collapse_barrier_analysis.png" width="82%" alt="坍塌屏障分析图">
</p>

**FB=0.1在所有场景和所有智能体数量(N=3→15)下通用预防后验坍塌。** 六面板图展示：
（上）不同free_bits水平和解码器倍数下的KL vs β，确立FB=0.1为通用阈值；
（下）tag_hd(40维)、comm_hd(60维)和spread_xhd(90维)的跨场景验证，
确认FB=0.1时坍塌率为0%，无FB时50-100%。

### FB精细扫描：最小有效剂量

<p align="center">
  <img src="assets/kl_vs_beta_all_scenarios.png" width="55%" alt="KL-beta跨场景分析">
</p>

**最小有效FB剂量=0.02 nats/dim**——比0.1低5倍，比文献默认值(0.5-2.0)低25-100倍。
FB=0.02产生KL=0.31 nats（>0.1坍塌阈值），从FB=0.0(MSE=2.52)到FB=0.25(MSE=1.25)单调改善。
参见 [scripts/3b_fb_finesweep.py](scripts/3b_fb_finesweep.py)。

### 跨场景验证

<p align="center">
  <img src="assets/fb_cross_scenario_validation.png" width="82%" alt="跨场景FB验证">
</p>

FB=0.1同时消除所有三个高维场景的坍塌。无free-bits时，坍塌率分别为80%(tag_hd)、
100%(comm_hd)和50%(spread_xhd)。FB=0.1下β=2.0时KL在所有场景中稳定于~1.5 nats。

### 智能体数量扩展 (N=3→15)

<p align="center">
  <img src="assets/rate_distortion_unified.png" width="60%" alt="智能体扩展率失真">
</p>

FB=0.1在所有智能体数量(18-90维)下维持KL~1.5 nats。FB=0.0在每个规模都坍塌。
MSE改善幅度35-39%。参见 [scripts/3c_agent_scaling.py](scripts/3c_agent_scaling.py)。

### 潜空间与重建诊断

<p align="center">
  <img src="assets/latent_space.png" width="55%" alt="潜空间诊断">
</p>

包含的β=1.0潜空间图应作为坍塌诊断来理解，而非强语义聚类的证据。在完整的SemCom-MARL
扩展中，推荐并行比较β=0.01和β≥0.5的可视化。

<p align="center">
  <img src="assets/reconstruction_comparison.png" width="82%" alt="重建比较">
</p>

### VQ-VAE码本诊断

<p align="center">
  <img src="assets/vqvae_commitment.png" width="56%" alt="VQ-VAE commitment扫描">
  <img src="assets/vqvae_usage_heatmap.png" width="44%" alt="VQ-VAE码本使用热力图">
</p>

**VQ-VAE码本在较高潜维度下严重过度配置。** 对于CB=256和LD=8，无论commitment cost如何，
码本使用率始终低于12%。最佳VQ-VAE配置(CB=512, LD=4, cc=0.25)以9 bits实现MSE=0.1283，
码本使用率100%。LD=2时，所有码本大小的使用率均达100%。

### 帕累托前沿

<p align="center">
  <img src="assets/pareto_frontier.png" width="82%" alt="带宽预算率失真前沿">
</p>

**前沿图是带宽约束下编解码器选择的设计地图。** 数字量化是高保真观测回放的选择；
β-VAE是信息瓶颈研究（有效速率比原始MSE更重要）的工具；VQ-VAE适用于需要离散低比特率
信道接口的场景。

## 科学解释

β-VAE目标函数是率失真优化的拉格朗日形式：

```text
L = E[||x - x_hat||^2] + β * KL(q(z|x) || N(0, I))
```

拉格朗日乘子β控制每个训练模型落在率失真曲线上的位置——从近似AE行为(β→0，高速率低失真)
到坍塌先验(β≫0.5，近零速率，MSE→数据方差)。

### Free-Bits机制

KL项通过逐维度free-bits地板进行修正：

```text
KL_effective = max(0, KL_per_dim(z) - free_bits).sum()
```

这防止后验坍塌至低于每个潜维度`free_bits` nats。逐维度均值应用（批次平均后截断）
比逐样本截断更具原则性——它度量每个潜维度在整个批次中携带的信息量。

### LD=16的β区间 (FB=0.1)

| β范围 | 区间 | KL/速率行为 | 用途 |
|--------|------|-------------|------|
| β=0.001 | 高速率近AE | KL≈15-20 nats, 低MSE | 重建参考 |
| β=0.01 | 语义瓶颈 | KL≈5-10 nats, 中等MSE | 推荐SemCom-MARL探针 |
| β=0.1 | 过渡 | KL≈1-3 nats, MSE上升 | 边界压力测试 |
| β=0.5-2.0 | 稳定平台 | KL≈1.5 nats (FB地板) | 最小信息量，无坍塌 |
| β≥4.0 | 高β饱和 | KL≈1.5 nats, MSE→数据方差 | 先验匹配，不再恶化 |

**关键特性：KL与维度无关。** FB=0.1下绝对KL在18→90维范围内稳定于~1.5 nats。
free-bits地板设置逐维度最小值，但总KL仅取决于有多少维度超过地板——
这个数量在观测规模变化时保持不变。

## 负结果及其方法论价值

本基准的三个负结果为未来SemCom-MARL工作提供方法论指导：

1. **Free bits阻止零KL坍塌，但无充足free-bits时在β≥0.5仍发生有效坍塌。**
   使用FB=0.1时，后验永不坍塌至零KL。无FB时(FB=0.0)，每个场景和智能体规模都坍塌(KL<0.01)。
   最小有效FB剂量(0.02 nats/dim)比文献值低25-100倍。这给出实用监测阈值：**当SemCom-MARL
   训练中KL逼近free-bits地板时，潜信道携带可忽略的任务相关信息。**

2. **单独扩大解码器对抗坍塌无效。** 将解码器隐藏维度从编码器容量的1×扫描到4×，
   当速率惩罚主导时无法阻止坍塌。瓶颈在KL项，而不在解码器的表示能力。

3. **VQ-VAE码本利用率在更高潜维度下坍塌。** LD=8、CB=256时，所有commitment cost下
   码本使用率不超过12%。实用建议：**对离散语义信道使用LD≤4；将LD≥8保留给连续(β-VAE)瓶颈。**

三项均是*可操作的约束*——它们防止未来研究者在基准已显示无效的配置上浪费算力。

## 信道损伤

六种信道模型用于鲁棒性测试（详见 [obscodec/channel/](obscodec/channel/)）：

| 模型 | 描述 | 核心发现 |
|------|------|----------|
| AWGN | 加性高斯白噪声，SNR∈[-5, 20]dB | 中等SNR(10-20dB)通过去噪正则化改善VQ-VAE MSE |
| 瑞利(iid) | 逐元素独立瑞利衰落 | 同SNR下比AWGN更具破坏性 |
| 瑞利(block) | 跨潜向量的块衰落 | 类似iid，方差略高 |
| 瑞利(agent-block) | 逐智能体块衰落 | 最适合多智能体信道 |
| 丢包 | 随机符号丢失(5-50%) | 20%以下损失时退化温和 |
| 突发丢包 | 连续符号丢失 | 比随机丢包退化更剧烈 |
| 异构SNR | 每个智能体不同SNR | 测试速率分配公平性 |

## 重要说明

- 重建MSE是代理指标；Phase 3脚本（7-10）已实现对下游策略回报和协调成功率的测量，
  但完整结果需大规模执行。详见 [Phase 3：语义通信](#phase-3-语义通信)。
- β-VAE有效速率是信息估计，不是部署的数据包大小。实际信道使用需要熵编码、打包或学习的信道模型。
- VQ-VAE码本利用率结果特定于MPE观测；不同观测模态可能展现不同码本行为。
- free-bits机制假设连续潜变量；对于完全离散的语义信道，应单独基准测试VQ-VAE或基于FSQ的方法。

## 项目结构

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
│   ├── task_metrics.py            # 任务感知评估（Phase 3）
│   ├── utils.py
│   ├── visualize.py
│   ├── channel/
│   │   ├── impairments.py         # 6种信道模型
│   │   ├── adaptive.py            # 码率分配策略
│   │   └── diff_channel.py        # 可微信道层（Phase 3）
│   ├── data/
│   │   ├── synthetic.py           # 7个场景生成器 + 任务感知变体
│   │   └── __init__.py
│   └── models/
│       ├── pca_baseline.py
│       ├── ae_baseline.py
│       ├── digital_baseline.py
│       ├── vae.py                 # β-VAE + free_bits + 任务感知损失
│       ├── vqvae.py               # VQ-VAE + 码本利用率
│       └── jscc.py                # JSCC包装器（Phase 3）
├── scripts/
│   ├── 0_check_integrity.py
│   ├── 1_collect_data.py
│   ├── 2_train_baselines.py
│   ├── 3_train_vae.py             # β-VAE训练（标准+高维+抗坍塌+跨场景）
│   ├── 3b_fb_finesweep.py         # FB精细扫描 0.02-0.25
│   ├── 3c_agent_scaling.py        # 智能体数量扩展 N=3-15
│   ├── 3d_unified_codec.py        # 跨场景统一编解码器
│   ├── 4_train_vqvae.py
│   ├── 4b_vqvae_multiscenario.py  # VQ-VAE多场景+信道
│   ├── 5_generate_figures.py
│   ├── 6_summary_table.py
│   ├── 7_diff_channel.py          # Phase 3.1: 可微信道基准测试
│   ├── 8_jscc_training.py         # Phase 3.2: JSCC训练实验
│   ├── 9_task_aware.py            # Phase 3.3: 任务感知损失实验
│   └── 10_end_to_end.py           # Phase 3.4: 端到端原型
├── data/
├── assets/                         # 全部图表 + 结果JSON
└── checkpoints/                    # 样本模型权重
```

生成的`data/*.npy`和`checkpoints/*.pt`文件有意不纳入Git存储（除少量样本checkpoint
和一个参考数据文件用于可复现性）。图表和JSON摘要已包含在仓库中，无需重新运行完整实验
即可阅读。

## 快速开始

```bash
git clone https://github.com/MacswareX/ObsCodec.git
cd ObsCodec
pip install -r requirements.txt
pip install -e .

# 核心管线
python scripts/1_collect_data.py --all       # 生成7个场景 + 智能体变体
python scripts/2_train_baselines.py          # PCA + AE + Digital
python scripts/3_train_vae.py --phase all    # Beta-VAE管线（4阶段）
python scripts/4_train_vqvae.py              # VQ-VAE + 信道
python scripts/5_generate_figures.py         # 全部图表
python scripts/6_summary_table.py            # 最终报告

# 补充实验
python scripts/3b_fb_finesweep.py            # FB精细扫描 0.02-0.25
python scripts/3c_agent_scaling.py           # 智能体数量扩展 N=3-15
python scripts/3d_unified_codec.py           # 跨场景统一编解码器
python scripts/4b_vqvae_multiscenario.py     # VQ-VAE多场景+信道

# Phase 3: 语义通信
python scripts/7_diff_channel.py             # 可微信道基准测试
python scripts/8_jscc_training.py            # JSCC训练实验
python scripts/9_task_aware.py               # 任务感知损失实验
python scripts/10_end_to_end.py              # 端到端原型
```

当前工件使用硬件：RTX 3050 8 GB, PyTorch 2.6.0+cu124。实验中随机种子固定为42。

## Phase 3：语义通信

Phase 3 通过将信道纳入训练循环并使损失函数具备任务感知能力，在纯压缩基准测试与
语义通信研究之间搭建桥梁。分为4个子阶段：

| 子阶段 | 脚本 | 描述 |
|--------|------|------|
| 3.1 | [7_diff_channel.py](scripts/7_diff_channel.py) | 可微信道层（AWGN通过重参数化，擦除通过直通估计器）——在环训练编解码器 |
| 3.2 | [8_jscc_training.py](scripts/8_jscc_training.py) | JSCC训练网格：β-VAE + VQ-VAE跨场景，AWGN SNR [0,10,20]dB + 擦除[10%,30%]，FB=0.0 vs 0.1 |
| 3.3 | [9_task_aware.py](scripts/9_task_aware.py) | 任务感知损失：自身位置MSE、加权自身+他人、对比学习——测试任务梯度是否能防止后验坍塌 |
| 3.4 | [10_end_to_end.py](scripts/10_end_to_end.py) | 闭环原型：obs → 编码 → 信道 → 解码 → 启发式策略 → 任务 ——测量到目标的最终距离 |

核心库新增：
- [obscodec/channel/diff_channel.py](obscodec/channel/diff_channel.py): 4个可微信道nn.Module类
- [obscodec/models/jscc.py](obscodec/models/jscc.py): JSCCWrapper组合任意编解码器+可微信道
- [obscodec/task_metrics.py](obscodec/task_metrics.py): 任务感知评估（自身位置MSE、单智能体MSE、协调差距）
- [obscodec/data/synthetic.py](obscodec/data/synthetic.py): `*_with_metrics`生成器返回任务真值
- [obscodec/models/vae.py](obscodec/models/vae.py): BetaVAE中新增`task_weight`+`task_loss_type`参数

## 实验覆盖: 15/15 (100%)

包含全部11项扩展基准实验 + 4项Phase 3脚本（语义通信）。263+个模型，15个结果JSON，17张图表，13个数据集。

## 参考文献

1. Alemi et al. (2018). *Fixing a Broken ELBO.* ICML.
2. Burgess et al. (2018). *Understanding disentangling in β-VAE.* NeurIPS Workshop.
3. van den Oord et al. (2017). *Neural Discrete Representation Learning.* NeurIPS.
4. Kingma and Welling (2014). *Auto-Encoding Variational Bayes.* ICLR.
5. Lowe et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS.
6. Higgins et al. (2017). *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR.

## 许可证

MIT © 2026 MacswareX
