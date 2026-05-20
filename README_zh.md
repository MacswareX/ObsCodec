# ObsCodec — 多智能体语义通信编解码器基准

ObsCodec 评估多智能体轨迹观测的压缩编解码器，面向**语义通信**场景——失真度量面向任务，信道不可靠。

## 路线B：高维扩展 + 坍塌预防 — 已完成 (100%)

当前分支遵循**路线B**：激进维度扩展（最高90维/15智能体）、通过free-bits预防后验坍塌、信道损伤鲁棒性测试、跨场景泛化——为Phase 3（语义通信+MARL）做适应性前置。

## 快速开始

```bash
pip install -e .
python scripts/1_collect_data.py --all       # 生成7个场景
python scripts/2_train_baselines.py          # PCA + AE + Digital
python scripts/3_train_vae.py --phase all    # Beta-VAE完整管线
python scripts/4_train_vqvae.py              # VQ-VAE + 信道
python scripts/5_generate_figures.py         # 全部图表
python scripts/6_summary_table.py            # 最终报告
```

**补充实验**（已完成）：
```bash
python scripts/3b_fb_finesweep.py            # FB精细扫描 0.02-0.25
python scripts/3c_agent_scaling.py           # Agent数量扩展 N=3-15
python scripts/3d_unified_codec.py           # 跨场景统一编解码器
python scripts/4b_vqvae_multiscenario.py     # VQ-VAE多场景+信道
```

## 仓库结构

```
ObsCodec/
├── obscodec/           # 核心库
│   ├── models/         # PCA, AE, Digital (baseline) + beta-VAE + VQ-VAE
│   ├── channel/        # AWGN, 瑞利衰落, 丢包, 自适应码率分配
│   ├── data/           # 合成多智能体轨迹生成器 (7个场景)
│   ├── config.py       # 中央配置
│   ├── metrics.py      # 评估指标 (MSE, KL, 码率, 码本利用率)
│   ├── trainer.py      # 训练循环
│   └── visualize.py    # 图表生成工具
├── scripts/            # 实验管线 (11个脚本, 编号)
├── data/               # 生成的.npy观测文件
├── assets/             # 结果JSON + 图表 + 项目说明
└── checkpoints/        # 已训练模型权重 (gitignore)
```

## 编解码器对比

| 编解码器 | 潜在类型 | 码率度量 | 坍塌风险 | 状态 |
|---------|---------|---------|---------|------|
| PCA | 连续 | LD (维度数) | 无 | 完成 |
| AE | 连续 | LD | 无 | 完成 |
| Digital | 离散 | bits/dim x dim | 无 | 完成 |
| beta-VAE | 随机 | KL nats -> bits | **已解决 (FB=0.1)** | 完成 |
| VQ-VAE | 离散 | log2(CB) x LD | 无 | 完成 |

## 核心发现：FB=0.1 通用抗坍塌

Free-bits在lambda=0.1 nats/dim时**在所有场景和所有Agent数量下通用防止后验坍塌**：

| 场景 | FB=0.0 坍塌率 | FB=0.1 坍塌率 | KL@beta=2.0 |
|------|:-:|:-:|:-:|
| tag_hd (40-dim) | 80% | **0%** | 1.55 |
| comm_hd (60-dim) | 100% | **0%** | 1.47 |
| spread_xhd (90-dim) | 50% | **0%** | 1.56 |

**FB精细扫描**: 最小有效剂量 = **0.02 nats/dim** — 比0.1低5倍，比文献常用值(0.5-2.0)低25-100倍。

## Agent数量扩展 (N=3→15)

FB=0.1将所有N值的KL维持在~1.5 nats。FB=0.0在所有规模都坍塌。MSE改善35-39%。

| N | 维度 | FB=0.0 状态 | FB=0.1 状态 | MSE 改善 |
|---|------|------------|------------|----------|
| 3 | 18 | 坍塌 | 正常 | -39.3% |
| 5 | 30 | 坍塌 | 正常 | -35.3% |
| 7 | 42 | 坍塌 | 正常 | -37.1% |
| 10 | 60 | 坍塌 | 正常 | -37.4% |
| 12 | 72 | 坍塌 | 正常 | -38.0% |
| 15 | 90 | 坍塌 | 正常 | -37.3% |

## VQ-VAE 多场景 + 信道

| 场景 | 最佳CB | 无噪声MSE | AWGN 10dB | AWGN 0dB | 瑞利 10dB |
|------|--------|-----------|-----------|----------|-----------|
| simple_spread (30-dim) | 512 | 0.658 | 0.533 | 0.853 | 0.921 |
| spread_hd (48-dim) | 512 | 0.898 | 0.852 | 1.120 | 1.292 |
| spread_xhd (90-dim) | 512 | 1.093 | 1.087 | 1.320 | 1.612 |

## 统一编解码器

单一BetaVAE在3个场景联合数据上训练，在spread_xhd上优于单场景模型（MSE -5.0%）。跨场景正迁移。

## 信道损伤模型

六种信道模型用于鲁棒性测试：AWGN、瑞利多径衰落(iid/block/agent-block)、随机丢包、突发丢包、异构SNR、级联损伤。

## 主要研究发现

1. **FB=0.1在所有场景和Agent数量(3→15)通用防坍塌**
2. **最小有效FB剂量=0.02** — 比文献值(0.5-2.0)低25-100倍
3. **单独扩大解码器对抗坍塌无效** — 瓶颈在码率项而非解码器
4. **跨场景信息结构差异**比原始维度对坍塌行为影响更大
5. **KL与维度无关** — 在18-90维范围恒稳于~1.5 nats
6. **统一编解码器优于单场景模型**（在困难任务上MSE降低5%）
7. **中等SNR的AWGN改善VQ-VAE的MSE**（去噪正则化效应）

## Route B 完成度: 11/11 (100%) — 263模型, 15结果JSON, 9图表

Phase 3（语义通信：任务感知压缩 + 联合信源信道编码）为下一阶段——详见 `scripts/6_summary_table.py`。

## 许可证

MIT
