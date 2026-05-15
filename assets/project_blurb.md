## ObsCodec：面向多智能体系统的学习型观测压缩

**一句话**：在5种编码范式、93组基准配置（89个训练模型 + 4个PCA拟合）中，β-VAE以约6.4 bits的有效KL码率揭示了多智能体观测在极低带宽下的语义压缩边界，后验坍塌在β≥0.5被系统实证。

**为什么做**：多机器人协同面临通信带宽瓶颈。传统方法要么无压缩传输原始观测，要么用固定比特率量化——缺乏自适应语义压缩能力。核心问题：**观测信号能被压缩到什么程度才会丢失任务相关信息？** 本项工作是语义通信-多智能体强化学习（SemCom-MARL）的前置预研——先隔离编解码器（codec）选型问题，再集成到完整强化学习流程。

**怎么做**：在MPE simple_spread环境中收集5万条多智能体观测，系统对比5种编码范式（PCA / AE / 数字量化 / β-VAE / VQ-VAE），所有模型共享统一的训练循环、数据划分和早停准则。

**关键发现**（证据锚定见正文图表）：
- **数字量化（Digital Quantization）** 是纯重建精度的最强基线（128 bits名义带宽下MSE≈0.0001，PSNR≈40 dB）→ 表1，图 rate_distortion
- **β-VAE（β=0.01, LD=8）** 提供约6.4 bits的有效KL码率（名义带宽256 bits）——在信息瓶颈意义下实现最大语义压缩 → 表2，图 ablation_heatmap
- **β≥0.5触发后验坍塌**：KL散度降至10⁻⁴–10⁻⁷量级，MSE饱和在≈0.545（先验方差），跨越所有潜在维度（LD=2/4/8/16/32均坍塌）→ 表2，图 kl_collapse
- **VQ-VAE在LD=8下码本利用率≤14%，LD=2达100%** ——低维离散潜在空间对MPE观测分布更高效 → 表3，图 vqvae_usage_heatmap
- 坍塌边界与Alemi et al. (2018)的率失真理论预测一致：β-VAE的Lagrangian形式 L = MSE + β·KL 在β过大时，KL项主导优化，编码器退化为先验N(0,I)

**重要限定**：重建MSE是代理指标——下游任务性能（覆盖率、碰撞避免、策略回报）需由SemCom-MARL闭环验证；有效KL码率是信息论估计而非可部署包大小，实际传输需级联熵编码（bits-back coding）或信道适配。

**技术关联**：PyTorch, PettingZoo/MPE, scikit-learn · 89个训练模型 + 4个PCA拟合 · 8张诊断图表 · 4张交叉对比表

**衔接SemCom-MARL**：β-VAE的率失真Lagrangian形式（Alemi et al. 2018; Burgess et al. 2018）使其天然适配信息瓶颈驱动的通信调度——β参数控制KL→码率，LD控制带宽上限，KL散度监测为坍塌预警工具。本项目确立的推荐配置为 β-VAE (β≈0.01, LD=8, 有效码率≈6.4 bits)。

[GitHub](https://github.com/MacswareX/ObsCodec) · [完整数值结果](results_summary_zh.md)
