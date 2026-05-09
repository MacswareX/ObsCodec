## ObsCodec: Learned Observation Compression for Multi-Agent Systems

**一句话**：在5种编码范式、93组基准配置中，β-VAE以约6.4 bits的有效KL码率揭示了多智能体观测在极低带宽下的语义压缩边界，后验坍塌在β≥0.5被系统实证。

**为什么做**：多机器人协同面临通信带宽瓶颈。传统方法要么无压缩传输原始观测，要么用固定比特率量化——缺乏自适应语义压缩。核心问题：**观测信号能被压缩到什么程度才会丢失任务相关信息？** 本项工作是SemCom-MARL的前置预研——先隔离codec选型问题，再集成到完整RL流程。

**怎么做**：在MPE simple_spread环境中收集5万条多智能体观测，系统对比5种编码范式，所有模型共享统一的训练循环、数据划分和早停准则。

**关键发现**：
- Digital Quantization是纯重建精度的最强baseline（128 bits名义带宽下MSE≈0.0001）
- β-VAE在β=0.01时提供约6.4 bits的有效KL码率（名义带宽256 bits）——在信息瓶颈意义下实现最大语义压缩
- β≥0.5触发**后验坍塌**：KL散度降至10⁻⁴量级，MSE饱和在≈0.545（先验方差），跨越所有潜在维度
- VQ-VAE在LD=8下码本利用率≤14%，但LD=2达到100%——低维空间离散化更有效
- 坍塌边界与Alemi et al. (2018)的率失真理论预测一致

**重要限定**：重建MSE是代理指标；有效KL码率是信息估计而非部署包大小（实际传输需熵编码）。下游任务性能需由SemCom-MARL闭环验证。

**技术关联**：PyTorch, PettingZoo/MPE, scikit-learn · 88个训练模型 · 8张图表 · 4张对比表

**衔接SemCom-MARL**：本项目确立了codec选型的率失真实证基础——β-VAE (β≈0.01, LD=8)为推荐配置，KL散度监测为坍塌预警工具。

[GitHub](https://github.com/MacswareX/ObsCodec) · [完整数值结果](assets/results_summary.md)
