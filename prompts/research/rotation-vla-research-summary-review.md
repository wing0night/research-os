# Prompt: 轮转期间 VLA 研究总结审阅

领域：research
状态：已验证
版本：v0.1
最后更新：2026-09-30

## 目标

审阅并改写轮转期间关于 tracker、flow、motion 特征、VLA 消融实验和 VQ-VAE 动作编码的阶段总结，检查概念边界、事实表述、证据充分性和后续研究计划，补充对基础理解与学习的重视。

## 适用场景与输入

- 用户提供的轮转研究总结草稿。
- 仓库已有 Track4Action/Track4World、tracker/fusion、VQ-VAE/semantic action flow 研究记录。
- 相关记录：`literature/2026-09-27-track4action-transfer-progress.md`、`literature/2026-09-22-track4action-semantic-action-flow.md`、`literature/2026-09-22-oat-vqvae-analysis.md`。

## 提示词正文

```text
请审阅以下 VLA 轮转期间研究总结：检查是否存在原理性错误、概念混淆、证据过度表述和遗漏内容。重点区分 tracker、optical flow、scene flow、point tracks 与一般 motion feature；区分已完成的实现/消融与已被真实实验验证的性能收益；解释 VQ-VAE 动作编码与空间先验融合时需要明确的目标、输入、训练/推理接口和潜在信息泄漏。给出可直接使用的改写版本，并补充后续对基础理论、模型结构、实验方法和研究能力的学习计划。不得编造成功率、论文结论或未提供的实验结果。
```

## 期望产出与验收标准

- 指出原文中需要澄清的概念和证据边界。
- 给出一版完整、稳妥、适合阶段总结/汇报的改写文本。
- 明确后续研究方向和基础学习内容。
- 结果包含目标、输入、产出、验收标准、证据、未验证项和下一步。
- 不把实验设计、代码测试或研究产出写成已验证的模型收益。

## 使用记录

| 日期 | 输入摘要 | 结果 | 证据 / 文件 | 是否达到目标 |
|---|---|---|---|---|
| 2026-09-30 | 用户轮转总结草稿与仓库已有 VLA/tracker/VQ-VAE 记录 | 完成概念边界审阅、证据风险提示、改写版本和基础学习补充 | [审阅结果](../../literature/2026-09-30-rotation-vla-research-summary-review.md) | 是 |

## 下一次修改

- 若补充了真实实验结果，加入任务、数据划分、baseline、指标、seed、统计方式和失败分析。
- 若继续 VQ-VAE/空间先验方向，补充空间先验的具体来源、接口位置、部署时是否可获得以及与动作 token 的对齐方式。
