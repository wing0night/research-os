# Prompt: 从 OAT 分析框架设计语义动作流研究

领域：research
状态：使用中
版本：v0.1
最后更新：2026-09-22

## 目标

围绕一篇具有分析型价值的离散表示/tokenizer 工作，提炼其核心分析框架、关键消融和可迁移方法，并据此设计 Track4Action 后续研究：使用 Track4World 的世界中心 3D 运动特征和 DINO 语义特征构造语义动作流，再通过 VQ-VAE/有序离散 tokenizer 编码并解码为机器人动作。

重点回答：

1. 该方向的科学问题和可行性是什么？
2. 它和 OAT、Track4Action 的关系与区别是什么？
3. 哪些 OAT 机制可以借鉴？
4. 可能形成哪些具有独立研究价值的创新？
5. 如何通过最小实验、单变量消融和证据记录验证，而不是只报告最终成功率？

## 适用场景与输入

- 输入论文：OAT、Track4Action、Track4World，以及相关 VQ-VAE、DINO 和 VLA 工作。
- 输入项目材料：`track4action.pdf`、已有 literature 文档、代码、数据说明、训练日志和失败记录。
- 适用场景：VLA、机器人动作表示、视频/世界模型、离散 latent、VQ-VAE tokenizer 研究。
- 时间对齐要求：Track4World/DINO 的未来片段只能作为训练期 privileged teacher；部署路径只能使用当前观测、语言和机器人状态。

## 提示词正文

```text
调研并分析 OAT: Ordered Action Tokenization，重点说明它为什么对 VQ-VAE 类离散表示工作有价值：

1. 解释 OAT 的核心思想、核心方法、FSQ、nested dropout、causal attention、prefix decoding；
2. 说明 reconstruction quality、decodability、modelability、token ordering 的区别；
3. 检查 OAT 的关键消融，区分论文明确报告、综合判断和未验证项；
4. 读取 track4action.pdf、Track4Action 和 Track4World 相关材料；
5. 分析如下后续方向：把 Track4World 的世界中心 3D 运动/轨迹特征和 DINO 的语义特征构造成动作对齐的 semantic action flow，通过 VQ-VAE 或有序离散 tokenizer 编码，再由 decoder 恢复机器人动作；
6. 明确训练时是否使用未来片段，如何避免部署时未来信息泄漏；
7. 比较该方向与 OAT、Track4Action 的输入、监督、latent、部署接口和科学问题；
8. 提出可借鉴机制，包括 nested dropout、causal latent attention、prefix decoding、codebook capacity 分析和 rate-distortion-modelability-decodability 四维评价；
9. 提出有独立研究价值的创新方向，例如语义/几何/接触分层 token、动作充分性约束、world-centric camera disentanglement、自适应 token 长度和跨 embodiment world-change interface；
10. 设计最小可验证实验：接口核验、四类 VQ-VAE 重构基线、当前观测到 token 的预测、ordering 消融和最终控制评测；
11. 为每个实验写清问题、假设、输入、操作、输出、验收标准、证据、风险和停止/否决条件；
12. 最后按 Task / Result / Verification / Uncovered / Next step 汇报，并把正式分析保存到 literature/YYYY-MM-DD-<topic>.md。
```

## 期望产出与验收标准

- 一份论文分析：核心思想、方法、公式、消融和证据边界。
- 一份研究方案：问题、假设、数据流、训练目标、部署接口和最小实验。
- 明确区分：未来 teacher target、当前观测 student、训练路径和部署路径。
- 至少有一个 alignment-free baseline 和单变量 ordering 消融。
- 同时评估 rate、distortion、modelability、decodability 与控制成功率。
- 不把未运行的实验写成已验证结果，不把简单特征拼接直接称为创新。
- 正式文档保存到 `literature/`；本 prompt 保存到 `prompts/research/`。

## 使用记录

| 日期 | 输入摘要 | 结果 | 证据 / 文件 | 是否达到目标 |
|---|---|---|---|---|
| 2026-09-22 | OAT 论文、`track4action.pdf`、Track4World 资料；提出语义动作流 VQ-VAE/VLA 后续方向 | 完成 OAT 分析、Track4Action/Track4World 对比、研究假设、风险和 E0–E4 实验方案 | [OAT 分析](../../literature/2026-09-22-oat-vqvae-analysis.md)；[研究方案](../../literature/2026-09-22-track4action-semantic-action-flow.md)；[Track4Action](https://arxiv.org/html/2608.03727v1)；[Track4World](https://arxiv.org/abs/2603.02573) | 部分达到；尚未运行 E0/E1 |

## 下一次修改

- 完成 E0 后，根据真实的 Track4World 坐标、DINO 时序对应和 confidence 字段修订输入定义。
- 若联合特征不改善 VQ-VAE 重构或 token 可建模性，删除无效特征分支，不继续堆叠模块。
- 若 ordering 有效，再加入自适应 token 长度和语义/几何/接触分层 token。
- 用真实训练日志、commit、配置和结果路径补充使用记录，保留失败实验。
