# Prompt: 实现轻量语义动作流 VLA 研究仓库

领域：research
状态：使用中
版本：v0.1
最后更新：2026-09-22

## 目标

基于已有 OAT 分析与 Track4Action 后续方案，交付可运行、可读、支持 VQ-VAE 范式对照的轻量研究代码，而非只搭配置或提出方案。

## 适用场景与输入

- `literature/2026-09-22-oat-vqvae-analysis.md`
- `literature/2026-09-22-track4action-semantic-action-flow.md`
- OAT 官方仓库与 Track4Action `new_0830` 分支。
- 数据优先 LeRobot；明确 action 维度、坐标、频率和控制时序。
- 前序研究 prompt：[OAT 与语义动作流方案](oat-track4action-semantic-action-flow.md)。

## 提示词正文

```text
先读取研究文档、代码和项目规则，核验 OAT 与 Track4Action 的具体实现与版本。
基于它们实现一个独立、轻量、可读的 PyTorch VLA 研究仓库：
1. 以 raw-action VQ-VAE 为主要基线，另有直接连续 BC 和 FSQ；
2. 输入可切换 action / motion / semantic / joint；未来 teacher 仅用于训练；
3. policy 只接收当前 RGB、语言和 state，避免采用重型 VLM 框架；
4. tokenizer 和 policy 分阶段训练，冻结 tokenizer，并保存归一化和版本证据；
5. 单独控制 causal attention、nested dropout、codebook 容量与辅助损失；
6. 优先官方 LeRobot reader，按 episode 划分，显式检查 action/teacher 时间对齐；
7. 实现带轨迹对应、单位、置信度与来源的 teacher 缓存协议，不把 pooled feature 冒充 dense flow；
8. 实现 prefix reconstruction、token entropy/usage、AR CE、free-running action error、
   随机/打乱 token、观察模态干预、参数量、延迟与成本记录；
9. 补齐架构、数据格式、运行命令、最小实验、单变量消融和否决条件文档；
10. 实际运行必要测试及最小训练，区分 fixture、真实数据验证和 rollout；
11. 保留已有记录，归档 prompt、结果、证据、未验证项与下一步。
```

## 期望产出与验收标准

产出：独立代码目录、测试、示例配置、结构说明、E0–E4 实验设计、明确证据边界。

验收标准：核心可安装与运行；两阶段接口连通；普通 VQ-VAE 对照清晰；未来 teacher 无法进入当前推理接口；episode/normalization 无泄漏；不虚构控制结果。

## 使用记录

| 日期 | 输入摘要 | 结果 | 证据 / 文件 | 是否达到目标 |
|---|---|---|---|---|
| 2026-09-22 | 两份研究文档、OAT、Track4Action new_0830；要求轻量 VLA 和 LeRobot | 完成独立初版、两阶段训练、分析和文档；20 tests 与合成 smoke 通过 | [结果归档](../../literature/2026-09-22-semantic-action-flow-codebase.md)、[代码入口](../../code/semantic-action-flow/README.md)、[验证记录](../../code/semantic-action-flow/docs/verification.md) | 初版实现达到；真实研究实验未完成 |

## 证据

上游 commit、测试环境、执行命令、输出路径与量化结果均保存在链接的验证记录和来源文档。

## 未验证项

真实 LeRobot 视频、Track4World/DINO 权重、坐标与尺度校准、GPU、rollout、多 seed 科学结论。

## 下一步与下一次修改

完成一条真实轨迹的 E0，再将 teacher 导出参数、数据版本和 benchmark 选择写入 prompt。只有真实实验显示瓶颈时才增加模型复杂度。

## Task / Result / Verification / Uncovered / Next step

- Task：实现轻量 VLA tokenizer 研究框架。
- Result：独立代码与可追溯文档。
- Verification：20 tests 和合成训练/重新加载成功。
- Uncovered：真实 teacher、数据和闭环结果。
- Next step：校准 E0、运行真实三组最小对照。
