# 语义动作流轻量 VLA 代码初版交付

日期：2026-09-22。状态：实现与合成验证完成；真实科研实验尚未开始。

## 目标

把 OAT 分析与 Track4Action 后续方案转化为一个可运行、可读、以 VQ-VAE 范式对照为中心的轻量 VLA codebase，支持充分分析，不依赖高成功率或重型 backbone 掩盖表示问题。

## 输入来源与检索范围

- [OAT 分析](2026-09-22-oat-vqvae-analysis.md)、[语义动作流方案](2026-09-22-track4action-semantic-action-flow.md)。
- [OAT](https://github.com/Chaoqi-LIU/oat)：commit `1da92695ef12c23b7000a0b1a76cab0aef4750e6`。
- [Track4Action new_0830](https://github.com/wing0night/Track4Action/tree/new_0830)：commit `897ac89298e3f4488f38321ac646819851438560`。
- LeRobot v0.4.4 官方 reader 源码，接口面向 LeRobot v3。

本轮检索用于核验 tokenizer、teacher、数据接口和许可边界，没有重新做全面论文综述，也没有用论文分数冒充本实现结果。

## 方法与产出

代码位于 [code/semantic-action-flow](../code/semantic-action-flow/README.md)，可独立安装。

方法：动作或未来世界变化 → register encoder → VQ/FSQ → state 条件动作 decoder；随后冻结 tokenizer，用当前图像、语言、state 训练小型 AR token policy。另提供共享观察编码器结构的连续动作 BC。

默认模型约 tokenizer 0.98M、policy 0.87M、direct BC 0.81M 参数；policy 推理还需冻结 decoder。CNN 与字节级语言 Transformer 从零训练，没有基础 VLM 的开放词汇能力；这是刻意控制复杂度的任务条件 VLA 基线，不宣称达到大型 VLA 分数。

产出包括：

- LeRobot 完整窗口、episode split、训练集归一化、teacher 缓存与 manifest 校验。
- Track4World dense world geometry + 绝对像素对应 → 轨迹桥接，DINO 沿轨迹差分；不把旧 pooled cache 或名为 flow_3d 的终点坐标直接当作世界位移。
- VQ-VAE / FSQ、四种 encoder 输入、独立 causal / nested 开关、辅助 flow/semantic 重构。
- tokenizer / policy / direct 三类训练命令、checkpoint、仅当前输入推理和重新评估。
- prefix、code entropy/usage、CE、free-running action MSE、干预诊断、参数量、延迟和训练代价。
- [结构与运行](../code/semantic-action-flow/README.md)、[数据协议](../code/semantic-action-flow/docs/data.md)、[实验设计](../code/semantic-action-flow/docs/experiments.md)、[上游核验](../code/semantic-action-flow/docs/upstream.md)。

## 验收标准

初版应具备可实际训练的 E1/E2 闭环，普通 VQ-VAE 与直接动作基线，E3 单变量开关，E0 可检验数据协议；未来信息与部署接口隔离；关键数据/量化/因果边界有测试；结果记录诚实可追溯。

通过合成验收不等于 E0 真实几何验证或 E4 控制成功。正式研究须遵循相同数据、token 预算、模型容量、训练预算与种子集合的对照协议。

## 证据

[验证记录](../code/semantic-action-flow/docs/verification.md) 保存完整命令和边界：

- 20 项测试通过，包含量化梯度、register/AR 因果隔离、prefix mask、冻结 teacher、episode 窗口、缓存转换、checkpoint 与微型 overfit。
- tokenizer、policy、direct 各 30 步合成训练成功；每组 16 个 held-out fixture 窗口；policy checkpoint 重新加载评估成功。
- 默认 FSQ ordered 配置运行成功；单 seed 因子矩阵生成 65 个顺序训练阶段，未运行大矩阵。
- 所有 fixture 报告标记 `fixture_only=true`、`success_rate=null`。没有提交权重、数据集、大日志或创建远端变更。

本轮使用 research-workflow 与 vibe-coding-workflow 将任务切成最小可验证实现，同时交付配置、必要测试、证据与未验证边界。仓库原有用户改动及原研究记录保留。

## 未验证项

真实 LeRobot 安装与视频 I/O、teacher 权重和真实对应、米制尺度与 confidence 校准、真实数据收敛、GPU 性能、rollout、跨任务/embodiment 泛化。FSQ 为独立实现，不与 OAT checkpoint 数值兼容；本代码不是上游结果复现。

初版不含双分支语义融合、自适应 token 停止、contact 因子化、cycle forward model、完整 resume CLI 或仿真 runner。这些应在最小对照形成证据后单独引入。

## 下一步

选择一个 LeRobot v3 子集及 action 定义，导出并人工核验一条 world-track+DINO 样本。以相同窗口先跑 action-VQ、joint-VQ 和 direct BC；若 oracle 改善却 policy 不改善，优先分析不可预测未来信息与 token entropy；若 oracle 也不改善，优先分析几何、时序或动作不唯一性。

可复用 prompt：[implement-lightweight-semantic-action-flow](../prompts/research/implement-lightweight-semantic-action-flow.md)。

## Task / Result / Verification / Uncovered / Next step

- Task：根据两篇分析及两套上游代码实现轻量研究初版。
- Result：独立 PyTorch/LeRobot 框架、VQ/FSQ 与连续 BC、两阶段 VLA、诊断和文档。
- Verification：20 tests、三条 30-step fixture 链路、默认 FSQ 和重新加载评估通过。
- Uncovered：真实 teacher/数据/GPU/rollout，尚无方法有效性的实验证据。
- Next step：真实 E0 后启动三组最小对照。
