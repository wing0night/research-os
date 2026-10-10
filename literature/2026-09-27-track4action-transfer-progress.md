# Track4Action 转投实验：当前进度与下一步计划

日期：2026-09-27。状态：In Progress；本记录是基于现有材料的进度盘点，不代表实验已经完成。

## 目标

简洁说明 Track4Action 转投实验目前做到哪里、哪些只是工程/协议准备、哪些真实实验尚未完成，并给出下一步最小可验证执行顺序。

## 输入

- [转投实验执行方案 PDF](/Users/nightwing/Desktop/my_projects/track_action/new/0830/track4action-vla/docs/private/Track4Action%20%E8%BD%AC%E6%8A%95%E5%AE%9E%E9%AA%8C%E6%89%A7%E8%A1%8C%E6%96%B9%E6%A1%88.pdf)：已用 macOS PDFKit 读取 22 页。
- `track4action-vla/docs/research-plan.md`、`docs/timeline.md`、`docs/verification.md`、`docs/experiment/stage1_pretrained.md`。
- 本仓库已有 Track4Action 语义动作流和代码初版记录；用于区分另一条 VQ-VAE 研究线与本次转投主线。

## 当前进度

### 已完成

- 已确定转投的核心控制原则：Vanilla、Camera、Scene、Motion、Direct Fusion、Gated Fusion 等 branch 都必须从同一个原始 pretrained VLA checkpoint 独立开始，使用相同 downstream 数据、post-training recipe 和 evaluation protocol；不能拿某个 branch 的 post-trained checkpoint 继续初始化其他 branch。
- 已把路线拆成 Stage I–V，并补充真机阶段：先测 pretrained VLA baseline，再做 Track4Action token/fusion 消融，选最佳 architecture，之后才做 1000–2000 h robot-data mid-training，最后重跑 LIBERO/RoboTwin 和真机。
- 已选定首轮两个底座候选：π0.5 `pi05_libero` 与官方 StarVLA Qwen3-VL-PI-LIBERO-4in1 100k。当前记录明确保留各自原生 action head、processor、normalization 和动作接口。
- 已完成启动器、配置校验、实验矩阵生成、因果历史/tracker provenance 接口、结果 JSONL 汇总和相关标准库测试；这些是工具/协议验证，不是模型性能结果。
- 已启动两份发布 LIBERO policy 的 LIBERO-Plus zero-shot rollout，作为 B0 原生参考线；当前记录状态仍是 `running`，结果表中的任务数、成功数和 success rate 尚待回收。

### 进行中

- B0：π0.5 与 StarVLA 的 LIBERO-Plus zero-shot rollout 及日志/视频/失败原因回收。
- 实验协议收敛：checkpoint、upstream、benchmark、task manifest、evaluation protocol hash，以及动作协议、相机/state、normalization 和 split 的冻结准备。
- 代码仓库的后续 tracker/fusion 接口设计；当前尚未实现或接通实际 tracker feature extraction、fusion network 和真实 VLA adapter。

### 尚未完成

- 尚无可用于论文结论的真实 B0 成功率；也没有真实 Stage II Camera/Scene/Motion/Direct/Gated 消融结果。
- 尚未完成匹配预算的 B1 post-training、history-matched baseline、time-shuffle/cross-episode negative control 和 causal leakage audit。
- 尚未选择并验证最佳 architecture，因此不能启动或宣称完成 Stage IV 1000–2000 h mid-training。
- 尚未进行 mid-trained checkpoint 的 LIBERO/RoboTwin 重测、ALOHA 真机实验、跨 seed 统计或论文主表级 bootstrap 区间。

## 产出

- 本记录：当前状态、证据边界、未验证项和执行顺序。
- 可复用 prompt：[Track4Action 转投进度摘要 prompt](../prompts/research/track4action-transfer-progress-summary.md)。
- 上游执行材料仍保留在外部 `track4action-vla` 项目中，本仓库不复制 checkpoint、数据集、模型权重或大日志。

## 下一步计划

1. **收齐并复核 B0 rollout**：核对预定任务数、有效分母、成功数、四个 suite、七类扰动、失败分类、执行步数、视频和 server/client 日志；补齐 checkpoint/upstream/benchmark/task/protocol hash。未完成复核前只把结果视为运行记录，不解释为 Track4Action 收益。
2. **做原始 LIBERO sanity 并冻结协议**：确认 reset、图像输入、反归一化、action chunk、success 统计链路；随后冻结 downstream split、动作定义、相机/state、normalization 和版本指纹。
3. **先在一个底座、开发子集、单 seed 做 B1**：从原始 pretrained checkpoint 独立跑 Vanilla、Camera、Scene、Motion、Direct Fusion、Gated Fusion；严格加入 history-matched、time-shuffle 和 cross-episode negative controls，并完成未来信息泄漏检查。
4. **设置 go/no-go**：只有在匹配预算下看到稳定增益、且排除额外历史/参数、时间错配和协议错误后，才冻结最佳 token/fusion 结构并扩展到 3 个 seed 与第二底座；若无增益，先分析 tracker token、gate 使用率、数据对齐和 history baseline，不扩大训练矩阵。
5. **再进入大规模阶段**：只有 Stage II/III 的 architecture evidence 足够，才从原始 pretrained checkpoint 重新启动最佳结构的 1000–2000 h mid-training；完成后重新做 LIBERO/RoboTwin post-training 和真机验证。

## 验收标准

- B0 有可追溯 run receipt、固定版本/协议 hash、完整分母和失败分类。
- B1 的所有 branch 由同一原始 pretrained checkpoint 独立初始化，使用相同数据、预算和评测协议。
- Stage II 同时包含正向消融和 history/time/episode 负对照，并完成 causal leakage audit。
- 只有在 Stage III 确定最佳结构后，才把 mid-training 标记为可启动；所有成功率、统计区间和真机结果均以实际日志为证据。

## 证据

- PDF 第 1–4、6–13、20–22 页：同一 pretrained checkpoint 独立 branch、Stage I/II/III 和最终证据链。
- `docs/research-plan.md`：B1 消融、控制变量、2×2 mid-training 和 go/no-go 原则。
- `docs/timeline.md`：当前 B0 rollout、后续时间安排、锁定/冻结要求和资源估算边界。
- `docs/verification.md`：测试/启动器完成情况及“未加载真实权重、未执行 simulator、未产生成功率”的明确边界。
- `docs/experiment/stage1_pretrained.md`：两个 baseline 的配置、初始化来源、运行约束和待回收结果表。

## 未验证项

- B0 实际成功率、每任务结果、扰动结果、延迟和失败分布：未验证/待补。
- 真实 checkpoint、数据、上游环境和 benchmark commit 是否已锁定：未验证。
- tracker 输出质量、Camera/Scene/Motion token 的有效性、fusion 收益和未来泄漏风险：未验证。
- 1000–2000 h mid-training 是否值得、跨底座迁移是否成立、RoboTwin/ALOHA 真机收益：未验证。

## 下一步

优先完成 B0 收尾与协议冻结，再用一个底座的小规模 B1 做最小正对照和负对照；在没有稳定 architecture gain 之前，不扩大到完整 post-training 矩阵、mid-training 或真机结论。

## Task / Result / Verification / Uncovered / Next step

- **Task**：根据转投执行方案和已完成计划，整理当前进度与下一步。
- **Result**：已完成计划/工具/协议准备和 B0 rollout 启动；真实消融、mid-training、benchmark 重测和真机仍未完成。
- **Verification**：已读取 22 页 PDF 及项目中的 research-plan、timeline、verification、stage1 记录；交叉核对当前状态字段与结果表，未发现可支持真实成功率的记录。
- **Uncovered**：缺少 B0 run receipt/最终统计、真实模型评测、Stage II 训练和后续阶段证据。
- **Next step**：收齐 B0，完成 sanity 与协议冻结；随后执行单底座、单 seed、开发子集的 B1 matched ablation 和负对照。

## 状态与交接

- **Status**：In Progress，不是 Done。
- **Owner**：未在输入材料中提供；待项目负责人确认。
- **Blockers**：真实 GPU/环境、checkpoint 与数据版本锁定、B0 rollout 结果回收；具体资源可持续占用量未在材料中明确。
- **Recovery**：保留已有运行记录；发现协议/泄漏/版本错误时新建 protocol version，将旧结果标记 invalidated，不覆盖历史。
