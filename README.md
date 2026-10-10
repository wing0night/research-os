# Research OS / Cyber Self

这是一个“赛博的自己”：用可检索、可复盘、可验证的文件，承载长期自我模型、当前注意力、项目推进、突然出现的想法，以及 Agent 的协作方式。

## 每次打开仓库先看什么

1. [`now/README.md`](now/README.md)：我目前处于什么阶段，正在把注意力放在哪里。
2. [`projects/README.md`](projects/README.md)：所有主线与项目的状态、下一步和证据入口。
3. [`inbox/README.md`](inbox/README.md)：还没有归档的想法、任务和记忆。
4. [`weekly/`](weekly/)：按时间回看发生了什么。

长期、不常变化的内容看 [`self/`](self/)，提示词看 [`prompts/`](prompts/)，实验与行动证据看 [`evidence/`](evidence/)。

## 信息架构

```text
self/       深层记忆：特性、价值、人生主线、长期偏好、边界
now/        当前阶段：注意力、节奏、重点、等待决定的事情
projects/   可推进对象：研究、创业、技能、生活事务及其下一步
inbox/      浅层记忆入口：想法、碎片、待办、待归档材料
diary/      原始日记与周期总结
prompts/    人与 Agent 的协作记忆：按领域归档的提示词、版本与效果
evidence/   事实锚点：实验、练习、学习和决策的可追踪证据
weekly/     时间线：每周发生了什么、学到了什么、下周做什么
literature/ 文献与研究方案
reports/    行业调研及来源
reflections/反思与方法论
decisions/  重要选择及复查
.agents/    本仓库可调用的技能；外部技能仍以其实际位置为准
```

这套结构借鉴长期记忆、工作记忆、情景记录和执行控制的分工，只是组织信息的类比，不是对脑科学的诊断或严格模拟。

## 最小工作循环

```text
捕获 -> 判断归属 -> 推进一个最小动作 -> 写下证据 -> 周期性复盘 -> 提炼为长期记忆
```

- 突然出现的东西先放 `inbox/`，不要要求自己当场分类完美。
- 日记先写入 `diary/entries/`；说“总结”时再由 `daily-journal` skill 归档。
- 只有稳定、反复出现、值得长期调用的内容才进入 `self/`。
- 每个项目都保留目标、输入、产出、验收标准、证据、未验证项和下一步。
- “已写下”“已开始”“已验证”“已完成”是不同状态，不混写。

## 给 Agent 的入口

```text
请先读 README.md、now/README.md 和 projects/README.md。
我的目标是：……
请把新事实、推测、未验证项和下一步分开；不要替我编造进度。
完成后按 Task / Result / Verification / Uncovered / Next step 汇报。
```

## 现有工作台

- 每周记录：复制 `weekly/_template.md`，运行 `./scripts/new_week.sh`。
- 日记记录与归档：使用 `daily-journal`，原文保留，周期性总结。
- 研究任务：使用 `research-workflow`，从最小可验证实验开始。
- 交付记录：使用 `research-delivery`，明确状态、验收和证据。
- 文献综述统一归档到 `literature/`；行业调研归档到 `reports/`。

## 常用命令

```bash
./scripts/new_week.sh
git add . && git commit -m "weekly: update"
```

## 记录原则

- 记录证据：实验编号、commit、日志、图表或链接。
- 结论和猜测分开写。
- 没验证的事情写“未验证”，不要写成完成。
- 不要把密钥、私有数据、checkpoint 和大文件放进仓库。

## 技能

- `research-workflow`：科研调研、实验设计、代码实现与验证闭环。
- `weekly-review`：每周总结、反思、优先级和下周计划。
- `research-delivery`：把工作变成可追踪、可交接的交付物。
- `literature-review-search`：检索、核验并撰写通俗易懂、带证据链的文献综述。

文献综述统一归档到 `literature/`，按 `YYYY-MM-DD-<topic-slug>.md` 命名，不覆盖历史版本。

- `industry-research`：从开放式问题主动检索研报、论文、官方资料、媒体与社媒，综合行业做法、争议和前景。报告归档到 `reports/YYYY-MM-DD-<topic-slug>/`，包含 `report.md`、`sources.md` 和 `search-log.md`。

- `daily-journal`：记录个人日记，按周期总结并把事实、项目行动和长期记忆候选分层归档。技能源文件位于 `skills/daily-journal/`，因为当前环境中的 `.agents/skills/` 不可写。

`literature-review-search` 侧重科研方法和实验，也可主动检索论文；`industry-research` 侧重多类资料支持的行业全貌，不要求先提供文献。

```text
请使用 $industry-research 调研 2023 年至今中国量化行业的做法、AI 应用与行业前景，并对照海外实践。综合研报、论文、官方资料及从业者社媒观点，解释基础术语，区分实盘证据与回测结果，保存报告。
```

技能文件遵循常见的 `SKILL.md` 约定：短描述负责触发，正文负责执行。可参考 [skillmd-examples](https://github.com/skillmdcreator/skillmd-examples)。
