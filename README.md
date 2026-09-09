# Research OS

个人科研工作台：记录每周进度、沉淀感悟、管理研究任务，并让 Codex 帮你完成调研、实现、复盘和验证。

## 3 分钟开始

1. 复制 `weekly/_template.md`，改名为 `YYYY-Www.md`。
2. 每天只记三件事：做了什么、得到什么、下一步是什么。
3. 周末在同一文件完成复盘。
4. 需要 Codex 时，从仓库根目录启动，并直接描述目标。

示例：

```text
请使用 weekly-review skill，帮我复盘本周工作，提炼 3 个结论和下周最重要的 3 个动作。
请使用 research-workflow skill，阅读这个实验仓库，找出最值得验证的 3 个假设，并给出最小实验。
```

## 目录

```text
weekly/                 每周工作记录
reflections/            长期感悟与方法论
projects/               项目索引与阶段目标
decisions/              重要决策记录
.agents/skills/         Codex 可复用技能
scripts/new_week.sh     创建新周报
```

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

技能文件遵循常见的 `SKILL.md` 约定：短描述负责触发，正文负责执行。可参考 [skillmd-examples](https://github.com/skillmdcreator/skillmd-examples)。
