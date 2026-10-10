# 从训练循环到具身智能：一份看得懂代码的基础指南

日期：2026-09-17。面向会一点 Python、但还没有系统学习深度学习的读者。

目标：看到一个模型项目时，能够回答“喂进去什么、预测什么、答案从哪里来、哪些参数在变、最后怎么用”。输入是基础训练知识、原始论文和官方资料；产出是这份入门指南。验收标准是能解释文中的数据形状、训练循环和三种动作预测方式，并能按第 12 节开展最小练习。

证据边界：这是教学材料，不是最新模型排行榜或论文复现报告。类比和小代码用于解释机制，不代表论文的完整实现。全文代码中，只有第 12 节标为“完整练习”，其他都是局部示意。当前环境没有 PyTorch，代码已做语法检查，但没有运行训练；未验证项和下一步见文末。

## 阅读路线

第一次读：1—5 节理解训练与 Transformer，7—9 节理解多模态与机器人。第二次读：6、10、11 节理解微调、工程和排错。想动手时直接看第 12 节。

始终带着一个例子：**让机器人根据“拿起红杯子”的指令行动。** 它要读懂文字、看到杯子、预测动作，再根据动作后的新画面修正。这些环节会反复出现。

## 1. 先把不同层次的概念分开

把模型比作学徒只是一个类比：训练数据像练习题，模型像可调的计算机器，损失像评分规则，优化器负责根据错误调整机器内部的数值。

| 概念 | 回答的问题 | 到代码里找什么 |
| --- | --- | --- |
| Transformer | 用什么结构处理信息？ | attention、MLP、norm、残差连接 |
| 自回归（autoregression） | 按什么依赖顺序预测？ | 因果掩码、错开一位的标签、生成循环 |
| 自监督（self-supervised learning） | 训练答案从哪里来？ | 从原始数据构造 `input` 和 `target` |
| 预训练、微调 | 在哪个训练阶段？ | 初始权重、数据集、可训练参数 |
| 多模态（multimodal） | 信息有哪几种形式？ | 图像、文本、声音等编码器和融合模块 |
| 具身智能（embodied intelligence） | 模型是否通过身体与环境交互？ | 观测、动作、控制器、环境反馈 |
| 扩散（diffusion） | 怎样学习并采样复杂分布？ | 加噪、去噪网络、迭代采样 |

它们可以同时成立：一个机器人模型可以使用 Transformer，以示范动作作为监督，自回归生成动作 token；另一个也用 Transformer，却通过扩散生成连续动作。

“大模型”主要描述规模，并不是独立的训练算法。把参数数量增大，也不会自动补齐动作数据、控制接口或评测。

## 2. 一次训练，具体在做什么？

### 2.1 数据、张量、参数

张量（tensor）可以先理解为多维数组。形状就是这批数据的“尺寸说明书”。下面的数值都是教学例子，不是统一标准。

| 数据 | 示例形状 | 各维含义 |
| --- | --- | --- |
| 文本 token 编号 | `[8, 32]` | 8 条文本，每条 32 个 token |
| 文本特征 | `[8, 32, 128]` | 每个 token 对应 128 个特征数 |
| RGB 图片 | `[8, 3, 224, 224]` | 8 张图，3 个颜色通道，高宽各 224 |
| 一段动作 | `[8, 16, 7]` | 8 条样本，各预测 16 步，每步 7 个数 |

后文常用 `B` 表示批量大小，`T` 表示序列长度，`D` 表示特征维度，`V` 表示词表大小，`H` 表示动作预测长度，`A` 表示每步动作维度。符号只在对应上下文中使用。

**参数（parameter）**是模型通过训练修改的数值，例如线性层 `y = x @ W + b` 中的 `W` 和 `b`。**超参数（hyperparameter）**是人设置的训练选项，例如学习率和批量大小。

**激活（activation）**是这次输入经过模型产生的中间结果。换一张图片，激活会变；不做训练更新，参数不会因此改变。

### 2.2 五行训练循环

```python
# 示意：model、loader 和 optimizer 已事先创建。
model.train()
for inputs, targets in loader:
    optimizer.zero_grad()               # 清掉上一次留下的梯度
    predictions = model(inputs)         # 前向：先做题
    loss = loss_fn(predictions, targets) # 评分：错了多少
    loss.backward()                     # 反向：计算参数该朝哪个方向调整
    optimizer.step()                    # 更新：实际修改参数
```

反向传播（backpropagation）利用链式法则算梯度。梯度告诉我们：某个参数稍微增加，损失倾向于增加还是减小，以及变化有多敏感。优化器使用这些信息更新参数；`backward()` 本身并不完成参数更新。[PyTorch 官方训练教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

学习率（learning rate）控制更新幅度。太大可能反复越过合适的位置，太小可能进展缓慢。梯度下降不是模型“理解自己哪里错了”的语言思考过程，而是一套数值计算。

### 2.3 loss 到底是什么？

损失函数（loss function）把预测与训练目标的差异变成一个可优化的数。

- **分类**：例如下一个 token 是哪个，常用交叉熵（cross entropy）。它鼓励模型提高正确类别的概率。
- **连续数值预测**：例如移动多少厘米，常用均方误差（mean squared error，MSE）或绝对误差。
- **分布学习**：例如同一画面下存在多种合理动作，可以学习动作的概率分布，而不只预测一个平均值。

```python
import torch.nn.functional as F

# logits: [B, V]，未经 softmax 的类别分数；target_ids: [B]，整数答案。
loss_text = F.cross_entropy(logits, target_ids)

# predicted_actions 和 target_actions 都是 [B, H, A]。
loss_action = F.mse_loss(predicted_actions, target_actions)
```

`cross_entropy` 接收原始分数，内部处理所需的对数概率计算；不要先自行 softmax 再当作 logits 传入。

目标函数是你要优化的整体准则；损失是它的一种具体表达，整体目标还可能包含正则化等项。评价指标是你真正关心的表现，例如机器人成功率。**训练 loss 下降，不保证任务成功率上升。** 模型也可能记住训练数据，或者只把容易的动作预测得更准确。

### 2.4 训练、验证、测试和推理

训练集用来更新参数，验证集用来选择训练设置，测试集用来做最终评估。不能根据测试成绩反复调参数，再把它当作独立测试。

推理（inference）是用已有参数产生输出，通常不更新参数：

```python
model.eval()                    # 切换 dropout 等模块的行为
with torch.no_grad():           # 不记录用于求梯度的计算图
    predictions = model(inputs)
```

`eval()` 不等于关闭梯度，`no_grad()` 也不等于设置评估模式。二者解决不同问题。

## 3. 文本怎样变成模型能计算的东西？

### 3.1 token：处理单位，不一定是一个字或一个词

分词器（tokenizer）把文本切成单位，再映射成整数编号。一个汉字可能对应一个或多个 token；一个英文单词也可能被拆开。

```text
原文：拿起红杯子
教学切分：[拿起] [红] [杯子]
教学编号：[ 17,   82,  305 ]
```

上述切分和编号是虚构例子。编号只像字典索引，`305` 并不比 `17` “含义更大”。

### 3.2 embedding：给编号查一行可学习的向量

嵌入（embedding）把离散编号转换成连续向量，使它们能够参与矩阵运算。

```python
embedding = torch.nn.Embedding(num_embeddings=1000, embedding_dim=128)
features = embedding(token_ids)  # [B, T] -> [B, T, 128]
```

可以把 embedding 表类比为“每个 token 的初始档案”。它会在训练中更新。这里得到的还不是充分结合上下文的含义；同一个 token 在不同句子里通常先查到同一行，后续网络再根据上下文加工。

位置编码（positional encoding）提供顺序信息。教学实现可以把位置向量加到 token 向量上；其他实现也可能在 attention 内通过旋转等方式表达位置。读代码时不要认定所有模型都使用同一种位置编码。

## 4. Transformer：让每个位置按需读取其他位置

### 4.1 attention 的直觉

注意力（attention）可以类比为查资料。读到“它”时，需要结合前文判断它指杯子还是桌子。模型会计算各位置之间的相关分数，再按权重汇总信息。这是数值加权，不代表网络具有人的注意意识。

自注意力（self-attention）的查询与被查询信息来自同一组特征；交叉注意力（cross-attention）可以让文字特征去查询另一组图像特征。Transformer 用 attention 进行位置间的信息交换，并结合逐位置的前馈网络等模块。[Vaswani et al., 2017，§3](https://arxiv.org/html/1706.03762v7)

### 4.2 Q、K、V 到代码里是什么？

把它们类比为“我想找什么”“我有什么标签”“我能提供什么内容”：

```python
# 单头 attention 示意。x: [B, T, D]；三个线性层输出维度均为 D。
q = query_layer(x)
k = key_layer(x)
v = value_layer(x)
scores = q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5)  # [B, T, T]
weights = scores.softmax(dim=-1)                       # 每行权重之和为 1
out = weights @ v                                     # [B, T, D]
```

`scores[b, i, j]` 表示第 `b` 条样本中，位置 `i` 对位置 `j` 的匹配分数。除以维度平方根用于控制分数尺度。最后，每个位置拿到一份按权重汇总的新特征。[原论文 §3.2](https://arxiv.org/html/1706.03762v7)

Q、K、V 不是人工标出的实体属性，而是线性层算出的向量。多头注意力就是让多组这样的计算并行进行，再合并结果；不能预先保证某一头专门负责“颜色”、另一头专门负责“空间”。

### 4.3 一个 block 不只有 attention

下面是教学用的 pre-norm 写法，归一化放在子模块之前；不是原论文 post-norm 结构的逐行复刻。

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

- 多层感知机（multilayer perceptron，MLP）：对每个位置的特征做非线性加工，通常包含线性层和激活函数。
- 残差连接（residual connection）：`x + ...` 保留原信息，让模块学习增量，也有助于梯度传播。
- 层归一化（layer normalization）：调整每个位置内部的特征尺度，使训练更稳定。

多个 block 堆叠后，同一位置的表示逐步结合更丰富的信息。attention 权重可以帮助观察信息流，但不能单凭一张热力图就证明模型为何作出某个决定。

### 4.4 encoder 和 decoder 不要只按字面记

编码器（encoder）通常把输入变成有用表示；解码器（decoder）通常把表示变成目标输出。但具体的可见范围、输入输出，必须看模型。

| 架构 | 常见信息可见范围 | 常见用途 |
| --- | --- | --- |
| Encoder-only | 输入内部双向可见 | 分类、检索、特征提取 |
| Decoder-only | 生成位置看已知前缀 | 自回归文本生成 |
| Encoder-decoder | 编码器读取输入，解码器读取前缀并查询编码结果 | 翻译等条件生成 |

这些是常见配置，不是名称本身强制的规则。BERT 是双向编码器的代表；自回归语言模型常使用带因果掩码的 decoder-only 结构。[BERT 原论文](https://arxiv.org/abs/1810.04805)

## 5. 自回归与自监督：一个说顺序，一个说答案来源

### 5.1 自回归：把上一次的输出接回去，继续预测

给定“机器人拿起”，先预测“红”，再根据“机器人拿起红”预测“杯子”。每一步都以已有前缀为条件。

训练时，可以把一段真实文本同时变成题目和答案：

```text
完整序列：[开始] 机器人 拿起 红 杯子 [结束]
输入 x ：[开始] 机器人 拿起 红 杯子
答案 y ：机器人 拿起   红   杯子 [结束]
```

```python
x = tokens[:, :-1]
y = tokens[:, 1:]
logits = model(x)  # [B, T, V]；model 内部必须正确限制未来信息。
loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
```

这叫 next-token prediction：每个位置预测下一个 token。这里手动错位一次；某些模型库接收 `labels` 后会内部错位，使用前要检查接口，避免错两次。

### 5.2 为什么必须遮住未来？

如果预测“红”的位置已经看见后面的“红”，它可以抄答案。因果掩码（causal mask）让位置只能看到自己和之前的输入：

```text
               输入位置
                0 1 2 3
查询位置 0      ✓ × × ×
查询位置 1      ✓ ✓ × ×
查询位置 2      ✓ ✓ ✓ ×
查询位置 3      ✓ ✓ ✓ ✓
```

```python
T = x.shape[1]
blocked = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
scores = scores.masked_fill(blocked, float('-inf'))
weights = scores.softmax(dim=-1)
```

这里 `True` 表示屏蔽，是因为我们亲自用 `masked_fill` 定义了含义。不同 attention 接口对布尔 mask 的约定可能不同，不能直接照搬。

训练时真实前缀都已经给出，因此可以在一次前向里并行计算多个位置的损失。这种提供真实历史的训练方式叫 teacher forcing。普通逐 token 推理时，下一个 token 尚未产生，通常要循环生成：

```python
model.eval()
with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_id], dim=1)
```

此片段省略结束标记处理和上下文长度限制。`argmax` 是贪心选择，也可以从概率分布采样。temperature 调节采样分布的集中程度，不等于直接调节事实准确率。

### 5.3 自监督：答案从数据自身构造

“自监督”并不是没有答案，而是不必为每道题另外人工标注答案。

| 方法 | 输入怎样构造 | 答案从哪里来 |
| --- | --- | --- |
| 自回归语言建模 | 文本前缀 | 原文的下一个 token |
| 遮盖预测 | 遮住一部分词或图像内容 | 被遮住的原始内容 |
| 去噪学习 | 给数据加入已知噪声 | 原始数据、加入的噪声等，取决于目标定义 |

例如“机器人拿起 `[MASK]` 杯子”，用原句里的“红”监督模型。BERT 的遮盖语言建模允许结合左右文；它不等同于从左往右续写。原始 BERT 还使用了句间预测目标，不能把整套方法概括为只有填空。[Devlin et al., 2018，BERT](https://arxiv.org/abs/1810.04805)

自监督并不意味着没有数据清洗或人工贡献；也不意味着学到的表示一定适合机器人控制。

### 5.4 监督学习、模仿学习和强化学习

监督学习（supervised learning）使用成对的输入与目标，例如图片和类别、问题和回答。模仿学习（imitation learning）向示范行为学习；其中行为克隆（behavior cloning，BC）直接用观测预测专家动作，是一种监督学习做法。

强化学习（reinforcement learning，RL）通过奖励来学习怎样行动，以提高长期累计回报。奖励不是每一步的“正确动作标签”：拿到杯子得到奖励，并不直接告诉模型上一秒各关节应该转多少。

```text
行为克隆：读取 (观测, 专家动作) → 预测动作 → 与专家动作比较 → 更新
强化学习：收集或读取交互轨迹 → 利用奖励估计回报 → 按具体算法更新策略
```

强化学习也可以利用已有离线轨迹，不一定每次都在线试错。策略（policy）就是根据观测决定动作的规则或模型，和“对话系统的提示词策略”不是同一个概念。

## 6. 预训练、微调、冻结、LoRA：究竟改了哪些参数？

预训练（pretraining）先用较广泛的数据学习可迁移的能力。微调（fine-tuning）从已有权重继续训练，使其适应具体数据或任务。它们描述训练阶段，不限定必须使用哪一种损失。

监督微调（supervised fine-tuning，SFT）常用指令与回答样本。训练格式可以是“用户提问 + 助手回答”，但只对回答区域计算损失：

```python
# 示意：input_ids 包含完整对话；assistant_mask 标记助手回答位置。
labels = input_ids.clone()
labels[~assistant_mask] = -100
# 若库负责 next-token 错位，直接交给库；若自己实现，需要同步错位。
```

`-100` 是常见交叉熵忽略标记。被忽略的提示词仍参与前向计算、为回答提供信息，并不是被从上下文删掉。

| 操作 | 代码上做什么 | 意义与代价 |
| --- | --- | --- |
| 从头训练 | 随机初始化参数 | 需要自己提供学习基础能力的数据与计算 |
| 全量微调 | 大部分或全部参数参与优化 | 自由度大，优化器状态与梯度占用较多 |
| 冻结骨干、训练预测头 | 骨干 `requires_grad=False` | 更新参数少，但适应能力受接口和原特征限制 |
| LoRA | 冻结原权重，添加可训练低秩分支 | 减少可训练参数及相应状态，仍需运行骨干网络 |

```python
for p in backbone.parameters():
    p.requires_grad = False
optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
```

冻结不等于 `eval()`。冻结控制参数是否求梯度，`eval()` 控制某些模块的运行行为。另外，如果需要让梯度穿过冻结模块传到更早的可训练模块，就不能随意用 `no_grad()` 包住整个冻结模块。

低秩适配（low-rank adaptation，LoRA）可以类比为给大机器加一个小调节器。设原线性层权重为 `W`，用两个较小矩阵 `A`、`B` 表示增量 `B @ A`；训练时主要更新小矩阵。这里的 `A`、`B` 是矩阵名称，与前文动作维度、批量大小符号无关。[Hu et al., 2021，§4](https://arxiv.org/html/2106.09685v2)

```python
# 示意：x [B, D_in]，A [r, D_in]，B_lora [D_out, r]。
y = F.linear(x, frozen_W) + scale * F.linear(F.linear(x, A), B_lora)
```

`r` 是低秩分支的宽度，`scale` 是缩放系数。LoRA 并不保证总显存按可训练参数比例下降；中间激活仍可能占很多显存，也不保证对任何新任务都优于全量微调。

检索增强生成（retrieval-augmented generation，RAG）则主要是在推理前查找资料并放入上下文。常见 RAG 流程本身不更新模型参数，与微调解决的问题不同。

## 7. 多模态：把不同来源的信息接到同一个任务里

### 7.1 图片如何变成一串特征？

视觉 Transformer（Vision Transformer，ViT）的一种基础做法是把图片切成小块，把每块映射为一个特征向量，再交给 Transformer。`224×224` 的图片按 `16×16` 切块，可得到 `14×14=196` 个块；实际序列可能还包含额外 token。[Dosovitskiy et al., 2020，ViT](https://arxiv.org/abs/2010.11929)

```python
# 用卷积完成互不重叠的切块与线性投影。
patch_embed = torch.nn.Conv2d(3, 128, kernel_size=16, stride=16)
features = patch_embed(images)               # [B, 128, 14, 14]
image_tokens = features.flatten(2).transpose(1, 2)  # [B, 196, 128]
```

图像 token 往往是连续特征，不一定具有文本 token 那样的词表编号。“token”在这里表示处理单位，不能据此认定所有模态都先转成文字。

声音可以先变成波形片段或频谱特征，再编码成序列；视频还需要考虑帧与时间顺序。共同问题是：怎样保留任务需要的信息，怎样让不同来源的特征相互对应。

### 7.2 CLIP：学会把图片与描述配对

对比学习（contrastive learning）让匹配样本的表示更接近，同时区分不匹配样本。CLIP 使用图像和文本两个编码器，在一个 batch 内构造图文相似度矩阵，训练正确配对得分更高。[Radford et al., 2021，§2.3 与 Figure 3](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)

```python
# 假设 batch 内第 i 张图与第 i 条文字是一对，且无重复正例。
image_z = F.normalize(image_encoder(images), dim=-1)  # [B, D]
text_z = F.normalize(text_encoder(texts), dim=-1)     # [B, D]
scores = image_z @ text_z.T / temperature            # [B, B]
targets = torch.arange(scores.shape[0], device=scores.device)
loss = (F.cross_entropy(scores, targets)
        + F.cross_entropy(scores.T, targets)) / 2
```

这里的 temperature 是对比损失的温度，不是生成采样的温度。图文配对本身是一种监督信号，不能因没有固定类别标签就说它“完全无监督”。重复描述可能使所谓负例其实也正确，需要结合数据处理。

操作意义：输入杯子照片，分别与“红杯子”“蓝杯子”的文字特征比较，可做匹配或分类。这个配对目标本身没有训练逐 token 回答的能力。

### 7.3 LLaVA：把视觉特征接到语言模型上

视觉语言模型（vision-language model，VLM）同时处理图像和语言。LLaVA 原始版本使用视觉编码器、投影层和语言模型，把视觉特征接入语言模型的表示空间，并用视觉指令数据训练回答。[Liu et al., 2023，§4](https://arxiv.org/html/2304.08485v2)

```text
图片 → 视觉编码器 → 投影层 ──┐
                            ├→ 语言模型 → 回答 token
问题 → tokenizer → embedding ┘
```

```python
# 数据流示意，不是可直接调用的 LLaVA API。
visual_features = vision_encoder(images)
visual_tokens = projector(visual_features)
text_tokens = text_embedding(input_ids)
inputs_embeds = torch.cat([visual_tokens, text_tokens], dim=1)
# 还需要对齐 labels、位置、padding 和 attention mask。
```

仅让形状相同还不够：投影层必须学会把视觉特征变成语言模型能使用的信息。原论文第一阶段冻结视觉与语言骨干，训练投影层；第二阶段继续冻结视觉编码器，更新投影层与语言模型。这里特指原始版本，不泛指后续所有 LLaVA 变体。[原论文 §4.2](https://arxiv.org/html/2304.08485v2)

如果想检查模型是否真的看图，可以在保持问题不变时替换图片，观察答案是否合理改变。只测常识问题，可能掩盖模型忽略视觉输入的情况。

## 8. 具身智能：输出会改变下一次输入

### 8.1 从“回答杯子在哪”到“真的把杯子拿起来”

具身智能关心具有身体的系统如何感知、行动并与环境交互；实体机器人和模拟身体都可以是研究对象，并不要求一定使用大模型。

与单次图片问答相比，一个关键变化是行动会改变环境：

```text
相机画面 + 身体状态 + 指令
            ↓
          策略模型
            ↓
      动作表示 → 控制器执行
            ↓
          环境变化
            ↓
       重新观测，再决策
```

本体感觉（proprioception）是身体自身状态，例如关节角度、夹爪开合、速度。只看图片未必能充分判断机械臂当前姿态。

### 8.2 机器人训练样本长什么样？

```python
# 示例字段，不是特定数据集格式。
sample = {
    "image": image_t,
    "instruction": "pick up the red cup",
    "proprio": joint_state_t,
    "actions": future_actions,  # [H, A]，从当前时刻对应的动作开始
}
```

一次完整任务尝试叫一个 episode；按时间记录的观测与动作组成轨迹（trajectory）。训练时通常从轨迹中切出窗口。

动作可能是关节目标角度，也可能是末端执行器的位置或位姿增量。末端执行器可以理解为机械臂最前端的夹爪或工具。教学上的 7 维动作例子是“三维平移增量 + 三维旋转参数 + 夹爪指令”，但不能把它当作所有机器人的标准。

实际操作前必须读清数据协议：坐标是相对相机还是相对机械臂基座？数值是米还是毫米？位置是绝对目标还是增量？旋转采用哪种表示？动作对应当前图像之后的哪个时刻？同样是 `[7]`，语义可能完全不同。

**数据对齐是模型的一部分。** 图片晚一帧、动作早一帧，也许形状全对、loss 能降，执行仍会错。

### 8.3 行为克隆：从示范预测动作

```python
# 连续动作回归示意。
pred_actions = policy(images, instructions, proprio)  # [B, H, A]
loss = F.mse_loss(pred_actions, target_actions)
```

训练数据中的动作是人或其他策略实际执行的示范。常见问题是：训练时见到的状态来自专家，部署时见到的状态来自模型自己。小偏差累积后，它可能进入没有见过的位置。这通常叫分布偏移（distribution shift），需要用闭环执行评估，而不是只看离线误差。

训练与验证数据应优先按 episode、场景或对象划分；随机打散相邻视频帧，容易让几乎相同的画面同时出现在两边，夸大泛化表现。

## 9. 三种动作生成方式：回归、token、扩散

### 9.1 动作分块：一次规划一小段

动作分块（action chunking）就是一次输出未来若干步，而不是只输出下一步。例如输出 `[B, 16, 7]`。它描述输出的时间组织方式，可以与不同生成方法组合。

ACT（Action Chunking with Transformers）是代表性方法之一：以 Transformer 和条件变分自编码器建模动作块，并使用时间集成融合不同预测时刻给出的动作。ALOHA 是论文中的机器人系统，ACT 是策略方法，两者不要混为一个模型名字。[Zhao et al., 2023，官方项目的 Method](https://tonyzhaozh.github.io/aloha/)

条件变分自编码器（conditional variational autoencoder，CVAE）可以先理解为：训练时引入一个压缩示范变化的潜变量，并用正则约束它的分布，再结合观测解码动作。ACT 不只是“把单步 MSE 的输出长度加大”；本文不展开其完整目标与潜变量实现。

预测一段动作后，可以执行其中一部分就重新观测。重规划更频繁通常更容易响应变化，但推理开销和延迟也必须考虑。

### 9.2 VLA：让语言模型的输出落到动作空间

视觉语言动作模型（vision-language-action model，VLA）将视觉、语言与动作联系起来。它是一类模型，不指定唯一的动作头；并非所有 VLA 都逐 token 生成动作。

原始 OpenVLA 将连续动作维度离散化为 token，并使用 next-token prediction 训练，只对动作 token 计算交叉熵。[Kim et al., 2024，§3.2](https://arxiv.org/html/2406.09246v3)

```text
图像 + “拿起红杯子”
          ↓
     视觉语言骨干
          ↓
    动作 token 序列
          ↓
  解码为动作数值并反归一化
          ↓
       机器人控制器
```

下面是通用的均匀分箱教学例子，不是 OpenVLA 分词器的逐行复刻：

```python
# normalized_action 已按约定归一化到 [-1, 1]。
num_bins = 256
bin_ids = (((normalized_action.clamp(-1, 1) + 1) / 2) * num_bins)
bin_ids = bin_ids.floor().long().clamp(0, num_bins - 1)
decoded = (bin_ids.float() + 0.5) / num_bins * 2 - 1  # 用箱中心还原
```

这样可把连续动作预测转成分类，但会引入量化误差。`bin_ids` 还需映射到模型约定的 token 编号；得到归一化动作后也不能直接当作电机命令。必须使用与训练一致的统计量和动作接口。

可迁移的想法是复用视觉语言预训练，再用机器人示范连接到动作。这并不意味着会描述抓取步骤的模型已经会抓取，也不保证更换机器人后可以免训练使用。

### 9.3 Diffusion Policy：从噪声逐步修出一段动作

当杯子前面有障碍物时，从左绕和从右绕都可能合理。简单 MSE 回归可能把两种答案平均，得到直穿障碍的动作。这里的“多峰分布”指多个合理行为模式，与“图像加文字”的多模态输入不是同一个概念。

Diffusion Policy 用条件扩散建模动作序列，利用观测作为条件，并结合滚动重规划执行。[Chi et al., 2023，官方项目](https://diffusion-policy.cs.columbia.edu/)

```python
# 常见噪声预测形式的教学示意，scheduler 与 denoiser 是占位对象。
noise = torch.randn_like(target_actions)
noisy_actions = scheduler.add_noise(target_actions, noise, noise_steps)
predicted_noise = denoiser(noisy_actions, noise_steps, observation_features)
loss = F.mse_loss(predicted_noise, noise)
```

训练时知道加入的噪声，教模型如何去掉它；推理时从随机噪声出发，按调度器多次更新，生成一段动作。噪声步 `noise_steps` 是去噪过程的位置，**不是机器人真实运动时间**。

此类方法可以表达多个合理行为模式，但采样有迭代成本；学习到分布也不等于保证无碰撞。骨干可以有不同选择，扩散与 Transformer 并不是互斥选项。

### 9.4 放在一起比较

| 路线 | 训练目标示意 | 推理过程 | 首先检查什么 |
| --- | --- | --- | --- |
| 直接连续回归 | 预测动作与示范动作的距离 | 前向输出数值 | 动作单位、归一化、多解平均问题 |
| 动作 token 自回归 | 正确动作 token 的交叉熵 | 逐 token 生成并解码 | token 映射、错位、量化误差、延迟 |
| 动作扩散 | 预测噪声等扩散目标 | 多次去噪得到动作序列 | 噪声调度、采样步数、条件输入 |

动作分块可以出现在这些路线中。选方法时要看动作是否多解、可用示范、硬件延迟和闭环效果，不能只按论文年份选择。

## 10. 训练工程里最常见的词：看到它们该做什么？

| 名词 | 白话解释 | 代码或操作中的意义 |
| --- | --- | --- |
| Batch size | 一次看多少样本 | 调整 DataLoader；太大可能显存不足 |
| Epoch | 完整遍历一次训练集 | 小数据多轮训练可能过拟合；流式数据未必适合按轮定义 |
| Step | 一个计数单位 | 区分 microbatch 步与 optimizer 更新步，尤其有梯度累积时 |
| AdamW | 根据梯度及历史统计更新参数的优化器 | 检查学习率、权重衰减和参数组 |
| Warmup | 开始时逐渐提高学习率 | 检查 scheduler 是否按正确的更新次数推进 |
| 梯度累积 | 多个小批次后再更新一次 | 为每批合理缩放 loss，并在累积结束后 step |
| 梯度裁剪 | 限制梯度过大 | 通常在 backward 后、step 前操作 |
| Dropout | 训练时随机屏蔽部分激活 | 推理时切换 `eval()`；不是随机删除数据样本 |
| 混合精度 | 部分计算使用较低精度 | 可减少部分显存和提高吞吐；依硬件、dtype 与算子而定 |
| 激活检查点 | 少存中间结果，反向时重算 | 用额外计算换激活显存，不保证训练更快 |
| Checkpoint 文件 | 训练状态存档 | 恢复训练通常还需 optimizer、scheduler、步数和随机状态 |
| KV cache | 推理时复用过去 token 的 key/value | 减少重复计算，增加缓存显存；不是训练权重 |

梯度累积的有效 batch 常按“每设备批量 × 累积次数 × 数据并行设备数”计算；这假设每次都累积足量且损失缩放正确。它在 dropout、批归一化等情况下未必与一次大 batch 完全等价。

显存至少要区分：模型参数、梯度、优化器状态、中间激活，以及推理时的缓存。参数少不代表激活少；图像分辨率、视频帧数、文本长度变大，都可能增加负担。

在显式构造 `[T, T]` 分数矩阵的 attention 中，序列长度加倍会让该矩阵元素数变成四倍。这不是所有实现峰值显存或总运行时间的精确预测：高效 attention 可以避免保存完整矩阵，其他模块也占成本。

测效率时分开记录峰值显存、样本或 token 吞吐、端到端耗时和任务质量。只说“显存省一半”无法推出“训练快一倍”。

## 11. 拿到一个陌生仓库，按这条线读

先追踪一条样本，不必从几千行模型定义第一页开始读。

1. **数据读取**：`Dataset.__getitem__()` 返回什么？图片、文字、动作各是什么形状与单位？
2. **组批**：`collate_fn` 如何补齐长度？padding 是否被正确忽略？
3. **前向**：`forward()` 的每个输入被哪部分网络使用？输出是特征、logits 还是动作？
4. **损失**：答案从哪里构造？哪些位置被计分？存在未来信息泄漏吗？
5. **更新**：optimizer 包含哪些参数？哪些被冻结？实际可训练参数是多少？
6. **推理**：训练答案是否被意外传入？生成如何停止？动作如何还原？
7. **评估**：用的是独立场景吗？记录的是离线误差还是闭环任务成功率？

```python
# 局部检查示意。
print({k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable parameters:", trainable)
```

| 现象 | 优先排查 |
| --- | --- |
| loss 几乎不变 | 标签、梯度、学习率、optimizer 参数列表、冻结设置 |
| 训练好，验证差 | 数据太少、过拟合、分布差异、预处理是否一致 |
| 验证异常地好 | 重复数据、相邻帧泄漏、未来信息、标签被输入模型 |
| VLM 换图后答案不变 | 视觉特征是否接入、问题是否依赖图像、视觉条件是否被忽略 |
| 动作数值误差小但抓取失败 | 时序对齐、单位、坐标、控制频率、闭环状态偏移 |
| 出现 NaN | 数据异常、全屏蔽的 attention 行、学习率、精度与损失计算 |

这些是排查方向，不是仅凭症状就能确定的诊断。

## 12. 完整最小练习：亲手训练一个“数字接龙”Transformer

这个玩具模型学习 `0 → 1 → 2 → … → 7 → 0`。它没有自然语言理解能力，但具备 token embedding、位置 embedding、因果 attention、next-token loss 和训练更新。

运行条件：Python 与 PyTorch。将下面完整代码保存为 `tiny_transformer.py` 后运行 `python3 tiny_transformer.py`。本次未安装依赖、未运行该训练；不会在此给出虚构的 loss 或准确率。

```python
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

VOCAB_SIZE = 8
CONTEXT = 8
DIM = 32


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.position_emb = nn.Embedding(CONTEXT, DIM)
        self.norm1 = nn.LayerNorm(DIM)
        self.attn = nn.MultiheadAttention(
            DIM, num_heads=4, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp = nn.Sequential(
            nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM)
        )
        self.output = nn.Linear(DIM, VOCAB_SIZE)

    def forward(self, ids):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        x = self.token_emb(ids) + self.position_emb(positions)
        # nn.MultiheadAttention 的布尔 attn_mask：True 表示禁止读取。
        mask = torch.triu(
            torch.ones(length, length, device=ids.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return self.output(x)


model = TinyTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

# 每行从不同数字开始；答案从同一序列向后错一位得到。
starts = torch.arange(VOCAB_SIZE).unsqueeze(1)
tokens = (starts + torch.arange(CONTEXT + 1)) % VOCAB_SIZE
x, y = tokens[:, :-1], tokens[:, 1:]

model.train()
for step in range(201):
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step={step}, loss={loss.item():.4f}")

model.eval()
with torch.no_grad():
    accuracy = (model(x).argmax(-1) == y).float().mean().item()
    print(f"training token accuracy={accuracy:.3f}")

    generated = torch.tensor([[0]])
    for _ in range(7):
        next_id = model(generated)[:, -1].argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
    print("generated:", generated[0].tolist())

    # 未来 token 改变，不应影响前 4 个位置的输出。
    changed = x.clone()
    changed[:, 4:] = (changed[:, 4:] + 3) % VOCAB_SIZE
    max_difference = (model(x)[:, :4] - model(changed)[:, :4]).abs().max()
    print("causal check max difference:", max_difference.item())
    assert max_difference.item() < 1e-5
```

怎样验收：loss 应明显下降；训练 token 准确率应接近 1；从 `0` 开始应学会接龙；因果检查应通过。前几项是预期，不是本次已获得的结果。这个练习只有 8 种起点且重复训练，验证的是实现能拟合简单规则，不是独立测试上的泛化。

再做一个单变量对照：去掉因果 mask，观察未来扰动检查。它用来检查结构是否允许未来信息影响过去；有限样本的数值检查不是对所有输入的形式证明，特定权重也可能恰好不使用未来信息。

后续练习按成本递增：

| 练习 | 输入与操作 | 产出及验收 | 未验证点 |
| --- | --- | --- | --- |
| 数字接龙 | 运行以上代码 | 保存实际 loss、生成结果与因果检查 | 本次未运行 |
| 图文配对 | 小规模授权图文对，训练一个投影头 | 在独立样本中报告检索准确率；对照打乱配对 | 数据和特征提取器待选 |
| 动作回归 | 简单模拟器的专家轨迹，先训练 MLP 策略 | 分开记录离线误差与闭环成功率 | 环境、数据、种子待确定 |
| 动作分块对照 | 固定数据与骨干，只改变输出时域 | 比较任务成功率、延迟及错误类型 | 未实施，不承诺收益 |

每次先让极小训练子集能够被拟合，再扩展数据。这一步能帮助发现输入、目标或梯度错误；它只是管线检查，不能取代验证集评估。

## 13. 代表性工作与证据索引

这些工作按教学价值选择，覆盖不同接口与训练目标，不代表完整历史或截至今天的最强方法。这里只解释方法，不比较不同数据与硬件下的成绩。

| 文献 | 核验来源与位置 | 要记住的实现切口 | 边界 |
| --- | --- | --- | --- |
| Vaswani et al., 2017, *Attention Is All You Need*, NeurIPS | [论文正文 §3](https://arxiv.org/html/1706.03762v7) | attention、位置、残差、因果掩码 | 原始翻译架构不等于所有现代语言模型 |
| Devlin et al., 2018 / NAACL 2019, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | [原论文摘要与元数据](https://arxiv.org/abs/1810.04805) | 用原文构造遮盖预测目标 | 本次未逐项核验完整预训练实现 |
| Dosovitskiy et al., 2020 / ICLR 2021, *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* | [原论文摘要与元数据](https://arxiv.org/abs/2010.11929) | 图片切块后构成序列 | 不据此断言其他视觉骨干都这样实现 |
| Radford et al., 2021, *Learning Transferable Visual Models From Natural Language Supervision*, ICML | [正文 §2.3、Figure 3](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) | batch 内双向图文匹配 | 匹配训练不等于生成式问答 |
| Hu et al., 2021 / ICLR 2022, *LoRA: Low-Rank Adaptation of Large Language Models* | [正文 §4](https://arxiv.org/html/2106.09685v2) | 冻结大矩阵、更新低秩增量 | 未在用户任务上验证效果或显存收益 |
| Liu et al., 2023, *Visual Instruction Tuning*, NeurIPS | [正文 §4、§4.2](https://arxiv.org/html/2304.08485v2) | 视觉投影接入语言模型、分阶段训练 | 只描述原始 LLaVA，不泛化到后续变体 |
| Zhao et al., 2023, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, RSS | [官方项目 Method](https://tonyzhaozh.github.io/aloha/) | ACT 动作块、CVAE、时间集成 | 核验到项目方法说明，未复现完整算法 |
| Chi et al., 2023, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS | [官方项目 Introduction、Highlights](https://diffusion-policy.cs.columbia.edu/) | 条件动作扩散、滚动执行 | 未比较采样器、架构或实测控制延迟 |
| Kim et al., 2024, *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv:2406.09246 | [正文 §3.2—3.4](https://arxiv.org/html/2406.09246v3) | 动作离散化、动作 token 损失、数据接口统一 | 原始方法，不包含后续动作头改进 |

跨方法可借鉴的操作包括：CLIP 的配对检查、LLaVA 的先对齐接口再微调、LoRA 的参数更新范围控制，以及动作模型的时序与归一化核查。它们的迁移效果都需要在目标数据上验证；本文没有提出新的融合方法，也没有复现论文的基线、消融和性能指标。

## 14. 检索记录、未验证项与下一步

### 检索范围与筛选

检索日期为 2026-09-17。本次采用按论文题名与已知官方地址定向核验，未执行穷尽式关键词搜索或系统综述。访问范围包括 arXiv、PMLR、作者项目页和 PyTorch 官方文档。定位条目为 `Attention Is All You Need`、`BERT`、`Vision Transformer`、`CLIP`、`LoRA`、`Visual Instruction Tuning`、`ACT ALOHA`、`Diffusion Policy`、`OpenVLA`、`PyTorch optimization tutorial`。

纳入标准是能清楚展示一种数据流、训练目标或动作接口，且存在可追溯原始来源；同一工作合并为一个条目。Transformer、LLaVA、OpenVLA、LoRA 访问了正文，CLIP 访问了出版页与论文 PDF，ACT 和 Diffusion Policy 使用官方方法介绍，BERT 和 ViT 仅用于摘要足以支持的基础定位。

访问 ACT 的 arXiv HTML 指定版本失败，改用官方项目页；未把失败页面作为证据。未纳入新闻、二手榜单、无法核验的性能数字，也未追求收录所有近期 VLA。本文的玩具张量形状、机器人例子和练习方案是教学设计，不是来源报告的实验设置。

### Uncovered / 未验证项

- 完整练习及局部代码已做 Python 语法检查；当前环境未安装 PyTorch，未进行数值运行、收敛或接口兼容性验证。
- 没有下载模型、数据或权重，没有执行真实机器人控制，也没有产生论文复现结果。
- 未详细展开强化学习算法、偏好优化、流匹配、世界模型、混合专家、分布式训练和完整扩散数学；先掌握数据流与训练目标后再补充。
- 没有对学习成本、GPU-hours、吞吐或显存作实测；方法收益不能从本文类比推导出来。

### Next step / 下一步

先运行第 12 节，把实际输出记录下来，并用自己的话解释 `x`、`y`、`mask`、`loss.backward()`、`optimizer.step()` 各做什么。随后选一个你正在看的真实仓库，沿第 11 节追踪一条样本，把输入形状、监督来源、可训练参数和推理输出写在同一页。能追通这条线后，再决定需要深入哪篇论文。
