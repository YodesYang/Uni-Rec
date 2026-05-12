# XHS / WeChat 情报 · 2026-05-12 晚（DECEM Tricks 2/2 + 杭仔 0.85+ LB + 群聊）

> **来源**：用户 2026-05-12 CEST 19:09 从小红书 + 微信群（"广小鹅的冒险岛" 500 / "广告算法大赛交流" 500）收集 37 张截图
> **原始文件**：本目录 37 张 `Screenshot_2026-05-12-*.jpg`
> **纪律**：本目录为"原始情报沉淀"，方法论结论已提炼到 `docs/10-research/community-intel.md` §DECEM-0512

---

## 📋 情报索引

37 张截图按内容分 4 类：

| 文件名时间戳范围 | 数量 | 内容类型 |
|---|---|---|
| 18-45-54 ~ 18-47-43 | 2 | 🟡 **用户自己平台 job 列表**（参考对比用） |
| 18-50-04, 18-53-23, 18-54-44 | 3 | 🟢 **DECEM 小红书主贴 + 配图 2/2 页** |
| 18-50-28, 18-50-36 | 2 | 🔵 **杭仔 platform LB 截图**（有 0.850012 / 0.848366 / 0.845012） |
| 18-51-39 ~ 18-52-55 | 4 | 🔵 **Seed "TAAC Industry Track 0.85+ 分享"** 帖子 |
| 18-55-11 ~ 19-04-41 | 26 | 🔵 **微信群聊讨论**（DECEM / 杭仔 / momo / PJH 等） |

---

## 🥇 DECEM 小红书主贴 `2026 腾讯广告算法大赛改进 Tricks 总结（个人观点仅供参考）`

**帖子位置**：小红书用户 DECEM（曾 Industrial Track Rank 1 LB 0.852181）发的付费笔记（¥39.9 已售 205）
**截图 18-54-44（配图 2/2 页）文字 OCR 摘录**：

### 前言标题

> **6.4.5 2026 腾讯广告算法大赛改进 Tricks 总结（个人观点仅供参考）**
>
> 改进 Tricks 汇总（持续更新中 →）

### 1. 时间分桶边界设计

- **BUCKET_BOUNDARIES** 是手工设计的 64 个对数边界，从 5 秒到 1 年。手工边界很可能在你的数据分布上有"空档"或"过度集中"桶，浪费 Embedding 容量
- 如果 `time_diff` 呈长尾分布，按等比例分位（1/100，2/100，...）重新边界
- 或者在距离较近的时间段（比如 7 天内）进行密集等距分桶
- 这里不同的 domain 序列时间 diff 对齐不同，建议各自分桶处理

### 2. 序列长度差异化优化

- a:256, b:256, c:512, d:512, c/d 域更长。但从 sample 看，d 域某字段长度超过 2000 个 token
- 对专个域统计真实非零长度的 `p99/p95`，按 domain 的真实情况设置 `seq_max_lens`
- 或配合 LongerEncoder (Top-K 稀疏注意力) 处理长序列

### 3. 缺失值差异化处理

- 所有 null / 0 / -1 统一映射为 0 (padding)，但是 sample 显示：大量 null（`86,94,99-104,108,109`）。如果这些字段缺失率与转化率相关（例如用户/物品 ID），**把缺失当 0 就丢了关键信号**
- 建议加一个 "missing" 专用 slot（vocab_size 配置区分），把 padding=0 独立
- **缺失指示位**：把"是否为空"作为额外的二值 `concat` 进模型（很多广告业务把"缺失本身作为信号"）

### 4. 重新设计 `ns_groups.json`

- baseline 默认配置走 rankmixer，不使用 `ns_groups.json`。GroupNSTokenizer 重同组特征共享一个 token，相关性高的特征共享
- **特征重要性加一遍**：按 domain 的真实情况设置 `seq_max_lens`
- 或改用 LongerEncoder (Top-K 稀疏注意力) 处理长序列
- 加 **SENet** 特征的重要度：对 NS tokens 使用 gate-style 的 channel-wise reweighting，SENet 的 weight 也可以用来指示特征重要性

### 5. 交叉特征增强

- 模型完全靠 Embedding+Attention 隐式学交叉，增加一些显式交叉结构在工业界 AUC 首选有用，且训练稳定
- 加 **FM/DCN-v2 分支**：把所有 user_int 和 item_int 的 embedding 先过一层 DCN/FM 交叉，与 Transformer 输出 concat 后过分类头
- (可能改 Ensemble？如不能因改在推理量大业务上)

### 6. Target Attention (DIN)

- baseline 四个 HyFormer Block 内部做 self-attention，没有显式 target item 或 target attention
- MultiSeqQueryGenerator 的 Q tokens 从 NS+MeanPool 生成的，但 item-side 的 ID embedding 没有作为 attention key 进行强化
- 在 MultiSeqHyFormerBlock 的 cross-attention 里，让 Q tokens 的初始值由 item embedding 引导：**Q_init = MLP(item_emb)** 而非 MLP(GlobalInfo)
- 或者加一个 DIN 分支：`q_i = softmax(Q_target · K_seq)`，结果与 Transformer 输出 fusion
- **DIN 的核心 insight 是说 "用户对广告 A 和广告 B 感兴趣的历史是不同的"**，target attention 能让模型对当前广告激活相关历史

---

## 🔵 杭仔 LB 截图（Screenshot 18-50-36 里贴的截图）

群聊 "广小鹅的冒险岛" 用户 **杭仔**（Sun 02:51 "熬了个夜 上大分" 附截图）贴出的自己的 platform evaluation history：

| Evaluation ID | Status | LB AUC | Inference Time | Last Updated |
|---|---|---|---|---|
| ..._1778373440941 | Success | **0.850012** | 170.23s | 2026-05-10 08:49:3.. |
| ..._1778365943951 | Success | **0.848366** | 182.99s | 2026-05-10 06:43:4.. |
| ..._eval_1778343916188 | Success | **0.845012** | 854.7s | 2026-05-10 00:50:0.. |
| ..._1778294954858 | Success | **0.841897** | 1024.06s | 2026-05-09 11:56:34 |
| ..._1778294176332 | Stopped | — | — | 2026-05-09 10:45:4.. |
| ..._1778291934737 | Stopped | — | — | 2026-05-09 10:26:0.. |

**重要**：
- 杭仔在 2026-05-10 凌晨已达 **LB 0.850012**（Top 20 阈值）
- 170.23s 推理时间极短（远快于我们 500s），暗示模型大小或 compile 效率不同
- 他自述 "6 小时 15 个 epoch"（单 epoch ~24 分钟，比我们 75 分钟快 3×）

---

## 🔵 群聊关键讨论摘录

### 讨论 A · 杭仔 (Sat 02:51~Sun 04:12) 技术路线

| 时间 | 发言人 | 内容 |
|---|---|---|
| Sun 02:51 | 杭仔 | 熬了个夜 上大分 🍑 + 贴 LB 截图 |
| Sun 02:56 | 杭仔 | "其实挺迷的，目前看用 **transformer+cross attention 提取序列特征的模块，全换成 din 的提升很大**" |
| Sun 03:10 | 天度 | "din 的 target 是只用 item 还是所有 ns token 呢" |
| Sun 03:10 | 杭仔 | "**我只用了 item**" |
| Sun 03:10 | 灏灏灏 | "@杭仔 序列 merge 了吗 还是说还是分开的" |
| Sun 03:10 | 杭仔 | "**分开的**" + "因为我之前出现过 best epoch 和上一个 epoch 测试集直接差了 5 个千分点，所以我觉得 valid 差不多的情况下就手动保存了一版中间的" |
| Sun 03:26 | bbj | "@杭仔 那换成 din 了，transformer 还在吗" |
| Sun 03:26 | 找实习破防版 | "换成 din 还有 global token 吗" |
| Sun 03:31 | 杭仔 | "**@bbj 不在了**" |
| Sun 03:31 | momo | "那其实就是相当于 din，然后接了一个增强模块?" |
| Sun 03:31 | momo | "可以这么理解嘛佬" |
| Sat 06:15 | 杭仔 | "当然还有其他一些常规的时间特征，和别的基本上都是**小红书偷来的上分点** 哈哈哈🤣" |
| ？ | bbj | "6 个小时 15 个 epoch" + "两小时就训练好了" |
| ？ | 杭仔 | "@bbj 但也确实挺快的，6 个小时 15 个 epoch" |

**技术要点**：
- DIN **完全替换** Transformer+cross-attention 模块（不是我们的 bypass 方式）
- DIN target **只用 item**（不是所有 ns tokens）
- **分开 query generator**（没 merge）
- **DIN 替代后 global token 不存在了**
- Best vs last epoch LB 差 **5‰**（过拟合信号）→ 手动保存中间 epoch

### 讨论 B · DECEM 的 Q&A

| 时间 | 发言人 | 内容 |
|---|---|---|
| Tue 18:06 | PJH | "200万训练集，53万测试集" |
| Tue 18:06 | momo | "@DECEM 大佬是怎么理解 din 的? 只在于那个权重参数吗?" |
| Tue 18:13 | DECEM | "**每个人对 DIN 的理解和实现都有大小出入，所以还是靠实验吧**" |
| Tue 18:13 | DECEM | "**作为一个模块嵌入到模型中**" |
| Tue 18:13 | DECEM | "**不要尝试过大的改动，我这里的提升是 +0.001**" |
| Tue 18:18 | 冰淇淋 | "@DECEM 跟我提升一样" |
| ？ | 27搜广推蛋仔 | "@DECEM 佬 交叉特征有用吗 今天试了下 掉了" |
| Yesterday 17:19 | DECEM | "**我没做，不知道**" |
| Sun 13:53 | momo | "@27搜广推蛋仔 用 onetrans 比官方提供的 hyformer 有提升" |
| ？ | 27搜广推蛋仔 | "有佬试过改结构吗 不要 hyformer 用 onetrans" |
| ？ | Gallagher 加拉格 | "backbone 直接改了还是作为一个模块" |
| ？ | 27搜广推蛋仔 | "@momo 佬 请问多序列还是类似 hyformer 处理吗" |
| 14:00 | momo | "@Gallagher 加拉格 单独写了个新的" |
| ？ | momo | "@27搜广推蛋仔 就是按照 onetrans 论文来的，每个 domain 的每个 event 生成 token 然后 concat 起来后面跟着 ns token" |

**技术要点**：
- DECEM 风格 = **小步迭代，每步 +0.001**（我们之前 T25 +0.004 已经是"大改动"级别，罕见）
- DECEM 自己 **没做 FM/DCN 交叉**，是 tricks 文档里的"推测方向"
- onetrans 被多人尝试，比 hyformer 有提升
- 杭仔用 onetrans 式的 token 生成 + 接 ns token

### 讨论 C · Scaling / 其他技术点

| 时间 | 发言人 | 内容 |
|---|---|---|
| 14:00+ | 心平 momo | "有 scaling up 的效应吗" |
| 14:11 | momo | "**暂时没有发现 scaling up**" + "就简单试了一下，还没细做这个方向" |
| 14:24 | 吃象伯 | "这个点推理和训练的速度都慢下来了" + "下午推理时间半小时，晚上就一个小时了" (资源紧张) |
| ? | DJ 使我平静 (学术赛道) | "工业 track 的时间特征是不是没用? 学术 track 都说用了涨分，我用了各自掉分" |
| ? | 闲闲超人 | "一样掉" |
| ? | Luminous | "掉的很彻底" |
| ? | Luminous | "但也有工业赛道用时间特征涨分的 没招了" |

**技术要点**：
- **Scaling up 确认无效**（再次验证，与 5/8 DECEM 情报一致）
- 工业赛道时间特征**取决于具体做法**（有人涨有人掉）

### 讨论 D · 其他有价值的碎片

| 发言人 | 内容 |
|---|---|
| 牛马艺术家-工业 | "不好意思，表述有问题，我没试过基线，一开始就把序列拉到 3000 了，昨天调回基线的 1500，**涨点了**" |
| 蟑螂恶霸 | "1500 不是 1024 吗" |
| 得之-学术 | "加上时间特征后 val auc 降" |
| 闲闲超人 (小红书评论) | "**我们 baseline 844ª，baseline 就可以打败很多队伍了**" |
| DJ 使我平静 (小红书评论) | "一模一 baseline 还是开了 amp + 编译得到的，不开都没那么高，后续各种修 bug 或者加复杂或者简单的时间特征都往下掉，我真的不知道咋回事了" |
| 擎烂 ing | "我改成 din 只涨了万分位 估计是波动" |
| Dyl.X | "ns group 有用吗" |
| 岚峰 | "好像没有" |
| 岚峰 | "这个比赛很难搞得一点是，**固定种子之后结果似乎也不固定**，看不明白操作有没有用" |
| lllll | "涨了就是有用" |

**技术要点**：
- 序列长度从 3000 回到 1500 **涨点**（印证 DECEM tricks #2 "按 p99/p95 自适应"）
- **fixed seed 下结果仍不固定**（CUDA nondeterminism 再次被社区观察到，与我们 team-memory 一致）
- baseline 可达 0.844（闲闲超人）
- 有人 改 DIN 只涨万分位（"姿势不对"）

---

## 🔵 Seed 另一篇 "TAAC Industry Track 0.85+ 分享"（Screenshot 18-51-39）

标题：**TAAC Industry Track 0.85+ 分享** (271 ❤️ / 419 ⭐ / 19 💬)
作者：Seed

首段内容（1/? 页）：
> 腾讯广告算法大赛 Day x
> 试了一下，感觉问题好多，没招了 🥲，简单说下目前方案
> 1. user dense feats 将 ue 分离出来，其他跟 int pair 加权
> 2. 建模为 raw embedding + 序列 merge+fafe+din

**（后续内容截图未包含）**

**这个 Seed 帖已在 `docs/10-research/community-intel.md` §工业赛道主楼 Seed 章节有记录（2026-05-05 pdt 凌晨处理）**，今天没新增内容。

---

## 文档使用指南

- 本 README 是**完整提取的原始文本**，方便 grep 和引用
- **方法论结论**已提炼到 `docs/10-research/community-intel.md` §DECEM-0512
- **行动路线**已记录到 `docs/20-tasks/active.md` T30/T31/T32 task 入口
- **铁律补充**已入 `.cursor/rules/ai-readme/team-memory.mdc`
