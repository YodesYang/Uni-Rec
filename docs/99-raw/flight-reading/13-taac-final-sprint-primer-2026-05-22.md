# TAAC Final Sprint Flight Primer · 2026-05-22

> **用途**：西班牙飞美国航班上的离线阅读材料。目标不是复述所有实验，而是在你落地前把本赛题的关键判断框架重建起来。
> **当前目标**：从团队 public SOTA **M123 = 0.851011** 冲击 **0.853+**，外部 SOTA 已到约 **0.85349**。
> **推荐阅读顺序**：先读本文，再读 `12-code-qa.md`，最后按需翻 `11-paper-roadmap.md` 和论文 PDF。

---

## 0. 一句话总纲

这个赛题现在不是“再试一个酷模型”的阶段，而是：

> 在强时间漂移、弱验证可信度、评估额度稀缺的约束下，围绕少数已被 public 验证的机制做低自由度放大。

当前最可信的 public-positive 家族只有：

1. **M106/M119/M123/M124 freshness family**：corrected T38 + UE61 split + ICQ + DIN + reduced valid ratio。
2. **MTL click+conversion regularization**：M123 把 M119/M117 一族推到当前 SOTA。
3. **checkpoint/selector/soup**：还没证明能涨 public，但它是低风险、可解释的最后放大器。

相反，很多听起来漂亮的方向已经被 public 或 clean-window 证伪：`vr015`、TIN-lite、history-CVR、time feature stacks、UE61/87 concat、MultiEmbK2、encoded-DIN 2.0、standard-valid seed search。

---

## 1. 赛题本质

这是一个广告 pCVR 排序问题，但平台把真实广告系统简化了很多：

| 工业广告系统 | 本赛题 |
|---|---|
| pCTR、pCVR、bid、budget、auction 一起决定收入 | 只看 pCVR AUC |
| 转化有延迟反馈和成熟窗口 | label 已经给好，但 train/test 时间差仍然很强 |
| 线上目标是未来 traffic | public/test 更像 train tail 之后的未来窗口 |
| 真实系统需要 calibration | 当前优化主要是 ranking AUC |

因此我们的核心难点不是建模容量不足，而是：

1. **train/valid/public 时间分布不一致**。
2. **正常 validation AUC 很容易被历史窗口骗**。
3. **clean-window 诊断又可能太小、太噪**。
4. **public quota 是最终真相，但每天只有 4 次**。

这解释了为什么很多 normal dev 高分模型 public 不涨，甚至大跌。

---

## 2. 数据心智模型

每一行是一次曝光/候选广告样本，目标是预测 conversion。特征大致分四类：

| 类型 | 例子 | 当前理解 |
|---|---|---|
| user_int | user sparse ids | 大量长尾，embedding 容易过拟合历史分布 |
| item_int | item/ad sparse ids | target item 表征是 ICQ、DIN、UE x item 的核心 |
| user_dense | fid61、fid87、count fids 62-66 等 | fid61 是已验证 UE；fid87 可能是外部情报提到的另一个 dense embedding |
| seq_a/b/c/d | 多域历史行为序列 | 时间尺度差异很大，T38 per-domain time buckets 是 confirmed positive |

几个必须记住的点：

- `fid61` 单独成 UE token 是 path B 的核心，早期 M49/M75/M84/M106 逐步验证了它。
- `fid87` 不是没有价值，而是我们之前把 `61+87` **拼成一个 token** 的实现不 work。
- 真正的 separate token 现在才用 M157/M158 测：`61` 和 `87` 各自一个 dense token。
- sequence 的 recency 不是一个全局尺度，四个 domain 的 time_diff 分布差异巨大，所以 corrected T38 是基础设施级收益。

---

## 3. 模型心智模型

当前主干不是普通 Wide&Deep，而是一个 “推荐特征 token 化 + HyFormer/RankMixer” 的混合体。

### 3.1 Token 布局

默认强主线配置大致是：

```text
seq query tokens: 2 queries * 4 seq domains = 8
NS tokens: user_ns 5 + user_dense/UE 1 + item_ns 2 = 8
total T = 16
d_model = 64, so 64 % 16 == 0
```

这就是为什么 M157/M158 的 true separate token 不能直接多加一个 UE token。直接加会变成 `T=17`，RankMixer full mode 会报错。我们采用：

```text
user_ns 4 + UE61 token + UE87 token + item_ns 2 = 8 NS tokens
total T remains 16
```

这叫 token budget swap。它牺牲一个 user_int chunk，换来一个真正独立的 fid87 dense token。

### 3.2 已验证组件

| 组件 | 状态 | 理由 |
|---|---|---|
| ICQ item-conditioned query | 保留 | 主线基础配置 |
| DIN classifier-time interest | 保留 | 与主线共存，未被当前 public-positive 家族否定 |
| UE61 split | 保留 | path B confirmed positive |
| corrected T38 per-domain buckets | 保留 | M84/M106 后的基础收益 |
| MTL click+conversion | 保留 | M123 public SOTA |
| dense-only weight soup | 待验证 | 低风险 side checkpoint，不改变主训练 |

### 3.3 已关闭或降级组件

| 组件 | 判决 |
|---|---|
| `valid_ratio=0.015` primary-led selection | public 过度乐观，M138/M144 关闭 |
| TIN-lite 当前实现 | M143 内部弱，M144 public 负，关闭 |
| UE61/87 concat token | M133/M136 不 work，不能代表 separate token |
| UE x item alpha=1.0 | M136 public 负，后续只能 mild alpha 或 separate-token 前提下再看 |
| history-CVR | M104 public 负 |
| time feature stacks / Beijing time | clean/public 多次负 |
| standard-valid seed search | M116 public 负，不能冲 0.853 |

---

## 4. 验证体系怎么想

把验证信号按可信度分层：

| 层级 | 信号 | 用法 |
|---|---|---|
| public LB | 每天 4 次 | 最终真相，但不能乱花 |
| public-positive family consistency | M119/M123/M124 | 当前最可靠的先验 |
| primary tiny-valid 2% | M119/M123 采用 | 对 freshness family 有用，但不能跨机制裸比 |
| aux m91 | leaky, 但有信息 | 只能辅助，不可单独选 public |
| clean windows | 防止大方向跑偏 | 对历史-CVR/time/feature 方向很有用，但对最终 ckpt 噪声较大 |
| normal 5% valid | 旧标准 | 现在对 public 相关性弱，不能主导冲刺 |

### 4.1 关键教训

M136 是最重要的反例：

- aux m91 很高：**0.856209**
- public 只有：**0.849015**

这说明高 aux 不是充分条件。只要机制伤害 primary 或 clean behavior，高 aux 可能是在拟合错误窗口。

M138/M144 是第二组反例：

- `vr015` tiny primary 很漂亮
- public 却掉到 0.849/0.848

这说明更小验证窗会产生 optimism，不是越贴近 tail 越好。

### 4.2 当前 public-positive anchors

| Model | Public | 内部读法 |
|---|---:|---|
| M106 | 0.849364 | seed5 corrected T38, 标准 valid 中最强 |
| M119 | 0.850754 | seed7 + valid_ratio=0.02 freshness-only |
| M123 | 0.851011 | seed5 + valid_ratio=0.02 + MTL, 当前 SOTA |
| M124 | 0.850718 | seed5 + valid_ratio=0.01 + MTL, aggressive freshness 仍 near-SOTA |

航班上想明白这张表就够了：我们现在所有合理冲刺都应该解释为 “如何从 M119/M123/M124 再榨 0.001-0.002”。

---

## 5. 明天如何看 M145-M158

落地后，如果日志很多，不要从 checkpoint 列表开始看。先看 Args，再看曲线，再看 side checkpoints。

### 5.1 先做 Args sanity check

所有主线任务都应该有：

```text
valid_ratio=0.02 or 0.01
enable_aux_ckpt_select=True
enable_weight_soup=True
enable_ue_split=True
ue_split_fid=61, unless M157/M158
enable_per_domain_buckets=True
enable_din_interest=True
item_conditioned_query=True
```

M157/M158 额外必须有：

```text
user_ns_tokens=4
ue_split_fids='61,87'
ue_split_separate_tokens=True
PCVRHyFormer model created: num_ns=8, T=16
```

如果 M157/M158 的日志没有 `T=16`，或者没有 `ue_split_separate_tokens=True`，直接当作 invalid readout。

### 5.2 anchor 数字

用这些数字做第一眼对比：

| Anchor | Primary AUC | LogLoss | Aux m91 | Public |
|---|---:|---:|---:|---:|
| M119 | 0.871196 | 0.221458 | 0.855080 | 0.850754 |
| M123 | 0.870584 | 0.221884 | 0.853945 | 0.851011 |
| M124 | 0.868723 | 0.225034 | 0.854380 | 0.850718 |
| M135 | 0.871377 | 0.221546 | 0.853113 | not evaluated |
| M138 | 0.872871 | 0.221286 | 0.854026 | 0.849451 |
| M144 | 0.871954 | 0.220568 | 0.855139 | 0.848183 |

注意 M138/M144 的存在：高 primary 或高 aux 都不是万能。必须结合机制家族。

### 5.3 任务解释

| Job | 读法 |
|---|---|
| M145 | M123 exact + auxselect/soup，最干净的 checkpoint-selection 测试 |
| M146 | M119 exact + auxselect/soup，判断 no-MTL 是否只是 checkpoint 没吃满 |
| M147 | M123 5ep，保留 E4/topk/soup，测试多一轮是否能平滑提升 |
| M148 | M141 思路 5ep，若 M141 健康则很重要 |
| M149/M150 | MTL loss shape，小改 conversion weight 1.5/3.0 |
| M157 | 真 separate `fid61`/`fid87` token，无 xitem，最干净机制 test |
| M158 | separate token + mild UE x item alpha=0.3，完整外部情报 test |
| M151-M156 | 旅行缓冲，优先级低于上述任务 |

### 5.4 Eval 候选排序规则

如果明天你只有很少决策时间，按这个顺序想：

1. **先看 M141/M137 是否健康**。它们是 seed7+MTL 缺口的重要证据。
2. **M145 如果不差于 M123 且 soup/aux/blend 更健康，优先评 M145 的最佳 side checkpoint**。
3. **M146 如果接近或超过 M119，优先级很高**，因为 M119 已 public-positive。
4. **M157 如果 primary/logloss 不伤且 aux 不差，值得至少一次 public calibration**。这是外部 `61/87` 情报的最后正统验证。
5. **M158 只有在 M157 或自身 primary 健康时才考虑**，不要因为 aux 高就评。
6. **M149/M150 只有在明显改善 M123-family primary/logloss 时评**。
7. **M151-M156 默认是补充，不要抢前面名额**。

红线：

- 不评 `vr015` 后代，除非出现极不寻常且可解释的新信号。
- 不评 TIN-lite 当前形态。
- 不评高 aux 但 primary/logloss 明显伤的 UE/xitem 变体。
- 不做 standard-valid seed 上沿 public eval。

---

## 6. 飞机上该读哪些已有文档

### 6.1 必读 1：当前状态

- `docs/00-context/platform-state.md`
- `docs/40-experiments/EXP-090-strategy-to-0853-2026-05-21.md`
- `docs/40-experiments/EXP-093-final-sprint-m123-family-m145-m150-2026-05-21.md`

读法：只读文件顶部 recent updates 和最后策略段，不要陷入完整历史。

### 6.2 必读 2：模型理解

- `docs/99-raw/flight-reading/12-code-qa.md`
- `docs/99-raw/flight-reading/08-model-py-key-sections.md`
- `docs/00-context/model-py-section-map.md`

重点理解：

1. NS token 是怎么构造的。
2. item target 是怎么进入 ICQ/DIN 的。
3. dense bypass 为什么能救 UE signal dilution。
4. RankMixer 为什么有 `d_model % T == 0` 限制。

### 6.3 必读 3：广告和论文

- `docs/99-raw/flight-reading/10-background-primer.md`
- `docs/99-raw/flight-reading/11-paper-roadmap.md`
- ESMM PDF：理解 CTR/CVR/CTCVR 和 MTL 动机。
- HyFormer/RankMixer PDF：只读 architecture 图和 method，不必逐字读。

---

## 7. 你可以在飞机上思考的 8 个问题

1. Public 是不是更像 “train tail 的未来同分布”，而不是普通随机 holdout？
2. M123 为什么 public 比 M119 高，尽管 M119 的 2% primary/aux 更高？
3. M138/M144 为什么内部漂亮但 public 崩？它们到底过拟合了什么？
4. fid87 是真的没用，还是 concat 实现错了表达形式？
5. MTL 的 click head 是 regularizer，还是引入了更接近 public 的 proxy target？
6. Weight soup 如果有效，应该看 primary、aux、logloss 哪个组合？
7. 如果 M157 训练健康但 M158 不健康，是不是说明 external intel 的 `UE x item` 部分过强？
8. 如果所有 M145-M158 都不涨，下一步是否应该承认 0.853 只能靠 submission-level ensemble 或平台外机制，而不是单模型？

---

## 8. 落地后 15 分钟决策清单

按顺序做：

1. 粘贴最新 `train_log.txt`。
2. 先让我解析 Args，确认没有 copy/paste 错误。
3. 对每个任务记录：E1-E5 primary AUC、logloss、aux m91、best/aux/blend/soup checkpoint。
4. 只挑最多 2 个 morning eval candidate，留 1-2 个 quota 给晚间新结果。
5. 若 M157 健康，至少认真讨论一次是否给 public calibration。
6. 若任何任务只靠 aux 高、primary/logloss 伤，默认不评。

一句很实用的准则：

> 明天不要问“哪个内部最高”，要问“哪个最像 M119/M123 这个 public-positive 家族的健康延伸”。

