# TAAC x KDD Cup 2026 工业组最终复盘

## 1. 最终状态

最终榜单确认后，本队在 Industrial Track 的结果是：

| 项目 | 结果 |
|---|---:|
| 排名 | **32 / 689** |
| 百分位 | **Top 4.6%** |
| 最佳 public AUC | **0.851365** |
| 最佳模型 | **M148 normal best_model** |
| 目标线 | 约 **0.852** |
| 距目标线 | 还差 **+0.000635** AUC |

结论要客观：结果不错，足够作为工作/项目/开源资产；但从比赛目标看，没有跨过 0.852 晋级线。真正有价值的是这套完整的工业推荐竞赛闭环。

## 2. 最终上分主线

最终正向主线不是“大模型堆模块”，而是逐步对齐目标分布：

```text
Early baseline 0.84381
  -> M84 corrected per-domain T38 bucket: 0.848854
  -> M106 corrected-T38 seed5 + aux window: 0.849364
  -> M119 fresh-tail valid_ratio=0.02 seed7: 0.850754
  -> M123 fresh-tail + MTL seed5: 0.851011
  -> M148 M119-lineage + MTL + 5ep seed7: 0.851365
```

| 模型 | Public AUC | 关键变化 | 赛季意义 |
|---|---:|---|---|
| Early baseline | 0.84381 | cleaned SwiGLU/HyFormer baseline | 快速可跑基座 |
| M84 | 0.848854 | corrected per-domain time buckets | 时间桶修正是真正 public-positive |
| M106 | 0.849364 | M84 seed5 + aux m91 | corrected baseline 上沿 |
| M119 | 0.850754 | `valid_ratio=0.02`, seed7 | fresh-tail 转折点 |
| M123 | 0.851011 | M119/M106 freshness + MTL | click/conversion 多任务有效 |
| M148 | 0.851365 | M119 seed7 + MTL + 5 epochs | 最终最好 |

## 3. 最终最佳 M148 的真实含义

M148 不是一个巨大机制突破，而是一个 public-positive 家族里的“小而可靠”组合：

```text
M148 = M119 fresh-tail seed7
     + MTL click/conversion objective
     + 5 epochs
     + corrected T38/per-domain buckets
     + UE61 split / DIN interest / ICQ 等既有 backbone
```

关键事实：

- M148 normal `best_model` public **0.851365**。
- M148 dense-only `weight_soup` public **0.850563**，低于 normal best_model。
- M159 lower dense LR public **0.851279**，接近但未超过 M148。
- M165 继续 6ep public **0.851162**，未超过 M148。
- M162 `conversion_loss_weight=1.5` public **0.851043**，未超过 M148。
- M146 no-MTL backup public **0.850222**，未转化。

因此，M148 的信息不是“5ep 总是好”或“soup 有效”，而是：

> seed7 fresh-tail + MTL 这一条 public-positive 路径在最后阶段被验证为最高；同族小扰动没有继续突破 0.852。

## 4. 最大认知收获：验证体系比模块更重要

本赛季最关键的判断不是某个模块，而是发现普通验证集会误导。

反复出现的模式：

```text
normal dev AUC / logloss 变好
public AUC 不升反降
```

因此我们从“单一 local AUC 选择”转向：

```text
normal validation
  + clean-window diagnostics
  + aux validation windows
  + public-positive family consistency
  + scarce eval quota decision
```

明年最该继承的是这套思维，而不是某个具体分支。

## 5. 被证伪或暂时不该继承的路线

这些方向听起来合理，但在本赛季没有转化成 public 分数：

| 路线 | 代表实验 | 结论 |
|---|---|---|
| naive Beijing/UTC absolute time features | M86/M89/M92/M99/M100 | local 可亮，public/clean-window 不稳定，默认不要开局押注 |
| history-CVR aggregate | M101-M104/M107-M108 | full dev 亮，但 public M104 负；机制/窗口不匹配 |
| seqstats / delayed feedback batch | M93-M100 | clean-window gate 大面积失败 |
| UE61/87 concat + xitem | M129-M136 | high aux 误导，M136 public 负 |
| true separate UE61/87 tokens | M157/M166/M167 | 当前实现弱，不是最后突破口 |
| TIN-lite classifier-time interest | M143/M144 | public 负，当前形态关闭 |
| dense-only weight soup | M141/M142/M144/M148 soup | M148 soup 低于 best_model |
| seed/logloss-only lottery | M151/M168 | high logloss/seed 信号不可靠 |
| sparse_lr 小扰动 | M155/M160/M161/M174 | M160 public 明确负，默认不要优先 |

明年可以重新探索这些方向，但不能把本赛季失败忘掉。重开必须有新的数据假设、新验证体系或明显不同实现。

## 6. 明年最该继承的资产

### 6.1 工程资产

- `src/dataset.py`: Parquet IterableDataset、time split、fixed window、aux windows、feature switches。
- `src/model.py`: PCVRHyFormer、RankMixer NS tokenizer、SwiGLU encoder、ICQ、UE split、DIN interest、MTL support。
- `src/train.py`: 完整训练参数入口、aux validation、MTL、soup/TIN/UE 等开关。
- `src/trainer.py`: top-k checkpoint、aux/blend side checkpoint、summary/logging。
- `src/infer.py`: checkpoint config-driven inference，默认关闭 eval-time EDA blob。

### 6.2 方法资产

- 先建 public-like validation，再谈模型。
- 任何新机制都需要 paired clean diagnostic，不只看 normal dev。
- public-positive family 的小改动，比脱离主线的高 local AUC 更可信。
- 失败实验必须进入“负结果表”，否则最后几天会重复踩坑。
- public eval quota 是稀缺资源，应该作为决策问题管理。

### 6.3 文档资产

- `docs/00-context/platform-state.md`: 最完整的赛季时间线。
- `docs/40-experiments/EXP-093...`: 最终冲刺族谱。
- `docs/60-season-review/`: 明年起手入口。
- public repo `KDDCup2026-TencentAds-UniRec`: 对外展示版本，私有 repo 是真实工作区。

## 7. 如果明年重来，第一周策略

1. **第一天**：跑官方 baseline + 复刻当前 repo 最稳 backbone，确认 schema/平台/推理路径。
2. **第二天**：做数据 EDA，尤其时间轴、label 成熟度、train/test gap、sequence length 分布。
3. **第三天**：建立 2-3 个 clean validation windows，不要等到中后期才做。
4. **第四天**：把所有实验分成 mechanism / validation / objective / seed，不混在一起。
5. **第五天后**：只让 public eval 服务于已通过 offline gate 的候选；保留 1-2 个信息评估额度，但不要每日乱烧。

## 8. 最后一句

这次比赛的最终教训很朴素：

> 在工业广告推荐竞赛里，真正难的不是想出 100 个模块，而是在 noisy validation 和稀缺 public eval 下，持续判断哪些信号值得相信。
