# EXP-073 · XHS 时间特征情报与 M85-M88 训练池

Date: 2026-05-18

## External Intel

小红书截图要点:

- 有选手称在 TokenFormer 上耗费近 2 周，分数上不去；回到 baseline 重写一遍特征，参数不变到约 **0.849**。
- 评论中另一个选手表示 TokenFormer 也到 **0.849** 左右，但不确定能否继续上。
- 有人问“什么特征”，作者回答：**时间**。
- 另一个评论称纯换 TokenFormer 比 baseline 高约前 3，但叠加旧特征工程反而掉约千 1.5。

## Interpretation

这条情报支持我们当前判断：

1. **不要把主资源投到大架构改写**。TokenFormer 不是 0.852 的稳定主钥匙，至多证明 0.849 附近可达。
2. **baseline family + 时间/特征工程仍是主线**。M75/M84 已经到 0.8488，和截图信息相符。
3. 我们已有 `--enable_time_of_day_features`，所以“时间特征”不是简单加 hour/dow。更值得查的是时间定义和目标分布。

## New Hypothesis

现有 `enable_time_of_day_features` 使用 UTC:

- hour sin/cos：UTC 与北京时间相差固定相位，模型大概率能学。
- DOW/weekend：UTC+8 会在北京时间 00:00-07:59 跨天，**不是简单相位旋转**，可能错标 weekday/weekend。
- 样本重加权代码已使用北京时间，说明我们内部也已把 test daypart 按北京口径理解。

因此新增 opt-in `--enable_beijing_time_features`，不改变旧默认行为。

新增 10 维 user_dense:

- 北京 hour 一阶 sin/cos
- 北京 hour 二阶 sin/cos
- 北京 DOW sin/cos
- 北京 weekend / workday
- 北京 09-14 test-window flag
- 北京 06-08 early-morning flag

## Training Pool

| Job | Purpose | Expected Readout |
|---|---|---|
| `M85-m84-t38fix-bjtime-gap0-seed42-4ep` | M84 stack，替换 UTC time 为北京时间增强 time | 测时间口径修正是否有效 |
| `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep` | M84 stack，UTC time + 北京时间增强 time 同时保留 | 测增量时间表达力，代价是略增参数/冗余 |
| `M87-m84-t38fix-tempweight25-gap0-seed42-4ep` | M84 stack + soft temporal weight 2.5 | M41 微正信号与 true T38 是否可叠加 |
| `M88-m84-t38fix-6ep-gap0-seed42` | M84 stack 延长到 6 epoch | 测 true T38 是否需要更久收敛 |

## M89 Prepared Candidate

2026-05-18 follow-up: implement but **do not launch until M85/M86回流**.

| Job | Purpose | Launch Condition |
|---|---|---|
| `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep` | M84 stack + Beijing time v1 + Beijing distribution time v2; no sample reweight | M85/M86 dev 不崩，且时间特征方向仍有 public-up 可能 |

新增 opt-in `--enable_beijing_time_v2_features`，追加 14 维 user_dense:

- 距离目标测试日 2026-03-04 的 signed / absolute day distance
- 2026-03-03 festival-day flag
- 2026-03-04 post-festival / target-day flag
- before-festival / after-target flags
- 距离北京 09-14 测试窗口的 hour distance
- 相对 11:30 测试窗口中心的 signed hour distance
- 5 个 coarse daypart one-hot: 00-05, 06-08, 09-14, 15-20, 21-23
- target-day × 09-14 flag

设计原则: 让模型自己学习 train/test 时间差异，不再用 x5/x25 这类硬重加权。

## M85/M86 Results

Source: user pasted `~/Downloads/train_log.txt` after M85/M86 completed.

Both jobs used corrected T38:

- schema timestamp resolve log present: `domain_a_seq_39`, `domain_b_seq_67`, `domain_c_seq_27`, `domain_d_seq_26`
- per-domain fit used nonzero samples: each domain `n_samples=5000000`

| Job | Time Features | E1 | E2 | E3 | E4 / Best | LogLoss | Readout |
|---|---|---:|---:|---:|---:|---:|---|
| `M85-m84-t38fix-bjtime-gap0-seed42-4ep` | Beijing time only | 0.865865 | 0.869207 | 0.870021 | **0.870887** | 0.221527 | Replacing UTC with Beijing does not hurt; slightly above M75 dev and +0.000397 vs M84 |
| `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep` | UTC time + Beijing time | 0.866497 | 0.869430 | 0.869953 | **0.871415** | 0.221523 | Clear dev winner; +0.000925 vs M84 and +0.000559 vs M75 |

Interpretation:

1. XHS “时间特征”情报得到强验证：时间特征不是噪声，且在 M84 corrected-T38/path-B 主干上有明显 dev 增益。
2. M85 说明北京时间口径本身可用；M86 说明旧 UTC time 与 Beijing-rich time 不是完全冗余，保留两者更强。
3. M86 E4 从 E3 跳升 +0.001462，尚未显示收敛；后续可考虑 M86 6ep / M86+v2。
4. 因 M82/M83 已证伪强 reweight，下一步继续 feature-only，不做 hard sample weighting。

Decision:

- Evaluation priority: **M86 E4 first**, **M85 E4 second** if two quotas available.
- Training priority: M89 launch condition is satisfied; start `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep` if a slot is free.
- Optional follow-up: launch `M90-m86-utc-bjtime-6ep-gap0-seed42` if another slot is free, because M86 curve is still rising at E4.

Prepared follow-up:

| Job | Purpose | Run script |
|---|---|---|
| `M90-m86-utc-bjtime-6ep-gap0-seed42` | M86 exact config extended to 6 epochs | `src/run_m90_t38_fix_utc_bjtime_6ep.sh` |

2026-05-18 14:13 CEST user update:

- `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep` started.
- `M90-m86-utc-bjtime-6ep-gap0-seed42` started.
- Training pool now has 2 in-flight jobs from the time-feature mainline.

## Upload Files

- New-code jobs M85/M86/M89 need upload: `src/dataset.py`, `src/train.py`, `src/infer.py`, corresponding `run.sh`.
- M87/M88 use current patched code safely; even without new flag，上传同一套代码可保持评估一致。

## Evaluation Policy

未来评估额度固定节奏:

- 西班牙早上：2-3 个。
- 西班牙晚上：1-2 个。

M85-M88 出分后，优先评估顺序按:

1. 是否超过 M84 dev 或接近 M75/M84 dev 且机制有 public-up 可能。
2. M85/M86 若 dev 不崩，优先级高于普通架构变体，因为它正面对应外部时间特征情报。
3. M87 即使 dev 小负，也保留一档观察，因为 M41 曾出现 dev↓/LB↑。
4. M88 只有 E5/E6 明显高于 E4 或稳定不掉时才消耗 quota。
