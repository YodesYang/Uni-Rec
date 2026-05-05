# EXP-023 · RankMixer Per-token FFN 验证（ADR-007 · T19）

**日期**：2026-05-04 夜  
**作者**：Agent  
**状态**：🔄 Local smoke 完成 + 单元级数值验证全过 → 等 S2 proxy 启动

---

## 背景

`docs/00-context/model-py-section-map.md` §RankMixer 已明确记录本仓库 `RankMixerBlock` 与 RankMixer 论文（arXiv 2507.15551）的核心差异：

- **论文原版**：Per-token FFN —— T 个 token 各有独立 FFN 参数（对抗 dominant feature 淹没长尾）
- **本仓库实现**：Shared FFN —— T 个 token 共用一套 `Linear(D, 4D) → Linear(4D, D)`

飞机伴读地图当时预测："如果论文里 Per-token FFN 是核心，而仓库用的 shared，可能你做一个真正的 Per-token FFN 版本就是 **+0.002~0.005 AUC 的子弹**"。

T19 / ADR-007 把这一条子弹落地为可选开关 `--rank_mixer_ffn_mode {shared, per_token}`，默认 `shared` 保持向后兼容。

---

## S1 验证（已完成 · 2026-05-04 晚）

### S1.a 单元级数值验证

| # | 检查 | 结果 |
|---|---|---|
| 1 | Shared 参数量（D=64, T=16, hm=4） | 33,344 ✓ |
| 2 | Per-token 参数量 | **529,664 = 15.88× shared** ≈ T ✓ |
| 3 | forward shape 两路径 | (2,16,64) → (2,16,64) ✓ |
| 4 | per-token 独立性（扰动 token 3 fc1_weight ×1000） | 只 token 3 变 188，其他严格 0 ✓ |
| 5 | backward 两路径 grad 非零 | ✓ |
| 6 | 非法 ffn_mode 抛 ValueError | ✓ |
| 7 | **Per-token token 5 输出 == 独立 nn.Linear（weight copy 对齐）** | **max\|Δ\| = 0.00e+00** bit-identical ✓ |
| 8 | `mode='ffn_only' + per_token` 兼容 | ✓ |
| 9 | `mode='none'` 两种 ffn_mode 都 0 params | ✓ |
| 10 | `MultiSeqHyFormerBlock` 透传 + 参数量增量 **+496k dense/block** | ✓ |

### S1.b 集成级 smoke（demo_1000, 10 step train）

```bash
--data_dir ./data/demo --ckpt_dir ./runs/smoke_ffn_{sh,pt}_ckpt
--num_epochs 1 --batch_size 64 --valid_split_strategy time --valid_ratio 0.2
--seq_encoder_type swiglu --limit_train_rgs 8 --limit_valid_rgs 2 --max_train_steps 10
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--emb_skip_threshold 1000000 --device cpu --seed 42 --num_workers 0
```

| 配置 | best AUC | best LogLoss | train loss @ step 9 |
|---|---|---|---|
| `--rank_mixer_ffn_mode shared`（默认） | **0.8737376434** | 0.3066 | 0.3740 |
| `--rank_mixer_ffn_mode per_token` | 0.8758085184 | 0.3097 | **0.3147** |

**核心观察**：

1. **向后兼容**：`shared` 默认路径与 T18 `loss_type=bce` backward-compat case 的 AUC **完全一致**（0.8737376434），证明"不传新 flag 行为不变"的红线严格满足
2. **容量生效**：`per_token` 路径 train loss 显著降低（0.31 vs 0.37），10 步 AUC +0.0021
3. **过拟合迹象**：`per_token` LogLoss +0.0030（10 步尺度微小，符合容量增加预期；half-train 尺度需要 LogLoss rebound guard patience=2 保护）

### S1.c train_config.json / best_model 落盘验证

```json
{
  "rank_mixer_mode": "full",
  "rank_mixer_ffn_mode": "per_token",
  ...
}
```

`training_summary.json` 正确记录；`best_model/` 自动含 `model.pt + train_config.json + schema.json` 三件套。

---

## S2 Proxy 计划（待启动）

**Job Name**：`t19-ptffn-t13tf-proxy-swiglu-vr005-rg128x32-s1500`

**配置**（在 t13tf 基线上只改 ffn_mode）：

```bash
--rank_mixer_ffn_mode per_token       # T19 核心切换
--seq_hash_vocab 500000
--enable_time_of_day_features
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--batch_size 128
--limit_train_rgs 128 --limit_valid_rgs 32 --max_train_steps 1500
--logloss_rebound_patience 0   # proxy scale 不可靠，禁用
```

**对照**：EXP-018 t13tf proxy dev AUC ≈ 0.8376~0.8385。

**决策阈值**（参考 team-memory 铁律："大参数 dense 分支 proxy 假负向"，ICQ/hash500k 先例）：

| proxy dev AUC | 相对 t13tf proxy | 动作 |
|---|---|---|
| ≥ 0.840 | ≥ +0.001 | 强信号，直接进 S3 |
| 0.836~0.840 | −0.002 ~ +0.001 | 中性，进 S3 |
| 0.833~0.836 | −0.005 ~ −0.002 | 大参数假负向规律，**仍进 S3** |
| < 0.833 | < −0.005 | 真崩盘，回滚或降 hidden_mult |

---

## S3 Half-Train 计划（推荐三 / 四合一搭车）

若 T19 独立推进：**Job Name** `t19-ptffn-t13tf-halftrain-swiglu-vr005-bs128`

但更推荐与 T16 / T18 / T17 holdout **搭车**：

**四合一 Job Name**：`t16-add-a01-t18-auc-w03-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128`

**Run.sh 关键参数**：
```bash
# T16 · target injection
--target_item_seq_injection additive --target_inject_alpha_init 0.1
# T18 · CombinedAUCLoss
--loss_type auc --auc_loss_weight 0.3 --auc_margin 0.5 --auc_max_pairs 256
# T19 · per-token FFN
--rank_mixer_ffn_mode per_token
# T17 · holdout 搭车
--holdout_ratio 0.05
# 保护
--logloss_rebound_patience 2
# baseline 继承
--seq_hash_vocab 500000 --enable_time_of_day_features
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu --valid_split_strategy time --valid_ratio 0.05
--batch_size 128 --num_epochs 4
```

**收益预期（加性保守估计）**：
- T16 +0.0003 ~ +0.001 AUC
- T18 +0.001 ~ +0.002 AUC
- T19 +0.0005 ~ +0.002 AUC
- 合计保守 **+0.0018 ~ +0.005**；乐观 **+0.007**
- 对应 E3 目标：`t13tf E3(~0.867) + 0.002 ~ 0.005 = 0.869 ~ 0.872`

**归因策略**（四合一 E3 实测偏差大时）：

| E3 实测 | 动作 |
|---|---|
| ≥ 0.872 | 强信号，四条正向叠加 → 直接进 full train |
| 0.870~0.872 | 中性叠加 → 进 full train |
| 0.869~0.870 | 至少一条子弹未协同 → full train + 观察 learned `α` / loss history / per-token diff token diversity |
| 0.867~0.869 | 两条子弹拖累 → 先关 T18（容量已大，BPR noise 可能干扰）对照 |
| < 0.867 | 组合退化 → 只留 T18（lightest delta）独立 half-train 确认 |

---

## S4 Full Train 计划（S3 达标后）

**Job Name**：`t16t18t19-full-swiglu-vr005-bs128`

**关键变量**：
- `--num_epochs 10 --batch_size 128`
- `--holdout_ratio 0.05`（ADR-005 搭车）
- `--logloss_rebound_patience 2`
- 阈值：best dev AUC ≥ **0.871** → 考虑 evaluation

---

## S5 Leaderboard（S4 达标后）

- 1 evaluation slot
- 阈值：LB ≥ **0.848**（t13tf LB 0.844956 + 0.003）
- 达标 → commit ADR-007，per-token FFN + 三合一组合设为新默认
- 不达标 → 分析转化率（若 local 涨 +0.005 而 LB 只涨 +0.0005，说明 valid→LB gap 吞掉了容量增加带来的优势，需回看 ADR-005 T17 的方案 C 留空窗）

---

## 风险点

- **过拟合**：per-token 容量 +15× 对 1.9M 数据可能过大。护栏：
  - AUC-select + LogLoss rebound guard patience=2
  - 若 full train E5+ LogLoss 反弹严重，降 `hidden_mult 4 → 3`
- **平台显存**：参数量 +~993k dense（相对现有 ~5M dense 是 +20%）。batch 128 下预计仍可跑，但若 OOM 降到 64
- **evaluation ckpt 兼容**：per-token ckpt **必须用自定义 infer.py**（state_dict 结构不兼容平台默认 shared path）

---

## 相关

---

## 四合一 Half-Train 最终提交版本（2026-05-04 晚 8 · auc_weight 从 0.3 → 0.2）

**Job Name**：`t16-add-a01-t18-auc-w02-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128`

**参数调整理由**：T19 per-token FFN 引入 +993k dense 参数（容量 ~15×）+ T18 BPR 梯度叠加可能放大噪声，首次组合保守先用 `auc_weight=0.2 + auc_margin=0.0`（vanilla BPR，最小扰动）。乐观叠加阈值: **E3 ≥ 0.872**。

**详见** [`EXP-022 §四合一 Half-Train 最终提交版本`](EXP-022-t18-auc-loss-smoke-2026-05-04.md#四合一-half-train-最终提交版本2026-05-04-晚-8--t16-e3086806-达标后)（含完整 run.sh + 本地 smoke 证据）。

## 相关

- [ADR-007](../30-decisions/ADR-007-rank-mixer-per-token-ffn.md)：本 ADR 完整决策推理
- [ADR-004](../30-decisions/ADR-004-target-item-seq-conditioning.md) / T16：可叠加
- [ADR-005](../30-decisions/ADR-005-valid-split-redesign.md) / T17：holdout 搭车
- [ADR-006](../30-decisions/ADR-006-combined-auc-loss.md) / T18：可叠加
- [EXP-018](EXP-018-t13tf-full-train-2026-05-04.md)：t13tf baseline 的完整 epoch 表
- 外部参考：RankMixer arXiv 2507.15551 §3.2 "Per-token FFN for Subspace Heterogeneity"
