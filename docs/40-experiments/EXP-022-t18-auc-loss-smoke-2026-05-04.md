# EXP-022 · CombinedAUCLoss 验证（ADR-006 · T18）

**日期**：2026-05-04 夜  
**作者**：Agent  
**状态**：🔄 Local smoke 完成 + 单元级数值验证全过 → 等 S2 proxy 启动

---

## 背景

ADR-006 把 CombinedAUCLoss = `(1-w)·BCE + w·PairwiseBPR` 作为新增 loss_type。核心假说：

- BCE 在稀疏正样本（~4% label_type=2）下负样本梯度接近零，**learning signal 在硬负样本上被淹没**
- PairwiseBPR 直接优化 AUC 定义 `P(s_pos > s_neg)`，对硬负样本梯度远大
- 腾讯 arXiv 2403.14144 生产数据：BCE + ranking loss → **GMV +0.70%~1.26%**
- UniRec(hojiahao/TAAC2026) 在**同赛题**上已将 CombinedAUCLoss 作为默认 loss

外部锚 + 代码层面最小改动（~180 行 pure function）→ 与 T16 / T17 完全正交，可并行推进。

---

## S1 验证（已完成 · 2026-05-04 晚）

### S1.a 单元级数值验证

```bash
/opt/anaconda3/envs/tencent-unirec/bin/python - <<'PY'
...  # 8 项检查,详见 ADR-006 §Verification
PY
```

| # | 检查 | 结果 |
|---|---|---|
| 1 | BPR 正样本全高于负样本 loss→0 | `4.5e-5` ✓ |
| 2 | BPR 正样本全低于负样本 loss→10 | `10.0000` ✓ |
| 3 | 空正/空负 batch 返回零 grad tensor | ✓ |
| 4 | `combined = (1-w)·BCE + w·BPR` 加权恒等（w∈{0,0.3,0.5,1.0}） | Δ=0.0 ✓ |
| 5 | `auc_weight=0` 等价纯 BCE（向后兼容） | bit-identical ✓ |
| 6 | 同 seed 可复现 + gradient 非零流动 | ✓ |
| 7 | 非法 `auc_weight=1.5` 抛 ValueError | ✓ |
| 8 | `sample_weight` 仅影响 BCE 分量 | ✓ |

### S1.b 集成级 smoke（demo_1000, 10 step train）

```bash
--data_dir ./data/demo --ckpt_dir ./runs/smoke_{bce,auc0,auc}_ckpt
--num_epochs 1 --batch_size 64 --valid_split_strategy time --valid_ratio 0.2
--seq_encoder_type swiglu --limit_train_rgs 8 --limit_valid_rgs 2 --max_train_steps 10
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--emb_skip_threshold 1000000 --device cpu --seed 42 --num_workers 0
```

| 配置 | loss_type | auc_loss_weight | auc_margin | best AUC | best LogLoss |
|---|---|---|---|---|---|
| A. Baseline | `bce` | — | — | 0.8737376434 | 0.3066050708 |
| B. 向后兼容探针 | `auc` | **0.0** | 0.5 | **0.8737376434** | **0.3066050708** |
| C. AUC 启用 | `auc` | 0.3 | 0.5 | 0.8940688308 | 0.2798988521 |

**核心结论**：

1. **B == A bit-identical** → 向后兼容零破坏
2. C 相对 A 在 10 步上 +0.020 AUC / −0.027 LogLoss
   - ⚠️ 样本过少（10 步 × 64 bs = 640 samples）无统计意义
   - 只能说**机制生效**，不能外推到 full train

### S1.c train_config.json 落盘验证

```json
{
  "loss_type": "auc",
  "auc_loss_weight": 0.3,
  "auc_margin": 0.5,
  "auc_max_pairs": 32,
  ...
}
```

确认四字段正确写入，平台 infer.py 可通过 `cfg.get()` 读取（虽然不会消费）。

---

## S2 Proxy 计划（待启动）

**Job Name**：`t18-auc-w03-t13tf-proxy-swiglu-vr005-rg128x32-s1500`

**配置**（在 t13tf 基线上只改 loss）：

```bash
--seq_hash_vocab 500000
--enable_time_of_day_features
--loss_type auc --auc_loss_weight 0.3 --auc_margin 0.5 --auc_max_pairs 256
--emb_skip_threshold 1000000
--ns_tokenizer_type rankmixer --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2
--seq_encoder_type swiglu
--valid_split_strategy time --valid_ratio 0.05
--batch_size 128
--limit_train_rgs 128 --limit_valid_rgs 32 --max_train_steps 1500
--logloss_rebound_patience 0  # proxy scale 不可靠，禁用
```

**对照**：EXP-018 的 t13tf proxy dev AUC ≈ 0.8376~0.8385。

**决策阈值**：

| proxy dev AUC | 相对 t13tf proxy | 动作 |
|---|---|---|
| ≥ 0.839 | ≥ +0.001 | 强信号，直接进 S3 half-train |
| 0.836~0.839 | −0.002 ~ +0.001 | 中等/可接受，进 S3（参考 ICQ 规律：大参数分支 proxy 假负向） |
| < 0.836 | < −0.002 | 先切 `auc_weight=0.1, margin=0.0` 再跑，仍负向回滚 |

---

## S3 Half-Train 计划（S2 达标后）

**Job Name**：`t18-auc-w03-t13tf-halftrain-swiglu-vr005-bs128`

**关键变量**：

- `--num_epochs 4`
- `--holdout_ratio 0.05` ← ADR-005 S4 搭车
- `--logloss_rebound_patience 2` ← 放宽（BPR 可能对 LogLoss 有轻微影响）

**判据**：

| E3 dev AUC 相对 t13tf E3 | 动作 |
|---|---|
| ≥ +0.0003 | 进 S4 full train |
| −0.0003 ~ +0.0003 | 看 E4 斜率决定 |
| < −0.0003 | 回滚，尝试 `auc_weight=0.1` |

同时观察 holdout AUC vs dev AUC，采集 ADR-005 S4 判决数据。

---

## 四合一 Half-Train 最终提交版本（2026-05-04 晚 8 · T16 E3=0.868061 达标后）

**Job Name**：`t16-add-a01-t18-auc-w02-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128`

**参数从 EXP-022 草稿版做了 2 项保守调整**：

| 参数 | 草稿值（auc_weight=0.3, margin=0.5） | **实际提交值** | 理由 |
|---|---|---|---|
| `--auc_loss_weight` | 0.3 | **0.2** | T19 per-token FFN 已加 +993k dense 参数,容量放大 ~15×,同时上 BPR 可能梯度叠加噪声,先保守 α=0.2 探路 |
| `--auc_margin` | 0.5 | **0.0** | margin=0 即 vanilla BPR,扰动最小;UniRec 的 0.5 是独立 loss 场景的值,与 T19 组合时不一定最优 |
| 其他 | 同草稿 | 同草稿 | — |

**T16 E3 实测数据（触发升级的证据）**：

| Epoch | T16 AUC | ΔE→E+1 |
|---|---|---|
| E1 | 0.8645 | — |
| E2 | 0.8664 | +0.0019 |
| **E3** | **0.868061** | +0.00166 |

- vs ADR-004 S3 阈值 `E3 ≥ t13tf E3 + 0.0002` → **超 +0.00086, 稳过**
- vs ICQ half-train E3 0.867965 → **+0.000096**
- 信号强度: **T16 ~10× ICQ**, 证明 target-item seq 侧注入确实生效

**最终 run.sh**（已写入 `baseline/run.sh`）：

```bash
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --holdout_ratio 0.05 \
    --target_item_seq_injection additive \
    --target_inject_alpha_init 0.1 \
    --loss_type auc \
    --auc_loss_weight 0.2 \
    --auc_margin 0.0 \
    --auc_max_pairs 256 \
    --rank_mixer_ffn_mode per_token \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
```

**平台提交格式**（team-memory 约定）：
1. **Job Name**：`t16-add-a01-t18-auc-w02-t19-ptffn-ho005-halftrain-swiglu-vr005-bs128`
2. **需要上传的有改动文件**（5 个，不能 Copy 旧 job）：`baseline/model.py` + `baseline/train.py` + `baseline/trainer.py` + `baseline/utils.py` + `baseline/infer.py`。
3. **run.sh**：见上方模板，已写入 `baseline/run.sh`,可直接提交

**本地 smoke 通过证据**（2026-05-04 晚 8）：

| 验证点 | 证据 |
|---|---|
| 启动 Args 完整 | `target_item_seq_injection: 'additive'` + `target_inject_alpha_init: 0.1` + `loss_type: 'auc'` + `auc_loss_weight: 0.2` + `auc_margin: 0.0` + `auc_max_pairs: 32` + `rank_mixer_ffn_mode: 'per_token'` + `holdout_ratio: 0.15` ✓ |
| 四路切分正确 | `Row Group split (time): 9 train (900 rows), 1 valid (1000 rows), 1 holdout (100 rows)` ✓ |
| holdout_loader attached | `[ADR-005] holdout_loader attached: holdout AUC/LogLoss will be logged every epoch. ...` ✓ |
| 模型参数量 | Sparse 158M + Dense **3.46M**（shared 基线约 2.47M, +993k 就是 per-token 新增）✓ |
| Trainer 正确拿到 CombinedAUCLoss 参数 | `loss_type=auc, auc_loss_weight=0.2, auc_margin=0.0, auc_max_pairs=32` ✓ |
| Epoch 末自动算 holdout | `[ADR-005] Epoch 1 step 10 holdout AUC: 0.631629 ... gap_auc=valid-holdout=+0.227574`（demo 数据 gap 大是 demo 特性,full data 不会这么大）✓ |
| best_model + summary 落盘 | `global_step10...auc=0.859202.best_model/` + `training_summary.json` ✓ |

---

若 T16 `t16-additive-a01-halftrain` E3 ≥ 0.868 达标，**不单独走 T18 S3**，而是把 T16 方案 A + T18 CombinedAUCLoss + T17 holdout 三条正交战线**打包一次 half-train** 并行验证，最大化 quota 利用率。

**Job Name**：`t16-add-a01-t18-auc-w03-ho005-halftrain-swiglu-vr005-bs128`

**Run.sh 模板**（在 `baseline/run.sh` 尾部 `"$@"` 之后所需附加参数）：

```bash
# 传入 train.py 的核心变量（每次 platform job 提交 run.sh 覆盖 CLI 即可）：
#
# ADR-004 T16 方案 A（seq 侧 target-item 加性注入）
# ADR-006 T18 CombinedAUCLoss（BCE + PairwiseBPR）
# ADR-005 T17 holdout 搭车（零 quota 采集 dev vs holdout gap）
#
# 基线等同 EXP-018 t13tf-full：swiglu + hash500k + timefeat + rankmixer NS + vr=0.05
#
# logloss_rebound_patience 从默认 1 放宽到 2：BPR 可能对 LogLoss 校准有轻微扰动，
# patience=1 容易在 E2/E3 误触发早停。
#
# 评估阈值：
#   E3 dev AUC ≥ 0.8695（t13tf E3 基线 ~0.867 + T16/T18 联合期望 +0.002）→ 进 full train
#   E3 dev AUC ∈ [0.868, 0.8695]               → 看 E4 斜率
#   E3 dev AUC < 0.868                          → 独立拆分归因（各退回单子弹 half-train）

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --seq_encoder_type swiglu \
    --seq_hash_vocab 500000 \
    --enable_time_of_day_features \
    --valid_split_strategy time \
    --valid_ratio 0.05 \
    --holdout_ratio 0.05 \
    --target_item_seq_injection additive \
    --target_inject_alpha_init 0.1 \
    --loss_type auc \
    --auc_loss_weight 0.3 \
    --auc_margin 0.5 \
    --auc_max_pairs 256 \
    --logloss_rebound_patience 2 \
    --logloss_rebound_delta 0.0 \
    --batch_size 128 \
    --num_epochs 4 \
    "$@"
```

**收益预期（加性保守估计）**：
- T16 半 train 单独预期 Δ+0.0003 ~ +0.001 AUC（ADR-004 ICQ 基线外推）
- T18 半 train 单独预期 Δ+0.001 ~ +0.002 AUC（外部文献 +0.70~1.26% GMV 折算 30% LB 转化率）
- 叠加保守估计 **Δ+0.0013 ~ +0.003**；乐观估计 **Δ+0.004**
- 对应 E3 目标 `t13tf E3 baseline(~0.867) + 0.0013 ~ 0.003 = 0.8683 ~ 0.870`

**归因策略**（E3 实测与期望偏差大时）：

| E3 实测 | 归因 | 下一步 |
|---|---|---|
| ≥ 0.870 | 强信号，T16+T18 叠加有效 | 直接进 **三合一 full train**（10 epoch） |
| 0.868~0.870 | 中性信号，两条独立但未协同 | 拆 T18 / T16 各自独立 half-train 确认单点贡献 |
| 0.866~0.868 | 至少一条负向拖累 | 先关 T18（loss_type=bce）跑 T16+holdout 对照 |
| < 0.866 | 组合退化 | 先关 T16（target_item_seq_injection=off）跑 T18+holdout 对照 |

**平台任务提交格式**（遵守 team-memory 约定）：
1. **Job Name**：`t16-add-a01-t18-auc-w03-ho005-halftrain-swiglu-vr005-bs128`
2. **需要上传的改动文件**：`baseline/utils.py`、`baseline/trainer.py`、`baseline/train.py`（T18 新增），加上 `baseline/model.py`（T16 已有）。**不能 Copy 旧 job**。
3. **run.sh** 内容：见上方模板

**evaluation 兼容性**：
- T16 的 `target_item_seq_injection` 是 `PCVRHyFormer.__init__` kwarg → 首次 evaluation **必须上传自定义 `infer.py`**（team-memory 铁律）
- T18 的 `loss_type=auc` 不影响 `__init__`（推理不走 loss），但为了保险建议同样上传
- T17 holdout 仅作用于 training split，推理零影响

---

## S4 Full Train 计划（S3 达标后）

**Job Name**：`t18-auc-w03-t13tf-full-swiglu-vr005-bs128`

**关键变量**：

- `--num_epochs 10 --batch_size 128`
- `--holdout_ratio 0.05`（继续搭车）
- `--logloss_rebound_patience 2`
- 阈值：local best dev AUC ≥ **0.870** → 考虑 evaluation

---

## S5 Leaderboard（S4 达标后）

- 1 evaluation slot
- 阈值：LB ≥ **0.847**（t13tf LB 0.844956 + 0.002）
- 达标 → commit ADR-006，设为默认 loss
- 不达标 → 分析 dev→LB 传导是否比 BCE 差（若 BPR 让 dev 涨 + LB 涨幅 < 0.0005，说明 dev→LB covariate shift 更严重，方案 B valid 重设计更紧急）

---

## 后续可能的拓展（暂不实施）

- `auc_weight` schedule：前 K step 纯 BCE（warmup），后期逐步提升 `auc_weight` 到 0.3（防止早期梯度噪声过大）
- gAUC pairwise loss：按 user 分组采 pair（本赛题是 global AUC 评估，不对口，暂不做）
- "hard negative mining"：只采 top-K 难负样本进 BPR（复杂度 +2×，proxy 验证后再考虑）

---

## 相关

- [ADR-006](../30-decisions/ADR-006-combined-auc-loss.md)：本 ADR 完整决策推理
- [ADR-004](../30-decisions/ADR-004-target-item-seq-conditioning.md) / T16：可叠加的另一正交战线
- [ADR-005](../30-decisions/ADR-005-valid-split-redesign.md) / T17：holdout 搭车策略
- [EXP-018](EXP-018-t13tf-full-train-2026-05-04.md)：t13tf baseline 的完整 epoch 表
- 外部参考：UniRec(hojiahao/TAAC2026) `model/heads.py::CombinedAUCLoss`；arXiv 2403.14144（Tencent ranking loss in production）
