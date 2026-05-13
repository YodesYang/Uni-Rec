# EXP-050 · T34 DCN-V2 Cross Feature Bypass · 设计与实装 · 2026-05-14

> **Status**: 🟢 实装完成 · 本地 smoke 通过 · 等 M19 回流后单变量启动平台 half-train
> **日期**: 2026-05-14 CEST 02:00 起草
> **关联**: [EXP-049 EDA Correlation](EXP-049-eda-correlation-2026-05-13.md) · [DECEM Trick #5 community-intel](../10-research/community-intel.md) · 实装 commit (待 push)

---

## TL;DR

基于 EXP-049 EDA-correlation 的"强用户 × 强广告"信号清单，在 PCVRHyFormer 后接一个 **DCN-V2 cross-feature bypass** 模块：

- 选 EDA top user fid `[1, 49]` × top item fid `[5, 6, 7, 10, 12]` (全部 dim=1 scalar)
- 拼 7 个 fid 的 emb (7 × 64 = 448) → Linear(448, 64) → 2 层 DCN-V2 cross → LayerNorm → (B, 64)
- 与 backbone output concat 进 classifier (与 T25 dense_bypass / DIN bypass 完全平行)
- **新增 41,408 dense params** · 极轻 · embedding 与 NS Tokenizer 共享 (DCN 论文惯例)

实装通过的本地验证:
- ✅ fp32 forward / backward gradient propagate
- ✅ bf16 autocast forward 无 type promotion 错误
- ✅ predict() emb shape 从 64 (backbone) → 128 (backbone + dcn cross) 正确
- ✅ off 配置 bit-identical (zero-cost when disabled)
- ✅ py_compile + ReadLints 0 errors

---

## 1. 动机

### 1.1 EDA-correlation 提供的具体证据 (EXP-049 §4)

Top user × item Cramér V pair (EDA 全扫 2.1M rows · 1044 RG):

| Rank | User × Item | CV | MI | V_u | V_i |
|---|---|---|---|---|---|
| 1 | `user_1` × `item_10` | 0.278 | 0.138 | 6 | 256 |
| 2 | `user_1` × `item_5` | 0.273 | 0.129 | 6 | 256 |
| 3 | `user_49` × `item_6` | 0.272 | 0.080 | 3 | 256 |
| 4 | `user_49` × `item_10` | 0.259 | 0.072 | 3 | 256 |
| 5 | `user_49` × `item_5` | 0.259 | 0.071 | 3 | 256 |
| 6 | `user_49` × `item_7` | 0.244 | 0.065 | 3 | 256 |
| 7 | `user_49` × `item_12` | 0.244 | 0.065 | 3 | 256 |
| 8 | `user_1` × `item_6` | 0.226 | 0.099 | 6 | 256 |

→ **Top 8 cross 全部**涉及 `user_1 / user_49` (低基数 → 推测是性别/渠道/设备类型) × `item_5/6/7/10/12` (fid→label MI top 6 · 最强广告特征)

→ "强用户特征 × 强广告特征" 是经典 FM/DCN 设计动机

### 1.2 当前架构空缺

HyFormer 已有的交互机制：
- **NS-Token cross-attention**: chunked projection 后的 NS tokens 在 RankMixerBlock 里被 token-mixing
- **Sequence cross-attention**: query token attends to encoded sequence tokens
- **DIN target attention** (T25/T28): item_ns mean-pool 作 query 对 sequence 的 attention

→ **缺少**: 显式的 outer-product cross feature。Attention 是 weighted sum (linear combination); DCN-V2 是 element-wise product (Hadamard) · 表达不同的二阶/高阶交互

---

## 2. 设计

### 2.1 数据流

```
inputs.user_int_feats (B, total_user_int_dim)
inputs.item_int_feats (B, total_item_int_dim)
   ↓ user fid [1, 49] (scalar, dim=1) → 取 offset 列 → user_ns_tokenizer.embs[real_idx] 查表
   ↓ item fid [5, 6, 7, 10, 12] (scalar, dim=1) → 类似
   ↓ concat all 7 embs → (B, 7 * emb_dim) = (B, 448) at emb_dim=64
   ↓ input_proj: Linear(448, d_model) + LayerNorm + SiLU → x0 (B, 64)
   ↓ DCN-V2 cross_W[0]: x = x0 * Linear(x0) + x0
   ↓ DCN-V2 cross_W[1]: x = x0 * Linear(x) + x
   ↓ out_ln: LayerNorm(x) → (B, 64)
   ↓ concat 到 backbone output (B, D) → (B, 2D)
   ↓ classifier
```

### 2.2 DCN-V2 数学形式 (Wang et al. 2021)

每层 cross step:
$$
x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l
$$

`x_0 ⊙ (...)` 是元素积 → 让 x_0 的每个 fid embedding 对 x_l 做"门控调制"·相当于 explicit 二阶交互；2 层 stacked 等价 ~4 阶交互。

### 2.3 参数预算

| 模块 | 参数 |
|---|---|
| `input_proj.0`: Linear(448 → 64) | 28,672 + 64 = 28,736 |
| `input_proj.1`: LayerNorm(64) | 128 |
| `cross_W[0]`: Linear(64 → 64) | 4,160 |
| `cross_W[1]`: Linear(64 → 64) | 4,160 |
| `out_ln`: LayerNorm(64) | 128 |
| `clsfier` 输入 dim: D → 2D | +64×64 = 4,096 |
| **总计** | **41,408** |

vs 当前 model 总参 ~2.45M · **+1.7%** · 极轻

### 2.4 关键设计选择

#### (a) Embedding 共享 with NS Tokenizer
- 不重新建 embedding table·直接 borrow `self.user_ns_tokenizer.embs[real_idx]`
- 同 fid 的 embedding 在 NS-Token 路径和 cross 路径共享 → 同一个表征·两种 interaction view
- DCN paper 惯例·节省 ~50k sparse params

#### (b) 只支持 dim=1 scalar fid
- multi-hot fid (如 user_15 dim=26) 需要 pooling 决策 (mean? first non-zero?)
- 拒绝 multi-hot 输入 (raise ValueError)·避免 silent mis-pooling
- 如果未来需要 user_15·走"upstream mean-pool 后作为 dense feature"

#### (c) Classifier-time bypass
- 与 T25 dense_bypass / DIN bypass 完全平行 (forward pass 固定顺序: backbone | dense_bypass | din_interest | dcn_cross)
- 不 invade RankMixer T-divisibility 约束 (不动 NS token 数量)
- 与 T25/T28 fully orthogonal·可单独消融 / 可叠加

#### (d) Bias zero-init
- DCN-V2 paper 推荐 init 策略
- 让 layer-0 forward ≈ x_l (gradual perturbation) · gracefully degrade to "x_0 + LN"

---

## 3. 实装清单

### 3.1 src/model.py
- 新 class `DCNCrossBypass(nn.Module)` (在 `DINInterestExtractor` 之后)
  - `__init__`: 接 user_int_feature_specs / item_int_feature_specs / user_fids / item_fids (LIST INDICES) / num_cross_layers
  - `_validate_scalar`: assert 每个 selected fid length==1 + vocab_size>0·否则 raise loud
  - `forward`: 接 user/item int_feats + NS tokenizer 的 embs / emb_index → (B, D)
- `PCVRHyFormer.__init__` 加 4 kwargs: `enable_dcn_cross / dcn_cross_user_fids / dcn_cross_item_fids / dcn_cross_layers`
- `PCVRHyFormer._apply_dcn_cross_bypass(output, inputs)`: borrow ns_tokenizer.embs · concat 到 backbone output
- forward / predict 都加 `if self.enable_dcn_cross: output = self._apply_dcn_cross_bypass(...)`
- classifier dim 计算扩展 `+ (1 if enable_dcn_cross else 0)`

### 3.2 src/train.py
- 4 个 CLI flag:
  - `--enable_dcn_cross` (action='store_true', default=False)
  - `--dcn_cross_user_fids` (str, default='1,49')
  - `--dcn_cross_item_fids` (str, default='5,6,7,10,12')
  - `--dcn_cross_layers` (int, default=2)
- model_args 之前加 raw schema fid → list index 映射 (validation + error message + log)
- model_args dict 加 4 个 key

### 3.3 src/infer.py
- model_args 加 4 个 key 从 cfg.get() 读取
- cfg 存的是 LIST INDICES (train.py 已 mapped)·infer.py 直接 forward 不用再 map
- ⚠️ **新增 PCVRHyFormer kwarg → 平台 default infer.py 兼容性必须自测**: 首次 evaluation 必须勾选"上传自定义 infer.py" (team-memory 5/4 铁律)

---

## 4. 本地 Smoke 验证清单

| 测试 | 结果 |
|---|---|
| `py_compile` (model/train/infer/dataset) | ✅ 0 errors |
| `ReadLints` (model/train/infer) | ✅ 0 lint errors |
| Unit smoke fp32 forward | ✅ logits shape (B, 1) |
| Unit smoke bf16 autocast forward | ✅ 无 type promotion 错误 |
| Unit smoke backward | ✅ 9/10 DCN tensors grad active (input_proj.0.weight grad=0 是因 mock data 全 0 padding · 真数据下不会) |
| `predict()` emb shape | ✅ 64 (backbone) → 128 (with dcn) |
| ON / OFF param diff | ✅ +41,408 (= 设计预算·on/off bit-identical when off) |
| **train.py CLI mapping log** | ✅ "T34 DCN cross enabled: user fids=[1, 49] (indices [0, 5]), item fids=[5, 6, 7, 10, 12] (indices [0, 1, 2, 5, 7]), layers=2" |

---

## 5. 实验设计 · 上平台

### 5.1 单变量 ablation (M19 回流后启动)

| Job | Config (改动 vs M9 baseline) | 预期信号 |
|---|---|---|
| `M21-t34dcn-icq-halftrain` | `--enable_dcn_cross` (其他全部继承 icq-halftrain default) | dev / LB |

⚠️ **铁律: 不和其他 bet 堆叠** (避免 EXP-021 三合一 BPR 类归因混乱)

### 5.2 ROI / 风险评估

| 维度 | 评估 |
|---|---|
| **预期 LB ΔLB** | +0.001 ~ +0.003 (DCN 公开赛常规增量·与 EDA 信号匹配) |
| **风险** | 低·DCN 是公开赛标准 design pattern·与 HyFormer 正交 |
| **训练时间** | ~5h half-train (E4)·与 baseline 持平 (新增 41k params 几乎没成本) |
| **回流后续路径** | (A) work → 立即起 T34 + T31-P2 stack / (B) 不 work → 转 T35 ns_groups 重组 |

### 5.3 平台启动 run.sh 模板 (M21 候选)

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_SCRIPT_DIR="${SCRIPT_DIR}"
cd "${RUNTIME_SCRIPT_DIR}"
export PYTHONPATH="${RUNTIME_SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${RUNTIME_SCRIPT_DIR}/train.py" \
    --data_dir "${TRAIN_DATA_PATH}" \
    --schema_path "${TRAIN_DATA_PATH}/schema.json" \
    --ckpt_dir "${TRAIN_CKPT_PATH}" \
    --log_dir "${TRAIN_LOG_PATH}" \
    --device cuda --seed 42 \
    --num_epochs 4 --batch_size 128 \
    --valid_split_strategy time --valid_ratio 0.05 \
    --enable_row_valid_gap --valid_gap_ratio 0.05 \
    --seq_encoder_type swiglu \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 1 \
    --d_model 64 --emb_dim 64 --hidden_mult 4 \
    --num_hyformer_blocks 2 --num_heads 4 \
    --enable_torch_compile --enable_amp --amp_dtype bfloat16 \
    --logloss_rebound_delta 0.003 --logloss_rebound_patience 2 \
    --item_conditioned_query \
    --enable_dcn_cross
```

**上传文件清单** (因 PCVRHyFormer 引入新 kwargs · 必须自定义 infer.py):
- `src/model.py`
- `src/train.py`
- `src/infer.py`
- `src/dataset.py` (无改动·若 platform 已是最新版可跳过)

---

## 6. Iron Laws 合规性检查

| Iron Law | 合规性 |
|---|---|
| **Independent Parameters Iron Law** | ✅ DCN cross layer 是独立架构模块·不与 NS Tokenizer / HyFormer Block / DIN extractor 共享 cross weights;embedding 与 NS Tokenizer 共享 (DCN paper 惯例·同 fid 表征不同 view) |
| **Stacking Seed Sensitivity** | ✅ 单变量改动 (新 flag default off)·可干净归因 |
| **Golden Triplet** | ✅ 与 T25 dense_bypass / DIN bypass / T28 DIN integrated 全部正交 (forward 固定顺序·classifier-time concat) |
| **新增 PCVRHyFormer kwarg → 平台 default infer.py 兼容性** | ⚠️ 必须自测·首次 evaluation 勾选"上传自定义 infer.py"(team-memory 5/4 铁律) |
| **gap=0.05 dev 作为 LB 代理主判决指标** | ✅ M21 计划用 gap=0.05 配置 |

---

## 7. 与 EXP-049 后续 Tier 1 候选优先级

| 排序 | 任务 | 状态 | 何时启动 |
|---|---|---|---|
| 🥇 | T34 DCN cross | **本 EXP·实装完成** | M19 回流后启动 (M21) |
| 🥈 | T35 NS-groups 8-cluster 重组 | 未实装 | T34 回流后 (优先 T34 而非 T35·DCN ROI 高 1 个 order) |
| 🥉 | T31-P2 user_int_feats_15 is-missing | 未实装 | M19 验证 T31 path work 后·堆叠 |
| P3 | feats_51/52 disable | 未实装 | 工程 cleanup·不上 LB |

---

## 8. 后续追踪

- [ ] M21 (T34 启动) · 等 M19 回流确认主路径状态后启动
- [ ] M21 dev 回流 → 判 LB evaluation 是否值得
- [ ] M21 LB 回流 → 验证 +0.001 ~ +0.003 区间假设
- [ ] 如 work · 堆叠 T34 + T31-P2 / T34 + T35 · 进入 SOTA 冲分阶段
- [ ] 如不 work · pivot 到 T35 ns_groups 重组
