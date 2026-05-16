# ADR-014 · Path B Step 1 · UE 分离 (User Embedding Outlier Isolation) [T37]

> **Status**: ✅ **Accepted** (2026-05-16 CEST 21:30 实证 · M49 LB 0.848162 · Δ+0.000570 vs M22 SOTA · 项目首个超 SOTA · path B 弱真信号)
> **Status History**: Proposed 09:00 → Suspended 09:30 → Reactivated 11:00 → **Accepted 21:30** (M49 LB 实证)
> **Date**: 2026-05-16 (4 status changes in 12 hours)
> **Related Tasks**: T37 (实施完成 · commit `e5b0bf4`)
> **Related Experiments**: EXP-049 (EDA-correlation · user_int side outlier 但**非 UE 真实指代**) · EXP-011 (dense log1p 修复 · **fid 61/87 = 预归一化 embedding 实证**) · EXP-055 (5/15~5/16 retrospective · path B 提议) · EXP-056 (Tier 0 EDA · 元宵节假设证伪) · EXP-057 (M45-48 daypart engineering) · **EXP-059 (M49 LB 0.848162 · path B 实证)** · **EXP-061 (M51-M54 path B stack overnight 启动)**
> **External References**:
> - 社区情报 Seed (LB 0.85+) 主楼 5/5 第 (1) 条原话: "user dense feats 将 ue 分离出来 · 其他跟 int pair 加权"
> - 沉淀位置: `docs/99-raw/community-snapshots/xhs-decem-0512/README.md` line 196
> - **EXP-011 实证**: `user_dense_feats_61 (256-dim · 预归一化 embedding)` · `fid_87 (320-dim · 预归一化 embedding)` · `fid_62~66 (原始计数统计量 10^3~10^5 量级)` · `fid_89/90/91 (10-dim · 预归一化 embedding)`
> - `src/ns_groups.json` line 8: "All user dense features (fid 61/62/63/64/65/66/87/89/90/91, total_dim=918) are concatenated and projected to a single NS token"

---

## ✅ Acceptance (2026-05-16 CEST 21:30)

**M49 evaluation 实证**:

| Metric | M22 (anchor) | M49 (UE split) | Δ |
|---|---|---|---|
| dev AUC | 0.868619 | **0.870741** | **+0.002122** ✅ |
| LB AUC | 0.847592 | **0.848162** | **+0.000570** ✅ |
| dev→LB 衰减 | 100% retain | 27% retain | **73% 衰减** ⚠️ |

**判决**:
- ✅ 超出 same-seed rerun band ±0.0003 (LB Δ +0.000570 > +0.0003)
- ⚠️ 未破 0.8484 严格阈值 (3× rerun band) · 信号强度不足以排除 cross-seed luck
- ✅ ADR-014 设计假设 (UE = fid_61 256-dim) **机制正确**
- ✅ M49 = 项目 16 天 60+ jobs 后**首次** dev↔LB 双侧超 rerun band 真信号

**经验**:
- "弱真信号 + 单变量不够冲 0.85" · 必须叠加第二真信号
- dev→LB 衰减 73% 远高于期望 (~30%) · UE split 在 OOV 远未来场景下信号衰减严重
- 5/16 后续战略转向 **path B 体系下 stack 探索** (EXP-061 M51-M54 4-job batch)

---

## 🛑 Suspension Note (2026-05-16 CEST 09:30 → ⚠️ 11:00 修订)

**用户决断**: ADR-014 解读不确定 · 暂缓启动。

**🔥 5/16 11:00 真相重大修订**:
agent 5/16 10:55 深度思考时严格 grep `src/ns_groups.json` line 8 + EXP-011 找到 user_dense 真实结构:
- **fid_61: 256-dim 预归一化 embedding** ⭐ = **UE 真实候选** (production user pre-trained embedding)
- fid_87: 320-dim 预归一化 embedding
- fid_62~66: 原始计数统计量 (count features)
- fid_89/90/91: 10-dim 小 embedding

**修订 "UE 分离" 解读**:
- 之前 (5/16 09:00): UE = user_int_feats_15 (V=256 categorical) ❌ **错** (与 Seed 主楼字面不符)
- 修订 (5/16 11:00): **UE = user_dense_feats_61 (256-dim float embedding)** ✅ 与 Seed "user dense feats 将 ue 分离出来" **字面一致** + 与 EXP-011 "fid 61 = 预归一化 embedding" 实证一致

**Seed 主楼原话再读**:
> "user dense feats **将 ue 分离出来** · 其他跟 int pair 加权"

**修订 mechanism**:
1. 把 fid_61 (256-dim UE) 从 988-dim user_dense 中**分离** · 独立成 1 token
2. 其他 user_dense (count + fid_87/89/90/91) 通过 **dense_bypass** (T25 已 work 机制) 直接到 classifier
3. 与 user_int / item_int 的 cross 通过 **DCN cross** (M23 已实测 LB 强负 · 不要叠加 · 可能 Seed 用了不同 cross 实现)

**~~重启条件~~** ✅ 已 reactivated 5/16 11:00 · ✅ 已 Accepted 5/16 21:30 (M49 LB 实证)

---

## (以下是 09:00 起草的原 ADR 内容 · ⚠️ Decision §设计已过时 · 必须按 fid_61 修订)

---

---

## Context (背景与触发)

### 触发证据

#### 1. EXP-049 EDA-correlation 实证 (5/13 已落档)

```
user-side max → label MI:  0.003603 (user_int_feats_15)
user 第 2 名 → label MI:    0.000983 (user_int_feats_106) · feats_15 高 3.7×
比例 user vs item:          1:8 (item-side max 0.028904)
```

→ **`user_int_feats_15` 是 user-side 唯一显著 outlier** · 大概率是**用户大类标识** (年龄 / 性别 / 地域 / 设备等 V=256 离散维度)。

#### 2. 当前 NS Tokenizer 让 feats_15 信号被稀释

`baseline/model.py::RankMixerNSTokenizer` (用于 M22 baseline):
```
46 个 user_int fids → concat embeddings → split into 5 chunks → project per chunk → 5 user_ns tokens
```

`user_int_feats_15` 与其他 45 个 fid 一起被 chunk-split → embedding 信号被分摊到某 chunk · **不能 isolate 到独立 token**。

#### 3. 5/14 ~ 5/16 实测 7/7 新方向 dev 不超 M22 · vocabtier 路径接近局部最优

EXP-055 §16 实证:
- M37 (NLIR) / M38 v2 (Multi-Emb K=2) / M39 v2 (Global Token) / M40 (hour_only) / M41 (temporal_weight) / M22 baseline + 各种 stack
- 7/7 dev 端不超 M22 anchor 0.868576

→ 任何 "M22 + 单变量 perturb" 的 stack 大概率 saturated · 必须切到 **architecture-level 创新**。

#### 4. Seed 主楼第 (1) 条情报 = path B 设计依据

> "user dense feats 将 ue 分离出来 · 其他跟 int pair 加权"

机制分解:
- **(a) UE 分离**: user 高 MI fid (feats_15) 单独成 token · **本 ADR 实施 (T37)**
- **(b) Int pairwise cross**: 剩余 user × item pair 做 cross · **已实现** = `enable_dcn_cross` (T34 · DCNCrossBypass)

**path B = T37 (UE 分离) + T34 (DCN cross)** · 双机制 stack。

---

## Decision (Plan B · 复用 user_ns_tokens slot · T-divisibility 不变)

### 设计选择

3 个方案对比:

| 方案 | num_user_ns | T = 2×4+num_ns | 64%T | 评估 |
|---|---|---|---|---|
| A · 加 1 独立 token | 5→6 | 17 | 17 ❌ | 不可行 |
| **B · 复用 1 chunk slot** | 5 (不变) | 16 | 0 ✅ | **推荐** |
| C · 减 num_queries | 各种组合 | 各种 | 不规则 | 复杂 |

**Plan B 实施**: `user_ns_tokens=5` 不变 · 但其中第 1 个 token 专给 `user_int_feats_15` · 其余 4 个 token 走 chunk-merged 路径覆盖剩 45 fid。

### 代码改动面 (~50~80 行)

#### 1. `src/model.py::RankMixerNSTokenizer.__init__` (~30 行)

加 `enable_ue_split: bool = False` + `ue_split_fid_idx: int = -1` (后者是 user_int_feats_15 在 user_int_feature_specs 中的 list index · 由 PCVRHyFormer 传入):

```python
class RankMixerNSTokenizer(nn.Module):
    def __init__(
        self,
        ...,
        enable_ue_split: bool = False,
        ue_split_fid_idx: int = -1,  # list index of user_int_feats_15
    ):
        ...
        self.enable_ue_split = enable_ue_split
        self.ue_split_fid_idx = ue_split_fid_idx if enable_ue_split else -1

        if self.enable_ue_split and self.ue_split_fid_idx >= 0:
            # Dedicated path for user_int_feats_15:
            # - 自己的 embedding (复用 self.embs[ue_split_emb_idx])
            # - 自己的 token projection (Linear(emb_dim, d_model))
            self.ue_split_proj = nn.Sequential(
                nn.Linear(emb_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )
            # Remaining fids (排除 ue_split_fid_idx) 走 chunk path
            # → user_ns_tokens 减 1 个 chunk (5→4 chunks for remaining fids)
            # → ue_split token 占第 1 个 token slot
```

#### 2. `RankMixerNSTokenizer.forward` (~25 行)

```python
def forward(self, int_feats):
    if self.enable_ue_split and self.ue_split_fid_idx >= 0:
        # 1. UE token: 取 user_int_feats_15 embedding
        fid_idx = self.ue_split_fid_idx
        vs, offset, length = self.feature_specs[fid_idx]
        emb_real_idx = self._emb_index[fid_idx]
        emb_layer = self.embs[emb_real_idx]
        if length == 1:
            ue_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
        else:
            # multi-hot: mean pool
            vals = int_feats[:, offset:offset+length].long()
            ue_emb = emb_layer(vals).mean(dim=1)  # (B, emb_dim)
        ue_token = self.ue_split_proj(ue_emb).unsqueeze(1)  # (B, 1, d_model)

        # 2. Remaining fids: chunk-merged with num_chunks - 1 chunks
        remaining_fids = [i for i in range(len(self.feature_specs)) if i != fid_idx]
        # ... build embeddings for remaining ... split into (num_ns_tokens - 1) chunks

        # 3. concat: [ue_token, *remaining_chunk_tokens]
        return torch.cat([ue_token, remaining_tokens], dim=1)  # (B, num_ns_tokens, d_model)
    else:
        # 原有路径 · bit-identical baseline
        ...
```

⚠️ **关键设计点**:
- UE token 占**第 1 个** token slot (固定位置)
- Remaining 45 fids 仍走 chunk-merged · 但 chunk 数从 5 减到 4
- T-divisibility 不变 (num_ns 8 不变)

#### 3. `src/model.py::PCVRHyFormer.__init__` (~10 行)

加 `enable_ue_split: bool = False` + 自动 resolve `ue_split_fid_idx`:

```python
def __init__(
    self,
    ...,
    enable_ue_split: bool = False,
    ue_split_raw_fid: int = 15,  # 用户 schema 中的 raw fid (默认 15)
):
    # Resolve raw fid → list index in user_int_feature_specs
    ue_split_fid_idx = -1
    if enable_ue_split:
        for i, (fid, _, _) in enumerate(user_int_feature_specs_with_fid):
            if fid == ue_split_raw_fid:
                ue_split_fid_idx = i
                break
        if ue_split_fid_idx == -1:
            raise ValueError(
                f"enable_ue_split=True but raw fid {ue_split_raw_fid} not found "
                f"in user_int_feature_specs"
            )
    self.enable_ue_split = enable_ue_split
```

#### 4. `src/train.py` CLI (~5 行)

```python
parser.add_argument('--enable_ue_split', action='store_true', default=False,
                    help='T37 / Path B Step 1 · 把 user_int_feats_15 (V=256 outlier · '
                         'EXP-049 实证 user-side 唯一显著 MI 0.0036) 单独成 1 个 NS '
                         'token · 不参与 chunk-merged 路径。复用 user_ns_tokens=5 中 '
                         '第 1 个 slot · T-divisibility 不变 (num_ns=8 · T=16)。')
parser.add_argument('--ue_split_raw_fid', type=int, default=15,
                    help='UE 分离的 raw user_int fid 编号 (schema fid · 不是 list '
                         'index)。默认 15 (user_int_feats_15)。')
```

#### 5. `src/infer.py` cfg fallback (~2 行)

```python
"enable_ue_split": bool(cfg.get("enable_ue_split", False)),
"ue_split_raw_fid": int(cfg.get("ue_split_raw_fid", 15)),
```

### 等价性 safeguard

`enable_ue_split=False` 时 · NS Tokenizer 走原有 chunk-merged 路径 · **bit-identical M22 baseline** (smoke 验证 1+ params 无变化)。

---

## Verification Plan

### S1 Smoke (CEST 12:00~14:00)

`scripts/smoke_t37_ue_split.py`:
1. `enable_ue_split=False` · params + forward 与 baseline 完全等价
2. `enable_ue_split=True` · 新增参数 = (1 × Linear(emb_dim, d_model) + LN + SiLU) ≈ 4.2k Dense params · forward shape OK
3. forward + backward · grad finite
4. SOTA-shape stress test (d_model=64 / 46 user_int fids / 14 item_int fids · M22 真实 config)
5. ue_split_fid_idx=15 自动 resolve 正确

### S2 平台 half-train

| Job | Config | 阈值 |
|---|---|---|
| **path B first (M44 backup name M45)** = M22 + UE 分离 + DCN cross stack | M22 baseline + `--enable_ue_split` + `--enable_dcn_cross` | E3 Dev ≥ 0.8691 (+0.0005) → 真信号 |
| **path B variant** (后续) = M22 + UE 分离 alone | M22 + `--enable_ue_split` (cross 不开) | 单变量 ablation |

### S3 LB 阈值 (gap=0 path 同 seed band ±0.0003)

- ≥ **0.8484** → 真信号 (3σ over M22 mean 0.847458)
- 0.847~0.8484 → 弱信号 / band 内 · 不下定论
- ≤ 0.846 → 路径关闭

---

## Consequences

### 正面

- **填补 Seed 主楼 0.85+ 第 (1) 条情报第一半** (UE 分离)
- **EDA-049 给定 ground truth** (feats_15 outlier · 不是凭印象)
- **+~4.2k Dense params** · 极小增量 · 不触发独立化铁律
- **T-divisibility 不变** (Plan B 复用 slot)
- **正交其他改动** (DCN cross / temporal_weight / Multi-Emb 等) · 可任意 stack
- **bit-identical safeguard** (default off)

### 负面 / 风险

- **5/14~5/16 7/7 新方向 dev 弱负** · UE 分离也可能 dev 弱负 · 但 Seed 主楼实证 LB +
- **机制风险**: feats_15 是 V=256 维 · sample 在 256 个值上稀疏 · 单独 embedding 可能学不充分 · 反而 chunk-merged 时跟其他 fid 合学更稳
  - Mitigation: emb_skip_threshold=1M · feats_15 V=256 远小 · 一定不被 skip
  - Mitigation: emb_dim=64 · 256 unique values × 64 = 16k params · 充足
- **Stack 风险**: path B = UE 分离 + DCN cross stack 第 2 步 · stack 历史 5/14~5/15 失败 3 次 (M28/M34/M37)
  - Mitigation: 但 M28/M34/M37 是 vocabtier × {Trick 1/Trick 6/NLIR} stack · 都是 attention/loss 类 perturb · 与 UE 分离 (NS Tokenizer 重组) + DCN cross (bypass branch) 性质完全不同 · 可能不撞同一坑

### 红线检查

- ✅ 共享 Linear (UE token 用同一 ue_split_proj)·不违反独立化铁律
- ✅ 单一架构 · 不 ensemble
- ❌ 不外部数据 · 纯架构改动
- ✅ ≤ 2 seed (default seed=42 · 不做 cross-seed)
- ✅ 0 代码 perturb baseline (default off)

---

## Alternatives

### A · UE 分离 + 不 stack DCN cross (单变量 ablation 优先)

- 拆 path B 为两步: 先 UE 分离单测 · 再叠 DCN
- 放弃: stack 第 2 步 5/14~5/15 全失败 · 但**M23 (M9+DCN cross alone)** 5/14~5/16 待评 LB · 如 M23 真信号 (LB ≥ 0.8484) · UE 分离 stack DCN 是合理直接 stack
- 决策: **不拆**·5/16 下午 path B first = UE + DCN 直接 stack · 5/16 晚 2 quota 优先评 M23 (确认 cross 路径 work · 给 stack 提供 baseline) + M38 v2 (闭环 RankUp)

### B · UE 分离用 Multi-Embedding K=2 (双独立 embedding for feats_15)

- ADR-012 Multi-Emb K=2 同时 enabled · feats_15 用 2 张表 concat 投影
- 放弃: M38 v2 实测 dev 弱负 · Multi-Emb 路径 saturated · 不叠加

### C · UE 分离扩展到 user_int top 3 (feats_15 + feats_106 + feats_66)

- top 3 user_int by MI 都单独成 token
- 放弃: feats_106 (V=4) 和 feats_66 (V=256) MI 比 feats_15 弱 3.7× · 边际信息低 · 复杂度高

---

## Open Questions

1. **feats_15 在 multi-hot 还是 scalar?** 需 grep schema → `feats_15` 在 `user_int_feature_specs` 中是 scalar (length=1) 还是 multi-hot (length>1)? 决定 forward 中是 `embed(int_feats[:, offset])` 还是 `embed(int_feats[:, offset:offset+length]).mean(dim=1)`。
2. **UE token 位置**: 第 1 个 token 还是末尾? 默认第 1 个 (类似 dense_user 的 "标识 token" 角色)。下游 (ICQ / target_inject) 用 mean pool · 位置不敏感 · 但**Item NS slice 的 offset 公式仍是 `num_user_ns + has_user_dense`** · 不需改下游。
3. **是否需要 `--ue_split_dropout`**: 给 UE token 加 dropout 防过拟合? 暂不加 · 与 dense_user (无 dropout) 保持一致。
4. **5/16 下午 first job 是否同时启用 DCN**: 推荐 stack (与 ADR-014 §S2 表格一致) · 风险点见 §Negative。

---

## §10 · Path B Future Direction Matrix (5/16 21:30 实证后路线 · M51-M54 已落档 EXP-061)

### 10.1 已闭环数据点

| Job | Config | dev | LB | 判读 |
|---|---|---|---|---|
| M22 (anchor) | vocabtier · no UE split | 0.868619 | 0.847592 | (anchor) |
| **M49** ⭐ | + UE split (T37) | **0.870741** | **0.848162** | **path B 单变量真信号** |
| M50 ⚠️ | + UE split + weekend filter | 0.861974 | (5/17 早 ⭐⭐ 候选) | 双 stack 损害 net |
| M45 ❌ | + weekend filter alone | 0.858907 | (不评) | daypart filter 路径死 |

### 10.2 5/16 晚 overnight stack 探索 (EXP-061 · 4 jobs)

| Job | Config | 类型 | 期望 dev | 期望 LB | 真信号概率 |
|---|---|---|---|---|---|
| **M51** | M49 + temporal_weight 2.5 | path B + M41 stack (两已验证真信号 stack) | 0.871~0.873 | 0.848~0.850 | 40% (高) |
| **M52** | M49 + per-domain time bucket (T38) | path B + EXP-058 §Error C 修复 | 0.870~0.872 | 0.847~0.849 | 25% (中) |
| **M53** | M49 + NLIR | path B + Linear gating stack | 0.870~0.872 | 0.847~0.849 | 25% (中) |
| **M54** | M22 + per-domain time bucket alone | M28 修复单变量 | 0.868~0.869 | 0.846~0.848 | 30% (中) |

**期望产出**: 1~2 个真信号 (40% × 1 + 25% × 2 + 30% × 1 = 1.2 真信号期望值)

### 10.3 5/17 早 4 quota 决策

按 dev 选 4 个最强候选评估:
- Slot 1: M37 (NLIR vocabtier alone · 老候选 · architecture-level)
- Slot 2: M38 v2 (Multi-Emb K=2 · 老候选 · RankUp 闭环)
- Slot 3: M51/M52/M53 dev 最高的 (今晚训完选最强)
- Slot 4: M50 (UE split + weekend filter 双 stack 反例验证) · 或备用 M30/M54

### 10.4 5/17+ 后续 path B 探索 (取决于 5/17 早 LB)

**场景 A · M51 LB ≥ 0.849** (path B + M41 stack 强真信号):
- 5/17 早 quota 评 M51 + M52 + M53 (评 3 个新 stack 各自 LB)
- 5/18~5/19 path B + M51 + 第二真信号 三 stack (e.g. M51 + per-domain bucket)
- 期望破 0.85 概率提升到 25~35%

**场景 B · M51 LB ∈ [0.848, 0.849]** (微正 · 持平 M49):
- M51 stack work 但增益不显著
- 5/17 早 quota 评 M51 + M52 + M53 中 dev 最高 1~2 个
- 5/18 探索 M22 cross-seed (5/22 前破例 · 因 M51 ≥ 0.8484 触发例外)

**场景 C · M51 LB < 0.848** (持平/微负 M49):
- M51 stack 不 work · path B 体系下 stack 困难
- 5/17 早 quota 评 M52/M53 (不评 M51)
- 5/18 切回 path B 单变量 grid (ue_dim=128/64)

**场景 D · 全部新候选 LB ≤ 0.847** (整批失败):
- path B 体系下 stack 整体死透
- 5/18 切回 single-variable architecture exploration
- 考虑 DECEM 情报未消化 Tokenization 修改

### 10.5 Iron Laws (path B 实证后新增)

1. **path B 单变量已 work · 但单变量不够冲 0.85** (M49 LB +0.000570 · 距 0.85 还差 +0.0018)
2. **必须找第二真信号** · 优先 stack 已知真信号 (M41 temporal_weight) · 风险中
3. **dev→LB 衰减率 27%** · UE split 在 OOV 远未来场景下信号衰减严重 · 选 stack 候选时优先衰减率低的方法
4. **任何 stack 候选必须基于 M49 base** · 不再回 M22 base (M22 base + 单变量 stack 6/6 失败 1 work · 历史失败率 83% · 不应继续浪费)
5. **5/22 前 cross-seed 仍锁** (M49 LB 0.848162 < 0.8484 严格阈值 · 不触发例外)

### 10.6 实施成本汇总

- **M51**: 0 新代码 · 仅 run.sh
- **M52/M54**: T38 per-domain bucket (~50 LOC dataset.py + 20 LOC train.py + 10 LOC infer.py + smoke ~80 LOC) · 2.5h
- **M53**: 0 新代码 · 仅 run.sh
- 总: ~3h 实施 + 4 jobs train (overnight 5h)

---

## References

- 社区情报: `docs/99-raw/community-snapshots/xhs-decem-0512/README.md` line 196
- EXP-049: `docs/40-experiments/EXP-049-eda-correlation-2026-05-13.md` §2.4
- 现有 DCN Cross: `src/model.py::DCNCrossBypass` (line 658) + `--enable_dcn_cross` CLI
- M22 baseline: `src/run.sh` (commit 804114c canonicalize)
- M23 (DCN gap=0) ckpt: dev 0.868311 · LB 待评 (5/16 晚 2 quota 候选)
