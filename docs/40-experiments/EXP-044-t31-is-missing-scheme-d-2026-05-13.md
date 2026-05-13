# EXP-044 · T31 is-missing 信号 Scheme D 双路冗余（代码落地 · 未验证）· 2026-05-13

> **Status**: 🧱 **Code landed · smoke passed · platform job pending** · CEST 13:30 代码完成 + 本地 smoke 通过
> **日期**: 2026-05-13 CEST 12:20-13:30（T30 + T31 并行开发窗口 · T31 占 70 分钟含方案调研）
> **关联**：[EXP-043](EXP-043-t30-time-bucket-config-2026-05-13.md)（同批 T30 改动）· [EXP-042](EXP-042-eda-piggyback-m16-2026-05-13.md)（M16 Q2 per-fid null_rate × CVR gap 将给出真实信号强度）· [docs/10-research/community-intel.md §DECEM-0512 Trick #3](../10-research/community-intel.md) · `docs/20-tasks/active.md §T31`

---

## TL;DR

**目标**：DECEM Trick #3 "大量 null 字段 baseline 统一映射为 0 (padding)，如果缺失率与转化率高度相关就丢了关键信号"。对选定的 `user_int_feats_{fid}` / `item_int_feats_{fid}` 标量 fids 开启 **"is-missing 双路冗余"**：
- **Path A · embedding slot 扩展**：给每个选中 fid 的 `vocab_size` 加 1 · 新的 top slot（`vs-1`）专供 missing 行 · missing 信号通过独立 embedding vector 进入 embedding path
- **Path B · dense bitmap**：把每个选中 fid 的 `is_missing ∈ {0, 1}` bit 填入 `user_dense_feats` 尾部 · 独立信号进入 dense path

**为什么两路都要？**（用户选 Scheme D · 2026-05-13 CEST 12:50 决策）：
- Path A 让模型学到 "missing 是一种特殊值而非 padding"（embedding 维度表达能力）
- Path B 提供 **模型无关** 的硬信号（即使 embedding lookup 失效或被其他信号淹没，bitmap 仍直接输入 classifier）
- 两路参数量都很小（A: +1 slot × d_model/fid · B: +1 float/fid）· 冗余换 robust
- 实测参数对比：
  - 选 10 个 fids 的总参数增量 ≈ 10 × 16 (d_model) + 10 × 1 = **170 个新参数**
  - vs T28 per-block DIN +7% params 曾证负向 · Scheme D 增幅可忽略（< 0.001%）· **不触发"独立化参数三连败铁律"**（T19/T28/FaFE 三次证伪的机制是"+34%/+7%/+4.5% 参数"）

**默认行为**：`--enable_is_missing` 关闭时 schema / vocab_size / user_dense_dim 全部与主线一致 · **backward-compat 严格保证**（smoke 验证通过）。

---

## 方案调研与决策路径

### Option 比较（2026-05-13 CEST 12:30-12:50）

| 方案 | Path | 参数量 | 信号强度 | 风险 |
|---|---|---|---|---|
| A1 · 仅 embedding slot | vocab +1 | ~16/fid | 中（被其他 token 稀释） | 低 |
| A2 · 仅 dense bitmap | +1 float | 0 | 中（dense path 信息瓶颈） | 低 |
| A3 · 独立 missing_indicator embedding table | 独立新 embedding | +2 × d_model/fid | 高 | 中（打破 "每参数必须大规模梯度" 铁律 · 参照 T19 失败） |
| **D · A1 + A2 双路冗余** | vocab +1 + dense bitmap | ~17/fid | 高（两个独立通路） | 低 |

**用户选 D**（CEST 12:50）：机制最"广告业务常识级"+ 参数量可忽略 + 两路冗余可相互兜底。

### 子决策：item_int 的 is-missing bitmap 放哪？

- **候选 1**：放 `item_dense_feats`
- **候选 2**：放 `user_dense_feats` 尾部
- **选择 2**（避免激活 `item_dense` 通路；部分 ckpt 历史上 `item_dense` 是空的）· 所有选中 item_int fids 的 is-missing bit 与 user_int fids 的 bit 共用 `user_dense_feats` 尾部

### 子决策：默认 fid 名单

- **user_int**: 空（opt-in）· 建议用户指定 DECEM 列表 `86,94,99,100,101,102,103,104,108,109`
- **item_int**: 空（不默认选任何 · DECEM 未列 item_int 高 null fid · 避免盲选）

---

## 代码改动（~120 行净增 · 全部在 `baseline/dataset.py` + CLI thread-through）

### `baseline/dataset.py`

#### 新增常量 & helper

```python
IS_MISSING_FEATURE_FID = 900006  # stable FeatureSchema key; avoids collision with real fids

def parse_int_csv(spec: str | None) -> list[int]:
    if not spec or not spec.strip():
        return []
    return [int(x.strip()) for x in spec.split(',') if x.strip()]
```

#### `PCVRParquetDataset.__init__`

- 新增 kwargs: `enable_is_missing: bool = False`, `is_missing_user_int_fids: Optional[List[int]]`, `is_missing_item_int_fids: Optional[List[int]]`
- 在 `_load_schema` 之后调用 `_resolve_is_missing_plan()` · 再在 `_col_idx` 建好后调用 `_link_is_missing_plan_col_idx()`

#### `_resolve_is_missing_plan()`

核心逻辑：
1. **非法 fid 过滤**（不在 schema / 非标量 dim!=1 / 不在 user_int∪item_int）· 只 warn 不 raise
2. **选中 fids 的 vocab_size +1**（mutating `self._user_int_cols[i][1]`、`self.user_int_vocab_sizes[offset]`）· 新 top slot 预留给 missing
3. **user_dense_schema 尾部扩展**：`self.user_dense_schema.add(IS_MISSING_FEATURE_FID, n_user_sel + n_item_sel)` · 记录 `self._is_missing_dense_offset`
4. **建 plan list**：`(col_idx=-1, int_buf_offset, fid)` · 按 fid 排序保证 bitmap 位置确定性

#### `_link_is_missing_plan_col_idx()`

遍历 plan list · 把 `col_idx=-1` 换成 `self._col_idx[f'user_int_feats_{fid}']`（item 同理）· 让 `_convert_batch` 可以 O(1) 查列。

#### `_convert_batch()` · 核心数据流

对每个选中 user_int fid：
```python
arr = col_data  # 原始 nullable scalar column (may contain null/<=0 as "missing")
missing_mask = (arr <= 0) | arr.isnull()   # 先算 mask（在 collapse 前）
arr_coalesced = np.where(missing_mask, 0, arr)  # baseline pre-existing: null→0
# Path A: mapping missing → vs-1 (new top slot); 其余值走原 clip 路径
arr_routed = np.where(missing_mask, vs - 1, arr_coalesced)
self._buf_user_int[:, ci] = arr_routed  # 放回 int buffer
# Path B: 同时填 bitmap 到 user_dense 尾部
self._buf_user_dense[:, self._is_missing_dense_offset + local_i] = missing_mask.astype(np.float32)
```

item_int 逻辑相同 · bitmap 共用 `user_dense_feats` 尾部（非 `item_dense`）。

### `baseline/train.py`

```python
parser.add_argument('--enable_is_missing', action='store_true', default=False, help='T31 ...')
parser.add_argument('--is_missing_user_int_fids', type=str, default='',
                    help='T31 · CSV user_int fids to treat as missing-aware. DECEM-listed: 86,94,99-104,108,109.')
parser.add_argument('--is_missing_item_int_fids', type=str, default='', help='T31 · CSV item_int fids (opt-in; DECEM未列).')

get_pcvr_data(
    ...
    enable_is_missing=args.enable_is_missing,
    is_missing_user_int_fids=parse_int_csv(args.is_missing_user_int_fids),
    is_missing_item_int_fids=parse_int_csv(args.is_missing_item_int_fids),
)
```

### `baseline/infer.py`

```python
from dataset import parse_int_csv
...
PCVRParquetDataset(
    ...
    enable_is_missing=bool(cfg.get("enable_is_missing", False)),
    is_missing_user_int_fids=parse_int_csv(cfg.get("is_missing_user_int_fids", "")),
    is_missing_item_int_fids=parse_int_csv(cfg.get("is_missing_item_int_fids", "")),
)
```

**model 层零改动**：`PCVRHyFormer.__init__` 从 dataset 读 `user_int_vocab_sizes` 和 `user_dense_schema.total_dim` · 自动看到 vs+1 和 user_dense_dim +N · 不需要任何新 kwargs。**平台 default infer.py 直接兼容**（与 T30 同理 · 都是 dataset 层改动）。

---

## 本地 Smoke 验证（demo_rg100 · tmp_t31_smoke.py · 已清理）

### Test 1: 关闭时 backward-compat

```
[disabled] user_dense_dim=755
[disabled] _is_missing_user_int_plan=[]
[disabled] _is_missing_item_int_plan=[]
[disabled] OK - backward compat
```

`user_dense_dim=755` 与主线完全一致 · plan list 为空 · `vs` / `user_int_cols` 未被修改。

### Test 2: 开启时 shape 正确

- 选 `86, 94` 2 个 user_int fids
- `base_dim=755` (关闭) → `on_dim=757` (开启) · **delta=2 精确等于选中 fids 数**
- fid 86 / 94 的 `vs_on - vs_off == 1`（vs bump 正确）

### Test 3: batch 数据数值正确

```
[batch] user_dense shape=(8, 757), user_int shape=(8, 234)
[batch] bitmap slice shape=(8, 2), unique=[0.0, 1.0]   ← bitmap 只有 0/1，无污染
[batch] fid 86: missing_rate=0.750
[batch] fid 94: missing_rate=0.625
[batch] OK - bitmap and top-slot routing consistent
```

关键验证：
- bitmap 值域严格 {0.0, 1.0}（没有连续值污染）
- bitmap=1 的行 `user_int[:, col] == vs - 1`（精确路由到 top slot）
- missing_rate 75% / 62.5% 在 demo 上 · 与 DECEM "86/94 高 null 率" 描述一致（虽然 demo 只 1000 行，信号方向对）

---

## 为什么不在本次做 model 层改动

**机制判断**：
- Path A 已经利用现有 `GroupNSTokenizer` / `RankMixerNSTokenizer` 的 embedding lookup · `vs+1` 在 `nn.Embedding` 初始化时自动加 1 slot · **零代码成本**
- Path B 已经走 `user_dense_feats` 的现有 dense→token 通路 · 也是零代码成本

换句话说，**Scheme D 是"纯 dataset 层"** 改动 · model 架构不变 · 这是它相对 Scheme A3（独立 missing_indicator table）的巨大优势 —— **不触发 "+X% 参数 必死" 铁律**（T19/T28/FaFE 三连败的共同机制）。

---

## 下一步

### Phase 1: 平台 half-train 验证（待 M16 EDA 回流后决策）

**前置条件**：M16 `training_summary.json['eda']['q2_user_int_missing_top']` 给出平台真实 fid 的 null_rate × cvr_gap top-20。

**判决 fids 子集**：
- 所有 |cvr_gap| > 2% 的 fid 纳入 `--is_missing_user_int_fids`
- 所有 |cvr_gap| < 0.5% 的 fid **排除**（信号太弱 · 反而引入噪声）
- 0.5% ≤ |cvr_gap| ≤ 2% 的 fid 视具体情况（建议宁少勿多）

**Job Name 占位**：`T31-ismiss-gap05-seed42-4ep`（config 同 M5 baseline · 只叠 `--enable_is_missing --is_missing_user_int_fids <EDA选出来的列表>`）

**判决阈值**（vs `icq-half-gap05` E3 Dev 0.842443）：
- E3 Dev ≥ 0.8437（+0.0013 = 3× rerun 噪声）→ 🎯 真信号 · eval
- [0.8433, 0.8437) → 🟡 seed=3 rerun 复现
- < 0.8430 → ❌ Deprecated

### Phase 2: 如 T31 work · 叠加测试

- T30 时间桶 + T31 is-missing 叠加（都是 dataset 层 · 完全正交）
- T25 bypass + T31 叠加
- 避免 T25+T29 叠加失败重蹈覆辙 · 每个叠加都单独验证

---

## 风险点 · 需要关注

1. **demo 数据的 missing_rate 不等于平台**：demo 里 fid 86 missing 75% 可能因为 demo 是小样本子集 · 平台 1.9M 上真实可能 20%~50%。M16 EDA 会给真实数字。
2. **某些 fid 的 vocab_size 已经很大**（如果原 vs = 1000，+1 变 1001 相对无感）· 但对 vs < 10 的小 fid 是 +10% 相对增幅 · 需注意 gradient signal 稀释（理论上影响不大 · 因为 missing slot 本身会有大量训练样本走）
3. **Path A 的 embedding 初始化**：`nn.Embedding(vs+1, d)` 的新 top slot 走默认 init · 一开始随机 · 靠梯度学到"这是 missing 特征"。**冷启动期（epoch 1~2）可能噪声大**。E3-E4 稳定性应足够（EXP-019 铁律 · E3-E4 是甜蜜点）。
4. **评估时 test set 的 missing 分布可能与 train 不同**（train 远未来 test 分布偏移已知问题）· 若某 fid train missing 率 30% · test missing 率 50% · Path A 的新 slot 在 test 上会被大量激活 · 其 embedding 是否泛化未知。Path B 的 bitmap 作为 0/1 硬信号不受此影响 · **这正是双路冗余的价值所在**。

---

## Materials

- Commit: 本次 T30+T31 一起 commit
- 代码改动：
  - `baseline/dataset.py` +120 行（`IS_MISSING_FEATURE_FID` / `parse_int_csv` / `_resolve_is_missing_plan` / `_link_is_missing_plan_col_idx` / `_convert_batch` 新 path）
  - `baseline/train.py` +30 行（3 个新 CLI flags · thread-through）
  - `baseline/infer.py` +5 行（import + 3 个 cfg.get fallback）
- Smoke 验证：tmp_t31_smoke.py（已清理）· 3 个 assertion 组 · ALL T31 SMOKE PASS
