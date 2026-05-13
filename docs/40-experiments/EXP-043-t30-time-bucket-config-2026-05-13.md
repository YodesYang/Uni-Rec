# EXP-043 · T30 时间桶边界可配置化（DECEM Trick #1 · 代码落地 · 未验证）· 2026-05-13

> **Status**: 🧱 **Code landed · smoke passed · platform job pending** · CEST 12:30 代码完成 + 本地 smoke 通过
> **日期**: 2026-05-13 CEST 12:20-12:30（T30 + T31 并行开发窗口）
> **关联**：[EXP-042](EXP-042-eda-piggyback-m16-2026-05-13.md)（M16 EDA 搭车 · Q5 time_diff 分位数将给出真实边界）· [docs/10-research/community-intel.md §DECEM-0512 Trick #1](../10-research/community-intel.md) · `docs/20-tasks/active.md §T30`

---

## TL;DR

**目标**：把 `baseline/dataset.py` 的 `BUCKET_BOUNDARIES`（63 个硬编码值、embedding vocab=65 包含 pad）从"手工调的硬编码常量" 升级为"可通过 CLI 传入的分位数边界"，**为 M16 EDA 回流后的精调版时间桶做准备**。本次不改默认值（backward-compat），**只开通道**。

**核心约束（Scheme B · 最小侵入）**：
- 保持 **边界数量固定 = 63**（对应 embedding vocab slots 不变 = 65）
- 只允许 **边界值** 变化 · 拒绝改变 bucket 数量
- CLI `--time_bucket_boundaries ""` 默认空字符串 → 走原 `DEFAULT_BUCKET_BOUNDARIES`
- 非空时校验长度必须 = 63 · 严格单调递增 · 全正整数

**机制假设**：DECEM 原话 "当前硬编码桶可能 5s~1h 集中 50% 边界但真实 time_diff p1 已在小时级" → 部分 embedding slot 几乎从不被激活 · Embedding 容量浪费。按真实分位数重设边界可让每 slot 大致均匀填充。

---

## 代码改动（~15 行净增）

### `baseline/dataset.py`

```python
DEFAULT_BUCKET_BOUNDARIES = np.array([5, 6, ..., 31536000], dtype=np.int64)
BUCKET_BOUNDARIES = DEFAULT_BUCKET_BOUNDARIES  # backward-compat alias

def parse_bucket_boundaries(spec: str | None) -> np.ndarray:
    """Parse comma-separated boundary string. Empty/None → default copy.

    Enforces: exactly 63 values, strictly increasing, all positive."""
    if not spec or not spec.strip():
        return DEFAULT_BUCKET_BOUNDARIES.copy()
    vals = [int(x.strip()) for x in spec.split(',') if x.strip()]
    if len(vals) != 63:
        raise ValueError(f"time_bucket_boundaries: got {len(vals)} values, expected 63")
    if any(v <= 0 for v in vals):
        raise ValueError("time_bucket_boundaries: all values must be positive")
    arr = np.array(vals, dtype=np.int64)
    if not (np.diff(arr) > 0).all():
        raise ValueError("time_bucket_boundaries: values must be strictly increasing")
    return arr
```

`PCVRParquetDataset.__init__` 新增 `bucket_boundaries: Optional[np.ndarray] = None` 存为 `self._bucket_boundaries` · `_convert_batch` 里 `np.searchsorted(self._bucket_boundaries, ...)` 替换原常量引用。

### `baseline/train.py`

```python
parser.add_argument('--time_bucket_boundaries', type=str, default='',
                    help='T30 · comma-separated 63 positive ints, strictly '
                         'increasing. Empty = DEFAULT_BUCKET_BOUNDARIES (backward-compat).')
...
get_pcvr_data(
    ...
    bucket_boundaries=parse_bucket_boundaries(args.time_bucket_boundaries),
)
```

CLI 参数通过 `vars(args)` 自动进 `train_config.json` · infer.py 读回后 `parse_bucket_boundaries(cfg.get("time_bucket_boundaries", ""))` 重建 · **model 层完全不涉及**（纯 dataset 层改动）。

### `baseline/infer.py`

```python
from dataset import parse_bucket_boundaries
...
PCVRParquetDataset(
    ...
    bucket_boundaries=parse_bucket_boundaries(cfg.get("time_bucket_boundaries", "")),
)
```

---

## 本地 Smoke 验证

### Unit-level（tmp_t30_smoke.py · 已清理）

1. **默认值 bit-identical**：`parse_bucket_boundaries("")` 与旧 `BUCKET_BOUNDARIES` 逐元素相等（63 values match）· **backward-compat 通过**
2. **长度校验**：`parse_bucket_boundaries("60,120,180")` → `ValueError: got 3 values, expected 63`
3. **单调递增校验**：乱序列表 → `ValueError`
4. **全正校验**：含 0 或负值 → `ValueError`
5. **自定义值生效**：63 个递增正整数传入后，`searchsorted` 对一组固定 `time_diff` 样本产生的 bucket_id 与默认版本**不同**（证明 boundary 真的切换了）

### E2E（已在 T30+T31 混合 smoke 中间接验证）

- `train_config.json` 正确序列化 `time_bucket_boundaries` 字段（空字符串或 CSV）· infer.py fallback 正确

---

## 为什么暂不改默认值

1. **demo 数据分位数不等于平台 1.9M 真实分布**。EXP-042 smoke 在 demo 上看到 `time_diff p1 ≈ 7.3h / p99 ≈ 149 天`，但 demo 只 1000 行 · 分位数噪声极大。
2. **M16 回流后从 `training_summary.json['eda']['q5_time_diff']` 取平台真实 p1/p5/.../p99.9 全谱 · 再生成 63 个边界值**。
3. **改默认值是有代价的风险动作**（影响所有历史 ckpt 的 infer · train_config 不带此字段的旧 ckpt 会走 default）· 本次只开通道、不动默认，**等真实 EDA 数据到手再提 PR 换默认**。

---

## 下一步

1. ⏳ **等 M16 CEST ~18:00-20:00 回流** · 看 `q5_time_diff` 平台真实分布
2. 若分布显示当前 boundaries 有显著浪费 → 按分位数生成新的 63 个边界值（`scripts/gen_time_bucket_quantile.py` · ~20 行）
3. **半程验证**（1 slot · 4 epoch · gap=0.05 baseline config）测新边界 · 判决阈值 vs `icq-half-gap05` E3 Dev 0.842443：
   - ≥ 0.8437（+0.0013 = 3× rerun 噪声）→ 🎯 真信号 · eval
   - [0.8433, 0.8437) → 🟡 seed=3 rerun 复现
   - < 0.8430 → ❌ Deprecated

---

## Materials

- Commit: 本次 T30+T31 一起 commit（详见 git log）
- 代码改动：
  - `baseline/dataset.py` +40 行（`parse_bucket_boundaries` + `DEFAULT_BUCKET_BOUNDARIES` rename + `PCVRParquetDataset` 构造参数 + `_convert_batch` 使用 `self._bucket_boundaries`）
  - `baseline/train.py` +4 行（CLI + 透传）
  - `baseline/infer.py` +2 行（import + fallback）
