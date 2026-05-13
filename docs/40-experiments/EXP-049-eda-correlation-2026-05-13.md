# EXP-049 · EDA Correlation Full Scan 1044 RG · 2.10M rows · 2026-05-13

> **Status**: ✅ 已完成 · 数据落档 + EXP-048 部分结论修正 + 第二批 Tier 1 方向定档 · CEST 23:55
> **日期**: 2026-05-13 CEST 23:50 起草
> **关联**: [EXP-048 quantile 全扫](EXP-048-eda-fullscan-2026-05-13.md) · [TODO active](../20-tasks/active.md) · 原始 JSON [`docs/99-raw/eda/eda-correlation-2026-05-13.json`](../99-raw/eda/eda-correlation-2026-05-13.json)

---

## TL;DR

平台跑 `src/eda_correlation.py` 全扫 1044 row groups · 2.1M rows · 46 user_int + 14 item_int fids · hash bucket=256 · top_k=100 · 总耗时 17 分 45 秒（user 552s + item 527s + correlation 计算 ~30s）· 数据通过 base64+gzip blob 从 stdout 完整回流。

### 3 条新结论 (重磅)

1. **🔄 EXP-048 "user-side 全弱" 部分修正**: `user_int_feats_15` MI=0.0036 → 比某些 item-side fid 还强 · user-side 不是绝对无信号 · 是**头比头弱 8x · 但有 outlier**
2. **🔥 user × user 多个高度冗余 cluster 实锤**: `feats_51 ↔ 52` CV=1.000 (完美决定) · `feats_94 ↔ 108` CV=0.979 + MI=1.37 · 其他多对 CV ≥ 0.5 · **强烈支持 ns_groups 8-cluster 重组（DECEM Trick #4）**
3. **🥇 user × item cross feature 候选清单到位**: top 3 是 `(user_1, item_10) / (user_1, item_5) / (user_49, item_6)` · `user_1` & `user_49` 反复出现 · `item_5/10/6` 都是 fid→label MI top 5 · **强强相交是经典 DCN/FM 设计动机**

### 第二批 Tier 1 候选定档

| 优先级 | 任务 | 依据 | 预期 LB 提升 |
|---|---|---|---|
| 🥇 T34 | DCN cross feature with top 5 user×item pairs | 信号 #3 + DECEM Trick #5 | +0.001 ~ +0.003 |
| 🥈 T35 | NS-groups 8-cluster 重组 | 信号 #2 + DECEM Trick #4 | +0.0005 ~ +0.0015 |
| 🥉 T31-P2 | user-side is-missing 仅 `user_int_feats_15` | 信号 #1 (user_15 outlier) | +0.0003 ~ +0.0008 |
| P3 | feats_51/52 disable / merge | 信号 #5 完美冗余 | ≈ 0 (节省参数) |

---

## 1. Run Metadata

```
parquet_dir:    /data_ams/industrial_training_data
schema:         46 user_int + 14 item_int fids (filtered by 'col' valid)
hash_buckets:   256 (limit contingency table size)
top_k:          100 (top correlations to output)
rows_scanned:   2,099,956
row_groups:     1,044
elapsed:        17m45s (user 552s + item 527s + correlation calc ~30s)
output:         stdout base64+gzip blob (43 KB JSON · 1977 lines pretty)
local fallback: TRAIN_LOG_PATH/eda_correlation.json (UI 不可见 · 但路径已写)
```

---

## 2. 信号 #1 · USER-SIDE 不是全弱 · 有显著 outlier

### 2.1 user-side 全部 46 fid → label MI 排序

| Rank | Fid | MI (nats) | V (vocab in sample) | 备注 |
|---|---|---|---|---|
| 1 | `user_int_feats_15` | **0.003603** | 256 | **outlier · 比第二名高 3.7x** |
| 2 | `user_int_feats_106` | 0.000983 | 4 | |
| 3 | `user_int_feats_66` | 0.000937 | 256 | |
| 4 | `user_int_feats_98` | 0.000918 | 4 | |
| 5 | `user_int_feats_65` | 0.000861 | 256 | |
| 6 | `user_int_feats_63` | 0.000802 | 47 | |
| ... (40 more 0.0008 → 0.000~) | | | | |

### 2.2 与 item-side 对照

```
user-side max  → label MI:  0.003603  (user_int_feats_15)
item-side max  → label MI:  0.028904  (item_int_feats_5)
比例: ~8x (item 比 user 强 8 倍)

user-side min  → label MI:  ~0.00001
item-side min  → label MI:  0.000993  (item_int_feats_81)

→ user_int_feats_15 (0.0036) 比 item_int_feats_81 (0.001) 还强 3.7x !
```

### 2.3 关键修正 EXP-048 / team-memory 的过度结论

**EXP-048 / team-memory 之前推论**: "user_id 一人一行 + Seed 情报 → user-side 信号弱 → user-side feature engineering ROI 极低"

**EXP-049 修正**: **user-side 整体弱 8x** 是正确的 · 但**绝对值不为 0** · 而且**最强 user fid (`user_int_feats_15`) 比最弱 item fid 还强**。

修订铁律 (2026-05-14 写入 team-memory):

> "user_id 一人一行" ≠ "user-side feature engineering ROI 为 0"。`user_int_feats_X` 是**用户分群**特征 (年龄/性别/地域/兴趣大类) · 即使每个 user_id 只出现一次 · 群体维度的统计信号依然能被模型学到。
> 
> 但 user-side 信号比 item-side 弱 ~8x · 头部 fid `user_int_feats_15` 是**显著 outlier** · ROI 优先级 < item-side · 但 > 完全忽视。

### 2.4 操作化结论

- **T31-P2**: user-side is-missing 不要全加 · 只加 `user_int_feats_15` (高基数 V=256 · MI 显著)
- 后续若做 user-side feature engineering · 优先级 = `feats_15` >> 其他 30+ fid

---

## 3. 信号 #2 · USER × USER 高度冗余 cluster · 强支持 ns_groups 重组

### 3.1 CV ≥ 0.5 的 user × user pair 列表

| Rank | Pair (a × b) | CV | MI | V_a | V_b | 解读 |
|---|---|---|---|---|---|---|
| 1 | `feats_51` × `feats_52` | **1.000** | 0.053 | 156 | 189 | **完美决定** (CV=1.0 数学上限) |
| 2 | `feats_94` × `feats_108` | **0.979** | 1.370 | 7 | 8 | **几乎完美 · MI 1.37 极高** |
| 3 | `feats_58` × `feats_59` | 0.936 | 0.759 | 3 | 15 | 强相关 |
| 4 | `feats_98` × `feats_106` | 0.715 | 0.650 | 4 | 4 | 强相关 (低基数 cluster) |
| 5 | `feats_97` × `feats_104` | 0.701 | 0.680 | 4 | 4 | 强相关 (低基数 cluster) |
| 6 | `feats_55` × `feats_58` | 0.678 | 0.456 | 44 | 3 | 强相关 |
| 7 | `feats_92` × `feats_93` | 0.675 | 0.461 | 3 | 38 | 强相关 |
| 8 | `feats_3` × `feats_4` | 0.626 | **4.227** | 256 | 256 | 高基数 + 极高 MI · 强相关 |
| 9 | `feats_57` × `feats_58` | 0.618 | 0.376 | 253 | 3 | 强相关 |
| 10 | `feats_82` × `feats_93` | 0.600 | 1.581 | 24 | 38 | 强相关 |
| ... (90 more · CV 0.4 → ...) | | | | | | |

### 3.2 用户特征聚类 (基于 CV ≥ 0.5 cluster)

```
Cluster A · 用户行为/偏好:
  feats_55, 56, 57, 58, 59     (CV 0.5~0.94 内部互相强相关)

Cluster B · 用户资料/类目:
  feats_3, 4                    (CV 0.626 · MI 4.23 极高)
  feats_51, 52                  (CV 1.0 · 完美冗余)
  feats_53, 64, 65              (与 52, 62 强相关)

Cluster C · 用户上下文:
  feats_92, 93, 82              (CV 0.5~0.68)
  feats_62, 63, 64, 65          (CV 0.5~0.55 · 互相相关)

Cluster D · 用户低基数标识:
  feats_94, 108                 (V=7,8 · CV=0.98)
  feats_97, 98, 104, 106, 107   (低基数 · 互相强相关 · 4 个)

孤立 outlier (高 MI 单独 player):
  feats_15  (V=256 · MI=0.0036 · 远超其他 user)
```

### 3.3 操作化结论

- **T35 NS-groups 重组**: 现有 NS Tokenizer 的 user-side 默认是 5 group · 应升到 **8 group** · 把 cluster A/B/C/D + outlier 分开
- **特殊处理**: cluster D (低基数 · 4个 fid `97/98/104/106/107`) 应共享同一个 embedding table · 减少参数浪费
- **重大风险**: 不动架构只改 ns_groups 配置可能 dev 没明显变化 · 验证 ROI 需要 A/B 对照

---

## 4. 信号 #3 · USER × ITEM Cross 候选清单 (FM/DCN 实证依据)

### 4.1 Top 20 user × item pairs by Cramér V

| Rank | User × Item | CV | MI | V_u | V_i |
|---|---|---|---|---|---|
| 1 | `user_1` × `item_10` | **0.278** | 0.138 | 6 | 256 |
| 2 | `user_1` × `item_5` | **0.273** | 0.129 | 6 | 256 |
| 3 | `user_49` × `item_6` | 0.272 | 0.080 | 3 | 256 |
| 4 | `user_49` × `item_10` | 0.259 | 0.072 | 3 | 256 |
| 5 | `user_49` × `item_5` | 0.259 | 0.071 | 3 | 256 |
| 6 | `user_49` × `item_7` | 0.244 | 0.065 | 3 | 256 |
| 7 | `user_49` × `item_12` | 0.244 | 0.065 | 3 | 256 |
| 8 | `user_1` × `item_6` | 0.226 | 0.099 | 6 | 256 |
| 9 | `user_15` × `item_81` | 0.217 | 0.046 | 256 | 3 | 
| 10 | `user_58` × `item_13` | 0.215 | 0.043 | 3 | 9 |
| ... (90 more) | | | | | |

### 4.2 战略价值 · 强强相交 = 经典 DCN/FM 设计动机

**Top 8 cross 里 6 对**涉及 `user_1` (V=6) 或 `user_49` (V=3) · 这俩**低基数 user 特征**很可能是**性别 / 渠道 / 设备类型** 等高粒度用户分类。

**Top 8 cross 里全部** 涉及 `item_5/6/7/10/12` · 这些都是 fid→label MI top 6 · **最强广告特征**。

→ "强用户特征 × 强广告特征" 的 cross 是经典 FM/DCN 学到的非线性交互 · 当前 HyFormer 走 attention 路径 · **缺少显式的 outer-product cross**·DCN 补 1-2 层 cross layer 是直接补充。

### 4.3 操作化结论 (T34 · DCN Cross Feature)

**实施方案 v1**:
- 选 top 5 user × item pairs (`(user_1, item_5/6/10), (user_49, item_5/6/10)` 或类似)
- 在 NS Tokenizer 输出后加一个 DCN-V2 cross layer · 输入是 user_int + item_int 的 NS token concatenation
- cross layer 复杂度 O(d^2) · d=64 时是 4096 参数 · 几乎没成本

**实施方案 v2 (FM)**:
- 直接拿 top 5 user × item pair 做手工 FM · concat 到 ModelInput

**预期 LB**: +0.001 ~ +0.003 (DCN 公开赛常规增量 · 与 EDA 信号匹配)

### 4.4 注意 · MI 是 feature ↔ feature 不是 cross → label

EDA 给的 0.138 nats 是 `user_1 ↔ item_10` 之间的 MI · **不直接代表 cross feature 对 label 的贡献**。要直接判断 cross 对 label 的实际贡献 · 需要算三元 MI `(u, i, label)` · 或直接做 ablation。但 cross feature 是经验上常 work 的方向 · 工程上先实施 + ablation 再说。

---

## 5. 信号 #4 · `feats_51 ↔ 52` 的 CV=1.0 + 低 MI 解读

```
feats_51 ↔ 52: CV=1.000  MI=0.053
```

CV=1.0 表示"知道一个能完全确定另一个" · 但 MI=0.053 nats 很低 · 这是数学上完全合理的：

```
H(X_51) = log(va) = log(156) ≈ 5.05 nats
H(X_52) = log(vb) = log(189) ≈ 5.24 nats
MI = 0.053 → 占 H(X) 仅 ~1%
```

→ **解读**: 这两个 fid 实际上**只取很少的几个共同值** · 其他 154/187 个值出现频次 ≪ 1/156 · 几乎可视为**长尾极端 ID 类**。CV=1.0 是因为非零交叉表里**每行只有一列非零** (完美决定) · 但**有效熵小**。

**关键推论**:
- 这俩 fid 既互相完美冗余 · 又对 label 几乎无贡献 (没进 user-side fid→label MI top 30)
- ROI 极低 · 但**完全可以 disable 一个或两个** · 节省参数 · 训练加速

---

## 6. 与 EXP-048 的对比 · 修正/确认

| EXP-048 结论 | EXP-049 验证 | 状态 |
|---|---|---|
| user-side 信号弱 | ✅ 弱 8× · 但有 outlier | 部分修正 |
| item-side 是主信号 | ✅ 完全确认 (item top 10 全部 > user top 1) | ✅ 确认 |
| `item_5/10/6/7` 是核心特征 | ✅ MI 排序与 quantile |gap| 排序高度一致 | ✅ 双确认 |
| `item_int_feats_84/85/83` 是 missing 高价值 | ✅ MI 排在中段 (0.0014~0.0018) · 不是 top 10 · 但远高于 user-side 长尾 | ✅ T31 决策依然成立 |
| user_id 一人一行 → user 特征无用 | ❌ **过度推论** · `user_15` 仍有显著信号 | ⚠️ 重大修正 |
| ns_groups 不需要重组 | ❌ user×user 多个 CV>0.6 cluster · 强支持重组 | ⚠️ 重大修正 |
| FM/DCN cross 是模糊候选 | ✅ Top 3 cross 已具体到 fid 对 | 可执行化 |

---

## 7. 第二批 Tier 1 候选 · 详细优先级

排在已启动 M18/M19/M20 之后:

### 🥇 T34 · DCN Cross Feature

- **依据**: EDA-correlation 信号 #3
- **实施**: 选 top 5 user × item cross pair · 加 1-2 层 DCN-V2 cross layer
- **预期 LB**: +0.001 ~ +0.003
- **风险**: 低 · DCN 是公开赛常规设计 · 与 HyFormer 正交
- **训练时间**: 0 (架构小改 · 几乎不增加参数)
- **何时启动**: M19 回流后 (无论 work 与否) · 单变量启动 · 不和其他 bet 重叠

### 🥈 T35 · NS-Groups 8-Cluster 重组

- **依据**: EDA-correlation 信号 #2
- **实施**: 把 user-side 5 group 升到 8 group · 按 CV cluster 划分
- **预期 LB**: +0.0005 ~ +0.0015 (信号比较隐性)
- **风险**: 中 · 配置改动 · 可能 dev 没明显变化
- **训练时间**: 0
- **何时启动**: T34 回流后 · 第二顺位

### 🥉 T31-Phase 2 · `user_int_feats_15` is-missing

- **依据**: EDA-correlation 信号 #1
- **实施**: M19 (item-only T31) 验证后 · 把 `user_int_feats_15` 加到 user-side bitmap
- **预期 LB**: +0.0003 ~ +0.0008 (作为 M19 的扩展)
- **风险**: 极低 · 只加 1 个 fid · 信号清晰
- **训练时间**: 0
- **何时启动**: 与 M19 work 与否绑定 · 如果 M19 work · 立即推 T31-P2 (Phase 2 = 加 user_15)

### P3 · `feats_51/52` Disable

- **依据**: EDA-correlation 信号 #4
- **实施**: 直接 disable · 节省参数
- **预期 LB**: ≈ 0
- **ROI**: 不上 LB · 只省参数 · 优先级低 · 留作工程 cleanup

---

## 8. 操作化 · 后续任务清单

- [ ] 写入 team-memory 修订铁律 (信号 #1 修正 user-side 全弱论)
- [ ] 写入 team-memory 新铁律 (信号 #2 user×user 冗余 cluster)
- [ ] 写入 team-memory 新铁律 (信号 #3 DCN cross 候选清单)
- [ ] 更新 active.md "最近一次审视" 反映 EXP-049 + 第二批 Tier 1 候选
- [ ] M18/M19/M20 任一回流后 · 重新评估第二批方向优先级
- [ ] T34 / T35 / T31-P2 实装代码（model.py + dataset.py）

---

## 9. Cost Tracking

- 1 training slot · 17m45s (远短于 5h half-train · 也短于 EDA-quantile 的 16min33s)
- GPU 卡空转（不调用 PyTorch · CPU-only correlation 计算）· 浪费但 wall-time 短
- 数据回流 100% 成功（base64+gzip blob · sentinels 完整）
- 0 quota 消耗在 evaluation 端

---

## 10. 后记 · stdout pretty section 的 UI 显示问题

平台 Logs UI 显示 `========== TOP USER × USER ==========` 后的 20 条 print 内容**全部消失**, 只看到分隔符。

**初步推测**: print 字符串里的 `→` `↔` `×` 等非 ASCII unicode 字符触发了平台日志系统的某种过滤 · 把整行 print 静默丢弃。但**JSON blob 里数据完整** · 不影响实际数据回流。

**建议改动 (P3)**: 改 `src/eda_correlation.py` 把 unicode 字符换成 ASCII (`x` 替换 `×` · `<->` 替换 `↔` · `->` 替换 `→`) · 让平台 logs 也能直观看到 top-N · 不需要本地解码 blob 才能看。
