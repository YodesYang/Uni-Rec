# TencentUniRec · TAAC 2026 内部 TODO

> 单一事实源（single source of truth）。任何 TODO 状态改动只在本文件落定。
> `document/knowledge_base.txt` §8 仅保留 stub，避免双源失同步。

**最近一次审视**：2026-04-26 · 结论：T2 完成；T5 暂缓；其余 `[~]` / `[!]` 状态保持。

---

## 审视清单（每次阅读本文件必跑 · 7 步）

按顺序执行。任何一步发现偏差 → 立刻动手修补，不留到「下次再说」。

1. **状态校验**：每条 `[~]` / `[!]` 状态是否仍准确？是否有暗中完成未打勾的工作？
2. **时间线压力**：对照 `knowledge_base.txt §2`（赛事时间表），距离最近 ddl 还有多少天？哪些 TODO 必须在此前推进？
3. **优先级重排**：高 ROI（杠杆大、阻塞少）的 TODO 是否排在前面？是否有「等结果型」TODO 可以并行启动？
4. **Recipe 时效**：每条 `[~]` 的 recipe 与当前 `baseline/` 代码、`schema.json`、依赖版本是否仍一致？
5. **红线复核**：执行路径是否仍满足 user rules 的四条红线？
   - 最小变更（不大改 baseline 文件）
   - 不新建 test 文件 / 文件夹
   - 生产安全（默认行为不变 / 配置开关 / 影响数据·采样·权重·特征的改动须给出验证方案）
   - 输出克制（无废弃 helper / debug 文件）
6. **下一步具象化**：对每条 `[~]` / `[ ]`，写一句「未来 30 分钟可做的最小动作」。写不出 → 拆分、合并或降级为 `[ ]`。
7. **沉淀回写**：在顶部「最近一次审视」打上 ISO 日期 + ≤ 1 行结论。任何状态变更须在对应 TODO 节点附 inline 注释 `(YYYY-MM-DD: ...)`。

输出规范：审视结论用结构化表格 / 列表呈现，不要埋在散文段落里。

---

## 状态约定

| 标记 | 含义 |
|------|------|
| `[x]` | 完成 |
| `[~]` | 进行中（含可复用 recipe） |
| `[ ]` | 待启动 |
| `[!]` | 阻塞（须注明阻塞条件 + 解除条件） |

---

## TODO 列表

### T1 `[~]` 跑通 baseline 在 1k demo 上，记录 AUC / FLOPs / Params

**阻塞点**：当前 workspace 无 demo 数据 + 无 torch 环境（Python 3.14.3 暂无 torch wheel）。

**Recipe（自洽，复制即可执行）**：

```bash
# ① 安装依赖（建议 conda / venv，需 Python 3.10~3.12）
pip install torch numpy pyarrow tqdm scikit-learn tensorboard

# ② 下载 demo 数据
mkdir -p data/demo
curl -L -o data/demo/demo_1000.parquet \
    https://huggingface.co/datasets/TAAC2026/data_sample_1000/resolve/main/demo_1000.parquet

# ③ 自动生成 schema.json（详见 T2）
python baseline/infer_schema.py \
    --parquet data/demo/demo_1000.parquet \
    --output  data/demo/schema.json

# ④ 启动训练（小数据建议 batch_size 调小，emb_skip_threshold 设大避免 OOB）
TRAIN_DATA_PATH=data/demo \
TRAIN_CKPT_PATH=runs/demo/ckpt \
TRAIN_LOG_PATH=runs/demo/log \
TRAIN_TF_EVENTS_PATH=runs/demo/tb \
bash baseline/run.sh --batch_size 64 --num_epochs 3 --num_workers 0

# ⑤ Params / FLOPs：训练日志会打印 Total parameters；FLOPs 用 fvcore 估算：
#   pip install fvcore
#   python -c "from fvcore.nn import FlopCountAnalysis; ..."
```

**待补**：跑通后把 `Val AUC / Params / FLOPs` 三个数字回填到本节。

---

### T2 `[x]` 写一份 schema.json 自动生成脚本

**实现**：`baseline/infer_schema.py`（2026-04-26 完成）

**关键点**：
- 从列名正则识别四类特征（`user_int` / `user_dense` / `item_int` / `domain_X_seq`）
- 整型列：`vocab_size = max(positive value) + 1`（可改 percentile）
- list 列：`dim = 实际最大长度`（同样可改 percentile）
- `ts_fid` 启发式：序列内某 fid 的正值中位数 `> 1e9`（典型 Unix 秒级时间戳）
- 输出 `domain_key` 用 baseline 兼容的 `seq_a / seq_b / seq_c / seq_d`，prefix 字段保留原始 `domain_a_seq` 等列名前缀

**CLI**：

```bash
python baseline/infer_schema.py \
    --parquet <file_or_dir> \
    --output  schema.json \
    [--vocab-percentile 1.0] \
    [--dim-percentile  1.0] \
    [--ts-min 1000000000] \
    [--sample-rows 100000]
```

**风险与回归点（拉到正式数据后必查）**：
- demo 1k 行可能 `vocab_size` 严重低估（真实数据上 fid 可能跳到 1e7+）；正式数据建议全量扫描或 `--vocab-percentile 0.999`
- `ts_fid` 启发式失败时（例如某 side-info 也是大数值），手工把对应 domain 的 `ts_fid` 字段改成正确 fid 即可
- 默认 `dim=max` 在长尾极端长序列下会撑出大显存；正式数据先用 `--dim-percentile 0.99`

---

### T3 `[!]` 拉取真实初赛数据后，更新 knowledge_base.txt §3 的 fid 列表

**阻塞点**：初赛数据需要登录 <https://taiji.algo.qq.com/> 下载。

**解除条件**：从太极平台下载到任意一份 `*.parquet` 训练分片。

**执行步骤（拿到数据后）**：

```bash
python baseline/infer_schema.py --parquet <real_data_dir> \
    --output schema.json --sample-rows 1000000
```

随后把脚本输出的 `user_int` / `user_dense` / `item_int` / `seq` 列数与 `knowledge_base.txt §3` 对照；如果差异显著，把 §3 的 fid 范围替换为真实值。

---

### T4 `[~]` 复现 HyFormer 三种序列编码器消融

baseline 中三种编码器都已实现：`baseline/model.py` 的 `SwiGLUEncoder` / `TransformerEncoder` / `LongerEncoder`，由 `--seq_encoder_type {swiglu, transformer, longer}` 切换。

**消融矩阵**（其它超参对齐）：

```text
共享基线参数：lr=1e-4, batch_size=512, num_hyformer_blocks=2,
            num_heads=4, d_model=64, num_queries=2,
            ns_tokenizer_type=rankmixer, user_ns_tokens=5, item_ns_tokens=2
```

```bash
# A. SwiGLU（最低延迟，无 attention）
bash baseline/run.sh --seq_encoder_type swiglu

# B. Transformer（最高表达，O(L^2)）
bash baseline/run.sh --seq_encoder_type transformer

# C. LONGER（折中，cross-attn top_k 压缩）
bash baseline/run.sh --seq_encoder_type longer --seq_top_k 50

# D. （可选）LONGER + causal mask
bash baseline/run.sh --seq_encoder_type longer --seq_top_k 50 --seq_causal
```

每组用相同 `--seed 42` 跑 ≥ 3 epoch，记录：`Val AUC / LogLoss / Params / FLOPs / Step latency / GPU peak mem`

**期望**（参考 HyFormer 论文 Table 1）：

```text
Transformer ≥ LONGER ≥ SwiGLU （AUC）
SwiGLU      ≪  LONGER ≪ Transformer （FLOPs）
LONGER 在初赛 100+ 字段、复赛 10× 数据上一般 ROI 最高。
```

---

### T5 `[~]` 复现 OneTrans Mixed-Param 版本（暂缓 · 等 T4 数字）

**决策（2026-04-26）**：选 (a)，暂缓启动。理由：
- HyFormer 论文 Table 1 中 OneTrans 仅领先 LONGER+RankMixer 0.05~0.08% AUC，在拿到本地 T4 数字之前贸然投 600~800 行新代码 ROI 低
- 复赛二期再启动可结合 T4 的最佳编码器选项做对比
- 维持「最小变更」红线，避免污染 `baseline/model.py`

**代价评估（启动时参考）**：约新增 600~800 行代码，集中改动 `model.py`。

**关键差异（vs 当前 baseline 的 HyFormer 风格）**：
- 把 S-tokens / NS-tokens 拼到同一注意力矩阵，不再用 Cross-Attention 桥接
- Attention Q/K/V 与 FFN 的权重做 Mixed Parameterization：S-token 共享一组 W；每个 NS-token 独立一组 W
- Causal Mask 摆放：NS-tokens 在 S-tokens 之后，吃完整 S 上下文
- 可选 Pyramid Stack（逐层缩 query 数）

**启动信号（满足任一即可重启）**：
- T4 的 LONGER 配置在 demo / 初赛上 AUC 已收敛，且距离公开 SOTA 差距 ≤ 0.1%
- 距离复赛截止 ≥ 3 周，有足够的离线 + 联调预算
- 你明确要冲创新奖（Unified Module Innovation Award）

---

### T6 `[!]` 草拟最终方案技术报告（KDD 答辩用）

**阻塞点**：缺至少 T1（baseline 数）+ T4（编码器消融数），否则报告会空洞。

**解除条件**：T1 + T4 中至少 2 组配置跑出可信 AUC 数字。

**模板规划**：参考 HyFormer / OneTrans / RankMixer 三篇论文的 Section 结构（Motivation / Method / Experiments / Online A/B / Discussion）。

---

## 编号迁移记录

`knowledge_base.txt §8.x` → 本文件 `Tx`，一一对应：

| 旧编号 | 新编号 | 状态 |
|--------|--------|------|
| §8.1 | T1 | `[~]` |
| §8.2 | T2 | `[x]` |
| §8.3 | T3 | `[!]` |
| §8.4 | T4 | `[~]` |
| §8.5 | T5 | `[~]` |
| §8.6 | T6 | `[!]` |

---

## Changelog

- **2026-04-26**：创建文件，从 `knowledge_base.txt §8` 迁出 6 条 TODO；T5 决议暂缓；新增 7 步审视清单。
