# TencentUniRec · TAAC 2026 内部 TODO

> 单一事实源（single source of truth）。任何 TODO 状态改动只在本文件落定。
> `document/knowledge_base.txt` §8 仅保留 stub，避免双源失同步。

**最近一次审视**：2026-04-29 · 结论：T1 环境从 venv 切到 conda（`tencent-unirec`，python 3.11），AUC 复现一致；T4 仍解阻塞；初赛开赛 5 天。

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

### T1 `[x]` 跑通 baseline 在 1k demo 上，记录 AUC / FLOPs / Params

**完成时间**：2026-04-28 · 平台：本地 macOS / CPU only / py3.11.7 / torch 2.11 CPU

**结果**（默认 run.sh 参数 + `--batch_size 32 --num_epochs 2 --device cpu --seq_max_lens seq_X:64`）：

| 指标 | 值 |
|------|---|
| Total Params | 160.94 M（Sparse 158.45 M + Dense 2.49 M） |
| Epoch 1 Val AUC / LogLoss | 0.6761 / 0.3380 |
| Epoch 2 Val AUC / LogLoss | 0.6799 / 0.3327 |
| 训练总耗时 | ~33 s（包含 2 次 reinit + 2 次 eval + 2 次 ckpt） |
| 单 batch CPU forward | 658 ms（B=32, seq_max_len=256） |
| ckpt 路径 | `runs/demo/ckpt/global_step58.layer=2.head=4.hidden=64.best_model/model.pt` |

**FLOPs**：未在 CPU 跑通时测量（fvcore 对自定义 nn.Module 不准）。GPU 上跑 T4 时再用 `torch.profiler` 测准。

**关键发现**（结合 PDF）：
1. **baseline 文件 ≡ 官方训练模板**：PDF p4 列出的 `dataset.py / model.py / run.sh / train.py / trainer.py / utils.py` 与 `baseline/` 一字不差。
2. **平台环境变量**（PDF p51 / p54）：
   - 训练：`USER_CACHE_PATH` / `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_TF_EVENTS_PATH`
   - 评估：`USER_CACHE_PATH` / `MODEL_OUTPUT_PATH` / `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `EVAL_INFER_PATH`
   - ⚠ baseline `train.py` 用了 `TRAIN_LOG_PATH`，但 PDF 未列；本地 OK，平台上需自行兜底默认。
3. **平台规格**（PDF p28 / p30）：Ubuntu 22.04、Python 3.10.20、CUDA 12.6、19GB GPU / 9 CPU / 55GB RAM。本地 py3.11 与平台 py3.10 接近，无明显兼容问题。
4. **ckpt 命名约束**（PDF p51-53）：必须以 `global_step` 前缀，目录名 ≤ 300 字符。baseline `_build_step_dir_name` 已遵守。
5. **公榜参考**（PDF p16）：HyFormer 配置 0.836~0.838 AUC；run-test 系列 0.851~0.861。

**踩坑 & 解法**：
| 现象 | 原因 | 解法 |
|------|------|------|
| `iter(train_loader)` 0 batch | demo 文件只有 1 row group，被 `valid_ratio=0.1` 取整后吃光 train | 用 `pq.write_table(t, ..., row_group_size=100)` 重写为 10 RG，训练 → 9 RG / 验证 → 1 RG。正式初赛数据多分片，不会有此问题。 |
| sysctlbyname IOError 警告 | macOS 沙箱 + pyarrow CPU detection | 警告无害，可忽略；平台 Linux 无此问题 |
| 大 vocab Embedding 占显存 | demo 里 ts 列被识别为 vocab≈1.7e9 | 已在 schema 中标 `ts_fid`（baseline `_load_schema` 自动剥离）；ID 列建议 `--emb_skip_threshold 1000000`（run.sh 默认） |

**复现 Recipe（自洽，复制即可执行）**：

```bash
# ① 创建 conda 环境（一键复现，详见仓库根 environment.yml）
conda env create -f environment.yml
conda activate tencent-unirec

# ② 下载 demo 数据
mkdir -p data/demo data/demo_rg100 runs/demo/{ckpt,log,tb}
curl -sSL -o data/demo/demo_1000.parquet \
    https://huggingface.co/datasets/TAAC2026/data_sample_1000/resolve/main/demo_1000.parquet

# ③ 重写 demo 为多 row group（避免 valid_ratio 把 train 吃光，仅 demo 需要）
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/demo/demo_1000.parquet')
pq.write_table(t, 'data/demo_rg100/demo_1000.parquet', row_group_size=100)
"

# ④ 自动生成 schema.json（详见 T2）
python baseline/infer_schema.py \
    --parquet data/demo_rg100/demo_1000.parquet \
    --output  data/demo_rg100/schema.json

# ⑤ 启动训练（CPU；GPU 上去掉 --device cpu 即可）
TRAIN_DATA_PATH=data/demo_rg100 \
TRAIN_CKPT_PATH=runs/demo/ckpt \
TRAIN_LOG_PATH=runs/demo/log \
TRAIN_TF_EVENTS_PATH=runs/demo/tb \
bash baseline/run.sh \
    --batch_size 32 --num_epochs 2 --num_workers 0 \
    --seq_max_lens "seq_a:64,seq_b:64,seq_c:64,seq_d:64" \
    --device cpu
```

> 平台注入：`TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_TF_EVENTS_PATH` 由 Tencent Angel 平台自动设置；上平台时 ⑤ 这块只保留 `bash baseline/run.sh` 即可。
>
> `TRAIN_LOG_PATH` 不在平台 env 列表中（已在 baseline 加 `/tmp` fallback，见 `bd1af1b`）。

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
| §8.1 | T1 | `[x]` |
| §8.2 | T2 | `[x]` |
| §8.3 | T3 | `[!]` |
| §8.4 | T4 | `[~]` |
| §8.5 | T5 | `[~]` |
| §8.6 | T6 | `[!]` |

---

## Changelog

- **2026-04-29**：T1 环境切换 venv → conda（`tencent-unirec`，python 3.11.15 / torch 2.10.0 CPU）。recipe 第 ① 步从 `python -m venv` 改为 `conda create`；后续步骤删除 `.venv/bin/` 前缀。conda env 跑通 1 epoch AUC 0.6761，与 venv 完全一致（seed=42 复现）。`.venv/` 已删除（释放 922MB）。
- **2026-04-28**：T1 完成。结合 PDF 全 58 页验证 baseline 即官方训练模板；本地 CPU 33s 跑通 demo，AUC 0.6799。沉淀「踩坑 & 解法」表 + Recipe（含 row_group_size 重写步骤）。回写 PDF 关键事实（环境变量清单、平台规格、ckpt 命名约束、公榜参考分数）。
- **2026-04-26**：创建文件，从 `knowledge_base.txt §8` 迁出 6 条 TODO；T5 决议暂缓；新增 7 步审视清单。
