# EXP-001 · 本地 CPU baseline 跑通 1k demo（AUC 0.6761/0.6799）

> **Date**: 2026-04-28（首跑）/ 2026-04-29（venv → conda 复跑确认 seed 复现）
> **Status**: Completed
> **Related Task**: T1（已 archived）
> **Platform**: 本地 macOS / CPU only / py3.11.7（首跑）/ py3.11.15（conda 复跑）/ torch 2.10/2.11 CPU
> **Commit**: bd1af1b 之前的 baseline tip

---

## Setup

- **数据**：`data/demo_rg100/demo_1000.parquet`（1k 行，被人工 `row_group_size=100` 重写为 10 个 row group）
- **配置**：默认 `run.sh` 参数 + `--batch_size 32 --num_epochs 2 --device cpu --seq_max_lens seq_X:64`
- **资源**：单核 CPU；无 GPU

---

## Result

### 关键指标

| 指标 | 值 |
|------|---|
| Total Params | 160.94 M（Sparse 158.45 M + Dense 2.49 M） |
| Epoch 1 Val AUC / LogLoss | **0.6761** / 0.3380 |
| Epoch 2 Val AUC / LogLoss | **0.6799** / 0.3327 |
| 训练总耗时 | ~33 s（包含 2 次 reinit + 2 次 eval + 2 次 ckpt） |
| 单 batch CPU forward | 658 ms（B=32, seq_max_len=256） |
| ckpt 路径 | `runs/demo/ckpt/global_step58.layer=2.head=4.hidden=64.best_model/model.pt` |

**FLOPs**：未在 CPU 跑通时测量（fvcore 对自定义 nn.Module 不准）。GPU 上跑 T4 时再用 `torch.profiler` 测准。

---

## Findings（关键发现）

1. **baseline 文件 ≡ 官方训练模板**：PDF p4 列出的 `dataset.py / model.py / run.sh / train.py / trainer.py / utils.py` 与 `baseline/` 一字不差
2. **平台环境变量**（PDF p51 / p54）：
   - 训练：`USER_CACHE_PATH` / `TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_TF_EVENTS_PATH`
   - 评估：`USER_CACHE_PATH` / `MODEL_OUTPUT_PATH` / `EVAL_DATA_PATH` / `EVAL_RESULT_PATH` / `EVAL_INFER_PATH`
   - ⚠ baseline `train.py` 用了 `TRAIN_LOG_PATH`，但 PDF 未列；本地 OK，平台上需自行兜底默认（已在 commit `bd1af1b` 加 `/tmp` fallback）
3. **平台规格**（PDF p28 / p30）：Ubuntu 22.04、Python 3.10.20、CUDA 12.6、19GB GPU / 9 CPU / 55GB RAM。本地 py3.11 与平台 py3.10 接近，无明显兼容问题
4. **ckpt 命名约束**（PDF p51-53）：必须以 `global_step` 前缀，目录名 ≤ 300 字符。baseline `_build_step_dir_name` 已遵守
5. **公榜参考**（PDF p16）：HyFormer 配置 0.836~0.838 AUC；run-test 系列 0.851~0.861

### 后续重要参照

- **2026-04-30 ADR-001 + ADR-002 落地后**，在同一 demo 数据上重跑得到 **AUC = 0.6761**，与本 EXP 完全一致 → 验证训练逻辑零副作用
- 任何后续修改 baseline 训练逻辑的实验，**必须先在 demo 上复现 AUC ∈ [0.6761, 0.6799]**，否则视为引入了 bug

---

## Reproduce Recipe（自洽，复制即可执行）

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

# ④ 自动生成 schema.json
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

> **平台注入**：`TRAIN_DATA_PATH` / `TRAIN_CKPT_PATH` / `TRAIN_TF_EVENTS_PATH` 由 Tencent Angel 平台自动设置；上平台时 ⑤ 这块只保留 `bash baseline/run.sh` 即可。
>
> `TRAIN_LOG_PATH` 不在平台 env 列表中（已在 baseline 加 `/tmp` fallback，见 commit `bd1af1b`）。

---

## 已知踩坑表

| 现象 | 原因 | 解法 |
|------|------|------|
| `iter(train_loader)` 0 batch | demo 文件只有 1 row group，被 `valid_ratio=0.1` 取整后吃光 train | 用 `pq.write_table(t, ..., row_group_size=100)` 重写为 10 RG，训练 → 9 RG / 验证 → 1 RG。正式初赛数据多分片，不会有此问题 |
| `sysctlbyname IOError` 警告 | macOS 沙箱 + pyarrow CPU detection | 警告无害，可忽略；平台 Linux 无此问题 |
| 大 vocab Embedding 占显存 | demo 里 ts 列被识别为 `vocab≈1.7e9` | 已在 `schema.json` 中标 `ts_fid`（baseline `_load_schema` 自动剥离）；ID 列建议 `--emb_skip_threshold 1000000`（`run.sh` 默认） |

---

## Changelog

- **2026-04-29**：环境从 venv 切到 conda（`tencent-unirec`，python 3.11.15 / torch 2.10.0 CPU）。recipe 第 ① 步从 `python -m venv` 改为 `conda create`；后续步骤删除 `.venv/bin/` 前缀。conda env 跑通 1 epoch AUC 0.6761，与 venv 完全一致（seed=42 复现）。`.venv/` 已删除（释放 922MB）
- **2026-04-28**：T1 完成（首次跑通 demo），AUC 0.6799
