# Tencent Angel Machine Learning Platform — User Guide（精简整理版）

> 来源：`docs/99-raw/platform-pdf/Tencent Angel Machine Learning Platform User Guide.pdf`（图片版 PDF，不进 git）。
> 整理范围：全 58 页；剔除空白排版页；保留所有可执行约束与官方公榜数字。
> 维护者复核日期：2026-04-28。

---

## 1. 概览 & 时间表（PDF p1-2）

- 主办：腾讯广告 / KDD 2026 联合赛事「TENCENT UNI-REC CHALLENGE 2026」
- 平台入口：在线 SaaS（侧栏 Home / Model Training / Model Management / Model Evaluation）
- 五个阶段（Phase 1 ~ 5）按时间线推进，对应：报名 → 初赛 → 复赛 → Top 50 评审 → 8/9 颁奖。

参赛队的核心产物在平台上分三步：
**STEP 1** 创建训练 Job → **STEP 2** 导出 Model → **STEP 3** 提交 Evaluation。

---

## 2. Workflow A · Model Training（PDF p4-10）

### 2.1 创建训练任务

字段填法（Create Training Job 页）：

| 字段 | 说明 |
|------|------|
| Job Name | 任务名（队内可见） |
| Job Description | 任务描述 |
| Training Dataset | 固定为 `TencentDR`（点 "View Demo Data" 可下载样本数据） |
| Training Resources | 资源数（默认 1，复赛 Top 50 可用 8 张 GPU） |
| Training Code | 上传训练脚本，**必须包含 `run.sh`** —— 平台启动时自动执行 |

> ⚠️ **`run.sh` 是 mandatory entry point**，平台容器启动后会自动 `bash run.sh`。

官方训练模板包含 6 个文件（与本仓库 `baseline/` 一字不差）：

```
dataset.py    28.1K
model.py      69.5K
run.sh         343B
train.py      11.4K
trainer.py    14.2K
utils.py       4.6K
```

### 2.2 启动 / 监控

- 列表页点 "Run" 启动；状态变为 "Training Job Running"
- 点 "Instances" 进入实例页，可查看 Output / Logs，或 Stop 任务
- Output 页右上角的 "Publish" 按钮把当前 ckpt 发布到 Model Management

### 2.3 编辑 / 复制 / 删除

任务卡右侧 "More" → Copy / Delete / Edit。"Edit" 直接打开 IDE 风格的 Script Editor。

---

## 3. Workflow B · Model Release & Management（PDF p11-15）

1. 在训练 Job 的 "Instances" → "Output" 页选目标 ckpt
2. 点 "Publish"，填 Model Name + Description，提交
3. Publish Status 变为 "Released" 即成功
4. 左侧 "Model Management" 可查看 / 编辑 / 删除已发布模型（**单队最多 20 个模型**）

ckpt 在平台上呈现的目录名样例（来自 PDF p13 截图）：

```
global_step3557.layer=2.head=4.hidden=64.best_model
```

与 baseline 的 `_build_step_dir_name` 输出格式一致。

---

## 4. Workflow C · Model Evaluation（PDF p16-26）

### 4.1 入口

- 方法 1：Model Management 卡片底部 "Model Evaluation" 按钮
- 方法 2：Model Evaluation 页顶部 "Create Evaluation"

### 4.2 提交评估

| 字段 | 约束 |
|------|------|
| Model Name | 从已 Released 的 model 列表中选 |
| Evaluation Name | 自定 |
| Evaluation Dataset | 固定 `Platform Test Set` |
| Evaluation Method | 固定 `Score` |
| Evaluation Resources | 默认 1 |
| Evaluation Docker Image | `Default Docker Image` |
| Inference Code | **必须包含 `infer.py`**，详见下文 |

### 4.3 `infer.py` 硬约束（极其重要）

```text
1. 文件名严格命名为 "infer.py"
2. 内含 main() 函数，且不接收任何参数
3. main() 必须在 EVAL_RESULT_PATH 目录下产出 "predictions.json"
4. 上传脚本总大小 ≤ 100 MB
5. 单次 inference 最长 30 分钟，超时强制 kill
6. 每队每滚动 24h 最多 4 次评估提交（Failed/Stopped 不计入）
```

`predictions.json` 格式：

```json
{
  "predictions": {
    "user_001": 0.8732,
    "user_002": 0.1245,
    "user_003": 0.5621
  }
}
```

⚠️ 每个 key 必须是测试集中的 `user_id`（字符串），value 是该用户的转化概率（float ∈ [0,1]）。**缺失或多余 user_id 都会影响最终分**。

### 4.4 Evaluation 状态机

```text
Pending  →  Waiting for Inference Resources
        →  Inference Running                  ← infer.py 在跑
        →  Waiting for Evaluation Resources
        →  Evaluation Running                 ← 平台打分
        →  Success / Failed
```

### 4.5 `prepare.sh`（可选）

如需在 inference 前装额外依赖，可在 Inference Code 中加 `prepare.sh`：

```bash
#!/bin/bash
# 1. 这个脚本会在 inference 逻辑启动前先执行
# 2. 在已激活的 conda 环境内运行
# 3. 可执行 conda install / pip install / 任意 setup 命令
```

---

## 5. Platform Specifications

### 5.1 硬件（PDF p28，单实例）

| 资源 | 规格 |
|------|------|
| Computing Power | 20% of a single GPU |
| GPU Memory | **19 GiB** |
| CPU Cores | 9 |
| Memory | **55 GiB** |

### 5.2 操作系统 & 关键 SDK（PDF p28-30）

| 软件 | 版本 |
|------|------|
| OS | Ubuntu 22.04 |
| CUDA | **12.6** |
| cuDNN | 9.5.1 |
| cuBLAS | 12.6.3.3 |
| NCCL | 2.26.2+cuda12.6 |
| conda | 26.1.1 |
| python | **3.10.20** |

### 5.3 Pre-installed Python Packages（PDF p30-49，节选）

平台 conda env 自带 200+ 包。**ML / 数据栈核心版本**（按本赛题相关性排序）：

| 包 | 版本 | 备注 |
|----|------|------|
| torch | **2.7.1+cu126** | 主框架 |
| torchaudio | 2.7.1+cu126 | |
| torchvision | 0.22.1+cu126 | |
| torchrec | **1.2.0+cu126** | 推荐场景的 KeyedTensor / Embedding 友好 |
| torch_cluster | 1.6.3+pt27cu126 | |
| torch_scatter | 2.1.2+pt27cu126 | |
| torch_sparse | 0.6.18+pt27cu126 | |
| torch-geometric | 2.7.0 | |
| triton | 3.3.1 | |
| transformers | 4.35.0 | |
| tokenizers | 0.14.1 | |
| numpy | **2.2.5** | 注意 NumPy 2.x 不兼容 1.x 二进制 |
| pyarrow | **23.0.1** | parquet 流式读入 |
| pandas | 2.3.3 | |
| polars | 1.39.2 | |
| scikit-learn | (见 PDF p43) | metrics 计算 |
| xgboost | 3.2.0 | |
| lightgbm | 4.6.0 | |
| catboost | 1.2.10 | |
| tensorflow | 2.19.0 | |
| keras | 3.12.1 | |
| accelerate | 0.32.1 | |
| peft | 0.13.2 | |
| bitsandbytes | 0.49.2 | 量化 |
| tensorboard | (见 PDF) | + tensorboard-data-server 0.7.2 |
| tensordict | 0.11.0 | |
| timm | 1.0.25 | |
| spacy | 3.8.11 | |
| nltk | 3.9.3 | |
| matplotlib | 3.10.8 | |
| seaborn | 0.13.2 | |
| Pillow | 12.1.1 | |
| opencv-python | 4.13.0.92 | |

> 完整 200+ 包列表请回看原 PDF p30-p49（按字母表排序）。本仓库只缓存关键版本号，避免维护成本。

---

## 6. 训练阶段：Environment Variables & Output Spec（PDF p50-53）

### 6.1 平台注入的环境变量

| Variable | Description |
|----------|-------------|
| `USER_CACHE_PATH` | 用户缓存目录，**配额 20 GB**；training 与 evaluation **共享**，可用于跨阶段传递特征 / 字典 / 中间产物 |
| `TRAIN_DATA_PATH` | 训练数据集根目录（parquet 分片所在路径） |
| `TRAIN_CKPT_PATH` | ckpt 输出根目录（`model.pt` 必须落到该目录子树下） |
| `TRAIN_TF_EVENTS_PATH` | TensorBoard event 文件目录 |

> ⚠️ baseline 的 `train.py` 还用了一个 `TRAIN_LOG_PATH`，此变量**未在官方 PDF 列出**。
> 平台上跑时建议在脚本里兜底：`os.environ.get("TRAIN_LOG_PATH", "/tmp/log")`，避免 KeyError。

shell / python 读法：

```bash
${TRAIN_DATA_PATH}
```

```python
import os
os.environ.get("TRAIN_DATA_PATH")
```

### 6.2 ckpt 输出规范

1. ckpt **必须**保存到 `TRAIN_CKPT_PATH` 指定的目录
2. 迭代型 ckpt **必须**放在以 `global_step` 为前缀的子目录里，否则平台不识别
3. 子目录名 **≤ 300 字符**
4. 子目录名仅可含：字母 `a-zA-Z`、数字 `0-9`、下划线 `_`、连字符 `-`、等号 `=`、点 `.`

附加参数可写在 `global_step` 后面，例如：

```text
global_step20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200
```

> baseline `_build_step_dir_name` 已遵守上述规范，无需改动。

### 6.3 TensorBoard 指标

平台**只支持 scalar metrics**。代码中按如下方式初始化：

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(os.environ.get("TRAIN_TF_EVENTS_PATH"))
```

---

## 7. 评估阶段：Environment Variables & Output Spec（PDF p54-57）

### 7.1 平台注入的环境变量

| Variable | Description |
|----------|-------------|
| `USER_CACHE_PATH` | 同上，与训练共享（20 GB 配额） |
| `MODEL_OUTPUT_PATH` | 已发布 model 的根目录 |
| `EVAL_DATA_PATH` | 测试数据目录 |
| `EVAL_RESULT_PATH` | 中间结果与 `predictions.json` 输出目录 |
| `EVAL_INFER_PATH` | 用户上传的 inference 脚本所在目录 |

shell / python 读法与训练阶段相同，把 `TRAIN_*` 替换为对应 `EVAL_*` 即可。

### 7.2 输出规范

唯一硬要求：`infer.py` 的 `main()` 在 `EVAL_RESULT_PATH` 下生成符合 §4.3 格式的 `predictions.json`。

---

## 8. 重要 ATTENTION 集中清单

> 这些是 PDF 中红字标注 / 反复强调的不可破坏约束。

### 8.1 训练侧

- `run.sh` 是**强制** entry point，平台自动执行
- ckpt 子目录必须 `global_step` 前缀 + ≤ 300 字符
- 复赛阶段需额外上传 `README.md`（Markdown）作为技术报告

### 8.2 评估侧

- `infer.py` 文件名严格、`main()` 无参 —— 否则 evaluation 直接 Failed
- `predictions.json` 必须放在 `EVAL_RESULT_PATH/`
- 24h 滚动窗口最多 **4 次**有效提交（Failed / Stopped 不计入）
- 单次 inference 30 分钟硬上限，超时强 kill
- inference 脚本总大小 ≤ 100 MB
- 单队最多 20 个 Released model

### 8.3 通用

- `USER_CACHE_PATH` 配额 20 GB，**train / eval 共享**
- 若需额外依赖：训练用 `run.sh` 内 `pip3 install ...`；评估用 `prepare.sh`
- 平台仅支持 scalar TensorBoard metrics

---

## 9. 平台公榜参考分数（PDF p16 截图节选）

仅供锚定本地实验进度：

| Evaluation Name | Status | Leaderboard Score | Inference Time |
|-----------------|--------|-------------------|----------------|
| `run-test_eval_*` ×4 | Success | **0.851 ~ 0.861** | 78~80 s |
| `4v hyformer 1rankmixer_eval` | Success | 0.837586 | 49.95 s |
| `transformer+mixer_separate_2_eval` | Failed | — | — |
| `longer mixer 1 rank_mixer_eval` | Success | 0.836481 | 48.24 s |

> 一个 baseline-style HyFormer 配置在公榜约 **0.836 ~ 0.838**；进阶配置（run-test 系列）可达 **0.851 ~ 0.861**。
> 本地若拿不到至少 0.83+，多半是数据 / 特征 / schema 出了问题，先回查再调网络。
