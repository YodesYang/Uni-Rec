# ADR-008 · 默认训练路径启用 torch.compile + AMP bf16（T27）

> **Status**: Proposed（2026-05-09 凌晨 · 待 T27 first half-train 验证通过后升 Accepted）
> **Date**: 2026-05-09
> **Related Tasks**: T27（见 `docs/20-tasks/active.md`）
> **Related Experiments**: EXP-031（`t27-compile-amp-halftrain-gap05`，3 epoch half-train）
> **External References**:
> - 小红书 · DECEM（Industrial Track Rank 1，LB 0.852181）2026-05-08 帖子："训练必开 `torch.compile` + AMP；**inference 不开 compile**"
> - PyTorch 2.7 `torch.compile` docs；`torch.autocast(device_type, dtype=torch.bfloat16)`

---

## Context（背景与触发原因）

### 触发证据

- **外部锚点**：DECEM 帖子（本项目 `community-snapshots/xiaohongshu/industrial-rank1-decem/`）明说"训练必开 compile + AMP；inference 不开"。这是 Rank 1（LB 0.852）作者的一手工程建议。
- **内部瓶颈现状**：本项目 full-train 单次 10h，**每天 4 training quota 已成为单 seed 实验 throughput 上限**。在 seed 噪声 ±0.005、方法论 ablation 信号 ±0.001~0.003 的当前格局下（EXP-028 / EXP-030 实测），**加速训练的价值不在于省时间，而在于解锁"同 budget 里塞更多容量"**（d_model / batch_size / num_epochs）。
- **时间压力**：5/25 ddl，2026-05-09 剩 16 天。若 E4 前能完成"默认加速配置"并把它用到后续 2~3 个架构级 bet 上，每 bet 省下的时间 = 多 1 次 half-train 采样 = 对 dev→LB 映射多一次校准。

### 问题域

| 维度 | 现状 | 目标 |
|---|---|---|
| Full train wall-time | ~10h | **~5~7h**（≥1.5× speedup） |
| 单卡显存占用（19GB GPU） | ~13~15GB（batch=128, d_model=64） | **≤13GB** 后仍有余量扩 d_model=96 或 batch=192 |
| 数值一致性 | fp32 | **bf16 autocast**，Dev AUC 必须在 ±0.001 内 |
| Inference 兼容性 | 平台 default infer.py 从 ckpt 动态构建 | **必须保持不变**：ckpt 的 `model.pt` 不含 `_orig_mod.` 前缀 |

### Trade-offs（放弃了什么）

- **放弃 fp16**：fp16 要走 `GradScaler` + sparse embedding 梯度特殊处理，实现复杂度高且 bf16 已够用（modern GPU 算力对 bf16 / fp16 基本等价，bf16 动态范围足以覆盖 BCE 梯度）。fp16 保留作为 `--amp_dtype float16` 可选路径但不默认。
- **放弃 `mode='reduce-overhead'` / `mode='max-autotune'`**：这两个模式对**固定 shape**优化更激进，但本项目有变 batch（最后 batch 可能 <128）+ sparse embedding lookup → 高概率触发 recompile。用 `mode='default'` + `torch._dynamo.config.suppress_errors = True` 保守兜底。
- **放弃 inference 路径的 compile/AMP**：DECEM 明说 inference 不开；+ 平台 default `infer.py` 完全动态读 ckpt 构建模型，若我方训练端 compile 后 `state_dict` 带 `_orig_mod.` 前缀，会让 `infer.py` 加载失败 → 我方选择"训练端保 raw_model 的 state_dict"这条兼容路线。

---

## Decision（最终决定做什么）

**在 `baseline/trainer.py` 新增 3 个可选 ctor 参数**（向后兼容默认全 False）：
- `enable_torch_compile: bool = False`
- `enable_amp: bool = False`
- `amp_dtype: str = 'bfloat16'`（可选 `'float16'`）

**训练路径改动**：
1. `Trainer.__init__` 保留 `self._raw_model`（原 model 引用）；若 `enable_torch_compile` 且 device 是 cuda，把 `self.model` 设为 `torch.compile(raw_model, mode='default')`，CPU device 自动降级并 warn。
2. `_train_step` 在 `enable_amp` 开启时，**forward + loss 整体包在 `torch.autocast(device_type, dtype=self.amp_dtype)`** 里；bf16 路径不走 GradScaler（无 underflow 风险），fp16 路径走 `torch.amp.GradScaler`。
3. `loss.backward` / `clip_grad_norm_` / `optimizer.step` 三个动作的参数集**全部走 `self._raw_model.parameters()`**，避免 compile wrapper 在 grad 收集阶段的歧义。
4. `state_dict` save（`_save_step_checkpoint`）走 `self._raw_model.state_dict()` → 保存的 `.pt` **无 `_orig_mod.` 前缀** → 与平台 `infer.py` 完全兼容，**推理端零改动**。
5. `reinit_high_cardinality_params` / `get_sparse_params` 两个 hook **走 `_raw_model`**（compile wrapper 不一定代理这些自定义方法）；每次 sparse 重初始化后**重新包装 `torch.compile`**（sparse param 指针变了，旧 compile graph 失效）。

**评估 / 推理路径改动**：
- `_evaluate_on_loader` 里的 `self.model.eval()` 改为 `self._raw_model.eval()`。
- `_evaluate_step` 里的 `self.model.predict(...)` 改为 `self._raw_model.predict(...)`。
- **评估路径完全不 autocast、不 compile**（与 DECEM 建议一致，也与平台 infer.py 行为一致）。

**配置持久化**：
- `enable_torch_compile` / `enable_amp` / `amp_dtype` 写进 `train_config.json`，但**不写进 `model_args`** → 平台 `infer.py` 只从 `model_args` 取 kwargs 构建模型，**完全看不见这三个 flag** → 推理端零知识、零兼容性风险。

---

## Consequences（改动带来的后果）

### 正面

- **训练 wall-time 预计 1.5~2.5× 加速**（modern Ampere+ GPU 上 bf16 通常 1.8×，compile 额外 1.2~1.5× overhead 减少）。半天就能跑完一个 3-epoch half-train。
- **显存占用预计降 30~40%**（activation 用 bf16 存，sparse embedding 用 fp32 存不变）。→ 解锁 d_model=96 或 batch_size=192 的实验空间。
- **推理完全不受影响**：`.pt` 格式不变，平台 default infer.py 不变。

### 负面 / 已知风险

- **bf16 数值误差累积**：Q×K/√d_k softmax 在极长序列上可能精度不够。Mitigation：若 Dev AUC 出现 >0.001 系统性低于 baseline，下一步把 attention 内部的 softmax 强制 cast 回 fp32（`scores.float()`）。
- **compile 重编译代价**：sparse embedding reinit 后必须重新 compile（已在 trainer 里处理）；variable batch shape 也会触发，但 `suppress_errors=True` 保证不 crash。
- **调试痛苦**：compile 之后的 stack trace 不友好。Mitigation：开发新 feature 时先跑 `--enable_torch_compile` 关掉验证逻辑，再打开。
- **fp16 路径未在本项目测过**：`--amp_dtype float16` 暂时视为实验性；本 ADR 只默认化 bf16。

### 红线 / 不变量

1. **ckpt 的 `model.pt` 不能含 `_orig_mod.` 前缀** — 已通过 `self._raw_model.state_dict()` 保证（本地 smoke test verified: `raw_model.state_dict()` keys have no prefix）。
2. **`train_config.json` 里 `model_args` section 不能多出 compile/AMP 相关字段** — 平台 default infer.py 如果撞到未识别 kwarg 会 TypeError。已通过"三个 flag 只出现在 trainer 签名、不进 `model_args`" 保证。
3. **Inference 路径永远不 compile、不 autocast** — DECEM 明说 + 本项目 `_raw_model.predict` 硬保证。

---

## Verification（验证路径）

### Stage 1（已完成 · 2026-05-08 晚）
- `py_compile baseline/trainer.py baseline/train.py`：✅
- 独立 toy-model smoke（`/tmp/smoke_t27.py`）：✅
  - `torch.compile(toy_model)` → `OptimizedModule`
  - `autocast(cpu, bfloat16)` forward → `logits.dtype=bfloat16`
  - `raw_model.state_dict()` keys 无 `_orig_mod.` 前缀
  - reinit + recompile 正常

### Stage 2（待跑 · 2026-05-09 早）— EXP-031
- **Job**: `t27-compile-amp-halftrain-gap05`（`run.sh` 已就位）
- **配置**: 完全复刻 `icq-half-gap05`（EXP-030 baseline, Dev E3 0.842443, LB 0.839223），**仅加** `--enable_torch_compile --enable_amp --amp_dtype bfloat16`
- **Go/No-go**:
  - ✅ **正确性**: E3 Dev AUC ∈ [0.8414, 0.8434]（baseline 0.8424 ±0.001）
  - ✅ **加速比**: epoch wall time ≤ 0.67 × 历史 baseline（≥1.5× speedup）
  - ✅ **稳定性**: 无 dynamo traceback、无 CUDA OOM
  - ✅ **推理兼容**: best_model 能被平台 default infer.py 加载并给出合理预测（evaluation 环节校验）

### Stage 3（若 Stage 2 通过）
- **升 Accepted**，把 `--enable_torch_compile --enable_amp` 设为所有后续 half-train / full-train 的**默认 flag**。
- 下一个实验：试 `d_model=96` + `batch_size=128` 同时开 compile+AMP，看是否能同时涨容量 + 不破预算（T28 候选）。

---

## Alternatives（考虑过但未选的方案）

### A. 只开 AMP 不开 compile
- Pros：简单、稳、不怕 dynamo bug。
- Cons：只拿 30~40% 显存收益，不拿 1.2~1.5× compile 加速。DECEM 明说两者都开。
- **结论**：若 Stage 2 出现 compile-specific 问题且短期修不好，降级到此方案。

### B. 只开 compile 不开 AMP
- Pros：数值完全 fp32，不担心精度。
- Cons：显存不降，batch/model capacity 扩展不了；加速也只有 1.2~1.3×。
- **结论**：不值得。AMP 是本次的主要收益来源。

### C. 手写 fused kernel（参考 DECEM pure-model 路径）
- Pros：MFU 可能更高。
- Cons：Seed 主楼明确 "pure model 在 1.9M 数据上掉分"（team-memory 铁律），**方向已排除**。
- **结论**：红线之外，不考虑。

---

## Open Questions（未决事项）

1. **bf16 在 MultiSeqHyFormerBlock cross-attention softmax 是否够用**？— Stage 2 观察 Dev AUC 即可回答。若出现系统性精度损失，补 softmax fp32 cast。
2. **sparse embedding reinit 后 recompile 的"首次编译"开销**是否显著（尤其 epoch 2 切 epoch 3）？— Stage 2 记录每个 epoch 的 wall time，若 epoch 2 比 epoch 1 慢 >20% 说明 recompile 是瓶颈，需要进一步诊断。
3. **未来 T28/T29 架构级实验（DIN-block、multi-task）是否要基于 compile+AMP 的 baseline** — 原则上是，但如果新 module 触发 dynamo bug（如自定义 einsum pattern），允许该实验临时关 compile 回 AMP-only。
