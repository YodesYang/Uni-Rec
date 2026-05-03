# EXP-013 · OneTrans Mixed-Param 架构：proxy 三轮调试 + full train 对比

> **Date**: 2026-05-02 ~ 2026-05-03
> **Status**: In Progress（v3 full train 仍在跑）
> **Related Tasks**: [T5](../20-tasks/active.md)
> **Platform**: Tencent Angel + macOS / CPU（smoke）
> **Commits**: bb88e29, 925cc5c, a1caf4e, 32a563d

---

## 背景

所有 P0 特征工程方向（P0-A/B/C/组合，共 6 轮全量训练）均无法超过 no-P0 reference 0.865884，差值系统性聚集在 -0.0007~-0.0009 之间，判断当前 HyFormer 架构对 dense 特征信号已饱和。

为冲 Top 1，切换方向实现 **OneTrans Mixed-Parameterization** 架构：
- S-token（各域序列压缩后的 token）使用 **共享** W_Q/K/V/O/FFN
- 每个 NS-token（用户/商品特征 token）使用**独立**一组 W_Q/K/V/O/FFN
- 注意力 mask：S-S 双向全注意力，NS-j 可 attend 所有 S + NS_0..NS_j（NS 内因果）
- 序列压缩：每域取最近 top_k=50 个 token（类 LONGER）
- NS conditioner：每个 NS-token 在进入 unified attention 前先接收 mean-pooled 序列初始化

**核心 classes（model.py）**：`OneTransMixedAttn`、`OneTransMixedFFN`、`OneTransNSConditioner`、`OneTransBlock`

**集成方式**：`PCVRHyFormer` 新增 `use_onetrans=True` 分支，`--use_onetrans` / `--onetrans_top_k` CLI flag，`infer.py` 同步更新。

---

## Proxy 三轮调试

### 公共配置
- `valid_split_strategy=time`, `valid_ratio=0.05`
- `limit_train_rgs=128`, `limit_valid_rgs=32`, `max_train_steps=1500`
- `batch_size=128`, `num_hyformer_blocks=2`, `onetrans_top_k=50`

| 版本 | 主要改动 | Proxy AUC | LogLoss | 问题/结论 |
|---|---|---|---|---|
| v1（`onetrans-proxy-swiglu-vr005-rg128x32-s1500`） | 原始实现，S-S 下三角因果 mask | 0.823635 | 0.24914 | S-S 因果方向错误（域间无时序意义） |
| v2（`onetrans-v2-proxy-swiglu-vr005-rg128x32-s1500`） | 修复：S-S 改为双向全注意力，NS-NS 保留因果 | 0.823355 | 0.24928 | mask 修复但 AUC 几乎不变 |
| v3（`onetrans-v3-proxy-swiglu-vr005-rg128x32-s1500`） | + NS conditioner（每 NS-token 接收 mean-pooled 域序列初始化） | 0.823194 | 0.24933 | 三次 proxy 一致落在 0.823，说明 1500 steps ≈ 0.72 epoch 对 OneTrans 不公平 |

**三次 proxy 均约 0.823**，远低于 EXP-008 P0-B swiglu 同规格 baseline（0.836734）。

**根本原因**：OneTrans NS-token 必须通过梯度学习 attend S-tokens 的模式，缺乏 HyFormer QueryGenerator 的 mean-pooled 显式先验。0.72 epoch / 266k 行远不足以收敛。**Proxy 对 OneTrans 信号失真**，不能作为决策依据，必须看 full train。

---

## Full Train 结果

### v2 full train（无 NS conditioner）

**Job Name**: `onetrans-full-swiglu-vr005-bs128`  
**配置**: `use_onetrans=True`, `onetrans_top_k=50`, `num_hyformer_blocks=2`  
**Dense params**: 1,616,705（vs HyFormer 2.5M）

| Epoch | AUC | 趋势 |
|---|---|---|
| E2 | ~0.848 | 在跑 |
| E13（最高） | **0.848712** | 14 个 epoch 后仍停在 0.848，架构不 work |

**结论**：v2 无 NS conditioner，14 epoch 后 best 0.848，远低于 HyFormer reference 0.865884（-0.017）。**v2 已停止，不 evaluation。** NS conditioner 是 OneTrans 有效工作的必要条件。

### v3 full train（with NS conditioner）

**Job Name**: `onetrans-v3-full-swiglu-vr005-bs128`  
**配置**: `use_onetrans=True`, `onetrans_top_k=50`, `num_hyformer_blocks=2`, NS conditioner active  
**Dense params**: 1,932,865

| Epoch | AUC | 趋势 |
|---|---|---|
| ~E1~E5 | 逐步提升中 | — |
| E10（最高） | **0.859408** | 仍在运行 |
| E11 | 运行中 | 待回流 |

**v3 收敛曲线** vs HyFormer（参考）：

```
HyFormer: E1 0.861 → E2 0.863 → E4 0.865* → plateau
OneTrans v3: ... E10 0.859408 → E11 ??? (in progress)
```

OneTrans v3 收敛更慢但还在提升。NS conditioner 的作用非常显著（v2 0.848 vs v3 0.859，+0.010）。

**当前状态**：v3 full train 仍在运行，等 E11~E15 收敛。无论最终 local AUC，**都应 evaluation 一次**——OneTrans 与 HyFormer 架构不同，local→leaderboard gap 可能更小。

---

## Findings

1. **proxy 对 OneTrans 不公平**：1500 steps ≈ 0.72 epoch，架构需要多 epoch 学习注意力模式，proxy AUC ~0.823 不代表 full train 结果

2. **NS conditioner 是关键**：v2（无）0.848 vs v3（有）0.859，full train 差 +0.010 AUC；proxy 阶段看不出来（均 0.823）

3. **S-S 双向 mask 无明显影响**：v1 因果 0.8236 vs v2 双向 0.8234，proxy 级别差异可忽略，但双向更符合语义

4. **OneTrans 收敛比 HyFormer 慢**：HyFormer E4 就达到最佳，OneTrans v3 到 E10 还在提升

5. **P0 特征工程路线正式关闭**（结合 EXP-009）：P0-A/B/C/组合 6 轮全量训练全部低于 reference，差值 -0.0007~-0.0009，系统性失效

---

## 待回填

- [ ] OneTrans v3 full train 完整 `[summary]`（v3 early stop 后回填）
- [ ] OneTrans v3 evaluation 结果（evaluation 后回填）
