# 改进想法 Backlog（Ideas Backlog）

> 来源：原 `knowledge_base.txt §6` 「关键经验 / 可改进方向」清单。
> **本文件只装"还没启动"的候选**。任何条目一旦决定启动 → **必须升为 `docs/20-tasks/active.md` 的一个 `T<N>` task**，并把本文件对应条目划掉或加上「升 T<N>」标注。
> 这是为了避免「TODO 双源」：active.md 是真相，本文件是预备役。

---

## 6.1 数据 / 特征侧（性价比高）

- [ ] 复核 label_type 全枚举值与样本占比：是否还有其它 label_type 是「弱正样本」可参与 multi-task。
  > 状态：已被 ADR-002 切片诊断工具覆盖一半 — 工具已打 `[diag] slice by label_type` 表，等平台 job 数据决定下一步。
- [ ] 针对 list 类特征做 stats encoding：长度、unique 数、最近一次时间差。
- [ ] 跨 domain 特征 alignment：同一 fid 在 user_int / user_dense 同时出现时，联合喂入同一组（baseline 默认是分开的）。
- [ ] 为序列 timestamp 设计更细的桶：当前 65 桶可能在「近 1 小时」内过粗。
- [ ] 检查 OOB id 比例（baseline `dump_oob_stats`）；OOB 比例高 → schema vocab 可能不准。
- [ ] 高基数 sparse 特征启用 `emb_skip_threshold`（默认 1e6）控制显存占用。

## 6.2 模型结构（中等收益）

- [ ] 把 `seq_encoder_type` 从 `transformer` 切到 `longer` 时，确认 top_k 与 `seq_max_lens` 配比（top_k 一般取 max_len 的 10%~25%）。
- [ ] `num_queries` / `num_hyformer_blocks` 的 scaling：参考 HyFormer Figure 3，AUC ~ FLOPs 满足 power-law；先 sweep ×2、×4 再决定。
- [ ] **RoPE**：当前默认 `use_rope=False`，开启对长序列通常 +0.1% AUC（小心 `head_dim` 要够大才不会过度旋转）。
- [ ] **RankMixer mode**：`full` > `ffn_only` > `none`（一般情况下）；`full` 受 `D % T == 0` 约束，crash 时优先放宽 `num_queries` / `num_ns_tokens`。
- [ ] **多任务化**：增加一个 `label_type==1` 的 CTR 头，主头依然是 PCVR，auxiliary loss 权重从 0.1 起 sweep。
  > 状态：与 ADR-002 切片表的 `label_type` 分支决策路径联动（若任务是 click-conditional CVR → 此项升级 P1）。

## 6.3 训练 / 优化（小收益但稳）

- [ ] `sparse_lr` 与 `dense_lr` 解耦时，sparse 通常 0.01~0.1 的 Adagrad，dense 1e-4 ~ 5e-4 AdamW。
- [ ] `reinit_cardinality_threshold > 0` + `reinit_sparse_after_epoch=1` 复现 MultiEpoch 效果，长期训练防过拟合。
- [ ] 早停 `patience=5` 偏紧，长尾收益场景可调到 8~10。
- [ ] **Focal Loss**：正样本极稀疏（CVR 普遍 < 5%）时尝试 `alpha=0.25, gamma=2.0`。
- [ ] **Mixed precision（bf16）+ torch.compile**：吞吐 ~1.3× 起步。
- [ ] `eval_every_n_steps`：复赛 10× 数据时务必启用 step-level 验证，避免一个 epoch 过长。

## 6.4 工程 / 平台（避免踩坑）

- [ ] Taiji 提交格式与本地一致即可（参考官方知识库）；CKPT 自带 schema.json + ns_groups.json 已在 trainer 中实现。
- [ ] **DataLoader num_workers**：单机 8~16 较稳；超过 16 容易 `/dev/shm` 爆（baseline 已切 `file_system` 共享策略）。
- [ ] 复赛免费 GPU 8 卡：用 DDP + sparse Embedding 单卡放置即可，不需要 ZeRO-3。
- [ ] `schema.json` 字段建议加版本号；当官方 demo 数据再次更新时方便回归测试。
