# 赛题 / 时间表 / 红线（Competition Context）

> **稳定文档**（赛季内一般不变）。来源：原 `knowledge_base.txt §0/§1/§2/§7`。
> 任何赛事时间或规则变更，必须**先**在本文件落定，再扩散到 `docs/20-tasks/active.md` 等下游。

---

## 速查链接（必收藏）

### 一手信息源（高优先级）

- 官方知识库（赛题 / 数据 / 规则 / 答疑）：
  https://lexiangla.com/spaces/9588b61dd0a64e22a1b8bb5ce9b19f8d?company_from=9cfe03d211a311f1a8588ad12dbe613c
- 赛事官网：https://algo.qq.com/
- 训练 / 联调平台（太极 Taiji）：https://taiji.algo.qq.com/
- 比赛规则：https://algo.qq.com/rules
- 官方 HuggingFace：https://huggingface.co/TAAC2026
- 1k 样本 demo（schema 参考）：https://huggingface.co/datasets/TAAC2026/data_sample_1000

### 社区参考（中优先级）

- 多 baseline 参考仓库（含 HyFormer/OneTrans/RankMixer）：https://github.com/hojiahao/TAAC2026
- 另一参考仓库：https://github.com/LC1332/TAAC_2026-ref1
- HyFormer 论文：https://arxiv.org/abs/2601.12681
- OneTrans 解读（datawhale）：https://datawhalechina.github.io/fun-rec/chapter_6_scaling/5.one_trans.html
- RankMixer 解读：https://www.huaxiaozhuan.com/applications/recommendation/ctr_prediction/chapters/2025_RankMixer.html

---

## 1. 赛题核心

**主题**：面向大规模推荐的「序列建模」与「特征交互」的统一。

**任务**：
- 设计统一的 Tokenization 方案与同构、可堆叠的骨干网络（Recommendation Block）
- 在「单一架构」内同时建模用户行为序列（多域）+ 非序列多域特征
- 输出对目标广告的 pCVR（Predicted Conversion Rate，点击后转化概率）

**公式**：

```
pCVR = P( conversion = 1 | Ad, User, Context )
```

**评估指标**：AUC（ROC 曲线下面积，越接近 1 越好）。
**损失函数**：典型为 BCEWithLogits（baseline 默认），可选 Focal Loss。

**关键背景**（来自 4/30 抓社区情报，详见 `docs/10-research/community-intel.md`）：
- **测试集时间不重叠且在训练集之后**（官方 4/26 答复） → 离线-在线一致性是头等问题（驱动 ADR-001）
- **`infer.py` / `model.py` 推理时必须与训练时完全一致**（官方 4/28 答复） → 任何模型结构改动都阻塞评估
- **推理 30 分钟硬上限**，超时即被系统 kill

---

## 2. 时间表（务必盯紧）

| 日期 | 事件 |
|------|------|
| 2026-03-19 | 报名开始 + 样本数据集发布 |
| 2026-04-23 | 报名截止 |
| 2026-04-24 ~ 05-25 | 第一轮（初赛） |
| 2026-05-25 ~ 06-24 | 第二轮（复赛），决出 Top 50 |
| 2026-08-09 ~ 08-13 | KDD 2026 大会现场颁奖 |

**资源配额**：
- 初赛数据集：100+ 真实业务脱敏特征字段
- 复赛数据量级：约为初赛的 10 倍
- 复赛 Top 50 每队赠 8 张 GPU（免费）
- **平台计算配额**：每队 2 quota，**严格串行**（4/27 官方答复，已迭代）
- **评估额度**：每天 3 次，**北京时间 12:00 刷新**（早期答 AOE 15:59:59 已迭代），失败/取消的不计数

---

## 3. 红线 / 反作弊

- **严禁多账号刷榜、严禁伪造身份**
- **严禁模型集成 / 模型融合（ensemble）**，最终提交必须是单一模型架构
- **数据使用范围**：仅限官方提供的训练 / 验证 / 测试数据；不得引入外部商业数据；学术公开 dataset 引入前先发邮件咨询主办方
- **复赛三重评审**：榜单 + 现场答辩 + 代码 reproducibility，三者均为评审依据；不可复现 / 提交模型与论文不一致 → 取消资格

特殊规则（与常见 Kaggle 比赛差异）：
- **设两项创新奖**（独立于排行榜，各 USD 45,000）：
  - 统一模块创新奖（Unified Module Innovation Award）
  - Scaling Law 创新奖
- **排行榜奖金池总额约 600 万人民币**，与 KDD 2026 联合

> 任何拿不准的规则解读，先到「鹅厂知识库」群答疑或发邮件给官方对接人，**不要自行假设**。
