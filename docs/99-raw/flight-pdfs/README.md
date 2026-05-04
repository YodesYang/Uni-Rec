# Flight Reading Package · 2026-05-06 出差论文精读包

> **用途**：16 小时飞行离线精读，拎去打印店一次搞定。
> **生成时间**：2026-05-04
> **总页数估算**：约 130-150 页（A4 双面打印 ≈ 70-80 张纸）

---

## 打印清单（按阅读顺序）

| # | 文件名 | 页数约 | 用途 | 推荐打印 |
|---|---|---|---|---|
| 0a | `00-flight-reading-list.pdf` | 6 | **阅读总指南**（时间分配 + 每篇论文的精读问题 + 验收标准） | 彩色双面 |
| 0b | `00-model-py-section-map.pdf` | 6 | **代码对照地图**（读完论文翻到哪行代码） | 彩色双面 |
| 1 | `01-Attention-Is-All-You-Need.pdf` | 15 | 论文 1 — 地基 | 双面 |
| 2 | `02-RoFormer-RoPE.pdf` | 15 | 论文 2 — RoPE | 双面 |
| 3 | `03-HyFormer.pdf` | 7 | 论文 3 — 赛题核心（抖音团队，arxiv 2601.12681） | 双面 |
| 4 | `04-RankMixer.pdf` | 14 | 论文 4 — RankMixer（arxiv 2507.15551） | 双面 |
| 5 | `05-OneTrans.pdf` | 12 | 论文 5 — OneTrans（WWW 2026，arxiv 2510.26104）| 双面 |
| 6 | `06-ESMM.pdf` | 4 | 论文 6 — 广告 pCVR 业务开山 | 双面 |
| 7 | `07-LONGER.pdf` | 12 | 论文 7（辅助）— 长序列建模 | 如有余量 |
| 8 | `08-model-py-key-sections.pdf` | 38 | **仓库代码打印版**（17 个关键类 + 每段读前提示） | **必须彩色**（蓝色标注要看清） |

总约 **130 页 / A4 双面 ≈ 65 张纸**

---

## 推荐打印规格

- **纸张**：A4
- **颜色**：彩色（`00-*` 和 `08-*` 的蓝色标注是设计过的视觉引导）
- **装订**：胶装或活页夹分装成三叠：
  - **叠 1 · 导航**：0a + 0b（随时翻阅）
  - **叠 2 · 论文**：1 → 2 → 3 → 4 → 5 → 6（按阅读顺序）
  - **叠 3 · 代码**：08-model-py-key-sections.pdf（单独一本，配论文 side-by-side）
- **辅助工具**：准备 **两色笔**（黑+蓝）+ **10 张 A4 白纸**（用来画 attention 矩阵、reshape 图、mask 图）

---

## 验证记录

| PDF | 验证动作 | 结果 |
|---|---|---|
| 01 | curl 下载 + Title 检查 | ✅ "Attention Is All You Need" |
| 02 | curl 下载 + Title 检查 | ✅ "RoFormer: Enhanced Transformer with Rotary Position Embedding" |
| 03 | curl 下载 + Title 检查 | ✅ "HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction"（抖音团队 v2） |
| 04 | curl 下载 + Title 检查 | ✅ "RankMixer: Scaling Up Ranking Models in Industrial Recommenders" |
| 05 | curl 下载 + Title 检查 | ✅ "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender" |
| 06 | curl 下载 + Title 检查 | ✅ ESMM |
| 07 | curl 下载 + Title 检查 | ✅ "LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders" |
| 08 | reportlab 生成 + pypdf 解析抽检 | ✅ 38 页，中文字体嵌入 (STHeiti)，代码带行号 |
| 00a/00b | reportlab 生成 + pypdf 解析抽检 | ✅ 中文 + 表格正常渲染 |

---

## 文件再生脚本（如需重新生成 08 / 00a / 00b）

```bash
# 1. 重新生成 model.py 关键段 PDF
/opt/anaconda3/envs/tencent-unirec/bin/python tools/build_model_snippets_pdf.py

# 2. 重新生成两份 md → pdf
/opt/anaconda3/envs/tencent-unirec/bin/python tools/build_md_to_pdf.py
```

论文 01-07 直接 curl 重新下载：
```bash
curl -sL -o 01-Attention-Is-All-You-Need.pdf "https://arxiv.org/pdf/1706.03762.pdf"
curl -sL -o 02-RoFormer-RoPE.pdf "https://arxiv.org/pdf/2104.09864.pdf"
curl -sL -o 03-HyFormer.pdf "https://arxiv.org/pdf/2601.12681.pdf"
curl -sL -o 04-RankMixer.pdf "https://arxiv.org/pdf/2507.15551.pdf"
curl -sL -o 05-OneTrans.pdf "https://arxiv.org/pdf/2510.26104.pdf"
curl -sL -o 06-ESMM.pdf "https://arxiv.org/pdf/1804.07931.pdf"
curl -sL -o 07-LONGER.pdf "https://arxiv.org/pdf/2505.04421.pdf"
```
