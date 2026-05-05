# Flight Reading Package · 2026-05-06 出差论文精读包

> **用途**：16 小时飞行离线精读，既可打印也可在 iPad Pro 上阅读。
> **生成时间**：2026-05-05（Typst 版重排）
> **纸张规格**：`00-*` 和 `08-*` 使用 **A5**（148×210mm，iPad Pro 11 寸竖屏最适尺寸，也可打印装订成口袋书）；论文 `01-07` 保持原 arxiv A4。
> **总页数**：A5 约 78 页，A4 论文约 79 页，合计 ≈ 157 页

---

## 打印清单（按阅读顺序）

| # | 文件名 | 页数 | 纸张 | 用途 |
|---|---|---|---|---|
| 0a | `00-flight-reading-list.pdf` | 10 | A5 | **阅读总指南**（时间分配 + 每篇论文的精读问题 + 验收标准） |
| 0b | `00-model-py-section-map.pdf` | 10 | A5 | **代码对照地图**（读完论文翻到哪行代码） |
| 1 | `01-Attention-Is-All-You-Need.pdf` | 15 | A4 | 论文 1 — 地基 |
| 2 | `02-RoFormer-RoPE.pdf` | 15 | A4 | 论文 2 — RoPE |
| 3 | `03-HyFormer.pdf` | 7 | A4 | 论文 3 — 赛题核心（抖音团队，arxiv 2601.12681） |
| 4 | `04-RankMixer.pdf` | 14 | A4 | 论文 4 — RankMixer（arxiv 2507.15551） |
| 5 | `05-OneTrans.pdf` | 12 | A4 | 论文 5 — OneTrans（WWW 2026，arxiv 2510.26104） |
| 6 | `06-ESMM.pdf` | 4 | A4 | 论文 6 — 广告 pCVR 业务开山 |
| 7 | `07-LONGER.pdf` | 12 | A4 | 论文 7（辅助）— 长序列建模 |
| 8 | `08-model-py-key-sections.pdf` | 58 | A5 | **仓库代码打印版**（18 个关键类 + 每段读前提示，代码带语法高亮） |

---

## 纸质打印推荐

- **A5 部分（0a / 0b / 08）**：A5 彩色双面，或 A4 纸 2-up 缩印（每面 2 页 A5）。
  - 2-up 打印命令示例（macOS Preview）：File → Print → Layout: 2 pages per sheet
- **A4 部分（1-7）**：原生 A4 彩色双面。
- **装订**：推荐活页夹分 3 叠：
  - **叠 1 · 导航**：0a + 0b（随时翻阅）
  - **叠 2 · 论文**：1 → 2 → 3 → 4 → 5 → 6（按阅读顺序）
  - **叠 3 · 代码**：08（单独一本，配论文 side-by-side）
- **辅助工具**：准备 **两色笔**（黑+蓝）+ **10 张 A4 白纸**

## iPad Pro 阅读推荐

- **0a / 0b / 08** 已按 A5 + 中英文混排优化字体（PingFang SC + Charter），在 11 寸竖屏显示字号舒适。
- 字体 + 排版参考主流技术书（O'Reilly / Manning / 人民邮电图灵系列）设计。
- 代码块自带 Python 语法高亮（紫色关键字 / 蓝色函数 / 红色字符串）。
- 建议阅读 App：**GoodNotes / Notability**（可直接在 PDF 上画 attention 矩阵图）。

---

## 文件再生脚本

整套 A5 PDF 通过 Pandoc + Typst 流水线生成：

```bash
# 源：tools/flight-reading-template.typ + tools/pandoc-typst-template.typ

# 重新生成所有 A5 PDF（00a / 00b / 08）
python tools/build_md_to_pdf.py      # 00a + 00b
python tools/build_model_snippets_pdf.py  # 08

# 依赖：
brew install pandoc typst
```

论文 01-07 直接从 arxiv 重新下载：
```bash
curl -sL -o 01-Attention-Is-All-You-Need.pdf "https://arxiv.org/pdf/1706.03762.pdf"
curl -sL -o 02-RoFormer-RoPE.pdf "https://arxiv.org/pdf/2104.09864.pdf"
curl -sL -o 03-HyFormer.pdf "https://arxiv.org/pdf/2601.12681.pdf"
curl -sL -o 04-RankMixer.pdf "https://arxiv.org/pdf/2507.15551.pdf"
curl -sL -o 05-OneTrans.pdf "https://arxiv.org/pdf/2510.26104.pdf"
curl -sL -o 06-ESMM.pdf "https://arxiv.org/pdf/1804.07931.pdf"
curl -sL -o 07-LONGER.pdf "https://arxiv.org/pdf/2505.04421.pdf"
```
