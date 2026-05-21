# Flight Reading Package · 2026-05-22 出差论文精读包

> **用途**：16 小时飞行离线精读。markdown 直接用 iPad 阅读器打开（iA Writer / Taio / Marked 2 / Obsidian / GoodNotes PDF+md），论文用原生 PDF 查看。
> **最近更新**：2026-05-21（新增 TAAC final sprint 航班 primer，面向 2026-05-22 西班牙飞旧金山）
> **总体量**：markdown 讲义 + 7 份论文 PDF。航班上优先读 13 / 12 / 11，不必从论文原文开始。

---

## 阅读清单（按顺序）

### A. 导航与指南（markdown，iPad 原生阅读）

| # | 文件 | 用途 |
|---|---|---|
| 0a | [`../../10-research/flight-reading-list-2026-05-06.md`](../../10-research/flight-reading-list-2026-05-06.md) | **阅读总指南（论文为主版）** — 原版，按论文顺序读 |
| 0a′ | [`../../10-research/flight-reading-list-blog-video-2026-05-06.md`](../../10-research/flight-reading-list-blog-video-2026-05-06.md) | **阅读总指南（blog+视频为主版）** ⭐ — 新版，针对"论文读不懂"诊断 |
| 0b | [`../../00-context/model-py-section-map.md`](../../00-context/model-py-section-map.md) | **代码对照地图** — 读完论文翻到哪行代码 |
| 08 | [`08-model-py-key-sections.md`](08-model-py-key-sections.md) | **仓库代码精读** — 18 个关键类 + 每段读前提示 + Python 代码块 |
| 10 | [`10-background-primer.md`](10-background-primer.md) ⭐ | **广告 + LLM 背景速通** — 30 分钟补齐广告/LLM 常识骨架 |
| 11 | [`11-paper-roadmap.md`](11-paper-roadmap.md) ⭐ | **7 篇论文演进地图** — 理清 HyFormer/RankMixer/OneTrans 三者关系 |
| 12 | [`12-code-qa.md`](12-code-qa.md) ⭐⭐ | **代码反问式讲义** — 10 个关键 class 的中文口语讲解（最核心） |
| 13 | [`13-taac-final-sprint-primer-2026-05-22.md`](13-taac-final-sprint-primer-2026-05-22.md) ⭐⭐⭐ | **TAAC 冲刺航班版** — 当前赛题理解、验证体系、M145-M158 判读和明天 eval 决策 |

### B. 论文 PDF（arxiv 原文，iPad 任何 PDF 阅读器）

| # | 文件 | 页数 | 说明 |
|---|---|---|---|
| 1 | `01-Attention-Is-All-You-Need.pdf` | 15 | 论文 1 — 地基（Vaswani et al., 2017） |
| 2 | `02-RoFormer-RoPE.pdf` | 15 | 论文 2 — RoPE（Su et al., 2021） |
| 3 | `03-HyFormer.pdf` | 7 | 论文 3 — 赛题核心（抖音，arxiv 2601.12681） |
| 4 | `04-RankMixer.pdf` | 14 | 论文 4 — RankMixer（arxiv 2507.15551） |
| 5 | `05-OneTrans.pdf` | 12 | 论文 5 — OneTrans（WWW 2026，arxiv 2510.26104） |
| 6 | `06-ESMM.pdf` | 4 | 论文 6 — 广告 pCVR 开山（Alibaba 2018） |
| 7 | `07-LONGER.pdf` | 12 | 论文 7（辅助）— 长序列建模 |

---

## iPad 阅读推荐

- **2026-05-22 航班建议顺序**：先读 `13-taac-final-sprint-primer-2026-05-22.md`，再读 `12-code-qa.md`，最后读 `11-paper-roadmap.md`。论文 PDF 当字典查，不必从第一页硬啃。
- **markdown 阅读器**：
  - [iA Writer](https://ia.net/writer)：最适合长文，字体 IBM Plex 系列，中英混排舒适，支持 dark mode
  - [Taio](https://taio.app/)：免费，支持代码 syntax highlight
  - [Marked 2](https://marked2app.com/)：配 dropbox/iCloud 同步本仓库自动实时预览
  - [Obsidian](https://obsidian.md/)：支持双链跳转（`08-model-py-key-sections.md` 的 TOC 锚点会直接工作）
- **PDF 阅读器**：GoodNotes / PDF Expert / Notability（可以在论文 PDF 上直接画 attention 矩阵图）
- **离线同步**：出发前把本目录 + `docs/10-research/` + `docs/00-context/` 三个目录同步到 iPad（iCloud Drive 或 Working Copy）

---

## 文件再生

### `08-model-py-key-sections.md`（从 `baseline/model.py` 抽取）

```bash
python tools/build_model_snippets_md.py
```

无第三方依赖，标准库 Python 即可。

### 论文 01-07（从 arxiv 下载）

```bash
cd docs/99-raw/flight-reading
curl -sL -o 01-Attention-Is-All-You-Need.pdf "https://arxiv.org/pdf/1706.03762.pdf"
curl -sL -o 02-RoFormer-RoPE.pdf           "https://arxiv.org/pdf/2104.09864.pdf"
curl -sL -o 03-HyFormer.pdf                 "https://arxiv.org/pdf/2601.12681.pdf"
curl -sL -o 04-RankMixer.pdf                "https://arxiv.org/pdf/2507.15551.pdf"
curl -sL -o 05-OneTrans.pdf                 "https://arxiv.org/pdf/2510.26104.pdf"
curl -sL -o 06-ESMM.pdf                     "https://arxiv.org/pdf/1804.07931.pdf"
curl -sL -o 07-LONGER.pdf                   "https://arxiv.org/pdf/2505.04421.pdf"
```

---

## 历史变更

- **2026-05-04**：首次生成（reportlab 出 PDF 版本）
- **2026-05-05 上午**：换 Pandoc + Typst 出 A5 PDF（`flight-pdfs/` → `flight-reading/`，代码段带语法高亮）
- **2026-05-05 下午**：放弃 PDF 流水线，回归 markdown；iPad markdown 阅读器本身排版已足够好，避免在中英文字体、分页、emoji 渲染上重复造轮子
- **2026-05-06（飞行中增补）**：新增 10/11/12 三份讲义 + blog+视频版阅读计划。起因：用户反馈"论文读不懂"。对症下药：先建背景骨架，再读代码讲义，论文作为字典而非主读物
- **2026-05-21（飞美国前增补）**：新增 13 TAAC final sprint primer，聚焦当前 public SOTA、验证体系、M145-M158 队列和 2026-05-22 落地后的 eval 决策。
