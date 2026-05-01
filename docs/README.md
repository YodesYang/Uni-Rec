# TencentUniRec 知识库（90 秒入门）

> TAAC 2026（腾讯广告算法大赛）参赛仓库的内部知识库。
> **每次新会话开始 + 任何对仓库不熟悉的协作者（含 AI agent），先读完本文件 90 秒**。

---

## 解决问题查这里（按高频问题排序）

| 你想 | 看这个 |
|------|--------|
| 知道当前在做什么 / 下一步做什么 | [`20-tasks/active.md`](./20-tasks/active.md) |
| 看赛题规则 / 时间表 / 红线 | [`00-context/competition.md`](./00-context/competition.md) |
| 看数据结构（120 列怎么分） | [`00-context/data-schema.md`](./00-context/data-schema.md) |
| 看 baseline 代码地图 | [`00-context/codebase-map.md`](./00-context/codebase-map.md) |
| 看平台手册（Tencent Angel） | [`00-context/platform-angel.md`](./00-context/platform-angel.md) |
| 看「为什么这么改 baseline」（决策推理） | [`30-decisions/`](./30-decisions/) |
| 看实验数字 / 历次 platform job 结果 | [`40-experiments/`](./40-experiments/) |
| 看技术报告草稿（KDD 答辩用） | [`50-paper/draft-v0.md`](./50-paper/draft-v0.md) |
| 看相关论文 | [`10-research/prior-art.md`](./10-research/prior-art.md) |
| 看社区情报（官方答复 / 选手讨论） | [`10-research/community-intel.md`](./10-research/community-intel.md) |
| 看候选改进想法（还没启动的） | [`10-research/ideas-backlog.md`](./10-research/ideas-backlog.md) |

---

## 知识库结构

```
docs/
│
├── README.md                              # 本文件（入口）
│
├── 00-context/                            # 长期不变（赛季内稳定）
│   ├── competition.md                     # 赛题 / 时间表 / 红线
│   ├── data-schema.md                     # 数据 120 列结构
│   ├── codebase-map.md                    # baseline/ 代码地图
│   └── platform-angel.md                  # Tencent Angel 平台手册
│
├── 10-research/                           # 知识沉淀（偶尔写）
│   ├── prior-art.md                       # 论文清单
│   ├── community-intel.md                 # 社区情报（官方答复抽取）
│   └── ideas-backlog.md                   # 候选改进想法（启动后升 task）
│
├── 20-tasks/                              # 任务跟踪（高频写入）
│   ├── README.md                          # 7 步审视清单 + 状态约定 + 编号约定
│   ├── active.md                          # 进行中 / 阻塞 / 待启动 任务
│   └── archive.md                         # 已完成 [x] 任务
│
├── 30-decisions/                          # 架构决策（永不删）
│   ├── README.md                          # ADR 模板 + 索引
│   ├── ADR-001-time-aware-valid-split.md  # T8 决策推理
│   └── ADR-002-slice-diagnostics.md       # T10 决策推理
│
├── 40-experiments/                        # 实验日志（结果只读）
│   ├── README.md                          # 实验编号约定
│   ├── EXP-001-baseline-cpu-demo.md       # 本地 demo baseline
│   └── EXP-002-platform-baseline-job.md   # 平台首次 job
│
├── 50-paper/                              # 技术报告（KDD 答辩用）
│   └── draft-v0.md                        # T6 草稿（迭代覆盖）
│
└── 99-raw/                                # 原始数据（备份用）
    ├── community-snapshots/               # 乐享社区抓取快照（带 diff 脚本）
    └── platform-pdf/                      # Tencent Angel 原始 PDF
```

---

## 三套独立编号体系

| 前缀 | 含义 | 在哪里 | 写入特征 |
|------|------|--------|----------|
| `T<N>` | Task | `20-tasks/active.md` 或 `archive.md` | 频繁 status 变更 |
| `ADR-<NNN>` | Architecture Decision Record | `30-decisions/` | **永不删，决策后只读** |
| `EXP-<NNN>` | Experiment | `40-experiments/` | **结果只读**，一次 job 一个 |

**升级路径**：

```
20-tasks/active.md  ─── 升决策推理 ──→  30-decisions/ADR-NNN-*.md
                  └─── 升实验数字 ──→  40-experiments/EXP-NNN-*.md
                  └─── 完成归档 ────→  20-tasks/archive.md
```

详细规则见 [`20-tasks/README.md`](./20-tasks/README.md)。

---

## 写入纪律

| 文件 | 写入频率 | 谁能改 |
|------|----------|--------|
| `00-context/*` | 极低（赛事规则变更才改） | 任何接触者 |
| `10-research/{prior-art, ideas-backlog}.md` | 中 | 任何接触者 |
| `10-research/community-intel.md` | 重新抓取时（用 `99-raw/community-snapshots/diff_with_previous.sh`）| 抓取者 |
| `20-tasks/README.md` | 极低（流程变更才改） | 任何接触者 |
| `20-tasks/active.md` | 高（每个工作日可能动） | 任何接触者，但**必须跑 7 步审视清单** |
| `20-tasks/archive.md` | 中（task 完成时一次性追加） | 完成 task 的人 |
| `30-decisions/ADR-*` | 永不修改（新决策写新 ADR）| 决策者 |
| `40-experiments/EXP-*` | 永不修改（实验跑完一次性写）| 跑实验的人 |
| `50-paper/draft-v*.md` | 中（迭代覆盖；文件名升 v0/v1/v2）| T6 owner |

---

## 当前赛事状态（一行快速看）

距初赛 ddl **25 天**（5/25）· 当前最佳 Leaderboard **0.84381**（swiglu encoder）· 社区 SOTA **0.85809**（差 +0.0143）· 评估额度每天 **4 次** / 资源 **4** 配额（榜单重置补偿）· T7 / T9 已结案 → 当前最高 ROI 动作：**重新发布 transformer step=38050 ckpt**（30 秒，+0.002 预期）+ 新一轮 platform job 验证 T8 时间切。

详见 [`20-tasks/active.md`](./20-tasks/active.md) 顶部「最近一次审视」一行。
