---
name: ai-readme-workspace
description: 基于子仓库已有的 ai-readme，生成多仓库项目群关系图和协作文档；当需要为多仓库项目群生成全局视角文档时使用
---

# 多仓库项目群 AI README 生成器

## 概述

本 Skill 用于在**父目录**下，基于各子仓库已有的 ai-readme 文档，自动生成项目群关系图和跨仓库协作文档。

> ⚠️ **模型要求**：本 Skill 涉及跨仓库分析和文档生成，**强烈推荐使用 Claude Opus 4.5 或更高版本模型**执行。

## 生成目标

本 Skill 生成的文档是**面向人和 AI 共同使用的项目群介绍手册**——新成员用来快速了解仓库间关系和协作模式，AI 编程助手用来评估跨仓库变更的影响范围。生成时**详尽优先**（宁可冗余不遗漏，AI 需要充分上下文才能给出高质量建议），优先用表格和 Mermaid 图等结构化格式呈现。

## 激活时机

- 用户需要为包含多个子仓库的父目录生成项目群文档
- 用户运行 `ai-readme-workspace` 命令

## 前置条件

**必须先在各子仓库中执行 `ai-readme` Skill**，生成单仓库的 ai-readme 文档。本 Skill 会读取这些文档，整合生成项目群级别的关系图。

## 核心价值

基于各子仓库已分析的模块信息，整理出：

- **仓库间的依赖关系图**
- **API 调用链路**
- **数据流转路径**
- **变更影响范围**

让 AI 能理解 **"这些仓库之间是什么关系？改了 A 会影响 B 吗？"**

## 输出目录结构

在**父目录**（包含多个子仓库的目录）下执行命令，生成：

```
parent-dir/
├── .cursor/
│   └── rules/
│       └── ai-readme/
│           ├── RULE.mdc               # 总览入口 + 生成进度
│           ├── 项目群总览.mdc          # 项目群包含什么、各仓库职责
│           ├── 系统拓扑.mdc            # 调用关系、数据流向
│           ├── 接口契约.mdc            # 跨仓库接口定义、共享类型
│           ├── 变更影响.mdc            # 改了这里还要改哪里（核心！）
│           └── 联调指南.mdc            # 本地联调开发方法
├── repo-a/
│   └── .cursor/rules/ai-readme/       # 已有 ai-readme（前置条件）
├── repo-b/
│   └── .cursor/rules/ai-readme/
└── repo-c/
    └── .cursor/rules/ai-readme/
```

## 执行流程总览

按以下 5 个阶段顺序执行：

| 阶段 | 内容                                             | 详细参考                                                                        |
| :--: | ------------------------------------------------ | ------------------------------------------------------------------------------- |
|  1   | 扫描父目录，识别子仓库，读取各子仓库的 ai-readme | [references/scanning-subrepos.md](./references/scanning-subrepos.md)            |
|  2   | 制定 TODO LIST（列出待生成/更新的文档清单）      | [references/generated-docs-spec.md](./references/generated-docs-spec.md)        |
|  3   | 基于读取的信息，生成项目群关系文档（5 个）       | [references/generated-docs-spec.md](./references/generated-docs-spec.md)        |
|  4   | 汇总统计，校验质量                               | [references/validation-and-update.md](./references/validation-and-update.md) §3 |
|  5   | 更新子仓库 ai-readme，添加项目群链接             | [references/validation-and-update.md](./references/validation-and-update.md) §4 |

## 关键约束摘要

### 边生成边更新（强制）

**每生成一个文档，必须立即更新 `RULE.mdc` 快速导航表格中对应行的「状态」列！**

```
✅ 正确：写入 项目群总览.mdc → 更新 RULE.mdc 状态列 [x] → 写入 系统拓扑.mdc → ...
❌ 错误：写入所有文档 → 最后统一更新 RULE.mdc
```

### 断点续写状态符号

| 符号  | 含义                 |
| ----- | -------------------- |
| `[ ]` | 未完成，需要生成     |
| `[x]` | 已完成（追加时间戳） |
| `[?]` | 待确认（中间状态）   |

### 禁止事项

- 禁止使用 Emoji 表情符号（`⚠️` 除外）
- Mermaid 节点 ID 禁止使用连字符，使用下划线或驼峰命名

### Mermaid 语法关键规则摘要

- 方向标识符全大写：`graph TB` / `graph LR`
- 节点 ID 只用字母/数字/下划线，不用连字符
- 箭头前后必须有空格：`A --> B`
- 特殊字符加引号：`A["用户:登录"]`

## 验收标准

1. **子仓库识别**：正确识别父目录下的所有代码仓库
2. **ai-readme 读取**：优先从子仓库已有的 ai-readme 提取信息
3. **关系图清晰**：AI 能从文档中理解"这些仓库之间是什么关系"
4. **变更影响明确**：AI 能从 变更影响.mdc 知道"改了 A 要改什么"
5. **每个文档**：包含 frontmatter + 至少 1 个 Mermaid 关系图
6. **RULE.mdc**：description 必须包含"必读入口"字样；必须使用 XML 标签结构化；快速导航表格包含「状态」列用于进度追踪
7. **进度更新**：每生成一个文档必须立即更新 RULE.mdc 进度状态

## 文件引用索引

| 文件                                                                         | 内容说明                               |
| ---------------------------------------------------------------------------- | -------------------------------------- |
| [references/scanning-subrepos.md](./references/scanning-subrepos.md)         | 阶段1：子仓库识别和 ai-readme 读取     |
| [references/generated-docs-spec.md](./references/generated-docs-spec.md)     | 阶段2：文件写入策略 + 5 个文档生成规格 |
| [references/validation-and-update.md](./references/validation-and-update.md) | 阶段3+4：汇总校验 + 更新子仓库         |
| [assets/rule-entry-template.md](./assets/rule-entry-template.md)             | 项目群 RULE.mdc 的完整 XML 模板        |
| [assets/doc-templates.md](./assets/doc-templates.md)                         | 5 个文档的模板内容                     |
