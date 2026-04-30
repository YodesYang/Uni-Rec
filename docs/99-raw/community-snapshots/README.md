# 社区抓取快照（Community Snapshots）

> 乐享比赛社区的原始抓取产物。**只用作回查 / diff，不直接读**（情报抽取后的精华版在 `docs/10-research/community-intel.md`）。

---

## 文件命名约定

每次抓取按时间戳命名：

```
lexiang_questions_YYYY-MM-DD-HH-MM-SS.{md,json}
```

`.md` 是给人读的；`.json` 是给 `diff_with_previous.sh` 自动 diff 用的。

`.json` 在 `.gitignore` 里被排除（避免仓库膨胀），抓完后**只在本地保留**。如需共享原始 JSON → 单独处理（不进 git）。

---

## 抓取工具

工具：`tools/lexiang_questions_crawl`（浏览器 console 粘贴即跑）

URL：https://lexiangla.com/questions?company_from=9cfe03d211a311f1a8588ad12dbe613c

抓完两个文件会自动下载到本地 Downloads，**手工 mv 到本目录**：

```bash
mv ~/Downloads/lexiang_questions_2026-MM-DD-HH-MM-SS.{md,json} \
   docs/99-raw/community-snapshots/
```

---

## 快照对比（增量发现新情报）

新抓取后 → 跑 `diff_with_previous.sh` 与上一次快照对比，**只看新增 / 状态变化的题**：

```bash
cd docs/99-raw/community-snapshots/
./diff_with_previous.sh
```

输出：
- 新增的题（按 organization 字段过滤 Administrators 后高亮）
- 已有题的新答复（views / ans 数变化）
- 删除的题（罕见）

把输出里有价值的部分手工合并到 `docs/10-research/community-intel.md`。

---

## 当前快照索引

| 时间戳 | 题数 | Admin/Official 答复数 | 沉淀位置 |
|--------|------|----------------------|----------|
| 2026-04-30 06:46:51 | 141 | 26 | `docs/10-research/community-intel.md` |
