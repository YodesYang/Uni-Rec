# 小红书 · DECEM（工业赛道 Rank 1）公开情报快照

> **采集时间**：2026-05-08（用户手动搬运截图，Agent 整理）
> **快照来源**：小红书账号 `DECEM`（昵称"撑鱼大队长"，IP 四川，Industrial Track）
> **精华版沉淀**：`docs/10-research/community-intel.md` §"工业赛道 Rank 1 · DECEM 公开情报（2026-05-08）"

---

## 截图清单（原文 raw 保留）

| 文件 | 帖子 / 评论概要 | 关键信息 |
|---|---|---|
| `01-profile-rank1-score-852181.png` | 个人主页 + 顶部评论（好烦呀 @ DECEM） | Best Score **0.852181** @ 2026-05-06 18:47:59；评论"让我猜猜，是 global token 里序列 pool 改为 din 吗？" DECEM 答"谢谢大佬"（未否认） |
| `02-comment-din-mlp-guess.png` | 同上帖子更完整截图 | 评论区确认 DIN + MLP 路线 |
| `03-post-paper-work.png` | 帖子《我不是故意想上分的》+ 评论「大佬有开编译吗」 | DECEM 回复："你说编译，是 flash-attn 吗？我没有折腾" → 追问 "torch.compile"：DECEM "这个必须得做"；"训推都加吗？"→ DECEM "推理没加"（**训练加 compile，推理不加**） |
| `04-comment-torch-compile.png` | 帖子《TAAC 简单分享一下最近的上分过程》 | 正文关键原话："做到初赛中期，目前做了 60 多个实验，work 的也就 6 个上分点，我也是尽量一直做单变量消融，当然整个实验的过程也是靠 Claude opus。最开始先做了常规的 torch.compile + 混合精度加速训练，简单测了一些数据集，后续就是常规的时间特征，目前只 work 了两个，也是比较常规的。然后是 Tokenization 的设计，简单做了一些修复，让其更加合理化。模型这块做了两点，一个是我在前期就做了的 din + mlp，我就不在再细细的说了，之前大佬发的帖子也都有提到，还有一个那天晚上看的一篇论文的方法，这个大家就先不要问了，我还在努力搞明白他是怎么 work 的。" + "大部分我认为都是面向 baseline 一些不够完善的地方去做小修小补，也没有做好 scaling。当前很多地方都是 baseline 原始的参数，序列长度没变，LR warmup 也没加……所以说大家的上分空间还是很大的，万一我只是找到了个好的 seed 呢" |
| `05-comment-valid-087.png` | 帖子《让酒精带来的灵感继续迸发》评论区 | 评论"好奇问下序列特征长度多长呢"→ DECEM "还是 baseline 的，没变"；评论"不知道是不是数据量太少，添加 user_id or item_id or scaling 都会掉分"→ DECEM "**是的，我试了 scaling 不掉分，但也不涨分**" |
| `06-comment-seqlen-userid-scaling.png` | 更早的主页截图（Best Score 0.851688 @ 2026-05-05 11:38:44） + 评论区 | 评论"好奇这个验证集得达到多少，工业赛道和学术赛道数据集一样吗"→ DECEM "**验证集 0.87+**" + "不一样" |

---

## 为什么单独建目录

本批快照来自小红书截图，不是 `tools/lexiang_questions_crawl` 抓取的乐享 JSON，不走 `diff_with_previous.sh` 流程。单独归档便于后续人工二次核对。

## 使用守则

1. 这些是**他人公开自述**，可能夸大成功点 / 略过负向实验，**不作为本仓库决策依据**。
2. 对口的方向级启示已写进 `community-intel.md` 精华段，**读精华段即可**，除非需要核对原文。
3. 如果后续 DECEM 发新帖，沿用同一目录追加 `07-*.png`... 并在 `community-intel.md` 相应段加增补。
