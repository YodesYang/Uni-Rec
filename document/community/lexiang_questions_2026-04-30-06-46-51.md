# 乐享比赛社区问答快照

- 抓取时间: 2026-04-30T06:46:51.774Z
- 来源: https://lexiangla.com/questions?company_from=9cfe03d211a311f1a8588ad12dbe613c
- 总题数: 141

---

## (无标题)

- ID: `create`
- 链接: https://lexiangla.com/questions/create?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 如下图：30号并没提交, 但是却告知我无法进行提交评测任务

- ID: `3cf0f63c443b11f1b6893a35d2f04c13`
- 链接: https://lexiangla.com/questions/3cf0f63c443b11f1b6893a35d2f04c13?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 模型评估过程显存被别的实例抢占爆了，这是官方bug吧，再次评估跑出分数了，但是就这也算评估了两次，这合理吗，老哥们

- ID: `cb21e0ac443011f1b4ad6abcdbd0613b`
- 链接: https://lexiangla.com/questions/cb21e0ac443011f1b4ad6abcdbd0613b?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

2026-04-29 13:21:02,005 - ERROR - Inference failed: CUDA out of memory. Tried to allocate 9.00 GiB. GPU 0 has a total capacity of 19.12 GiB of which 19.12 GiB is free. Process 2052065 has 12.60 GiB memory in use. Process 1094656 has 46.96 GiB memory in use. Process 2010256 has 14.62 GiB memory in use. Process 166351 has 14.48 GiB memory in use. Process 2115634 has 2.20 GiB memory in use. Of the allocated memory 1.37 GiB is allocated by PyTorch, and 411.94 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
&nbsp;
#####跪求修一下这个评估bug，把我的评估还给我

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 训练的schema是如何获取的？如果要做特征工程的话，是否需要另外上传schema.json?

- ID: `63466e8843fb11f1b88c8e742a80aeca`
- 链接: https://lexiangla.com/questions/63466e8843fb11f1b88c8e742a80aeca?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 请问有经验的选手，目前这一届比赛的选手整体水平在历届中算是什么水平？分数都好高啊

- ID: `2366e82643ef11f1bac5825bb877947b`
- 链接: https://lexiangla.com/questions/2366e82643ef11f1bac5825bb877947b?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

请问有经验的选手，目前这一届比赛的选手整体水平在历届中算是什么水平？分数都好高啊

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 太极这个训练任务框能不能拉长点呀，太短，使用太不方便了

- ID: `5fd64d7243e711f1b1759e47a5de1825`
- 链接: https://lexiangla.com/questions/5fd64d7243e711f1b1759e47a5de1825?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

如题

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 为什么训练完成后 看不到输出的checkpoint

- ID: `51797d4c43cb11f1919d9eb605d32106`
- 链接: https://lexiangla.com/questions/51797d4c43cb11f1919d9eb605d32106?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-29 22:36:30

因为一大轮还没训练完

---

## 同一任务能调用两张GPU卡吗

- ID: `134637f443c111f184d8be5dec493471`
- 链接: https://lexiangla.com/questions/134637f443c111f184d8be5dec493471?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

这两张GPU能不能正常通信，通过何种方式通信

### 回答 (1)

#### #1 @(匿名) · 2026-04-29 22:16:16

试了下不行

---

## 离线数据量太少了，官方在huggingface上的demo数据能给多一点吗？或者有那位大佬知道哪里有可靠的数据集吗

- ID: `89cf91b443c011f1b6c8aa5005b73072`
- 链接: https://lexiangla.com/questions/89cf91b443c011f1b6c8aa5005b73072?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 建议：可视化训练与评估的资源分配队列

- ID: `8bd9670843b311f1bd80ea260899f34e`
- 链接: https://lexiangla.com/questions/8bd9670843b311f1bd80ea260899f34e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

在本次竞赛的社区中，可以看到许多关于训练任务和评估任务资源分配的问题。此外，资源分配失败、任务中断、队列顺序变更等相同问题也频繁出现。因此，我们建议增加一项功能，让参加者也能可视化查看资源分配等待队列的顺序。
① 参加者执行训练或评估时，任务进入队列堆栈。② 若需等待资源，任务保持堆栈状态；资源分配完成后，任务从队列中出栈。
希望能够在UI上简单直观地显示上述信息，让参加者一目了然地掌握资源分配的顺序。

### 回答 (1)

#### #1 @(匿名) · 2026-04-29 22:33:29

我几乎没遇到呀

---

## Evaluation Failed log没有输出信息，怎么确定哪里出了问题。

- ID: `21defcd243b311f1a5910a647d4965e7`
- 链接: https://lexiangla.com/questions/21defcd243b311f1a5910a647d4965e7?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

训练成功。Evaluation Status Failer 但是log 只显示到

=== Competition Environment Ready ===

Complete setting network policy rules.

Complete setting taiji user.

如何确定错误啊。

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 现在计算资源显示有2个了，意思是可以同时跑2个训练任务了吗？有人验证过没有

- ID: `df82ccc043ac11f1a989728ea5686482`
- 链接: https://lexiangla.com/questions/df82ccc043ac11f1a989728ea5686482?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (2)

#### #1 @(匿名) · 2026-04-29 22:18:35

可以但是不能在一个训练任务使用两张卡

#### #2 @(匿名) · 2026-04-29 17:46:06

是的
&nbsp;

---

## predictions.json的设计是不是有问题

- ID: `0d5c15ee439d11f19b66aa5005b73072`
- 链接: https://lexiangla.com/questions/0d5c15ee439d11f19b66aa5005b73072?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

这个结构的话，相同用户不同物品的预测值会冲突

### 回答 (2)

#### #1 @(匿名) · 2026-04-29 17:47:12

There are no duplicate users in the prediction data&mdash;each user appears only once, so no conflict will occur.

#### #2 @(匿名) · 2026-04-29 17:34:05

没有相同用户

---

## 请问每一轮训练完发布的模型，是否对应每一轮的checkpoint？意思是我每一轮都发布的话，每一轮的模型进行的评估分数都会不同，这样对吗？

- ID: `af5753a8439411f188a1ba0336b6e028`
- 链接: https://lexiangla.com/questions/af5753a8439411f188a1ba0336b6e028?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

请问每一轮训练完发布的模型，是否对应每一轮的checkpoint？意思是我每一轮都发布的话，每一轮的模型进行的评估分数都会不同，这样对吗？

### 回答 (1)

#### #1 @(匿名) · 2026-04-29 17:48:49

Yes, the model released after each training round corresponds to the checkpoint of that round, so the evaluation scores will differ each round

---

## 模型训练好后，Storage Size显示为0，是怎么回事？

- ID: `c5e0f8ba438d11f19ca3fae25ca28989`
- 链接: https://lexiangla.com/questions/c5e0f8ba438d11f19ca3fae25ca28989?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-29 14:44:51

Please read the Tencent Angel Machine Learning Platform User Guide carefully.

---

## 学术赛道 第一轮和第二轮是都有公开测试集和私人测试集吗，赛题说的第一轮排行榜前50名指的是公开测试集上的排名还是私人测试集上的排名，二者分布差异大吗？

- ID: `8a1b6be0438c11f18633c6e5e564e96a`
- 链接: https://lexiangla.com/questions/8a1b6be0438c11f18633c6e5e564e96a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (2)

#### #1 @(匿名) · 2026-04-29 22:35:41

很公平
&nbsp;

#### #2 @(匿名) · 2026-04-29 19:00:43

应该是类似于kaggle那样的，最终排名是由公开测试集中最好的那个模型放到官方的私人测试集测试后的分数排序的

---

## 在训练过程中程序被kill，且log中没有打印出任何error或warning，请问这是什么原因？（前人已有提问类似问题但未得到合理答复）

- ID: `f9038406437e11f1bba4c6e5e564e96a`
- 链接: https://lexiangla.com/questions/f9038406437e11f1bba4c6e5e564e96a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

前人帖下的非对口answer：A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.The timeout does not include built-in waits like sleep 120, nor does it count queue waiting time (e.g., "Waiting for evaluation resources"). The 30‑minute evaluation timer is based on your code&rsquo;s actual execution log time.

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 不懂就问，工业赛道数据为啥有时间穿越

- ID: `0052d0c2437a11f1aa7a6a8ce5bc633c`
- 链接: https://lexiangla.com/questions/0052d0c2437a11f1aa7a6a8ce5bc633c?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

训练日期：
数据最早时间: 2026-02-28 00:35:09
&nbsp;数据最晚时间: 2026-03-04 01:49:50

测试日期：

&nbsp;
理论上，过去训练的模型预测未来，但是为啥这个比赛测试数据还有比训练数据早的？？？
请问主办方为啥如此设置。

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## infer可以加一个copy按钮吗

- ID: `9f6634fe437711f1a5baea260899f34e`
- 链接: https://lexiangla.com/questions/9f6634fe437711f1a5baea260899f34e?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 无法查看所有推理log，提示“Network Error! Contact the competition assistant for more helps”

- ID: `1c333d78434211f1b7524ee504b554c4`
- 链接: https://lexiangla.com/questions/1c333d78434211f1b7524ee504b554c4?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-29 04:52:52

解决了，是去广告插件导致的

---

## 时间有被脱敏处理吗，样本里面time是绝对时间吗

- ID: `29ff8bc8431f11f1bc86324e01955a1c`
- 链接: https://lexiangla.com/questions/29ff8bc8431f11f1bc86324e01955a1c?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 请问一下第一次跑demo的时候显示fail to initialize user storage怎么办

- ID: `546f6f54431011f199d5be0aa5498713`
- 链接: https://lexiangla.com/questions/546f6f54431011f199d5be0aa5498713?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 我记得之前说的每个组不是有三张GPU可用吗，但是现在只能用两张，训练效率较低，而且也无法本地训练

- ID: `c16f2928430b11f1950c825bb877947b`
- 链接: https://lexiangla.com/questions/c16f2928430b11f1950c825bb877947b?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 您好，为什么我这个训练失败了，output却没有了

- ID: `c0fc1806430711f1966f5600a622cc9a`
- 链接: https://lexiangla.com/questions/c0fc1806430711f1966f5600a622cc9a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

看日志报错应该是265 Killed python3&nbsp; &nbsp;job id&nbsp;

angel_training_ams_2026_1393421075977191638_20260428152555_09709c9a

&nbsp;

### 回答 (2)

#### #1 @(匿名) · 2026-04-30 07:52:06

那这个会跟没有output有关系吗，我刚刚又遇到这种情况了，这次是我先看到了里面output结果，退出之后output又消失了

#### #2 @(匿名) · 2026-04-28 22:10:02

检查一下模型代码传对没
&nbsp;

---

## 重新组织训练样本

- ID: `a16212b4430411f1b8ad12fb614a689d`
- 链接: https://lexiangla.com/questions/a16212b4430411f1b8ad12fb614a689d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

hi，我在比赛的时候如果不希望以现有的方法组织训练样本，是否有位置可以存储新的样本呢，每次都重新组织过于花费时间，而且现有的空间似乎不足以下载完整的样本数据。

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 22:11:00

样本数据哪能让你下载下来

---

## 每队分配的RAM是多少？

- ID: `6830eb56430311f19b530a647d4965e7`
- 链接: https://lexiangla.com/questions/6830eb56430311f19b530a647d4965e7?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 怎么突然多了张卡了？能直接给三张吗？ Why is there an extra card all of a sudden? Could you give three directly?

- ID: `2a07716242fc11f1b19de6414582ea1d`
- 链接: https://lexiangla.com/questions/2a07716242fc11f1b19de6414582ea1d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

昨天还是1张的。It was only one yesterday.

### 回答 (1)

#### #1 @(匿名) · 2026-04-29 22:15:26

(空)

---

## HELP:模型评估后，显示成功和对应AUC，但是一整天后也不能查询排行，从外部进入排行榜也看不到自己的提交和排名。(已有修改infer文件和经过对应时间)

- ID: `49061ebc42f511f1946b0ab7809cf5b9`
- 链接: https://lexiangla.com/questions/49061ebc42f511f1946b0ab7809cf5b9?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

NotFoundError: Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node.
    at GO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:26100)
    at tn (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27391)
    at XO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27851)
    at tn (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27541)
    at XO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:30592)
    at tn (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27541)
    at XO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27851)
    at tn (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27541)
    at XO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27851)
    at tn (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27541)

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 还没有修复评估次数重置的bug吗？目前为止还是无法提交。

- ID: `3d6bccaa42f511f1ba7cee8dccbbec8c`
- 链接: https://lexiangla.com/questions/3d6bccaa42f511f1ba7cee8dccbbec8c?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

还没有修复评估次数重置的bug吗？目前为止还是无法提交。

### 回答 (1)

#### #1 @(匿名) · 2026-04-29 15:59:13

Please try again and see.

---

## 请问这个只能在线上平台训练的话，那是不是意味着无法调适啊？调适只能提交任务看线上日志吗？

- ID: `81454c6242eb11f1b16c8e742a80aeca`
- 链接: https://lexiangla.com/questions/81454c6242eb11f1b16c8e742a80aeca?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 新的demo数据在原来demo数据基础多了一个序列域特征，请问这个序列域特征代表什么意思

- ID: `2c56562442eb11f1812d6adecbf36ff9`
- 链接: https://lexiangla.com/questions/2c56562442eb11f1812d6adecbf36ff9?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 18:39:58

All features have been anonymized&nbsp;

---

## 如果在训练推理中安装新的包？

- ID: `3acbd3dc42e611f1bab8825bb877947b`
- 链接: https://lexiangla.com/questions/3acbd3dc42e611f1bab8825bb877947b?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

如何在环境中安装包？

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:42:22

Tencent Angel Machine Learning Platform User Guide&nbsp;

---

## Where is action feature?

- ID: `3402ce1242e511f1bfe4e6414582ea1d`
- 链接: https://lexiangla.com/questions/3402ce1242e511f1bfe4e6414582ea1d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Sorry which feature corresponds to the action feature in each seq category (a/b/c/d)?

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:40:19

All features have been anonymized&nbsp;

&nbsp;

---

## 为啥我的成绩没在leaderboard里面显示出来

- ID: `5cc4055242e311f1a0d1fa9c40bfe1bd`
- 链接: https://lexiangla.com/questions/5cc4055242e311f1a0d1fa9c40bfe1bd?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

为啥我明明有成绩却没在leaderboard里面显示出来呢？

我的team name

### 回答 (3)

#### #1 @(匿名) · 2026-04-28 20:28:31

工业赛道

#### #2 @(匿名) · 2026-04-28 18:09:11

请问一下你在什么赛道呀

#### #3 @(匿名) · 2026-04-28 17:28:09

This is most likely caused by the&nbsp;Delayed Feedback mechanism, which is normal behavior. Here's the explanation:

To prevent&nbsp;leaderboard overfitting, some submission results during Round 1 will be&nbsp;delayed, focusing on evaluating model generalization.
The leaderboard consists of a&nbsp;Public Leaderboard&nbsp;(based on partial test set with real-time feedback) and a&nbsp;Private Leaderboard&nbsp;(based on the full test set).
Your score may have already been generated but not yet displayed on the public leaderboard.

Please wait a bit longer. If it still doesn't appear after an extended period, please contact official support for further confirmation.

---

## 算力这么紧张吗？我一个Training Job Running跑了6个小时

- ID: `539c2a9a42de11f19e7e8e742a80aeca`
- 链接: https://lexiangla.com/questions/539c2a9a42de11f19e7e8e742a80aeca?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:25:06

According to platform rules, an evaluation task stuck in "waiting for inference" status is a&nbsp;normal queuing behavior. Here's why:

Each team is allocated&nbsp;0.2 GPU, meaning all tasks must be executed&nbsp;serially&nbsp;(one after another).
Any team member can initiate evaluation, but due to the resource quota limit, tasks need to wait in queue.

---

## 每天3次模型评估，究竟是北京时间几点更新次数啊？一整天都没法进行评估。

- ID: `1cf4f86e42de11f1b3af466c822786ed`
- 链接: https://lexiangla.com/questions/1cf4f86e42de11f1b3af466c822786ed?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

每天3次模型评估，究竟是北京时间几点更新次数啊？一整天都没法进行评估。

### 回答 (2)

#### #1 @(匿名) · 2026-04-28 17:09:15

请问能帮忙解决这个问题吗？已经过了北京时间12点，次数还是没有更新。到现在还是无法提交评估。

#### #2 @(匿名) · 2026-04-28 17:03:22

Would it be possible to provide the error message/details from the failed submission?

---

## Why i cant see my submission on the leaderboard?

- ID: `b9eda59042d811f1976bbef3d5422991`
- 链接: https://lexiangla.com/questions/b9eda59042d811f1976bbef3d5422991?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Evaluation ID is 30335, Leaderboard Score auc: 0.846108, Inference Time 625.28s and last timstamp from log is 2026-04-27 15:47:38.044.
Do I need to do anything more to submit this solution for leaderbord or its done automatically?

### 回答 (3)

#### #1 @(匿名) · 2026-04-28 17:38:54

Once an evaluation task is submitted, the system&nbsp;automatically runs through the full evaluation pipeline.
The&nbsp;public leaderboard reflects real-time feedback based on partial test set results&mdash; meaning your AUC score should appear automatically.
However, Round 1 uses a&nbsp;Delayed Feedback mechanism: some submission results will be delayed to prevent leaderboard overfitting.

#### #2 @(匿名) · 2026-04-28 16:37:52

But this was submitted 2026-04-27 15:47 (before 2026-04-28 12:00), so I would expect to see it after 2026-04-28 15:00 ...

#### #3 @(匿名) · 2026-04-28 16:31:06

The leaderboard is updated daily with results that have been finalized before 12:00 noon Beijing Time, and you will be able to view your ranking at 3:00 PM tomorrow.

---

## 有的 instance 示例没有办法删除，一直卡在 started 这里，没法 stop 也没法删除

- ID: `72caecca42d511f196760ab7809cf5b9`
- 链接: https://lexiangla.com/questions/72caecca42d511f196760ab7809cf5b9?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:00:33

Status updates may occasionally be delayed. If this does not affect quota usage, it can be ignored for now.

---

## 提交次数更新不是北京时间中午12点更新吗，为什么显示：You can only submit up to 3 tasks per day (failed tasks are not counted). Contact the competition assistant for more helps

- ID: `574a898242d111f1a94dea71aef76897`
- 链接: https://lexiangla.com/questions/574a898242d111f1a94dea71aef76897?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:10:26

&nbsp;The daily submission limit resets every 24 hours at&nbsp;15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24-hour period. Failed or stopped tasks do not count toward the daily limit.

---

## 能提供训练用的卡型信息嘛

- ID: `e065056e42cf11f1bd88ba0336b6e028`
- 链接: https://lexiangla.com/questions/e065056e42cf11f1bd88ba0336b6e028?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

rt，想针对卡型进行针对性优化，能提供卡型信息
&nbsp;

### 回答 (2)

#### #1 @(匿名) · 2026-04-29 01:18:09

h20 20gb

#### #2 @(匿名) · 2026-04-28 17:17:48

Thank you for your question. The specific GPU card model is not publicly disclosed as part of the platform's operational environment.

---

## 建议在网页上显示当前剩余的评测次数或下次评测更新时间

- ID: `641c96fc42cf11f1b2af56a046cec7fa`
- 链接: https://lexiangla.com/questions/641c96fc42cf11f1b2af56a046cec7fa?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 测评任务任务一直处于“waiting for inference"状态

- ID: `2614714842cc11f18bbf9e47a5de1825`
- 链接: https://lexiangla.com/questions/2614714842cc11f18bbf9e47a5de1825?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

测评任务任务一直处于&ldquo;waiting for inference"状态，已经5个小时

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:24:52

According to platform rules, an evaluation task stuck in "waiting for inference" status is a&nbsp;normal queuing behavior. Here's why:

Each team is allocated&nbsp;0.2 GPU, meaning all tasks must be executed&nbsp;serially&nbsp;(one after another).
Any team member can initiate evaluation, but due to the resource quota limit, tasks need to wait in queue.

---

## 推理模型load错误

- ID: `fc30652242ca11f1b3110e7eff9d52a2`
- 链接: https://lexiangla.com/questions/fc30652242ca11f1b3110e7eff9d52a2?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

在我导出模型后，我遇到一个奇怪的问题，推理模型加载错误。我记得在昨晚我做了两个近似的实验，但是今天看train中的model.py似乎不一致，我随机选择一个进行eval，显示报错，另一个不报错。在文件中我看到加载模型不是从ckpt中加载，为什么会遇到这个问题呢？看起来是一个bug似乎，因为我有改model.py的模型无法load成功。

2026-04-28 06:08:51,120 - ERROR - Inference failed: Error(s) in loading state_dict for PCVRHyFormer:

&nbsp;

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:23:54

Recommendation:&nbsp;Re-train with your&nbsp;modified code, or ensure that the&nbsp;infer.py&nbsp;/&nbsp;model.py&nbsp;used during inference is&nbsp;exactly identical&nbsp;to the one used during training.

---

## training开始后的instance无法停止，请求管理员手动停止

- ID: `e1a2d54042c611f183a4c61a13687317`
- 链接: https://lexiangla.com/questions/e1a2d54042c611f183a4c61a13687317?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

instant ID 95d1b2ad9dca2466019dd20c1a590dbe 这里我们不知道 删除这个instance是否会停止 不太敢删除。

### 回答 (2)

#### #1 @(匿名) · 2026-04-28 14:08:16

Solved 青天大老爷管理员给我停掉了

#### #2 @(匿名) · 2026-04-28 13:57:43

95d1b1c49dca22ef019dd26977bd0e76 我的也是，管理员看到麻烦也帮我的停一下

---

## 请问能否为特征列提供 nominal / ordinal / numeric / timestamp 类型标注？

- ID: `d44df4d442c511f1b1eef26125df00d6`
- 链接: https://lexiangla.com/questions/d44df4d442c511f1b1eef26125df00d6?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

发现 baseline 的 embedding table 比较大，因此想确认一下：是否所有整数特征都是真正的类别 ID？还是其中有一些特征其实是有序特征、数值型信号或时间相关信号，可能更适合使用数值变换、分桶、归一化等方式处理，而不是直接使用 embedding table？
具体来说，能否确认以下特征组是否都属于类别 ID？
user_int_feats_*item_int_feats_*domain_a_seq_*domain_b_seq_*domain_c_seq_*domain_d_seq_*
如果其中有部分特征不是类别 ID，也希望能进一步说明它们更适合归为 ordinal / numeric / timestamp 中的哪一类。这样可以帮助我们更合理地设计特征处理方式。

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:29:54

all features have been anonymized, and the organizing committee cannot confirm specific feature type classifications

---

## 我这个训练停止下来，实际上也没有训练，能不能帮我停一下

- ID: `8d70584e42a811f1b334aa74fb716d19`
- 链接: https://lexiangla.com/questions/8d70584e42a811f1b334aa74fb716d19?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-28 14:07:51

Please note that there may be a delay when stopping some tasks.

---

## 想问一下为什么已经评估好的模型没有入排行榜？

- ID: `1383b7b242a611f1a9fc3600151239df`
- 链接: https://lexiangla.com/questions/1383b7b242a611f1a9fc3600151239df?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 10:33:59

每天15点出新的榜单，别急，成功出分了就行

---

## Model Evaluation一直显示Waiting for Inference Resources这是什么原因？

- ID: `f097249c429f11f19ff5ea260899f34e`
- 链接: https://lexiangla.com/questions/f097249c429f11f19ff5ea260899f34e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Model Evaluation一直显示Waiting for Inference Resources这是什么原因？

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 10:33:16

比赛是串行，一次只能一个任务，train就不能infer，或者你看看log报什么错

---

## 团队名字叫Semantic_Shards，提交了两版结果，为啥看不到分数排序？（工业界榜单）

- ID: `162b5c40429711f191d93a35d2f04c13`
- 链接: https://lexiangla.com/questions/162b5c40429711f191d93a35d2f04c13?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (3)

#### #1 @(匿名) · 2026-04-28 11:24:22

请问有解决么？一样的现象

#### #2 @(匿名) · 2026-04-28 10:13:44

请问最后有解决么？我也遇到了没法上榜单的同样的情况
&nbsp;

#### #3 @(匿名) · 2026-04-28 10:00:31

我们也碰到了类似的问题，提交了几个版本，但是榜单上没有对应的名字

---

## What do “label_type=1” and “label_type=2” mean, respectively?

- ID: `c4753276425611f18591b2aef88eec92`
- 链接: https://lexiangla.com/questions/c4753276425611f18591b2aef88eec92?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (2)

#### #1 @(匿名) · 2026-04-28 18:38:40

In the dataset, label_type=2 means a positive sample (user converted within 24 hours after an ad click), while label_type=1 means a negative sample (no conversion within 24 hours). The code in dataset.py remaps them to 1 and 0 for model input.

#### #2 @(匿名) · 2026-04-28 00:37:49

You can infer it from the official baseline.

---

## 请问已经publish过的model，在model manage页面被删除了之后，可以重新publish吗，目前显示已被publish过的ckpt不允许再被publish

- ID: `f45326d4425011f1b5c6766d32605923`
- 链接: https://lexiangla.com/questions/f45326d4425011f1b5c6766d32605923?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## model eval看到能推理出来但是fail

- ID: `a2459898424d11f1a5ddbe87ea613f06`
- 链接: https://lexiangla.com/questions/a2459898424d11f1a5ddbe87ea613f06?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

model eval看到能推理出来但是fail，显示&ldquo;&nbsp;ERROR - Missing 310000 answer user_ids in predictions (need full coverage)&rdquo;

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 10:36:46

我遇到了95NAN的问题，但5%有结果还是给我出了分，你这个怕不是100%都NAN？检查一下是不是模型代码，FP16/32设置的问题

---

## 我一个人的团队，为什么算力资源也只给我分配了一个？而且还是只有0.2的GPU。虽然我是一个人，但也需要进行多项训练协同验证，这相当于其他多人的团队训练速度是我的4倍，这样是否公平呢？

- ID: `b4c98606424711f1823f2229879a38e2`
- 链接: https://lexiangla.com/questions/b4c98606424711f1823f2229879a38e2?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

我一个人的团队，为什么算力资源也只给我分配了一个？而且还是只有0.2的GPU。虽然我是一个人，但也需要进行多项训练协同验证，这相当于其他多人的团队训练速度是我的4倍，这样是否公平呢？

### 回答 (2)

#### #1 @(匿名) · 2026-04-28 10:14:47

多人团队也是一张卡，也和单人一致的

#### #2 @(匿名) · 2026-04-28 02:41:11

No, actually, regardless of the number of people on the team, you can only have one run per team

---

## 1、推理结束不出分？ 2、每个队伍 每天多少次提交推理呀，有没有限制呀

- ID: `5dd6b5e4424711f1911a825bb877947b`
- 链接: https://lexiangla.com/questions/5dd6b5e4424711f1911a825bb877947b?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-28 10:37:15

看看log显示啥呢

---

## 为什么会被kill，是代码有问题吗

- ID: `91843d68422311f19182f2f0f4c72958`
- 链接: https://lexiangla.com/questions/91843d68422311f19182f2f0f4c72958?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-27 19:55:56

A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

&nbsp;

---

## label_type为1的表示啥

- ID: `3bda1ed2421e11f1a20ddec6795b114a`
- 链接: https://lexiangla.com/questions/3bda1ed2421e11f1a20ddec6795b114a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

是表示有点击无转化还是没有点击，还是说是混着的

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 19:33:06

In this dataset, a label_type value of 1 for the CVR label indicates that the user has performed a conversion action within 24 hours following an ad click.

---

## 可用的显存是多少唉？我看到其他评论区是0.2 的显卡，假设是80G的A100 每组可用的显存就是16G吗？超过16G就会报错还是

- ID: `09480e5a421b11f1bf814263052e84cf`
- 链接: https://lexiangla.com/questions/09480e5a421b11f1bf814263052e84cf?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 19:02:16

Sorry, that information is not public.

---

## Hey everyone, has anyone run into this situation?

- ID: `5cbf435a421611f1bba79e9532461105`
- 链接: https://lexiangla.com/questions/5cbf435a421611f1bba79e9532461105?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

&nbsp;Today I'm limited to using just one GPU, but during training, the GPU utilization goes over 100%. Normally, it shouldn't exceed 100%. Has anyone else experienced this? Also, just quietly wondering&mdash;any idea when we'll be able to resume parallel training?

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 19:57:04

Each team is allocated a single resource quota, which is shared across both training and inference tasks. If a team submits multiple tasks, they will be executed sequentially: a new task will not begin until the currently running task has completed.

---

## What are the training and evaluation AUC scores of the official baseline? Has anyone run it?

- ID: `d0ec51ec421011f18cd8f2f0f4c72958`
- 链接: https://lexiangla.com/questions/d0ec51ec421011f18cd8f2f0f4c72958?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

What are the training and evaluation AUC scores of the official baseline? Has anyone run it?

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:47:27

To understand the actual performance of the official baseline, you may want to check if any teams on the leaderboard have submitted results using the unmodified official baseline code as a reference point.

---

## 推理时间最长是多少啊？What is the inferencing time limit?

- ID: `1bf43120420f11f1ae153a63274c50fb`
- 链接: https://lexiangla.com/questions/1bf43120420f11f1ae153a63274c50fb?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

评估任务运行到一半被stop了，大概在inferencing状态用了十来分钟，不知道是什么原因。初步怀疑是超时了？但是没找到推理最长时间是多少？有人知道吗？
Eval task was stopped during inferencing. It was stopped after being inferencing status for ten minutes. Is this caused by timeout on inferencing time limit? Does anyone know the inferencing time limit?

### 回答 (2)

#### #1 @(匿名) · 2026-04-27 21:33:49

A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

#### #2 @(匿名) · 2026-04-27 16:48:51

试出来了，推理时限是30min

---

## 请问评测次数不是北京时间12点刷新的吗？为什么我今天12点后提交评测显示我评测次数用完了呢？

- ID: `14329f14420d11f1ac2826b09727b8b8`
- 链接: https://lexiangla.com/questions/14329f14420d11f1ac2826b09727b8b8?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-28 17:50:46

The daily evaluation submission quota does NOT reset at 12:00 Beijing time&nbsp;&mdash; it resets every&nbsp;24 hours at 15:59 AOE .

---

## Can we get the `schema.json`?

- ID: `23ead234420b11f1afe53a773f0e4cdd`
- 链接: https://lexiangla.com/questions/23ead234420b11f1afe53a773f0e4cdd?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Because of impressed OOM in the platform, we need do some experiments locally. However the train.py needs a schema.json, can anyone give us this file?

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 20:15:09

The schema.json is tied to the full competition data and cannot be shared. If you want to train locally on the 1000 public samples, you can generate a schema.json yourself.

---

## Why the system returned that OOM error? are computing nodes shared for all teams? how to get stable GPU and CPU nodes for training?

- ID: `cce114a8420511f1a1ee76202f67a800`
- 链接: https://lexiangla.com/questions/cce114a8420511f1a1ee76202f67a800?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

2026-04-27 14:38:16.009 torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 182.00 MiB. GPU 0 has a total capacity of 19.12 GiB of which 19.12 GiB is free. Process 2160295 has 73.12 GiB memory in use. Process 2287370 has 13.93 GiB memory in use. Process 756592 has 6.48 GiB memory in use. Process 26149 has 1.30 GiB memory in use. Of the allocated memory 932.34 MiB is allocated by PyTorch, and 25.66 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 任务无法stop

- ID: `01c6e1b241ec11f1805526c5667219ff`
- 链接: https://lexiangla.com/questions/01c6e1b241ec11f1805526c5667219ff?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Job: angel_training_ams_2026_1393421075977197453_20260427005421_b800f26f
instance id： 95a8cc5c9dca2162019dcab6c136017f

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 15:01:50

Please note that there may be a delay when stopping some tasks.

---

## 训练数据在呢？

- ID: `d502e66a41e811f1aa1ddabb3ffa631d`
- 链接: https://lexiangla.com/questions/d502e66a41e811f1aa1ddabb3ffa631d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

是说只基于这1000个数据样例开发完脚本，然后在平台上提交代码再进行完整数据训练模型吗？

### 回答 (2)

#### #1 @(匿名) · 2026-04-28 00:15:35

所以我们拿不到完整的训练数据样本对吗？模型的训练一定只能在Tencent Angel Machine Learning Platform进行训练吗。

#### #2 @(匿名) · 2026-04-27 15:52:33

对

---

## job can not be stopped

- ID: `fb7a842641e611f1908c7e6b586d458c`
- 链接: https://lexiangla.com/questions/fb7a842641e611f1908c7e6b586d458c?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

we clicked stop but it was not stopped
job id:
&nbsp;

angel_training_ams_2026_1029735554728163763_20260426205043_4c6dff52

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 一队三个人只有一张卡实在是吃紧，后面算力会放开一点吗

- ID: `0e97e29641e311f1825f4686ed2487a3`
- 链接: https://lexiangla.com/questions/0e97e29641e311f1825f4686ed2487a3?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 今天4.27 我都没发布过 告诉我 up to 3 tasks, 逗我玩呢。有没有真人客服啊？

- ID: `403f9f6e41d911f1898b3a63274c50fb`
- 链接: https://lexiangla.com/questions/403f9f6e41d911f1898b3a63274c50fb?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (3)

#### #1 @(匿名) · 2026-04-27 14:57:48

确实有问题。官方能解答一下吗？北京时间几点到几点算一天？Submission limits.&nbsp;Each team may submit up to&nbsp;three&nbsp;submissions per day. 这句话该怎样理解？

#### #2 @(匿名) · 2026-04-27 13:21:11

但是就按照aoe时间昨天也跑了5个左右啊 他这个3 tasks是怎么算的

#### #3 @(匿名) · 2026-04-27 09:36:40

不是按北京时间，是按AOE的16：00的。

---

## 1、如图，任务完成之后为什么不能停止？ 2、模型eavl 之后显示分数了，为什么排行榜却没有

- ID: `607d683a41d711f19dceeed6b9c02d85`
- 链接: https://lexiangla.com/questions/607d683a41d711f19dceeed6b9c02d85?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-27 10:41:16

1.因为没有在运行 2.没在AOE16:00前记录

---

## Request to Address Bottleneck in the Model Build → Publish → Submit & Score Workflow

- ID: `3731bd6641d511f1adf2ca8321426fe4`
- 链接: https://lexiangla.com/questions/3731bd6641d511f1adf2ca8321426fe4?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

It appears that resource allocation for evaluation is currently taking longer than model training itself. Under these conditions, evaluation has become a bottleneck in the training &rarr; model publishing &rarr; evaluation pipeline, creating a very high risk that trained jobs &mdash; which are deleted after 24 hours &mdash; will expire before evaluation completes.
While multiple bugs seem to be occurring, the issues we would like prioritized are:

Resolving the resource allocation issue in the evaluation phase
Ensuring the training phase stops reliably and the next job begins automatically

These two points are critical tasks required for participants to complete the intended workflow as documented, and represent the minimum conditions for all participants to compete on equal and fair footing in this competition. Therefore, prompt action is strongly requested.
We understand the team is managing a large number of issues, and we sincerely apologize for the additional burden. However, we ask for your assistance &mdash; both for the benefit of all participants and to help reduce the volume of support requests the organizing team will need to handle going forward.

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 19:59:30

1. Inference Time Limit A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system. 
2. Resource Allocation Policy Each team is allocated a single resource quota, which is shared across both training and inference tasks. If a team submits multiple tasks, they will be executed sequentially: a new task will not begin until the currently running task has completed.

---

## Evaluation Stopped in Model Evaluation

- ID: `eccc4bc441d411f1b8db32136c2f0ae8`
- 链接: https://lexiangla.com/questions/eccc4bc441d411f1b8db32136c2f0ae8?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Eval ID 28664 appears to have been forcibly transitioned to "Stopped" status approximately 24 hours after submission, and is no longer being evaluated. Please look into this.
Eval ID 29403 has been stuck at "Waiting for Inference Resources" for over 12 hours since submission. Could you let us know the average time required for resource allocation?

### 回答 (2)

#### #1 @(匿名) · 2026-04-27 20:17:13

What status does an inference job get when it exceeds the time limit?

#### #2 @(匿名) · 2026-04-27 19:58:45

A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

---

## Training job can not be stopped

- ID: `bb7e7acc419a11f185f56e84f4d34a4d`
- 链接: https://lexiangla.com/questions/bb7e7acc419a11f185f56e84f4d34a4d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

&nbsp;This instance can not be stopped:

95d1b1c49d8ff032019dc923c70e084a
&nbsp;

job ID:
angel_training_ams_2026_1393421075977191678_20260426172836_8fa80343

### 回答 (2)

#### #1 @(匿名) · 2026-04-27 15:01:05

Please note that there may be a delay when stopping some tasks.

#### #2 @(匿名) · 2026-04-27 07:48:58

同理angel_training_ams_2026_1393421075977198222_20260427053812_df22f387

95d1B1C49DCA22ef019dcbbaba8D021e&nbsp;&nbsp;

---

## 点了stop停不下来

- ID: `45d70790419411f1969cfea4113eb7a6`
- 链接: https://lexiangla.com/questions/45d70790419411f1969cfea4113eb7a6?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

angel_training_ams_2026_1029735554728158017_20260427004732_83502e01&nbsp; &nbsp;下面的两个实例95d1b2ad9dca2466019dcab086c50178和95d1b2ad9dca2466019dcab075fd0177都点了stop但是没有停止

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 15:50:48

我也是，现在有解决吗

---

## 训练最优节点是否会自动保存？评估的时候是都会自动使用最优轮数的参数？

- ID: `cb5f0e8e419011f1b9045663b438421f`
- 链接: https://lexiangla.com/questions/cb5f0e8e419011f1b9045663b438421f?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

例如，我训练了4轮，第三轮分数是最优的，第四轮已经下降了，但是我并没有备份第三轮的参数，最后直接用第四轮的数据输出模型并提交评估，那么这个时候评估的模型是我第三轮的最优参数还是第四轮的？

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 01:52:52

默认trainer.py只会保存AUC最高的checkpoint，如果要保存其他checkpoint需要将其保存到设定好的目录下并且按要求命名，具体见文档。
&nbsp;

---

## 现在一个团队只能用一个训练资源嘛，那这限制也太大了吧

- ID: `f56bc5fc418911f1a213ca80602fc2c4`
- 链接: https://lexiangla.com/questions/f56bc5fc418911f1a213ca80602fc2c4?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

现在一个团队只能用一个训练资源嘛，那这限制也太大了吧

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 00:09:56

我也觉得不太合理，至少训练和评估得分开吧

---

## job can not be stopped

- ID: `a5841bd6417d11f19deb4e413d2770cd`
- 链接: https://lexiangla.com/questions/a5841bd6417d11f19deb4e413d2770cd?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

job id angel_training_ams_2026_1029731852466345516_20260426214341_58cd1f4e

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 14:49:54

Please note that there may be a delay when stopping some tasks.

---

## Question about whether the 0.2 H20 GPU limit applies throughout the entire First-round competition

- ID: `0ff00fda417811f1b3f8427473545811`
- 链接: https://lexiangla.com/questions/0ff00fda417811f1b3f8427473545811?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Hi organizers,
&nbsp;
I noticed from another official reply that &ldquo;each team is limited to 0.2 GPU, which means all tasks need to be executed serially.&rdquo;
&nbsp;
Could you please clarify whether this 0.2 H20 GPU limit is only a temporary restriction during the current evaluation phase, or whether it will remain the same throughout the entire First-round competition?
&nbsp;
For a 3-member team, sharing 0.2 H20 GPU means that experimentation and evaluation have to be extremely limited, especially since tasks must be executed serially. It would be helpful to know whether we should plan our whole first-round strategy around this resource constraint, or expect any adjustment later.
&nbsp;
Thank you!

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 模型评估成功出auc之后不在排行榜上出现

- ID: `b967e8b4417611f1b555eed6b9c02d85`
- 链接: https://lexiangla.com/questions/b967e8b4417611f1b555eed6b9c02d85?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

模型评估成功出auc之后不在排行榜上出现，user center中也不显示

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 21:59:38

Based on the results submitted daily before 16:00 AOE time, sort the teams in descending order according to their historical best scores, and update the leaderboard at 19:00 AOE time on the same day.

---

## 什么时候能开放并行，为什么明明有三卡的资源只能用一张卡

- ID: `75449a2e417111f191f9b22f0a600adf`
- 链接: https://lexiangla.com/questions/75449a2e417111f191f9b22f0a600adf?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (3)

#### #1 @(匿名) · 2026-04-27 14:02:06

Currently, each team has one computing unit, equivalent to the officially announced 0.2 GPU, and the issue displaying 3 computing units has been fixed.

#### #2 @(匿名) · 2026-04-27 10:43:40

同问

#### #3 @(匿名) · 2026-04-26 21:17:36

同问

---

## 请问大家训练分数与评估分数相差大吗？

- ID: `fcfdadc8416911f18e7dead2caac9371`
- 链接: https://lexiangla.com/questions/fcfdadc8416911f18e7dead2caac9371?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

请问大家训练分数与评估分数相差大吗？

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 08:39:50

过拟合了

---

## 为什么评估模型会在处理完样本后不评分就停止？

- ID: `8a62f590416711f19450dabb3ffa631d`
- 链接: https://lexiangla.com/questions/8a62f590416711f19450dabb3ffa631d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:04:32

baseline吗？还是改过了？

---

## 模型评估，一直处于等待推理资源状态，大半天了，算力那么紧张啊？

- ID: `0648b302416311f1a8a6eaa59d583e05`
- 链接: https://lexiangla.com/questions/0648b302416311f1a8a6eaa59d583e05?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

模型评估，一直处于等待推理资源状态，大半天了，算力那么紧张啊？

### 回答 (5)

#### #1 @(匿名) · 2026-04-26 21:13:14

不是有三张卡吗，为什么一个job只用了一张卡，job之间还要串行

#### #2 @(匿名) · 2026-04-26 22:58:45

same quesion for me as well.In my chain, no task is running yet. All wait for resource/infer resource/evaluation resource.

#### #3 @(匿名) · 2026-04-26 20:49:59

但是我已经停止了训练进程，只有评估 还是在等待推理资源&nbsp; 这正常吗 感觉像是平台的bug而且终于跑起来之后，点击日志还会报错：Network Error! Contact the competition assistant for more helps

#### #4 @(匿名) · 2026-04-26 20:31:59

可是我从下午1点半等到现在，还是处于等待推理资源状态，请问这正常吗？

#### #5 @(匿名) · 2026-04-26 20:16:01

1. Any team member can initiate evaluation, but each team is limited to 0.2 GPU, which means all tasks need to be executed serially.

---

## 停止评估后也会浪费评估次数吗？

- ID: `a212c27a415c11f1a47d2e3d16a5b7db`
- 链接: https://lexiangla.com/questions/a212c27a415c11f1a47d2e3d16a5b7db?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

因为一直处于资源等待中，于是我停止了评估。此时已经成功评估两次，但是想要第三次评估时显示次数不够。

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 20:07:41

The daily submission limit resets every 24 hours at 15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24‑hour period. Failed or stopped tasks do not count toward the daily limit.

#### #2 @(匿名) · 2026-04-26 19:29:42

同问，如果提交评估任务在AOE 23:59:59之前，但等待资源后实际执行在AOE 23:59:59之后，算哪一天的评估次数呢？
楼主，问你一个问题，你前两次评估成功时那个时候有在训练吗？

---

## Job Cannot Be Stopped

- ID: `2f61eea2415911f1a3d726b09727b8b8`
- 链接: https://lexiangla.com/questions/2f61eea2415911f1a3d726b09727b8b8?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

以下の手順に従えば、ランニングトレーニングの作業を止めることは不可能になるだろう。
1. ジョブのインスタンスステータスが「リソースキューから退出」のままだったので、停止を選択して再起動しました。2. 停止を選択したにもかかわらず、インスタンスステータスは「トレーニングジョブ実行中」に変わりました。3. トレーニングが完了したらジョブを停止したいのですが、停止ボタンが選択できず、今後新しいジョブを起動できなくなってしまいました。
これは将来の雇用創出と運営能力に重大な影響を与えるため、迅速なご支援をいただければ大変ありがたく存じます。
ジョブID: angel_training_ams_2026_1029735554728163556_20260426001448_5ffaca8a

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## 能快点修复排队状态吗，两个半小时了还在排队，早上也是下午也是

- ID: `f043d672415311f18a5a0a27492a833e`
- 链接: https://lexiangla.com/questions/f043d672415311f18a5a0a27492a833e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:16:30

1. Any team member can initiate evaluation, but each team is limited to 0.2 GPU, which means all tasks need to be executed serially.

---

## 平台代码上传功能求优化！Quick suggestion for code upload!

- ID: `12bb94b0414a11f19b27b2ca7d28dfcd`
- 链接: https://lexiangla.com/questions/12bb94b0414a11f19b27b2ca7d28dfcd?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

现在的代码上传很不方便：1. &nbsp;一次只能传一个文件，项目文件多的时候要点好几次，太慢了。2. &nbsp;不能直接传文件夹，这很不符合现代写代码的习惯，而且就算拆开来一个一个交也很慢希望能改成支持多选文件和直接上传文件夹，这样大家提交代码会舒服很多！
The current code upload process is quite inconvenient:1. &nbsp;It only allows uploading one file at a time. For projects with multiple files, this requires repeated clicks and is really slow.2. It doesn't support uploading folders directly, which goes against modern coding practices. Even if we split and submit files one by one, it's still time-consuming.Hope you can update the feature to support multi-file selection and direct folder uploads. This would make the submission process much smoother and more user-friendly for all participants!

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 18:31:29

我也有两点关于文件管理的建议：
1. 禁止上传同名文件

如果不小心上传了同名文件并保存或提交，将导致需要重新创建 Job。
由于 run.sh 无法删除，一个 Job 中可能会存在多个无法删除的 run.sh 文件，最终不得不全部重新创建。

2. 支持多文件或批量下载，而不仅限于上传

逐个下载 7 个文件非常繁琐，希望能为所有参与者提供多选文件或一键全部下载的功能。

&nbsp;
I would also like to share two suggestions regarding file management:
1. Prevent uploading files with duplicate names

If a file with the same name is accidentally uploaded and saved or submitted, it requires rebuilding the job from scratch.
Since run.sh cannot be deleted, multiple non-deletable run.sh files can end up in a single job, forcing a full rebuild.

2. Support bulk or multi-file downloads, not just uploads

Downloading seven files one by one is quite tedious. A feature to select multiple files or download all at once would be very helpful for all participants.

&nbsp;
&nbsp;

#### #2 @(匿名) · 2026-04-26 19:36:59

还有一个问题就是当我直接点击编辑某个文件时，想要退出是退出不了的，希望也可以修复这个问题，优化直接编辑的界面和体验。There is another issue: when I click to directly edit a file, I cannot exit the editing interface. I hope this problem can be fixed, and the interface and user experience for direct editing can also be improved.还有一个问题：当我点击直接编辑文件时，无法退出编辑界面。我希望这个问题可以得到解决，并且直接编辑的界面和用户体验也可以得到改善。

---

## Issue of automatic stop during the model evaluation process

- ID: `4692d74c414711f1a5da4a2c978b479c`
- 链接: https://lexiangla.com/questions/4692d74c414711f1a5da4a2c978b479c?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Problem Description:
&nbsp;
When running evaluation tasks, both the official baseline model and our custom model will stop automatically within 10 to 20 minutes after the evaluation starts. No results are exported, and the logs are interrupted halfway. Neither the Leaderboard Score nor the Inference Time is displayed in the end, and there are no error prompts in the logs. Please check and confirm this issue. Thank you very much.
&nbsp;
Attachment:
&nbsp;
Evaluation ID：29169 / 29109 / 28267

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:17:30

All processes for the previously problematic tasks have been terminated. You may now retry.

---

## 每人一张卡都不能保证是不是不合理？

- ID: `fe16e168414411f1aad0ca267b1c911d`
- 链接: https://lexiangla.com/questions/fe16e168414411f1aad0ca267b1c911d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:18:45

Any team member can initiate evaluation, but each team is limited to 0.2 GPU, which means all tasks need to be executed serially.

---

## 这个比赛显示的3个训练资源，但是训练任务开三个之后立马秒炸Failed，每个人可用的计算资源到底有多少个啊

- ID: `d6fa3f16414011f1914de6513740ffb4`
- 链接: https://lexiangla.com/questions/d6fa3f16414011f1914de6513740ffb4?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 20:15:15

1. Any team member can initiate evaluation, but each team is limited to 0.2 GPU, which means all tasks need to be executed serially.

&nbsp;

#### #2 @(匿名) · 2026-04-26 19:40:52

同问，我每次只能起一个任务，训练和推理还不能同时开

---

## 训练任务无法停止

- ID: `155fcc62413711f1a0db6e84f4d34a4d`
- 链接: https://lexiangla.com/questions/155fcc62413711f1a0db6e84f4d34a4d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Overall Registration No.3906Track Registration No.2602
job ID
angel_training_ams_2026_1393421075977196559_20260426081500_2a2c7a2e
instance id
95d1b1c49d8ff032019dc843bdd105da

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:19:01

All processes for the previously problematic tasks have been terminated. You may now retry.

---

## 我训练到中途，突然看不自动停止，且看不到输出

- ID: `e6b57504413311f1ae4d76202f67a800`
- 链接: https://lexiangla.com/questions/e6b57504413311f1ae4d76202f67a800?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

11:53开始跑，12:58已经保存了第一个epoch的模型，val的auc也出了，但是跑到13:18分的时候，连output那个按钮都不见了，同时也自动结束了，日志上啥都没显示，你们这个平台怎么bug这么多啊
&nbsp;
&nbsp;
&nbsp;
&nbsp;

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 20:21:52

All processes for the previously problematic tasks have been terminated. You may now retry.

#### #2 @(匿名) · 2026-04-26 13:52:28

就算异常结束，我之前保存的模型不该不见吧，就连output那个按钮都不见了

---

## "3" in Training Resources fails

- ID: `55f61704413211f18a724263052e84cf`
- 链接: https://lexiangla.com/questions/55f61704413211f18a724263052e84cf?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

If I choose '3' in Training Resources while creating a new job, the job always finishes with 'Finished (Failed)', even though the hint suggests that '3' is a possible value.
&nbsp;
&nbsp;

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:35:25

According to the competition resource rules:

Each team is limited to&nbsp;0.2 GPU
All tasks need to be executed&nbsp;serially
When one task is in "Training Job Running" status, other tasks (including evaluation tasks) will enter the waiting queue, with status cycling through: "Exited Resource Queue" &rarr; "TRAINING_INIT" &rarr; "Waiting for Resources"

The scheduling order follows the task submission order.

---

## Training Job Cannot Be Stopped (Possible Platform Bug)

- ID: `4bb9ac04412511f1b1b22e3d16a5b7db`
- 链接: https://lexiangla.com/questions/4bb9ac04412511f1b1b22e3d16a5b7db?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

This seems to be caused by the fact that waiting jobs cycle through "Exited Resource Queue" &rarr; "TRAINING_INIT" &rarr; "Waiting for Resources" &rarr; "Exited Resource Queue" &rarr; ... repeatedly, suggesting that queued jobs are internally being enqueued and dequeued in a loop rather than holding their position.
&nbsp;
Overall Registration No.5018 Track Registration No.1743
Job ID-1:&nbsp;angel_training_ams_2026_1029731869646212955_20260425204233_c1a09ffe (Training Job Running)
Job ID-2: angel_training_ams_2026_1029731869646212955_20260425220554_508f5181("TRAINING_INIT" &rarr; "Waiting for Resources" &rarr; "Exited Resource Queue" &rarr; ... repeatedly)

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:22:02

All processes for the previously problematic tasks have been terminated. You may now retry.

---

## Title: Question about Queue Order

- ID: `4d62ae36411911f1a16cdec6795b114a`
- 链接: https://lexiangla.com/questions/4d62ae36411911f1a16cdec6795b114a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

When multiple jobs are built and Run is clicked for all of them, only the first job starts Running while the others transition to a waiting/queued state. However, once the first job completes, it is not the second job that starts next &mdash; instead, the third or fourth job begins executing, making the queue appear to have no meaningful order.
This seems to be caused by the fact that waiting jobs cycle through "Exited Resource Queue" &rarr; "TRAINING_INIT" &rarr; "Waiting for Resources" &rarr; "Exited Resource Queue" &rarr; ... repeatedly, suggesting that queued jobs are internally being enqueued and dequeued in a loop rather than holding their position.
As a potential fix, we would suggest managing the queue strictly based on the order in which Run was clicked, regardless of each waiting job's wait time or connection status.
We would appreciate it if you could look into this behavior.&nbsp;

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:23:05

asks are scheduled in the order they were submitted.

---

## 这个比赛有运维工程师吗，或者真人客服吗，模型训练停不了，stop功能不可用，一只占资源，只是平台bug，怎么提bug啊，没人解决呢，都一天了

- ID: `f45a2256411311f1b46076202f67a800`
- 链接: https://lexiangla.com/questions/f45a2256411311f1b46076202f67a800?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

这个比赛有运维工程师吗，或者真人客服吗，模型训练停不了，stop功能不可用，一只占资源，只是平台bug，怎么提bug啊，没人解决呢，都一天了

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 11:15:48

Please provide your Team ID and Task ID so we can locate the issue more accurately.

---

## 训练模型停不了，stop功能提示，该功能不可用，怎么办，都跑了一晚上了

- ID: `326ad66a411111f1a91552764530454e`
- 链接: https://lexiangla.com/questions/326ad66a411111f1a91552764530454e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

一个epoch2个半小时，等早停结束，好几天就过去了，现在一个队伍只有一张卡，我这张卡还被占用了，相当于比赛的好几天就训练不了模型了，你们平台有bug啊，还很严重，赶紧修复一下啊

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 11:16:22

Please provide your Team ID and Task ID so we can locate the issue more accurately.

&nbsp;

#### #2 @(匿名) · 2026-04-26 09:44:34

要么就延长一下比赛时间，这bug耽误好几天，等你们bug修复了，时间过去一半，本来训练一个epoch就很慢

---

## I want to know how to stop the training of the model, the current model training is taking up the only resource, but I can't stop it?

- ID: `7ec6fea4411011f1bb183e75ec0e9dd0`
- 链接: https://lexiangla.com/questions/7ec6fea4411011f1bb183e75ec0e9dd0?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

bug trigger: Click stop when the model is not actually trained, the subsequent model will still train, but stop will not be pressed

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## How can we download the available training data?

- ID: `fe1df7f8410f11f18f03eaa59d583e05`
- 链接: https://lexiangla.com/questions/fe1df7f8410f11f18f03eaa59d583e05?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

How can we download the available training data?

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 10:12:59

It seems like we can't due to privacy reasons.
I am still wondering if there is a way to plot some feature distributions for analysis

#### #2 @(匿名) · 2026-04-26 10:52:00

Could the organizers release a larger sample dataset this year? The current 1000 row sample is helpful, but a larger subset would better capture the distribution and sparsity for exploratory analysis. If feasible, could you share an approximate timeline and target size?

---

## 无法正常进入是怎么回事呢

- ID: `e9c8526e410d11f1896e2e84d7011911`
- 链接: https://lexiangla.com/questions/e9c8526e410d11f1896e2e84d7011911?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## in the first-round of the competition, can the team captain be changed?

- ID: `a80076ba40e311f1954752764530454e`
- 链接: https://lexiangla.com/questions/a80076ba40e311f1954752764530454e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:40:45

Before 23:59:59 AOE on April 23, 2026: Team members can be adjusted (including adding, removing, or replacing members)
After April 23, 2026: No team changes are allowed in any form (including joining, withdrawing, or renaming)

---

## Is the Round 1 baseline/starter kit available yet? I checked the Model Management section on AngelML, but the files have no content. Thanks for the help!

- ID: `68767b1040d511f1bf08eaa59d583e05`
- 链接: https://lexiangla.com/questions/68767b1040d511f1bf08eaa59d583e05?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 03:46:35

After clicking the `Create a New Job` button in the `Model Training` tab, you can download the scripts in it:
&nbsp;

#### #2 @(匿名) · 2026-04-26 05:56:55

thank you

---

## 我的推理任务为啥总是等待中啊

- ID: `c8490ef440d011f1b1b2328795a383ba`
- 链接: https://lexiangla.com/questions/c8490ef440d011f1b1b2328795a383ba?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

怎么回事，都登录快一小时了

### 回答 (2)

#### #1 @(匿名) · 2026-04-28 14:31:42

解决了吗，我的从早上九点一直waiting到下午两点了

#### #2 @(匿名) · 2026-04-26 02:01:59

都快等待一小时了，还是等待推理资源

---

## 模型的训练任务没法停下来，咋么办

- ID: `1a45531240d011f192b63eb432f4e775`
- 链接: https://lexiangla.com/questions/1a45531240d011f192b63eb432f4e775?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

这个模型的训练任务已经开始了，我想把它停止，但是停止的按钮是灰色的，怎么办，现在搞得我没有资源训练和推理了，在线急等

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 01:58:01

如果等它跑完估计得好几天，没法停，资源又被占着，这是平台的bug吗

---

## With one task already running and free quota remaining, a newly submitted task keeps switching states and never starts

- ID: `64cec8ca40ce11f1a9ba7e6b586d458c`
- 链接: https://lexiangla.com/questions/64cec8ca40ce11f1a9ba7e6b586d458c?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

When one task is already running and my account still has free GPU quota for another, any newly submitted task can't actually start. Its instance status keeps switching between TRAINING_RESOURCE_WAITING , RESOURCE_WAIT_TRANSITION , and TRAINING_INIT indefinitely &mdash; it briefly reaches TRAINING_INIT , then gets bounced back, and never reaches TRAINING_RUNNING . pod_log stays empty throughout. The fact that it does briefly reach TRAINING_INIT suggests the scheduler agrees a slot exists, so this doesn't look like an actual resource shortage.
Could you check whether the scheduler is correctly handling new submissions when there's already a running task? Thanks!

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## I cant stop my instance on angelml

- ID: `63a51e3c40cd11f1b6ee964e57ab7fcf`
- 链接: https://lexiangla.com/questions/63a51e3c40cd11f1b6ee964e57ab7fcf?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

该instance对应stop一直灰色，想contact assistant，扫二维码提示该用户不存在！

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 11:18:04

Please provide your Team ID and Task ID so we can locate the issue more accurately.

---

## basic questions on dataset

- ID: `0a46968040c411f1a62ba24b4f44eca3`
- 链接: https://lexiangla.com/questions/0a46968040c411f1a62ba24b4f44eca3?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

1.can we save tmp files on disk for training only purpose? what's size limit? USER_CACHE_PATH is only 20G but what if I save to other paths like TRAIN_DATA_PATH?2.I note most training files with timestamp 2026-02-28~2026-03-04 so it is not divided in time order. Also baseline validation set didn't use time series split. How leadboard test set is splitted? does it comes from same period or non-overlapping future?

### 回答 (1)

#### #1 @(匿名) · 2026-04-26 20:29:14

1. TRAIN_DATA_PATH is currently the only accessible path, but it is read-only and cannot be used for storage.
2. The test set timestamps do not overlap with the training set and are from a later time period.

---

## What is the inference time limit for the industrial track?

- ID: `dbae962040c211f199f57e6b586d458c`
- 链接: https://lexiangla.com/questions/dbae962040c211f199f57e6b586d458c?company_from=9cfe03d211a311f1a8588ad12dbe613c

_(未取到结构化回答, 请查看 JSON 中的 answers_raw)_

---

## Existed Resource Queue

- ID: `92f52b3640bf11f1aeb03e75ec0e9dd0`
- 链接: https://lexiangla.com/questions/92f52b3640bf11f1aeb03e75ec0e9dd0?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Why did I find "Exited Resource Queue" several times for my second submitted job？Shouldn't it always wait in the same queue？If adjusted, should be only changed to a faster queue and should not be adjusted too many times, right?

### 回答 (2)

#### #1 @(匿名) · 2026-04-26 00:42:17

I encountered the same situation as the person who asked the question.
Status:

One job has an Instance Status of "Training Job Running"
When I click Run on any subsequent jobs, the Instance Status cycles repeatedly: "Exited Resource Queue" &rarr; "TRAINING_INIT" &rarr; "Waiting for Resources" &rarr; "Exited Resource Queue" &rarr; ...
This cycle for the subsequent jobs appears to loop through "Exited Resource Queue" &rarr; "TRAINING_INIT" &rarr; "Waiting for Resources" roughly every 5&ndash;10 minutes

I have two questions:

Is it not possible for a participant to have multiple jobs in the "Training Job Running" state simultaneously?
Is my understanding correct that the job queue is managed on a per-participant basis, rather than the queue and available GPUs being managed across the entire team?

#### #2 @(匿名) · 2026-04-26 20:07:08

The daily submission limit resets every 24 hours at 15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24‑hour period. Failed or stopped tasks do not count toward the daily limit.

---

## Is there a time limit for the Single training task?

- ID: `c45bc22e40b611f1852a52764530454e`
- 链接: https://lexiangla.com/questions/c45bc22e40b611f1852a52764530454e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

I&rsquo;d like to add some intermediate evaluations during training to help me track the model&rsquo;s progress. I&rsquo;m wondering if this might cause a timeout and result in the training failing?

### 回答 (1)

#### #1 @(匿名) · 2026-04-27 14:17:37

Hello, each team has one computing unit. If you need to start another task, you will need to queue and wait for the previous task to complete. However, queued tasks will not affect the currently running task.

---

## 为什么训练时显示等待资源

- ID: `24fe50c440a811f187a8fea4113eb7a6`
- 链接: https://lexiangla.com/questions/24fe50c440a811f187a8fea4113eb7a6?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

我们已经开始了1个使用1份资源的任务，现在这个新的1个资源的任务一直在pending，我们总共剩下2份资源。

### 回答 (1)

#### #1 @(匿名) · 2026-04-25 21:15:26

This is normal behavior. Please refer to the user guide for instructions：https://doc.weixin.qq.com/doc/w3_ALQATQZyACgCNqA7rGAELTHOn2rc5?scode=AJEAIQdfAAoeeprvmVAPUALQY-AEk&nbsp;

---

## Is this training platform open? I clicked on the button but can't get in.

- ID: `eb87051440a411f19c8e5663b438421f`
- 链接: https://lexiangla.com/questions/eb87051440a411f19c8e5663b438421f?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文



### 回答 (2)

#### #1 @(匿名) · 2026-04-25 20:52:24

Me too
&nbsp;

#### #2 @(匿名) · 2026-04-25 20:52:10

Yes, it is now open. Please check your browser or clear your cache and try again.&nbsp;

&nbsp;

---

## Tencent Angel Machine Learning Platform User Guide

- ID: `648de516409d11f195ef964e57ab7fcf`
- 链接: https://lexiangla.com/questions/648de516409d11f195ef964e57ab7fcf?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

User Guidance

Model Training

Create and submit a training task
1.&nbsp; Fill in "Job Name".
2.&nbsp; Fill in "Job Description".
3.&nbsp; Click "Local Upload" to upload scripts from local files OR click "New Script" to start writing a new script.
&nbsp;

ATTENTION: The training template includes a mandatory run.sh file as the execution entry point, which will be executed automatically at task startup!

&nbsp;
4.&nbsp; After confirmation, click "Submit" to save the task.

Run a task
Click "Run" to start running a task.

Instance status check
After you click "Run", the "Instance Status" automatically changes to "Training Job Running".

Output and Logs check:
Select the task to check and click "Instances". You will be directed to a page containing "Output" and "Logs", both of which can be clicked for more details. You could also click "Stop" to terminate the running task.

Copy Delete Edit example
Click "More" to copy, delete, or edit current task.

Script example

&nbsp;
Model Release And Management
Model release steps
1.&nbsp; Click "Instances" from the training task.

2.&nbsp; Click "Output".

3.&nbsp; Select the ckpt you want to release, click "Publish" and fill in the "Model Name" and "Model Description", after confirmation click "Submit".

4.&nbsp; "Publish Status" will automatically switch to "Released" if submitted successfully.

5.&nbsp; Click "Model Management" (left sidebar) to review and manage released models.

Model edit or delete

&nbsp;
Model Evaluation

&nbsp;
Entrance
Method 1: On the "Model Management" page, locate the model you want to evaluate and click the "Model Evaluation" button at the bottom of its card.

Method 2: On the "Model Evaluation" page, click "Create Evaluation".

Submit evaluation task
 ●&nbsp;Steps
1.&nbsp; Select Model: Choose the trained model you want to evaluate from the dropdown list.
2.&nbsp; Upload Inference Code: Upload your inference scripts via &ldquo;Upload from Local&rdquo; or create new scripts via &ldquo;New Script&rdquo;.
 ●&nbsp;You must include an entry script named "infer.py", which must contain a main() function that takes no arguments.
 ●&nbsp;You may also upload other supporting scripts (e.g., "dataset.py", "model.py", or any custom modules) that your "infer.py" imports. All uploaded files will be placed in the EVAL_INFER_PATH directory.
 ●&nbsp;The total size of all uploaded scripts is limited to 100 MB.
3.&nbsp; Submit: Click the &ldquo;Submit&rdquo; button to create the evaluation task.
&nbsp;

ATTENTION: 
1.&nbsp; The script must be strictly named "infer.py" and must contain a main() function that takes no arguments, otherwise the evaluation would fail!
2.&nbsp; The daily submission limit resets every 24 hours at 15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24‑hour period. Failed or stopped tasks do not count toward the daily limit.
3.&nbsp; A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

&nbsp;

&nbsp;

 ●&nbsp;Evaluation Status
Once submitted, the evaluation task will go through the following stages:

Status

Description

Pending

The task has been submitted and is queued.

Waiting for Inference Resources

Waiting for available compute resources.

Inference Running

Your "infer.py" script is executing.

Waiting for Evaluation Resources

Inference completed. Waiting for scoring resources.

Evaluation Running

The platform is scoring your predictions.

Success

Evaluation completed. You can view your score.

Failed

Evaluation failed. Check the logs for details.

&nbsp;
Install dependencies
 ●&nbsp;If you wish to execute shell commands to install dependencies &mdash; similar to the "run.sh" script used during training &mdash; you can upload a "prepare.sh" script (which must be named prepare.sh) when creating an evaluation task. This script runs automatically before inference starts, directly within the pre-activated Conda environment. A template is provided below:
&nbsp;
#!/bin/bash&nbsp;# 1. This script will be executed before the inference logic starts.# 2. This script runs inside the activated conda environment.# 3. You can execute your own preparation commands here, such as:#      conda install &lt;package&gt;#      or any other setup commands you need.
&nbsp;
&nbsp;
Platform Specifications
Operating Environment
Model training and evaluation run in a consistent hardware and software environment.
1.&nbsp; Hardware Environment 
The platform deploys high-performance GPUs with virtualized resource allocation. Each GPU partition meets the following specifications:

Resource 

Specification

Computing Power

20% of a single GPU

GPU Memory

19GiB

CPU Cores

9

Memory

55GiB

2.&nbsp; Software Environment 
 ●&nbsp;The operating system is Ubuntu 22.04. The following foundational software packages are pre-installed:

Software

Version

cuda

12.6

cudnn

9.5.1

cublas

12.6.3.3

nccl

2.26.2+cuda12.6

conda

26.1.1

python

3.10.20

 ●&nbsp;Installed Python packages
&nbsp;
Package                      Version---------------------------- ----------------absl-py                      2.4.0accelerate                   0.32.1aiohappyeyeballs             2.6.1aiohttp                      3.13.3aiosignal                    1.4.0albucore                     0.0.24albumentations               2.0.8alembic                      1.18.4annotated-doc                0.0.4annotated-types              0.7.0astunparse                   1.6.3async-timeout                5.0.1attrs                        25.4.0audioread                    3.1.0beautifulsoup4               4.14.3bitsandbytes                 0.49.2blis                         1.3.3Bottleneck                   1.4.2catalogue                    2.0.10catboost                     1.2.10certifi                      2026.2.25cffi                         2.0.0charset-normalizer           3.4.6click                        8.3.1cloudpathlib                 0.23.0cloudpickle                  3.1.2colorlog                     6.10.1confection                   0.1.5contourpy                    1.3.1cycler                       0.12.1cymem                        2.0.13datasets                     2.14.7decorator                    5.2.1dill                         0.3.7duckdb                       1.4.4einops                       0.8.2en_core_web_sm               3.8.0fbgemm_gpu                   1.2.0+cu126filelock                     3.25.2flatbuffers                  25.12.19fonttools                    4.61.0frozenlist                   1.8.0fsspec                       2023.10.0gast                         0.7.0gensim                       4.4.0google-pasta                 0.2.0graphviz                     0.21greenlet                     3.3.2grpcio                       1.78.0h5py                         3.16.0huggingface-hub              0.17.3idna                         3.11ImageIO                      2.37.3importlib_metadata           8.7.1iopath                       0.1.10Jinja2                       3.1.6joblib                       1.5.3jsonlines                    4.0.0keras                        3.12.1kiwisolver                   1.4.9lazy-loader                  0.5libclang                     18.1.1librosa                      0.11.0lightgbm                     4.6.0lightning-utilities          0.15.3llvmlite                     0.46.0Mako                         1.3.10Markdown                     3.10.2markdown-it-py               4.0.0MarkupSafe                   3.0.3matplotlib                   3.10.8mdurl                        0.1.2mkl_fft                      2.1.1mkl_random                   1.3.0mkl-service                  2.5.2ml_dtypes                    0.5.4mpmath                       1.3.0msgpack                      1.1.2multidict                    6.7.1multiprocess                 0.70.15murmurhash                   1.0.15mypy_extensions              1.1.0namex                        0.1.0narwhals                     2.18.0networkx                     3.4.2nltk                         3.9.3numba                        0.64.0numexpr                      2.14.1numpy                        2.2.5opencv-python                4.13.0.92opencv-python-headless       4.13.0.92opt_einsum                   3.4.0optree                       0.19.0optuna                       4.8.0orjson                       3.11.7packaging                    25.0pandas                       2.3.3patsy                        1.0.2peft                         0.13.2pillow                       12.1.1pip                          26.0.1platformdirs                 4.9.4plotly                       6.6.0polars                       1.39.2polars-runtime-32            1.39.2pooch                        1.9.0portalocker                  3.2.0preshed                      3.0.12propcache                    0.4.1protobuf                     5.29.6psutil                       7.2.2pyarrow                      23.0.1pyarrow-hotfix               0.7pycparser                    3.0pydantic                     2.12.5pydantic_core                2.41.5Pygments                     2.19.2pyparsing                    3.2.5PyQt6                        6.10.2PyQt6_sip                    13.11.0pyre-extensions              0.0.32python-dateutil              2.9.0.post0pytorch-lightning            2.6.1pytz                         2026.1.post1pyvers                       0.2.2PyYAML                       6.0.3regex                        2026.2.28requests                     2.32.5rich                         14.3.3sacremoses                   0.1.1safetensors                  0.7.0scikit-image                 0.25.2scikit-learn                 1.7.2scipy                        1.15.3seaborn                      0.13.2sentencepiece                0.2.1setuptools                   80.10.2shellingham                  1.5.4simsimd                      6.5.16sip                          6.15.1six                          1.17.0smart_open                   7.5.1soundfile                    0.13.1soupsieve                    2.8.3soxr                         1.0.0spacy                        3.8.11spacy-legacy                 3.0.12spacy-loggers                1.0.5SQLAlchemy                   2.0.48srsly                        2.5.2statsmodels                  0.14.6stringzilla                  4.6.0sympy                        1.14.0tensorboard                  2.19.0tensorboard-data-server      0.7.2tensordict                   0.11.0tensorflow                   2.19.0tensorflow-io-gcs-filesystem 0.37.1termcolor                    3.3.0textblob                     0.19.0thinc                        8.3.10threadpoolctl                3.5.0tifffile                     2025.5.10timm                         1.0.25tokenizers                   0.14.1tomli                        2.4.0torch                        2.7.1+cu126torch_cluster                1.6.3+pt27cu126torch-geometric              2.7.0torch_scatter                2.1.2+pt27cu126torch_sparse                 0.6.18+pt27cu126torchaudio                   2.7.1+cu126torchmetrics                 1.0.3torchrec                     1.2.0+cu126torchtext                    0.18.0torchvision                  0.22.1+cu126tornado                      6.5.4tqdm                         4.67.3transformers                 4.35.0triton                       3.3.1typer                        0.24.1typer-slim                   0.24.0typing_extensions            4.15.0typing-inspect               0.9.0typing-inspection            0.4.2tzdata                       2025.3urllib3                      2.6.3wasabi                       1.1.3weasel                       0.4.3Werkzeug                     3.1.6wheel                        0.46.3wrapt                        2.1.2xgboost                      3.2.0xxhash                       3.6.0yarl                         1.23.0zipp                         3.23.0
&nbsp;
 ●&nbsp;To install other packages, use conda or pip3.
 For instance: to install a package like pandas (Note: pandas is pre-installed in the default environment; this is just an example).
&nbsp;
# use condaconda install -y pandas&nbsp;# use pip3pip3 install pandas
&nbsp;
Model Training: 
1.&nbsp; Environment Variables 
The platform provides directories required for training (e.g., training datasets, outputs, logs, etc.), and passes them into the container as environment variables. You can read these variables in your script.

Variable Name

Description

USER_CACHE_PATH

User cache storage path (quota: 20GB). 
This variable is provided for both training and evaluation, allowing you to share files between these two stages.

TRAIN_DATA_PATH

Path to training datasets.

TRAIN_CKPT_PATH

Path for saving model checkpoints. 

TRAIN_TF_EVENTS_PATH

Path for TensorBoard event files.

In a shell script, you can read environment variables as follows:
&nbsp;
${TRAIN_DATA_PATH}
&nbsp;
In a Python script, you can read environment variables as follows：
&nbsp;
import osos.environ.get("TRAIN_DATA_PATH")
&nbsp;
2.&nbsp; Output Specifications
2.1. Model Output
a. Model weights must be saved to the directory specified by TRAIN_CKPT_PATH.
b. Iterative checkpoint files must be placed in a directory prefixed with global_step; otherwise, the platform will not recognize them. The directory name must not exceed 300 characters.
c. Directory names may only contain letters (a-z, A-Z), numbers (0-9), underscores (_), hyphens (-), equal signs (=), and periods (.).
&nbsp;

To include additional parameter information in the directory name, append it after "global_step". 
For example, a checkpoint from the 20th iteration with learning rate (lr=0.001), layer count (layer=2), head count (head=1), hidden size (hidden=50), and max sequence length (maxlen=200) should be saved under: "global_step20.lr=0.001.layer=2.head=1.hidden=50.maxlen=200"

&nbsp;
2.2. TensorBoard Metrics
&nbsp;

For PyTorch TensorBoard usage, refer to:https://docs.pytorch.org/docs/2.7/tensorboard.html#

&nbsp;
As shown in the following code, you can read the "TRAIN_TF_EVENTS_PATH" environment variable to initialize a SummaryWriter. The platform only supports scalar metrics.
&nbsp;
from torch.utils.tensorboard import SummaryWriterwriter = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
&nbsp;
Model Evaluation:
1.Environment Variables
The platform provides directories needed during the evaluation process (such as evaluation datasets, outputs etc.) and passes them into the container as environment variables. You can read these variables in your script.

Variable Name

Description

USER_CACHE_PATH

User cache path, quota 20GB.
This variable is provided for both training and evaluation, allowing you to share files between these two stages.

MODEL_OUTPUT_PATH

Model output path.

EVAL_DATA_PATH

Path to test data directory for inference.

EVAL_RESULT_PATH

Output path for intermediate results and "predictions.json".

EVAL_INFER_PATH

Directory for user-uploaded inference script files.

In a shell script, you can read environment variables as follows:
&nbsp;
${EVAL_DATA_PATH}
&nbsp;
In a Python script, you can read environment variables as follows：
&nbsp;
import osos.environ.get("EVAL_DATA_PATH")
&nbsp;
2.&nbsp; Output Specifications
The script must be strictly named "infer.py" and must contain a main() function that takes no arguments, otherwise the evaluation would fail!
The main() function must produce a "predictions.json" file and save it under the EVAL_RESULT_PATH directory.
&nbsp;
 ●&nbsp;predictions.json Format
The JSON file must contain a predictions field, which is a mapping from user_id (string) to the predicted conversion probability (float, range 0&ndash;1).

Field

Data Format

Description

predictions

map&lt;string, float&gt;

A mapping where each key is a user_id (string) and each value is the model&rsquo;s predicted probability that the user&rsquo;s interaction is positive (i.e., a conversion).

&nbsp;
 ●&nbsp;Example of predictions.json:
&nbsp;
{    "predictions": {        "user_001": 0.8732,        "user_002": 0.1245,        "user_003": 0.5621    }}
&nbsp;
&nbsp;

Attention：Each key in the predictions map must be a valid user_id from the test dataset, and the corresponding value must be the model&rsquo;s predicted conversion probability for that user&rsquo;s interaction. Missing or extra user_ids may affect the final score!

&nbsp;

### 回答 (4)

#### #1 @(匿名) · 2026-04-26 18:37:27

How can we access Tencent Angel Machine Learning Platform User Guide without "WeCom"?

#### #2 @(匿名) · 2026-04-25 20:19:39

Since the competition has already begun, we highly recommend that all participants review the official User Guide. This manual contains essential operational workflows and technical instructions to help you navigate the platform and optimize your performance.
You can access the document via the following link:
Tencent Angel Machine Learning Platform User Guide
Good luck with the competition!

#### #3 @(匿名) · 2026-04-29 19:24:24

I submitted the official baseline script for successful training and inference, but there is no ranking. Is there a problem with the system?&nbsp;

#### #4 @(匿名) · 2026-04-28 00:00:31

Hi. I submitted the provided sample submission (without changing anything) and it failed.&nbsp;
![]( https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Apr-2026/e1.png )
And then when i view the logs, the logs are empty. Any advice on how to submit the sample submission successfully?
&nbsp;
UPDATE: The problem may be that the sample submission code doesn't set the `TRAIN_LOG_PATH` variable. Trying this now.
UPDATE2: Fixing variable above solved first error and now it begins training and makes a log. However now there is an OOM error. Fixing that now...

---

## How to train and submit on the official platform?

- ID: `2890fd603fca11f1885e06932b6e0744`
- 链接: https://lexiangla.com/questions/2890fd603fca11f1885e06932b6e0744?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (3)

#### #1 @(匿名) · 2026-04-24 18:41:42

Regarding the detailed user manual for the Angel machine learning platform, it will be officially released after the competition starts. Please stay tuned for relevant announcements on the official website for detailed guidance.&nbsp;

#### #2 @(匿名) · 2026-04-25 17:06:33

Yes, our team also didn't find the corresponding website address and dataset of the computing power platform on the official website. I want to know exactly when it will be released. According to the AOE time schedule, shouldn't the competition have already started by now?

#### #3 @(匿名) · 2026-04-25 16:46:43

Excuse me, has the announcement about the competition dataset and computing resources been published yet? Our team hasn't seen it.

---

## 📘 2026 Tencent Advertising Algorithm Competition Player Guide

- ID: `55556dce3ef611f1bd34366a677e2e06`
- 链接: https://lexiangla.com/questions/55556dce3ef611f1bd34366a677e2e06?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Tencent LearnShare AI Knowledge Base &mdash; Official Knowledge Management Platform for the Competition
🎯 1. Why Choose Tencent LearnShare AI Knowledge Base?
As the official knowledge management platform designated for the competition, Tencent LearnShare AI Knowledge Base helps participating teams:

✅Centralized Management:Unified storage of scattered papers, code, and experimental records
✅AI Assistance:Automatic parsing of literature, generating overviews, and extracting key information
✅Team Collaboration:Multi-user real-time editing, task assignment, and progress tracking
✅Knowledge Accumulation:Transform competition experience into reusable knowledge assets
✅Dialogue Integration:Support integration with Openclaw, Workbuddy, and mainstream IM platforms
✅Unlimited Tokens:Unlimited supply of Knowledge Agent, literature parsing, infographics, PPT generation, and other features

📋 2. Quick Start: 5 Steps to Build Your Team's Exclusive Knowledge Base
Step 1: Login
Access the competition's exclusive invitation link to log in to the knowledge base:
https://url.LearnShare-asset.com/0HH95xkm
(Participants are registered by default and can log in directly)
Step 2: Build Knowledge Base Directory Structure
Create a knowledge base, then use the natural language "Ask AI" function: Please refer to the knowledge base structure of "2026 Tencent Algorithm Competition - Team ABC", create a knowledge base with the same structure in Team ABC's knowledge base, or directly copy the official template into the dialogue box and let the Agent create it.

📚 3. Core Function User Guide
Function 1: Batch Upload Literature Materials
Applicable Scenario:Quickly import large quantities of PDF papers and dataset documents
Operation Steps:

Click [Upload] button &rarr; Select [Upload Folder]
Select locally organized literature folder (supports 10-level directories, 1000 files per level)
Supported formats: PDF, Word, PPT, Excel, Markdown, and 103 other formats
Maximum 200MB per file, system will automatically parse content for AI Q&amp;A

Pro Tips:

Recommend organizing folders by topic (e.g., "Transformer Papers", "CTR Prediction Methods")
File naming convention: Author_Year_PaperTitle.pdf

Function 2: AI-Generated Literature Overview
Applicable Scenario:Quickly understand core paper content, save reading time
Operation Steps:

Open uploaded paper file
Click [Ask AI] button on the right
Select question mode:

Quick Answer:Suitable for simple questions (e.g., "What is the main contribution of this paper?")
Deep Thinking:Suitable for complex analysis (e.g., "Compare the advantages and disadvantages of this method with BERT")
Professional Research:Suitable for systematic research (e.g., "Summarize the development trends of sequence modeling in the past three years")

Recommended Question Templates:
&nbsp;
- "Please summarize the core ideas of this paper in 300 words"
- "Extract the key formulas and algorithm workflows from the paper"
- "What aspects of the experimental setup in this paper are worth learning from?"
- "Where can I download the datasets mentioned in the paper?"
&nbsp;
Function 3: Generate Mind Maps to Organize Knowledge Context
Applicable Scenario:Organize literature reviews, clarify technical roadmaps
Operation Steps:

Select documents to organize (can @ up to 20 files)
Click [Mind Map] function
AI automatically generates structured mind maps, supporting:

Zoom to full screen view
Mind map/outline dual mode switching
Export as image for sharing

Application Examples:

Organize "Recommendation System Development History" (from collaborative filtering to deep learning)
Compare "Different Feature Interaction Methods" (FM, DeepFM, DCN, AutoInt)

Function 4: Intelligent Classification and Organization of Literature
Applicable Scenario:Too much literature, need automatic categorization
Operation Steps:

Enable [Agent Mode] (check in Ask AI interface)
Issue instructions to AI:
"Please help me classify the 50 papers in '02-Literature Repository' according to the following dimensions:
- Sequential modeling methods (RNN/Transformer/Mamba, etc.)
- Feature interaction methods (FM series/Attention mechanisms/Graph neural networks)
- Application scenarios (Advertising recommendation/E-commerce recommendation/Video recommendation)
And create subfolders for each category, then move the corresponding papers"

AI will automatically execute classification tasks and generate classification reports

Function 5: Generate Knowledge Infographics
Applicable Scenario:Visualize complex concepts for team learning or defense presentations
Operation Steps:

Select documents for infographics (e.g., "Transformer Principle Explanation")
Click [Knowledge Infographics]
AI generates key point infographics, clarifying knowledge context
Can be directly inserted into defense PPT or technical reports

Function 6: One-Click PPT Generation
Applicable Scenario:Final defense, mid-term report
Operation Steps:

Prepare technical report or experimental record documents
Click [PPT Creation]
Fill in requirements:

Target audience: Expert judges
Page count requirement: 15-20 pages
Style preference: Academic professional style

AI automatically generates PPT based on document content, including:

Competition background and challenges
Technical solution architecture diagram
Experimental results comparison table
Innovation summary

👥 4. Team Collaboration Best Practices
Collaboration Scenario 1: Multi-User Real-Time Editing of Experiment Logs

Create online document "Model Experiment Record Table"
Team members edit simultaneously, real-time synchronization
Support historical version rollback, avoid accidental deletion

Collaboration Scenario 2: Task Assignment and Progress Tracking

Use [To-Do Task] function in documents
Example:
## Weekly Tasks (April 1-7, 2026)
- [ ] Zhang San: Complete data preprocessing script @Zhang San Deadline: 4.3
- [ ] Li Si: Research Transformer variant models @Li Si Deadline: 4.5
- [ ] Wang Wu: Build baseline model @Wang Wu Deadline: 4.7

Check off when task is completed, automatically update progress

Collaboration Scenario 3: Annotate and Discuss Technical Solutions

Add annotations to key paragraphs in technical reports
Teammates can reply to annotations, forming discussion threads
Suitable for asynchronous communication, avoid information loss

🔍 5. Advanced Application Scenarios
Scenario 1: Literature Research and Review Writing

Upload 50+ related papers to knowledge base
Use [Professional Research] mode to ask:
"Based on the uploaded papers, write a 3000-word review on sequential modeling techniques, 
including: development history, comparison of mainstream methods, and future trends"

AI automatically generates review draft, manually polish and save to knowledge base

Scenario 2: Experimental Results Comparison Analysis

Create smart table "Model Performance Comparison Table"
Enter AUC, LogLoss, and other metrics for different models
Use [analyze_table] function:
"Analyze which model has the best overall performance in the table, 
and generate a bar chart to compare the performance of each model across different metrics"

Scenario 3: Code and Document Linkage Management

Upload Python scripts to knowledge base
Create online document "Code Description" next to script
Use AI assistance:
"Read the train.py script and generate detailed code comments and usage instructions"

Scenario 4: Automatic Meeting Minutes Generation

Use Tencent Meeting to record team discussions
Import meeting recording to knowledge base (supports Tencent Meeting one-click import)
AI automatically extracts:

Key meeting conclusions
To-do items
Technical difficulty discussions

🛡️ 6. Data Security and Permission Management
Permission Setting Recommendations

Team Leader:Management permission (can delete, move files)
Core Members:Edit permission (can upload, modify documents)
Advisor:View-only permission (can view all content, cannot edit)

Sensitive Information Protection

Enable page watermark (display viewer name + time)
Prohibit downloading core code and experimental data
Set document expiration date (automatic expiration reminder after competition ends)

📊 7. Knowledge Base Usage Effect Evaluation
Data Dashboard Function
Regularly check knowledge base data statistics:

Team member activity (who contributes the most content)
Knowledge popularity ranking (which documents are viewed most)
AI Q&amp;A usage trends (which questions the team invests effort in)

Success Case Reference
Effect of a Top 3 team using LearnShare Knowledge Base:

Literature management efficiency increased10x(from manual organization to AI automatic classification)
Team communication cost reduced60%(asynchronous collaboration reduces meeting time)
Technical report writing speed increased3x(AI-assisted draft generation)

🎓 8. Frequently Asked Questions (FAQ)
Q1: Is there a limit on knowledge base capacity?
A: Supports single files up to 10GB, unlimited total capacity.
Q2: Will AI Q&amp;A leak our technical solutions?
A: No. AI accesses as user identity, strictly follows permission control, only visible to team members.
Q3: Can the knowledge base still be used after the competition?
A: No. This knowledge base is only for use during the competition. After the competition, it is recommended to migrate to a personal knowledge base as a team knowledge asset.
Q4: Is mobile access supported?
A: Yes! Accessible anytime via mini-program or WeChat .
Q5: How to export knowledge base content?
A: Supports batch export to PDF, Word, and other formats, convenient for paper submission.
🚀 9. Advanced Tips

Use @ Function:When asking AI, @ multiple documents for cross-literature comparative analysis
Build Template Library:Save commonly used experiment record tables and weekly report templates as templates
Regular Inspection:Tell Agent to check knowledge base when asking AI, delete duplicate and outdated content, maintain high purity
Tag Management:Add tags to documents (e.g., #Transformer #Feature Engineering) for easy retrieval

&nbsp;
🏆 Conclusion
Tencent LearnShare AI Knowledge Base is not just a file storage tool, but your intelligent teammate. It can help you:

📖Read Papers:Automatically extract key information
🧠Analyze:Generate comparison reports and visualization charts
✍️Write Documents:Assist in writing technical reports and defense PPTs
🤝Promote Collaboration:Enable efficient team progress synchronization

Start building your team's knowledge base now and let AI become your secret weapon for winning the championship!
This guide is written by LearnShare Agent based on Tencent LearnShare AI Knowledge Base official documentation, more features are continuously being updated...

&nbsp;

### 回答 (1)

#### #1 @(匿名) · 2026-04-25 20:21:06

We have provided a free AI Knowledge Base for all participants. Please refer to the operational instructions provided above. 
Good luck with the competition!

---

## "Registration successful" remains unchecked after team formation – Is further action required?

- ID: `254d4a343e2911f1befaf242e4f09224`
- 链接: https://lexiangla.com/questions/254d4a343e2911f1befaf242e4f09224?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Dear Organizers,
I have a question regarding my registration status on the dashboard.
Current Status:

I have completed all steps up to "form team".

My team is created, and all members have successfully joined.

However, the final "Registration successful" step remains unchecked.

Questions:

Should we simply wait for the competition to begin, as this is the farthest we can proceed right now?

Or are there any remaining actions required on our end (e.g., uploading a document via the "upload resume" button)?

Thank you for your support!

### 回答 (1)

#### #1 @(匿名) · 2026-04-22 17:16:08

Please check your registration status within your Personal Center. You must complete the following steps: Registration -&gt; Real-Name Verification -&gt; Team Formation; only upon completing these steps will your registration be considered successful. Steps 1 and 2 will appear grayed out with a checkmark, while Step 3&mdash;"Registration Successful"&mdash;will appear in purple.

---

## Why can my teammates not find my team via my invitation code, and why is the Verify Now button unclickable for Identity Verification?

- ID: `c02e8c583e2811f19e7d2ab698a0006d`
- 链接: https://lexiangla.com/questions/c02e8c583e2811f19e7d2ab698a0006d?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-22 16:53:59

Dear participants,The team formation function has now been restored following an emergency investigation and repair by our technical team. We sincerely apologize for any inconvenience this may have caused.Please refresh the page and try again to create or join a team. If you continue to experience any issues, feel free to contact us at the official email address: taac@tencent.com. We will assist you promptly.Thank you for your understanding and patience. We wish you the best of luck in forming your team and preparing for the competition!

---

## Hello, I joined the team of others and entered the invitation code, but it shows that the team doesn't exist. How can I solve this problem?

- ID: `ae813e983e2711f1b9643e4c6182b081`
- 链接: https://lexiangla.com/questions/ae813e983e2711f1b9643e4c6182b081?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-22 16:46:48

Dear participants,The team formation function has now been restored following an emergency investigation and repair by our technical team. We sincerely apologize for any inconvenience this may have caused.Please refresh the page and try again to create or join a team. If you continue to experience any issues, feel free to contact us at the official email address: taac@tencent.com. We will assist you promptly.Thank you for your understanding and patience. We wish you the best of luck in forming your team and preparing for the competition!

---

## 已经创建队伍，为什么邀请码队员识别显示队伍不存在，所有人已完成注册验证前提下。

- ID: `8abcacb23d6a11f185fe56dcfbd34f2b`
- 链接: https://lexiangla.com/questions/8abcacb23d6a11f185fe56dcfbd34f2b?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-04-21 18:17:58

For inquiries regarding team information, please contact our official email address at taac@tencent.com, and attach a screenshot of the issue concerning your team's details.

---

## platform details

- ID: `2538adf8395c11f1a5445e0ebcfdd39e`
- 链接: https://lexiangla.com/questions/2538adf8395c11f1a5445e0ebcfdd39e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Since my team's preparation plan depends heavily on the platform details, could you please clarify:

ardware: GPU model and count per team (e.g., 1&times; / 2&times; / 8&times; A100 80G? H100? V100?), CPU/RAM, and whether multi-node training is supported.

Compute quota: Is it unlimited during the competition phase, or capped by GPU-hours / concurrent-job limit per team? Is the quota shared across Phase 3 (first round) and Phase 4 (second round)?

Data access: Will the full dataset be pre-loaded on the platform? Is data download to local machines allowed, or must all development happen on-platform?

Environment: What framework versions are pre-installed (PyTorch, CUDA)? Is internet access available for&nbsp;pip install&nbsp;of additional packages? Can we upload our own code / pretrained weights / open-source models?

Latency evaluation: Will the inference latency be measured on the same hardware as the training platform, or a separate evaluation machine? If separate, what is its spec?

### 回答 (1)

#### #1 @(匿名) · 2026-04-16 14:19:32

The official competition platform is GPU-based, please refer to the "RCOMPETITION SETTING" at "algo.qq.com/rules".&nbsp;(1) Round 1. We have prepared resources to support up to 5,000 teams participating concurrently.(2) Round 2. Each team will be provisioned with approximately eight high-performance GPUs on the official platform, subject to platform policies, fair-use constraints, and resource availability.&nbsp;(3) The dataset is not allowed to be downloaded to the local device.
Regarding more detailed user manual for the Angel machine learning platform, it will be officially released after the competition starts. Please stay tuned for relevant announcements on the official website for detailed guidance.&nbsp;

---

## Will the data be compressed using any algorithm? Will the organizing committee provide sufficient storage space for participants to store preprocessed datasets?

- ID: `1ad388b6395b11f199f55ae828b833da`
- 链接: https://lexiangla.com/questions/1ad388b6395b11f199f55ae828b833da?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

We would like to consult the organizing committee on several specific details:1. How many data records will be provided in the official competition dataset?2. What is the approximate total size of the data files?3. Will the data be compressed using any algorithm?4. Will the organizing committee provide sufficient storage space for participants to store preprocessed datasets?Thank you very much for your time and assistance.

### 回答 (1)

#### #1 @(匿名) · 2026-04-16 14:14:21

The complete training dataset will be released, synchronized with the Round 1 start time.Data scale: The training set contains about 100 million samples, and the validation set contains about 10 million samples. The dataset will be available for download on the official competition website after the competition begins.We have also published a sample dataset of 1,000 records on HuggingFace (https://huggingface.co/datasets/TAAC2026/data_sample_1000) for participants to familiarize themselves with the data format.Round 1. We have prepared resources to support up to 5,000 teams participating concurrently.For more detailed information about the dataset, it will be officially released after the competition starts. Please stay tuned for relevant announcements on the official website for detailed guidance.&nbsp;

---

## Update Identification Verification Info

- ID: `177ebcee370811f1b9dfca168d19eca9`
- 链接: https://lexiangla.com/questions/177ebcee370811f1b9dfca168d19eca9?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Hi! The identification verification page was constantly failing for me. So for the latest attempt I put a wrong id but it succeded. Seems like I'm not able to modify the ID info anymore, can someone help assist to update this for me? Thanks!

### 回答 (1)

#### #1 @(匿名) · 2026-04-13 15:15:17

Please contact the official email address at taac@tencent.com to request a modification to your personal information.Please include a copy of your ID card or passport, along with a screenshot of your Personal Center profile.

---

## Are recent postgraduate graduates eligible for the Academic Track?

- ID: `ba569f1c34a711f1b1132e05d1235a3a`
- 链接: https://lexiangla.com/questions/ba569f1c34a711f1b1132e05d1235a3a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

I graduated last August and will begin employment within one year of graduating. I&rsquo;m currently exploring opportunities while continuing to build my experience in LLM-based ranking.

### 回答 (1)

#### #1 @(匿名) · 2026-04-10 14:38:41

Detailed information about track eligibility can be found in the 「Rules」 section on our official website: algo.qq.com.&nbsp;Regarding the enrollment requirement for the Academic Track, you are eligible as long as you maintain your student status by the registration deadline (23:59:59 AOE, April 23).&nbsp;

---

## Inquiry: How to access the official TAAC Communication Group and AI Bot mentioned in the Q&A?

- ID: `e31c6516324b11f1bfeac20ea8da74e8`
- 链接: https://lexiangla.com/questions/e31c6516324b11f1bfeac20ea8da74e8?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Hi TAAC Organizers,
I recently read the "2026 TAAC Rules and Q&amp;A" document. Under the "Operations and Support" section, it explicitly mentions a comprehensive multi-level Q&amp;A system, including:

A 7*24 AI Bot.
An official community group with tech staff online.
Fixed expert Q&amp;A time windows.
Curated QA summaries pushed to the group.

However, I couldn't find the entry points or QR codes for these resources on my user dashboard or the main website.
Could you please clarify where we can interact with the AI Bot and&nbsp;how we can join the official community group (e.g., WeChat/QQ)? Are the entry links going to be released soon?
Thank you!

附中文原意（致运营同学）： 您好，在释出的【对外版】26年算法大赛规则常见Q&amp;A.xlsx中，官方明确承诺会提供&ldquo;专属 AI Bot&rdquo;、并配有专人的&ldquo;官方社群&rdquo;以及&ldquo;固定时间窗专家答疑&rdquo;。 但目前我在个人中心和官网上均未找到这些相关社群的加群二维码或 AI Bot 的接入口。请问该如何加入具体的官方社群？是否会在近期开放具体入口？感谢解答！

### 回答 (1)

#### #1 @(匿名) · 2026-04-07 14:37:32

This Community section on the official website serves as the sole official communication channel for this competition.

---

## How to get access to the dataset?

- ID: `d7a31fae324b11f1852bf63c890bc997`
- 链接: https://lexiangla.com/questions/d7a31fae324b11f1852bf63c890bc997?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

According to the announcement, the dataset has been released. However, the link seems not available for Chinese mainland.

### 回答 (1)

#### #1 @(匿名) · 2026-04-07 14:40:30

The baseline will be released after the competition officially starts. Stay tuned!

---

## If an architecture uses conditional computation like MoE, where sub-networks are jointly trained end-to-end under a single objective, does it violate the ensemble rule?

- ID: `3aaa43562f4111f19cc6a685989d5f48`
- 链接: https://lexiangla.com/questions/3aaa43562f4111f19cc6a685989d5f48?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (2)

#### #1 @(匿名) · 2026-04-19 02:34:47

If the key rational here is the "joint trained" with a loss function? What If we train different experts separately and merged with gating layers fine-tuning?&nbsp;

#### #2 @(匿名) · 2026-04-03 17:43:12

No, it does not violate the ensemble rule.
The key distinction is that ensemble methods require multiple independently trained models whose predictions are combined (e.g., via averaging, voting, stacking). In contrast, MoE is a single model where experts and the gating network are jointly trained end‑to‑end under one objective. The experts are not independent&mdash;they cooperate and specialize through shared training dynamics.&nbsp;

---

## Is seq_feature wrong in hf dataset

- ID: `176993802cc911f18c14f63c890bc997`
- 链接: https://lexiangla.com/questions/176993802cc911f18c14f63c890bc997?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

for introduction, the seq_feat look like the above picture, but in hf, the seq look like the following picture

and i want to know what the feature_id is in action_seq\content_seq\item_seq，it seem like the feature_id in seq_feat is not from item or user feat.
so what is the meaning of feature_id in seq_feature?

### 回答 (1)

#### #1 @(匿名) · 2026-04-01 14:41:04

The seqs in the introduction is just a simple example. The final version will be similar to the demo dataset on HuggingFace. The feature ids in action_seq/content_seq/item_seq refer to sideinfo in the seqs, which are different from the original item_feat/user_feat. For example, for the action_seq, there exist timestamp, action type and so on. For the item_seq, there also exist timestamp and item categories, etc. And for the content_seq, there exist some seq features about the content domain.

---

## Is there any official group for the competition? How can I join it?

- ID: `5fb69f922be911f1a9ba2ef473905ed5`
- 链接: https://lexiangla.com/questions/5fb69f922be911f1a9ba2ef473905ed5?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

比赛有官方群吗

### 回答 (2)

#### #1 @(匿名) · 2026-04-01 15:44:42

你可以去kaggle里面看看

#### #2 @(匿名) · 2026-03-30 11:35:55

This Community section on the official website serves as the sole official communication channel for this competition.

---

## Hello, may I ask how registered teams can claim computing resources? What means first registered, first admitted. Can they be claimed after the competition starts? Thank you.

- ID: `90b5ab2429b911f1a4918e218cf48ea7`
- 链接: https://lexiangla.com/questions/90b5ab2429b911f1a4918e218cf48ea7?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-03-27 16:47:54

1.After the competition officially starts, Tencent will provide a dedicated computing platform to ensure a fair contest environment.2.Capacity-aware admission (if needed). We have prepared resources to support up to 5,000 teams participating concurrently. If the total number of registered teams across all tracks exceeds 5,000, the organizers may introduce additional rules to determine (i) which teams are eligible to enter the competition and (ii) for eligible teams, when they may enter.&nbsp;Detailed information &nbsp;can be found in the 「Rules」 section on our official website: algo.qq.com.&nbsp;

---

## When will the baseline be available online?

- ID: `44b40aea272a11f189bceeeffbc12236`
- 链接: https://lexiangla.com/questions/44b40aea272a11f189bceeeffbc12236?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-03-24 10:37:30

The baseline will be released after the competition officially starts. Stay tuned!

---

## Users should be allowed to preview and verify their personal information before submission.

- ID: `d3ad7ff4265f11f1b1312290b1e4dbc4`
- 链接: https://lexiangla.com/questions/d3ad7ff4265f11f1b1312290b1e4dbc4?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Users should be allowed to preview and verify their personal information before submission. Currently, key details like ID numbers are partially masked, preventing users from confirming their accuracy.

### 回答 (2)

#### #1 @(匿名) · 2026-03-23 14:20:38

Hello, contestants must confirm their personal information during the filling and editing process. Certification will succeed only when all real-name details&mdash;country/region, ID number, and full name&mdash;match exactly. If any information is incorrect, an error prompt will appear during verification. Other details (such as institution, education level, company, job title, etc.) can be updated in your profile. Contestants must ensure that all information provided is accurate and valid.

#### #2 @(匿名) · 2026-03-23 14:58:08

Thanks for your reply. Just to confirm: for users in Mainland China, if the name and ID number do not match during registration, the system will trigger an error prompt. Once the system accepts the input, it means the real-name authentication has been successfully verified.&nbsp;

---

## How many teams are allowed to participate in each track respectively?

- ID: `c83c5b8a25fc11f1bdd0ce838936e099`
- 链接: https://lexiangla.com/questions/c83c5b8a25fc11f1bdd0ce838936e099?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

How many teams are allowed to participate in each track respectively?
How can I confirm that my registration has been successful?

### 回答 (2)

#### #1 @(匿名) · 2026-03-23 00:22:49

All my status is shown in the screenshot above. Have I successfully registered?
&nbsp;
&nbsp;

#### #2 @(匿名) · 2026-03-22 22:38:56

There is currently no limit on the number of participating teams for each track.
Please check your registration status in the Personal Center; it appears that your registration has already been successful.

---

## What is the rationale behind the rule that participants in the academic track must attend the same school?

- ID: `045fd4ee25a211f1b7d5227d4a1cee1e`
- 链接: https://lexiangla.com/questions/045fd4ee25a211f1b7d5227d4a1cee1e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 回答 (1)

#### #1 @(匿名) · 2026-03-22 11:49:23

Teams can be formed across different universities. The official 【Rules】 state: &ldquo;The Academic Track is available to teams whose members are all affiliated with a university or college.&rdquo; This means&nbsp;that students&nbsp;in the Academic Track must all be university or college students, and therefore, cross-track teaming is not allowed.

---

## 学术赛道组队问题

- ID: `5966dab025a111f1a3ba5a8be7d65508`
- 链接: https://lexiangla.com/questions/5966dab025a111f1a3ba5a8be7d65508?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

请问一个队伍里面的成员可以不是来自同一所高校吗？个人认为以下规则："Academic Track.&nbsp;The Academic Track is available to teams whose members are&nbsp;all affiliated with a university or college. The Organizers may request documentation to verify academic affiliation at any time."会严重阻碍参赛的积极性。不知道主办方是从什么角度来考量从而设立这个规则，能否给个相应的解释。

### 回答 (1)

#### #1 @(匿名) · 2026-03-22 11:48:07

Teams can be formed across different universities. The official 【Rules】 state: &ldquo;The Academic Track is available to teams whose members are all affiliated with a university or college.&rdquo; This means&nbsp;that students&nbsp;in the Academic Track must all be university or college students, and therefore, cross-track teaming is not allowed.
可以跨学校组建，但根据规则要求，所有学术赛道的选手都必须是学生身份，但不允许跨赛道组队。

---

## How can I check which track I have specifically signed up for?

- ID: `95429c2624d511f1b4e4f2ac39f114a6`
- 链接: https://lexiangla.com/questions/95429c2624d511f1b4e4f2ac39f114a6?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

How can I check which track I have specifically signed up for? The user center interface does not show which track I have registered for.

### 回答 (1)

#### #1 @(匿名) · 2026-03-21 11:35:46

Upon checking, we can see that you are currently in the Academic track. If you need to verify this yourself, you can click on the User Center.

---

## Will winning solutions be open-sourced after the competition?

- ID: `1c7b41b2244911f1b52c267dadf44ef8`
- 链接: https://lexiangla.com/questions/1c7b41b2244911f1b52c267dadf44ef8?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Outstanding solutions will be invited to present at the KDD Cup Workshop at ACM SIGKDD 2026, and the corresponding technical papers will be included in the ACM SIGKDD Cup 2026 proceedings. Participants may choose whether to open-source their solutions and, if so, which platform(s) to use.

### 回答 (1)

#### #1 @(匿名) · 2026-03-21 11:27:08

Outstanding solutions will be invited to present at the KDD Cup Workshop at ACM SIGKDD 2026, and the corresponding technical papers will be included in the ACM SIGKDD Cup 2026 proceedings. Participants may choose whether to open-source their solutions and, if so, which platform(s) to use.

---

## Code sharing and open-source policy during the competition

- ID: `06943c50244911f197584aac023bdd89`
- 链接: https://lexiangla.com/questions/06943c50244911f197584aac023bdd89?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Publicly sharing core code &nbsp;during the competition is strictly prohibited and may lead to disqualification.

### 回答 (1)

#### #1 @(匿名) · 2026-03-21 11:26:31

Publicly sharing core code &nbsp;during the competition is strictly prohibited and may lead to disqualification.

---

## Can multiple team members submit independently on the same day?

- ID: `ef8812ac244811f1afd5e6832550468e`
- 链接: https://lexiangla.com/questions/ef8812ac244811f1afd5e6832550468e?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Submissions are team-based. The team shares a daily quota; members can take turns .

### 回答 (1)

#### #1 @(匿名) · 2026-03-21 11:27:56

Submissions are team-based. The team shares a daily quota; members can take turns .

---

## Are postdoctoral researchers eligible for the Academic Track? What are the specific criteria?

- ID: `d5db3cf8244811f1b51a92ec18cfff06`
- 链接: https://lexiangla.com/questions/d5db3cf8244811f1b51a92ec18cfff06?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

&nbsp;Yes. Postdoctoral researchers are eligible if they meet either of the following conditions:
1. Currently working at a university-affiliated postdoctoral station within two years of receiving their highest degree;2. Must begin employment within one year of graduation, after leaving the postdoctoral station.

### 回答 (1)

#### #1 @(匿名) · 2026-03-21 11:26:49

&nbsp;Yes. Postdoctoral researchers are eligible if they meet either of the following conditions:
1. Currently working at a university-affiliated postdoctoral station within two years of receiving their highest degree;2. Must begin employment within one year of graduation, after leaving the postdoctoral station.

---

## Can Tencent employees participate in the competition?

- ID: `2e5384c2242f11f19dc686007e4ff72a`
- 链接: https://lexiangla.com/questions/2e5384c2242f11f19dc686007e4ff72a?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

No. To ensure fairness, current Tencent employees (including full-time, intern, and outsourced staff, etc.) are not eligible to participate once the competition starts. For details, please refer to the "Rules" on the official website (algo.qq.com).

### 回答 (2)

#### #1 @(匿名) · 2026-03-21 11:27:34

No. To ensure fairness, current Tencent employees (including full-time, intern, and outsourced staff, etc.) are not eligible to participate once the competition starts. For details, please refer to the "Rules" on the official website (algo.qq.com).

#### #2 @(匿名) · 2026-03-27 21:40:41

If I am not currently, but be an intern during the competition, if this is ok.

---

## What are the eligibility criteria regarding graduation for the Academic Track?

- ID: `eb386acc242e11f1b2db06c6ffa2c200`
- 链接: https://lexiangla.com/questions/eb386acc242e11f1b2db06c6ffa2c200?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

You must maintain your student status through the registration deadline (23:59:59 AOE, April 23, 2026).&nbsp;

### 回答 (3)

#### #1 @(匿名) · 2026-03-24 21:34:18

It is recommended to participate in the Industrial Track.

#### #2 @(匿名) · 2026-03-23 15:57:58

Regarding the enrollment requirement for the Academic Track:You must maintain your student status by the registration deadline (23:59:59 AOE, April 23).

#### #3 @(匿名) · 2026-03-21 11:27:52

You must maintain your student status through the registration deadline (23:59:59 AOE, April 23, 2026).&nbsp;

---

## What if I qualify for the Academic Track but do not have a verifiable .edu email address?

- ID: `2d0da68e242e11f1addaf6c26a8ea416`
- 链接: https://lexiangla.com/questions/2d0da68e242e11f1addaf6c26a8ea416?company_from=9cfe03d211a311f1a8588ad12dbe613c

### 正文

Please refer to the "Rules" at algo.qq.com for eligibility. If you qualify but lack an .edu email, complete your registration on the website first, then contact taac@tencent.com for manual verification. Please provide:&nbsp;1. Registration details (Username/UID);&nbsp;2. Your primary email address;&nbsp;3. Valid proof of student status (e.g., Student ID card, National ID, or Chsi verification).

### 回答 (1)

#### #1 @(匿名) · 2026-03-21 11:26:01

Please refer to the "Rules" at algo.qq.com for eligibility. If you qualify but lack an .edu email, complete your registration on the website first, then contact taac@tencent.com for manual verification. Please provide:&nbsp;1. Registration details (Username/UID);&nbsp;2. Your primary email address;&nbsp;3. Valid proof of student status (e.g., Student ID card, National ID, or Chsi verification).

---
