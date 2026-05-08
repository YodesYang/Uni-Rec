# 工业赛道情报抽取 (基于 lexiang 社区抓取)

- 数据源: `docs/99-raw/community-snapshots/lexiang_questions_2026-04-30-06-46-51.json` (141 题)
- 工业赛道相关题数: 47
- 含官方答复 (Administrators / TAAC): 26

> ⚠️ **历史快照·部分官方答复中的规则数字 / 工程结论已被后续迭代覆盖**（原文保留用于回溯；不要直接当作当前规则使用）。
> **当前规则 / 工程结论口径**：以 `docs/00-context/competition.md` + `docs/00-context/platform-angel.md` + `.cursor/rules/ai-readme/team-memory.mdc` + `docs/40-experiments/` 为准。

---

## 📘 2026 Tencent Advertising Algorithm Competition Player Guide
- 🏛 OFFICIAL · views=2034 · ans=1 · 2026-04-23 17:25
- URL: https://lexiangla.com/questions/55556dce3ef611f1bd34366a677e2e06?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: Tencent LearnShare AI Knowledge Base — Official Knowledge Management Platform for the Competition
🎯 1. Why Choose Tencent LearnShare AI Knowledge Base?
As the official knowledge management platform designated for the competition, Tencent LearnShare AI Knowledge Base helps participating teams:

✅Centralized Management:Unified storage of scattered papers, code, and experimental records
✅AI Assistance:Automatic parsing of literature, generating overviews, and extracting key information
✅Team Collab

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-25 20:21  👍2
  We have provided a free AI Knowledge Base for all participants. Please refer to the operational instructions provided above. 

Good luck with the competition!

---

## Tencent Angel Machine Learning Platform User Guide
- 🏛 OFFICIAL 🏭 IND-PEER · views=1920 · ans=4 · 2026-04-25 19:53
- URL: https://lexiangla.com/questions/648de516409d11f195ef964e57ab7fcf?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: User Guidance

Model Training

Create and submit a training task
1.  Fill in "Job Name".

2.  Fill in "Job Description".

3.  Click "Local Upload" to upload scripts from local files OR click "New Script" to start writing a new script.

 

ATTENTION: The training template includes a mandatory run.sh file as the execution entry point, which will be executed automatically at task startup!

 

4.  After confirmation, click "Submit" to save the task.

Run a task
Click "Run" to start running a task.



**回答**:

  🏭 **[IND]** @(匿) · 2026-04-26 18:37  👍3
  How can we access Tencent Angel Machine Learning Platform User Guide without "WeCom"?

  🏛 **[OFFICIAL]** @(匿) · 2026-04-25 20:19  👍1
  Since the competition has already begun, we highly recommend that all participants review the official User Guide. This manual contains essential operational workflows and technical instructions to help you navigate the platform and optimize your performance.

You can access the document via the following link:

Tencent Angel Machine Learning Platform User Guide

Good luck with the competition!

  🏭 **[IND]** @(匿) · 2026-04-29 19:24
  I submitted the official baseline script for successful training and inference, but there is no ranking. Is there a problem with the system?

  🏭 **[IND]** @(匿) · 2026-04-28 00:00
  Hi. I submitted the provided sample submission (without changing anything) and it failed. 

![]( https://raw.githubusercontent.com/cdeotte/Kaggle_Images/refs/heads/main/Apr-2026/e1.png )

And then when i view the logs, the logs are empty. Any advice on how to submit the sample submission successfully?

 

UPDATE: The problem may be that the sample submission code doesn't set the `TRAIN_LOG_PATH` variable. Trying this now.

UPDATE2: Fixing variable above solved first error and now it begins training and makes a log. However now there is an OOM error. Fixing that now...

---

## 模型评估，一直处于等待推理资源状态，大半天了，算力那么紧张啊？
- 🏛 OFFICIAL 🏭 IND-PEER · views=279 · ans=5 · 2026-04-26 19:28
- URL: https://lexiangla.com/questions/0648b302416311f1a8a6eaa59d583e05?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 模型评估，一直处于等待推理资源状态，大半天了，算力那么紧张啊？

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-26 21:13  👍1
  不是有三张卡吗，为什么一个job只用了一张卡，job之间还要串行

  🏭 **[IND]** @(匿) · 2026-04-26 22:58
  same quesion for me as well.
In my chain, no task is running yet. All wait for resource/infer resource/evaluation resource.

  🎓 **[ACA]** @(匿) · 2026-04-26 20:49
  但是我已经停止了训练进程，只有评估 还是在等待推理资源  这正常吗 感觉像是平台的bug
而且终于跑起来之后，点击日志还会报错：Network Error! Contact the competition assistant for more helps

  🏭 **[IND]** @(匿) · 2026-04-26 20:31
  可是我从下午1点半等到现在，还是处于等待推理资源状态，请问这正常吗？

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 20:16
  1. Any team member can initiate evaluation, but each team is limited to 0.2 GPU, which means all tasks need to be executed serially.

---

## basic questions on dataset
- 🏛 OFFICIAL · views=165 · ans=1 · 2026-04-26 00:30
- URL: https://lexiangla.com/questions/0a46968040c411f1a62ba24b4f44eca3?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 1.can we save tmp files on disk for training only purpose? what's size limit? USER_CACHE_PATH is only 20G but what if I save to other paths like TRAIN_DATA_PATH?
2.I note most training files with timestamp 2026-02-28~2026-03-04 so it is not divided in time order. Also baseline validation set didn't use time series split. How leadboard test set is splitted? does it comes from same period or non-overlapping future?

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 20:29  👍1
  1. TRAIN_DATA_PATH is currently the only accessible path, but it is read-only and cannot be used for storage.

2. The test set timestamps do not overlap with the training set and are from a later time period.

---

## label_type为1的表示啥
- 🏛 OFFICIAL · views=159 · ans=1 · 2026-04-27 17:48
- URL: https://lexiangla.com/questions/3bda1ed2421e11f1a20ddec6795b114a?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 是表示有点击无转化还是没有点击，还是说是混着的

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-27 19:33  📌  👍1
  In this dataset, a label_type value of 1 for the CVR label indicates that the user has performed a conversion action within 24 hours following an ad click.

---

## 停止评估后也会浪费评估次数吗？
- 🏛 OFFICIAL · views=142 · ans=2 · 2026-04-26 18:42
- URL: https://lexiangla.com/questions/a212c27a415c11f1a47d2e3d16a5b7db?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 因为一直处于资源等待中，于是我停止了评估。此时已经成功评估两次，但是想要第三次评估时显示次数不够。

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 20:07  👍1
  The daily submission limit resets every 24 hours at 15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24‑hour period. Failed or stopped tasks do not count toward the daily limit.

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 19:29
  同问，如果提交评估任务在AOE 23:59:59之前，但等待资源后实际执行在AOE 23:59:59之后，算哪一天的评估次数呢？

楼主，问你一个问题，你前两次评估成功时那个时候有在训练吗？

---

## Is the Round 1 baseline/starter kit available yet? I checked the Model Management section on AngelML, but the files have no content. Thanks for the help!
- 🏛 OFFICIAL · views=125 · ans=2 · 2026-04-26 02:34
- URL: https://lexiangla.com/questions/68767b1040d511f1bf08eaa59d583e05?company_from=9cfe03d211a311f1a8588ad12dbe613c

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 03:46  📌  👍1
  After clicking the `Create a New Job` button in the `Model Training` tab, you can download the scripts in it:

  🎓 **[ACA]** @(匿) · 2026-04-26 05:56
  thank you

---

## What do “label_type=1” and “label_type=2” mean, respectively?
- 🏛 OFFICIAL · views=124 · ans=2 · 2026-04-28 00:33
- URL: https://lexiangla.com/questions/c4753276425611f18591b2aef88eec92?company_from=9cfe03d211a311f1a8588ad12dbe613c

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 18:38
  In the dataset, label_type=2 means a positive sample (user converted within 24 hours after an ad click), while label_type=1 means a negative sample (no conversion within 24 hours). The code in dataset.py remaps them to 1 and 0 for model input.

  🎓 **[ACA]** @(匿) · 2026-04-28 00:37
  You can infer it from the official baseline.

---

## 新的demo数据在原来demo数据基础多了一个序列域特征，请问这个序列域特征代表什么意思
- 🏛 OFFICIAL · views=123 · ans=1 · 2026-04-28 18:15
- URL: https://lexiangla.com/questions/2c56562442eb11f1812d6adecbf36ff9?company_from=9cfe03d211a311f1a8588ad12dbe613c

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 18:39
  All features have been anonymized

---

## Can we get the `schema.json`?
- 🏛 OFFICIAL · views=118 · ans=1 · 2026-04-27 15:31
- URL: https://lexiangla.com/questions/23ead234420b11f1afe53a773f0e4cdd?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: Because of impressed OOM in the platform, we need do some experiments locally. However the train.py needs a schema.json, can anyone give us this file?

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-27 20:15
  The schema.json is tied to the full competition data and cannot be shared. If you want to train locally on the 1000 public samples, you can generate a schema.json yourself.

---

## 每天3次模型评估，究竟是北京时间几点更新次数啊？一整天都没法进行评估。
- 🏛 OFFICIAL 🏭 IND-PEER · views=112 · ans=2 · 2026-04-28 16:41
- URL: https://lexiangla.com/questions/1cf4f86e42de11f1b3af466c822786ed?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 每天3次模型评估，究竟是北京时间几点更新次数啊？一整天都没法进行评估。

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-28 17:09
  请问能帮忙解决这个问题吗？已经过了北京时间12点，次数还是没有更新。到现在还是无法提交评估。

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:03
  Would it be possible to provide the error message/details from the failed submission?

---

## 推理时间最长是多少啊？What is the inferencing time limit?
- 🏛 OFFICIAL · views=112 · ans=2 · 2026-04-27 16:00
- URL: https://lexiangla.com/questions/1bf43120420f11f1ae153a63274c50fb?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 评估任务运行到一半被stop了，大概在inferencing状态用了十来分钟，不知道是什么原因。初步怀疑是超时了？但是没找到推理最长时间是多少？有人知道吗？

Eval task was stopped during inferencing. It was stopped after being inferencing status for ten minutes. Is this caused by timeout on inferencing time limit? Does anyone know the inferencing time limit?

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-27 21:33
  A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

  🎓 **[ACA]** @(匿) · 2026-04-27 16:48
  试出来了，推理时限是30min

---

## When will the baseline be available online?
- 🏛 OFFICIAL · views=108 · ans=1 · 2026-03-24 10:36
- URL: https://lexiangla.com/questions/44b40aea272a11f189bceeeffbc12236?company_from=9cfe03d211a311f1a8588ad12dbe613c

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-03-24 10:37  👍1
  The baseline will be released after the competition officially starts. Stay tuned!

---

## 我训练到中途，突然看不自动停止，且看不到输出
- 🏛 OFFICIAL 🏭 IND-PEER · views=92 · ans=2 · 2026-04-26 13:50
- URL: https://lexiangla.com/questions/e6b57504413311f1ae4d76202f67a800?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 11:53开始跑，12:58已经保存了第一个epoch的模型，val的auc也出了，但是跑到13:18分的时候，连output那个按钮都不见了，同时也自动结束了，日志上啥都没显示，你们这个平台怎么bug这么多啊

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 20:21
  All processes for the previously problematic tasks have been terminated. You may now retry.

  🏭 **[IND]** @(匿) · 2026-04-26 13:52
  就算异常结束，我之前保存的模型不该不见吧，就连output那个按钮都不见了

---

## 为啥我的成绩没在leaderboard里面显示出来
- 🏛 OFFICIAL 🏭 IND-PEER · views=84 · ans=3 · 2026-04-28 17:19
- URL: https://lexiangla.com/questions/5cc4055242e311f1a0d1fa9c40bfe1bd?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 为啥我明明有成绩却没在leaderboard里面显示出来呢？

我的team name

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-28 20:28
  工业赛道

  🎓 **[ACA]** @(匿) · 2026-04-28 18:09
  请问一下你在什么赛道呀

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:28
  This is most likely caused by the Delayed Feedback mechanism, which is normal behavior. Here's the explanation:

To prevent leaderboard overfitting, some submission results during Round 1 will be delayed, focusing on evaluating model generalization.
The leaderboard consists of a Public Leaderboard (based on partial test set with real-time feedback) and a Private Leaderboard (based on the full test set).
Your score may have already been generated but not yet displayed on the public leaderboard.

Please wait a bit longer. If it still doesn't appear after an extended period, please contact official support for further confirmation.

---

## What are the training and evaluation AUC scores of the official baseline? Has anyone run it?
- 🏛 OFFICIAL · views=81 · ans=1 · 2026-04-27 16:12
- URL: https://lexiangla.com/questions/d0ec51ec421011f18cd8f2f0f4c72958?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: What are the training and evaluation AUC scores of the official baseline? Has anyone run it?

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:47
  To understand the actual performance of the official baseline, you may want to check if any teams on the leaderboard have submitted results using the unmodified official baseline code as a reference point.

---

## 如果在训练推理中安装新的包？
- 🏛 OFFICIAL · views=78 · ans=1 · 2026-04-28 17:40
- URL: https://lexiangla.com/questions/3acbd3dc42e611f1bab8825bb877947b?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 如何在环境中安装包？

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:42
  Tencent Angel Machine Learning Platform User Guide

---

## 提交次数更新不是北京时间中午12点更新吗，为什么显示：You can only submit up to 3 tasks per day (failed tasks are not counted). Contact the competition assistant for more helps
- 🏛 OFFICIAL · views=75 · ans=1 · 2026-04-28 15:10
- URL: https://lexiangla.com/questions/574a898242d111f1a94dea71aef76897?company_from=9cfe03d211a311f1a8588ad12dbe613c

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:10
  The daily submission limit resets every 24 hours at 15:59:59 AOE time. Each team may submit up to 3 evaluation tasks per rolling 24-hour period. Failed or stopped tasks do not count toward the daily limit.

---

## 请问每一轮训练完发布的模型，是否对应每一轮的checkpoint？意思是我每一轮都发布的话，每一轮的模型进行的评估分数都会不同，这样对吗？
- 🏛 OFFICIAL · views=73 · ans=1 · 2026-04-29 14:28
- URL: https://lexiangla.com/questions/af5753a8439411f188a1ba0336b6e028?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 请问每一轮训练完发布的模型，是否对应每一轮的checkpoint？意思是我每一轮都发布的话，每一轮的模型进行的评估分数都会不同，这样对吗？

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-29 17:48
  Yes, the model released after each training round corresponds to the checkpoint of that round, so the evaluation scores will differ each round

---

## Request to Address Bottleneck in the Model Build → Publish → Submit & Score Workflow
- 🏛 OFFICIAL · views=64 · ans=1 · 2026-04-27 09:05
- URL: https://lexiangla.com/questions/3731bd6641d511f1adf2ca8321426fe4?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: It appears that resource allocation for evaluation is currently taking longer than model training itself. Under these conditions, evaluation has become a bottleneck in the training → model publishing → evaluation pipeline, creating a very high risk that trained jobs — which are deleted after 24 hours — will expire before evaluation completes.

While multiple bugs seem to be occurring, the issues we would like prioritized are:

Resolving the resource allocation issue in the evaluation phase
Ensur

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-27 19:59
  1. Inference Time Limit A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system. 

2. Resource Allocation Policy Each team is allocated a single resource quota, which is shared across both training and inference tasks. If a team submits multiple tasks, they will be executed sequentially: a new task will not begin until the currently running task has completed.

---

## 请问能否为特征列提供 nominal / ordinal / numeric / timestamp 类型标注？
- 🏛 OFFICIAL · views=59 · ans=1 · 2026-04-28 13:48
- URL: https://lexiangla.com/questions/d44df4d442c511f1b1eef26125df00d6?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 发现 baseline 的 embedding table 比较大，因此想确认一下：是否所有整数特征都是真正的类别 ID？还是其中有一些特征其实是有序特征、数值型信号或时间相关信号，可能更适合使用数值变换、分桶、归一化等方式处理，而不是直接使用 embedding table？

具体来说，能否确认以下特征组是否都属于类别 ID？

user_int_feats_*
item_int_feats_*
domain_a_seq_*
domain_b_seq_*
domain_c_seq_*
domain_d_seq_*

如果其中有部分特征不是类别 ID，也希望能进一步说明它们更适合归为 ordinal / numeric / timestamp 中的哪一类。这样可以帮助我们更合理地设计特征处理方式。

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:29
  all features have been anonymized, and the organizing committee cannot confirm specific feature type classifications

---

## Issue of automatic stop during the model evaluation process
- 🏛 OFFICIAL · views=42 · ans=1 · 2026-04-26 16:09
- URL: https://lexiangla.com/questions/4692d74c414711f1a5da4a2c978b479c?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: Problem Description:
 
When running evaluation tasks, both the official baseline model and our custom model will stop automatically within 10 to 20 minutes after the evaluation starts. No results are exported, and the logs are interrupted halfway. Neither the Leaderboard Score nor the Inference Time is displayed in the end, and there are no error prompts in the logs. Please check and confirm this issue. Thank you very much.
 
Attachment:
 
Evaluation ID：29169 / 29109 / 28267

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-26 20:17  👍1
  All processes for the previously problematic tasks have been terminated. You may now retry.

---

## Evaluation Stopped in Model Evaluation
- 🏛 OFFICIAL 🏭 IND-PEER · views=37 · ans=2 · 2026-04-27 09:03
- URL: https://lexiangla.com/questions/eccc4bc441d411f1b8db32136c2f0ae8?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: Eval ID 28664 appears to have been forcibly transitioned to "Stopped" status approximately 24 hours after submission, and is no longer being evaluated. Please look into this.
Eval ID 29403 has been stuck at "Waiting for Inference Resources" for over 12 hours since submission. Could you let us know the average time required for resource allocation?

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-27 20:17
  What status does an inference job get when it exceeds the time limit?

  🏛 **[OFFICIAL]** @(匿) · 2026-04-27 19:58
  A maximum time limit of 30 minutes is imposed on each inference task in both the Industry and Academic tracks. Any inference job that exceeds this time limit will be automatically terminated by the system.

---

## Why i cant see my submission on the leaderboard?
- 🏛 OFFICIAL 🏭 IND-PEER · views=33 · ans=3 · 2026-04-28 16:03
- URL: https://lexiangla.com/questions/b9eda59042d811f1976bbef3d5422991?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: Evaluation ID is 30335, Leaderboard Score auc: 0.846108, Inference Time 625.28s and last timstamp from log is 2026-04-27 15:47:38.044.

Do I need to do anything more to submit this solution for leaderbord or its done automatically?

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:38
  Once an evaluation task is submitted, the system automatically runs through the full evaluation pipeline.
The public leaderboard reflects real-time feedback based on partial test set results— meaning your AUC score should appear automatically.
However, Round 1 uses a Delayed Feedback mechanism: some submission results will be delayed to prevent leaderboard overfitting.

  🏭 **[IND]** @(匿) · 2026-04-28 16:37
  But this was submitted 2026-04-27 15:47 (before 2026-04-28 12:00), so I would expect to see it after 2026-04-28 15:00 ...

  🎓 **[ACA]** @(匿) · 2026-04-28 16:31
  The leaderboard is updated daily with results that have been finalized before 12:00 noon Beijing Time, and you will be able to view your ranking at 3:00 PM tomorrow.

---

## 还没有修复评估次数重置的bug吗？目前为止还是无法提交。
- 🏛 OFFICIAL · views=32 · ans=1 · 2026-04-28 19:27
- URL: https://lexiangla.com/questions/3d6bccaa42f511f1ba7cee8dccbbec8c?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 还没有修复评估次数重置的bug吗？目前为止还是无法提交。

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-29 15:59
  Please try again and see.

---

## 推理模型load错误
- 🏛 OFFICIAL · views=28 · ans=1 · 2026-04-28 14:25
- URL: https://lexiangla.com/questions/fc30652242ca11f1b3110e7eff9d52a2?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 在我导出模型后，我遇到一个奇怪的问题，推理模型加载错误。
我记得在昨晚我做了两个近似的实验，但是今天看train中的model.py似乎不一致，我随机选择一个进行eval，显示报错，另一个不报错。
在文件中我看到加载模型不是从ckpt中加载，为什么会遇到这个问题呢？看起来是一个bug似乎，因为我有改model.py的模型无法load成功。

2026-04-28 06:08:51,120 - ERROR - Inference failed: Error(s) in loading state_dict for PCVRHyFormer:

**回答**:

  🏛 **[OFFICIAL]** @(匿) · 2026-04-28 17:23
  Recommendation: Re-train with your modified code, or ensure that the infer.py / model.py used during inference is exactly identical to the one used during training.

---

## 不懂就问，工业赛道数据为啥有时间穿越
- (无标记) · views=168 · ans=0 · 2026-04-29 11:17
- URL: https://lexiangla.com/questions/0052d0c2437a11f1aa7a6a8ce5bc633c?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 训练日期：
数据最早时间: 2026-02-28 00:35:09
 数据最晚时间: 2026-03-04 01:49:50

测试日期：

 
理论上，过去训练的模型预测未来，但是为啥这个比赛测试数据还有比训练数据早的？？？
请问主办方为啥如此设置。

---

## 请问大家训练分数与评估分数相差大吗？
- (无标记) · views=156 · ans=1 · 2026-04-26 20:18
- URL: https://lexiangla.com/questions/fcfdadc8416911f18e7dead2caac9371?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 请问大家训练分数与评估分数相差大吗？

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-27 08:39
  过拟合了

---

## 我的推理任务为啥总是等待中啊
-  🏭 IND-PEER · views=141 · ans=2 · 2026-04-26 02:01
- URL: https://lexiangla.com/questions/c8490ef440d011f1b1b2328795a383ba?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 怎么回事，都登录快一小时了

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-28 14:31
  解决了吗，我的从早上九点一直waiting到下午两点了

  🏭 **[IND]** @(匿) · 2026-04-26 02:01
  都快等待一小时了，还是等待推理资源

---

## 模型评估成功出auc之后不在排行榜上出现
-  🏭 IND-PEER · views=133 · ans=1 · 2026-04-26 21:49
- URL: https://lexiangla.com/questions/b967e8b4417611f1b555eed6b9c02d85?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 模型评估成功出auc之后不在排行榜上出现，user center中也不显示

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-26 21:59  👍1
  Based on the results submitted daily before 16:00 AOE time, sort the teams in descending order according to their historical best scores, and update the leaderboard at 19:00 AOE time on the same day.

---

## 团队名字叫Semantic_Shards，提交了两版结果，为啥看不到分数排序？（工业界榜单）
-  🏭 IND-PEER · views=114 · ans=3 · 2026-04-28 08:13
- URL: https://lexiangla.com/questions/162b5c40429711f191d93a35d2f04c13?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-28 11:24
  请问有解决么？一样的现象

  🎓 **[ACA]** @(匿) · 2026-04-28 10:13
  请问最后有解决么？我也遇到了没法上榜单的同样的情况

  🏭 **[IND]** @(匿) · 2026-04-28 10:00
  我们也碰到了类似的问题，提交了几个版本，但是榜单上没有对应的名字

---

## 训练最优节点是否会自动保存？评估的时候是都会自动使用最优轮数的参数？
- (无标记) · views=109 · ans=1 · 2026-04-27 00:55
- URL: https://lexiangla.com/questions/cb5f0e8e419011f1b9045663b438421f?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 例如，我训练了4轮，第三轮分数是最优的，第四轮已经下降了，但是我并没有备份第三轮的参数，最后直接用第四轮的数据输出模型并提交评估，那么这个时候评估的模型是我第三轮的最优参数还是第四轮的？

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-27 01:52  👍1
  默认trainer.py只会保存AUC最高的checkpoint，如果要保存其他checkpoint需要将其保存到设定好的目录下并且按要求命名，具体见文档。

---

## 1、如图，任务完成之后为什么不能停止？ 2、模型eavl 之后显示分数了，为什么排行榜却没有
- (无标记) · views=107 · ans=1 · 2026-04-27 09:21
- URL: https://lexiangla.com/questions/607d683a41d711f19dceeed6b9c02d85?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-27 10:41  📌  👍2
  1.因为没有在运行 2.没在AOE16:00前记录

---

## 模型的训练任务没法停下来，咋么办
-  🏭 IND-PEER · views=103 · ans=1 · 2026-04-26 01:56
- URL: https://lexiangla.com/questions/1a45531240d011f192b63eb432f4e775?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 这个模型的训练任务已经开始了，我想把它停止，但是停止的按钮是灰色的，怎么办，现在搞得我没有资源训练和推理了，在线急等

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-26 01:58
  如果等它跑完估计得好几天，没法停，资源又被占着，这是平台的bug吗

---

## 离线数据量太少了，官方在huggingface上的demo数据能给多一点吗？或者有那位大佬知道哪里有可靠的数据集吗
- (无标记) · views=70 · ans=0 · 2026-04-29 19:42
- URL: https://lexiangla.com/questions/89cf91b443c011f1b6c8aa5005b73072?company_from=9cfe03d211a311f1a8588ad12dbe613c

---

## 为什么评估模型会在处理完样本后不评分就停止？
- (无标记) · views=66 · ans=1 · 2026-04-26 20:00
- URL: https://lexiangla.com/questions/8a62f590416711f19450dabb3ffa631d?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-26 20:04
  baseline吗？还是改过了？

---

## 1、推理结束不出分？ 2、每个队伍 每天多少次提交推理呀，有没有限制呀
- (无标记) · views=65 · ans=1 · 2026-04-27 22:42
- URL: https://lexiangla.com/questions/5dd6b5e4424711f1911a825bb877947b?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-28 10:37
  看看log显示啥呢

---

## 想问一下为什么已经评估好的模型没有入排行榜？
- (无标记) · views=58 · ans=1 · 2026-04-28 10:00
- URL: https://lexiangla.com/questions/1383b7b242a611f1a9fc3600151239df?company_from=9cfe03d211a311f1a8588ad12dbe613c

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-28 10:33
  每天15点出新的榜单，别急，成功出分了就行

---

## 请问已经publish过的model，在model manage页面被删除了之后，可以重新publish吗，目前显示已被publish过的ckpt不允许再被publish
- (无标记) · views=50 · ans=0 · 2026-04-27 23:51
- URL: https://lexiangla.com/questions/f45326d4425011f1b5c6766d32605923?company_from=9cfe03d211a311f1a8588ad12dbe613c

---

## 建议：可视化训练与评估的资源分配队列
- (无标记) · views=49 · ans=1 · 2026-04-29 18:09
- URL: https://lexiangla.com/questions/8bd9670843b311f1bd80ea260899f34e?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 在本次竞赛的社区中，可以看到许多关于训练任务和评估任务资源分配的问题。此外，资源分配失败、任务中断、队列顺序变更等相同问题也频繁出现。因此，我们建议增加一项功能，让参加者也能可视化查看资源分配等待队列的顺序。

① 参加者执行训练或评估时，任务进入队列堆栈。
② 若需等待资源，任务保持堆栈状态；资源分配完成后，任务从队列中出栈。

希望能够在UI上简单直观地显示上述信息，让参加者一目了然地掌握资源分配的顺序。

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-29 22:33
  我几乎没遇到呀

---

## 无法查看所有推理log，提示“Network Error! Contact the competition assistant for more helps”
-  🏭 IND-PEER · views=45 · ans=1 · 2026-04-29 04:37
- URL: https://lexiangla.com/questions/1c333d78434211f1b7524ee504b554c4?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

**回答**:

  🏭 **[IND]** @(匿) · 2026-04-29 04:52  👍1
  解决了，是去广告插件导致的

---

## HELP:模型评估后，显示成功和对应AUC，但是一整天后也不能查询排行，从外部进入排行榜也看不到自己的提交和排名。(已有修改infer文件和经过对应时间)
- (无标记) · views=43 · ans=0 · 2026-04-28 19:27
- URL: https://lexiangla.com/questions/49061ebc42f511f1946b0ab7809cf5b9?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: NotFoundError: Failed to execute 'removeChild' on 'Node': The node to be removed is not a child of this node.
    at GO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:26100)
    at tn (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27391)
    at XO (https://fe-static-1258344696.cos.accelerate.myqcloud.com/algo_fe_prod/assets/react-Do9DFjP3.js:32:27851)
    at tn (https://fe-static-1258344696.co

---

## model eval看到能推理出来但是fail
- (无标记) · views=39 · ans=1 · 2026-04-27 23:27
- URL: https://lexiangla.com/questions/a2459898424d11f1a5ddbe87ea613f06?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: model eval看到能推理出来但是fail，显示“ ERROR - Missing 310000 answer user_ids in predictions (need full coverage)”

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-28 10:36
  我遇到了95NAN的问题，但5%有结果还是给我出了分，你这个怕不是100%都NAN？检查一下是不是模型代码，FP16/32设置的问题

---

## 模型评估过程显存被别的实例抢占爆了，这是官方bug吧，再次评估跑出分数了，但是就这也算评估了两次，这合理吗，老哥们
- (无标记) · views=25 · ans=0 · 2026-04-30 09:06
- URL: https://lexiangla.com/questions/cb21e0ac443011f1b4ad6abcdbd0613b?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: 2026-04-29 13:21:02,005 - ERROR - Inference failed: CUDA out of memory. Tried to allocate 9.00 GiB. GPU 0 has a total capacity of 19.12 GiB of which 19.12 GiB is free. Process 2052065 has 12.60 GiB memory in use. Process 1094656 has 46.96 GiB memory in use. Process 2010256 has 14.62 GiB memory in use. Process 166351 has 14.48 GiB memory in use. Process 2115634 has 2.20 GiB memory in use. Of the allocated memory 1.37 GiB is allocated by PyTorch, and 411.94 MiB is reserved by PyTorch but unallocat

---

## 为什么训练完成后 看不到输出的checkpoint
- (无标记) · views=24 · ans=1 · 2026-04-29 20:59
- URL: https://lexiangla.com/questions/51797d4c43cb11f1919d9eb605d32106?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

**回答**:

  🎓 **[ACA]** @(匿) · 2026-04-29 22:36
  因为一大轮还没训练完

---

## 训练的schema是如何获取的？如果要做特征工程的话，是否需要另外上传schema.json?
- (无标记) · views=15 · ans=0 · 2026-04-30 02:44
- URL: https://lexiangla.com/questions/63466e8843fb11f1b88c8e742a80aeca?company_from=9cfe03d211a311f1a8588ad12dbe613c

---

## 请问一下第一次跑demo的时候显示fail to initialize user storage怎么办
- (无标记) · views=15 · ans=0 · 2026-04-28 22:41
- URL: https://lexiangla.com/questions/546f6f54431011f199d5be0aa5498713?company_from=9cfe03d211a311f1a8588ad12dbe613c

**问题正文**: [图片]

---

## 同赛题公开开源方案情报（2026-05-04 晚深调研）

> **来源**：GitHub 搜索 "TAAC 2026" 找到 3 个公开 repo，深读前两个的代码与 README。原始 repo 已 `git clone --depth=1` 到 `docs/99-raw/external-refs/{hojiahao,suyanli,lc1332}/`（`.gitignore` 屏蔽，不 push）。
> **本节只装"方法学级可借鉴点"**；具体数字 / 代码细节见原 repo。

### 方案 1：UniRec · `hojiahao/TAAC2026`（⭐ 41 stars，最详细）

**整体架构**：融合 6 篇论文的统一 backbone
- OneTrans (WWW 2026)：Mixed Parameterization（S-tokens 共享 / NS-tokens 独立）
- InterFormer (Meta CIKM 2025)：Feature Cross Layer（预骨干 NS self-attn）
- HSTU 2.0 (Meta arXiv 2026-02)：SiLU Gated Attention + Semi-Local Attention + Attention Truncation + MoT
- HyFormer：MoT 多序列独立建模
- DIN (Alibaba 2018)：Target-aware Interest Extraction（item_pool query 全序列 cross-attn）
- Kimi (arXiv 2026)：Block Attention Residuals（per-layer pseudo-query + softmax over block summaries）

**本仓库可直接借鉴的方法学级单点**（按 ROI 排序）：

1. **CombinedAUCLoss = WeightedBCE + PairwiseBPR** (`model/heads.py`)
    - 腾讯自己 arXiv 2403.14144 生产数据：+0.70%~1.26% GMV
    - 我们完全没做 ranking loss，loss 家族是未消化的工业级杠杆
    - **→ T18 / ADR-006 已落地（2026-05-04 晚）**

2. **DIN-style 显式 target-query cross-attention**（生成 `interest_token` 独立 NS-token）
    - 我们的 ICQ 只做 query 端 gate；DIN 做 target 对全序列的完整 attention，信息量级不同
    - 与 T16 方案 A（seq token 侧加性注入）正交，理论可叠加

3. **Block Attention Residuals (Kimi)**
    - per-layer pseudo-query + softmax over block summaries，~60 行
    - 论文宣称 "1.25× compute 等效" scaling 优势，<2% 推理开销

4. **RankMixer Per-token FFN 正确实现**
    - 我们的 `RankMixerBlock` 用 shared FFN，与论文原版差异
    - Per-token FFN 专门对抗 dominant feature 淹没长尾，本赛题 item 长尾极端

5. **Muon 优化器（Linear 层专用）**
    - MuonRec (arXiv 2603.00416) 实测 NDCG@10 +12.6%, 收敛步数 −32%
    - 与现有 Adagrad(sparse) + AdamW(dense) 架构兼容（把 AdamW(dense) 换为 Muon+AdamW 混合）

**不可直接借鉴**：
- UniRec 的 `data/dataset.py` 按 list<dict> 原始格式消费（57 user + 16 item + 3 seq），与本仓库扁平 120 列 parquet 不兼容；**方法学可移植，数据层代码不能**
- HSTU 2.0 SiLU Gated Attention 替换 backbone attention：骨干级重写，时间成本过高
- Attention Truncation / Semi-Local Attention：需 Triton kernel，与现有 SDPA + RoPE 路径耦合大

### 方案 2：DeepContextNet · `suyanli220/TAAC-2026-Baseline`（⭐ 10 stars，轻量）

**核心**：HSTU core + Muon optimizer，346 行单文件
- 亮点：Muon 实现可直接抄（~40 行 `MatrixUnitaryOptimizer`）
- 其余架构（HSTU block）对本仓库启发不大

### 方案 3：`LC1332/TAAC_2026-ref1`（⭐ 2 stars）

仅含框架脚手架，无独立方法学贡献，不详细分析。

### 外部论文数字锚（用于本仓库 ROI 估算）

| 方法 | 来源 | 官方数字 | 本赛题适配预期 |
|---|---|---|---|
| BCE + PairwiseBPR | Tencent production (arXiv 2403.14144) | GMV +0.70%~1.26% | LB +0.002~0.005 |
| InterFormer Feature Cross | Meta CIKM 2025 | +0.14% AUC 基准 / +0.15% Meta 生产 | LB +0.001~0.002 |
| DIN target attention | Alibaba KDD 2018 | 离线 AUC +0.5~1% | LB +0.002~0.004 |
| CAN Co-Action | Alibaba WSDM 2022 | CTR +12% / RPM +8% | LB +0.002~0.004（T16 方案 C 预埋） |
| Muon 替 AdamW | MuonRec arXiv 2603.00416 | NDCG@10 +12.6%, 收敛 −32% | 收敛提速 + AUC 持平~+0.001 |
| HSTU 2.0 全量 | Meta arXiv 2602.16986 | 训练 5×, 推理 21×, 业务 4~8% | 骨干级改造，本赛季时间不够 |

**关键纪律提示**：上述数字是各自论文 / 生产环境的结果，在本赛题（post-click CVR、1.9M 行、test 远未来 valid→LB gap 0.023）的**转化率不可等比外推**。历史经验是 local AUC → LB AUC 约 **30~40% 转化率**（EXP-018 观测）。所以上表"本赛题适配预期"都按保守 30% 系数折算。

---

## 工业赛道 Rank 1 · DECEM 公开情报（2026-05-08 小红书抓取）

> **来源**：小红书账号 `DECEM`（工业赛道当时榜一「撑鱼大队长」，定位四川），6 张截图用户手动搬运 · 2026-05-08 整理
> **赛题身份**：Industrial Track · Rank 1 · Best Score **0.852181**（2026-05-06 18:47:59）/ 此前 0.851688（2026-05-05 11:38:44）
> **本节纪律**: 外部自述可能夸大或略过负向实验；数字 / 方法论**只作方向性参考**，不作为本仓库决策依据。对口实验仍以 holdout AUC + LB 小样本校准为准。

### 作者公开自述（汇总 · 按主题合并多帖）

**背景**：
- 第一次做推荐相关比赛。自 2021 年起做过 NLP、语音、CV、LLM、AI4S、AI+6G、3D 分割等多个方向。
- TAAC 中期已做 **60+ 个实验**，实际 work（上分点）**约 6 个**。一直尽量做**单变量消融**，整个过程靠 **Claude Opus** 辅助。

**分享的"work 了"的方向**（作者明确说 work 的）：
1. **torch.compile + 混合精度加速训练**（见 Q&A 2）。作者明确说 "必须得做"，但**只加在训练，推理不加**。
2. **时间特征**（作者原话："简单测了一些数据集，后续就是常规的时间特征，目前只 work 了两个"）。具体哪两个未公开。
3. **Tokenization 设计的小修改**（作者原话："简单做了一些修复，让其更加合理化"）。具体未公开。
4. **模型两块**：
   - "一个是我在前期就做了的 **DIN + MLP**"（多次提到，用户多次追问"就是 global token 里序列 pool 改为 DIN 吗？"，作者未明确否认）。
   - "还有一个那天晚上看的**一篇论文的方法**，这个大家就先不要问了"（明确保留）。

**未 work 的方向**（作者明确说掉分 / 不涨分）：
- **user_id / item_id 加入**：作者原话 "添加 user_id or item_id or scaling 都会掉分"（与 Seed 帖观测完全一致 → 再次证实 id OOV 过拟合是系统性问题）。
- **Scaling up**（模型变大）：作者原话 "**scaling 不掉分，但也不涨分**"。暗示本赛题数据规模（1.9M）下模型参数量已饱和。
- **序列长度**：作者原话 "还是 baseline 的，没变"（即没改 baseline 默认序列长度截断）。

**Baseline 相关**：
- 作者原话："大部分我认为都是面向 baseline 一些不够完善的地方去做小修小补，也没有做好 scaling。当前很多地方都是 baseline 原始的参数，序列长度没变，LR warmup 也没加"
- 暗示：**榜一方案对 baseline 做的是"精修补丁"而非架构级重写**；与 Seed 帖主"主流 rankmixer/onetrans/hyformer/mixformer 改成 pure model 都掉分"相互印证 → **本赛题在 1.9M 数据规模下不支持架构级 scaling**。

**对 seed 的观点**：
- 作者原话："所以说大家的上分空间还是很大的，万一我只是找到了个好的 seed 呢"
- 与本仓库 2026-05-06 EXP-028 观测（icq-half 5-seed 跨度 0.010、seed 上沿 0.846）**完全印证** → 榜一的 0.852181 里 **约 ±0.005 是 seed luck**，真实方法学峰值估算在 **0.847 左右**。

**验证集分数**：
- 作者原话："验证集 0.87+"
- 同日本仓库 C6 `rowgap15-icq-halftrain` dev=0.838（gap=0.15 row-level） / EXP-030 `icq-half-gap05` dev=0.841（gap=0.05） / icq-half（无 gap）dev=0.868
- **推测**：作者"验证集 0.87+"大概率是**无 row-level gap 的 baseline 切分**，与本仓库 `valid_split_strategy=time` 无 gap 的 dev 0.868 同量级 → **方法学绝对贡献约 +0.002~0.003 dev AUC，放大到 LB 约 +0.006~0.008**；比我们多出的 LB 差距（0.852 - 0.846 = 0.006）中约一半是方法学，一半是 seed。

### 对口本仓库的路线决策启示

| DECEM 线索 | 对口本仓库当前状态 | 决策启示 |
|---|---|---|
| DIN + MLP work | T16 ICQ additive 方向（query 侧 gate，非 target cross-attn）；DIN 未实现 | **优先实装 DIN-style target cross-attention（G1 原方案）**，把 item_pool 对全序列的完整 softmax attention 接入，位置可选：① NS token 侧生成 `interest_token`；② 序列 pool 替换 mean-pool |
| 时间特征 work 2 个 | T12 时间特征已做（年月日 / 星期 / 小时 / 距离最近时间 / diff log bucket）；已包含可能的两个 | **当前 T12 已大概率覆盖**；不重投入 |
| user_id / item_id scaling 掉分 | 本仓库 T21 idmask Δ−0.004 / T23 disable47 Δ−0.006 已知掉分 | **方向关闭确认**；不再做 id embedding 相关 ablation |
| Scaling 不涨分 | 本仓库 T11/d_model=128 Δ gap +0.028 | **d_model=64 定锁**；不再试更大 |
| torch.compile + AMP 训练加速 | 本仓库未做 | **优先级中等**：不提升 LB 但可加快迭代；若能让 half-train 从 5h → 3h，实验吞吐 +60%。**实装成本**：`model = torch.compile(model)` + `torch.amp.autocast('cuda')`，~20 行代码，但要小心 SDPA kernel 兼容性；**推理不加**（作者明确）|
| Tokenization 小修改 work | 本仓库 RankMixerNSTokenizer / GroupNSTokenizer 已成熟 | **未知具体修改**，无可执行方向 |
| seed 作用 ±0.005 | 本仓库 2026-05-06 已确认 | 已纳入 team-memory 铁律 |
| "一篇论文的方法" 保留 | 未知 | 无法行动 |

### 本次情报最重要的 3 条判决

1. **G1 DIN target attention 方向从"候选"升级为"强优先"**：榜一公开承认 DIN + MLP 是 work 的，与 UniRec / 方案 1 推荐路线、外部论文数字锚全部一致。建议下一个 half-train 直接实装。
2. **torch.compile + AMP 纳入工程优化队列**：ROI 不在 LB 而在迭代速度。半个月内若想多跑 2~3 批 ablation，训练加速是必要前置条件。建议作为独立小任务（T?? "train acceleration"）单独落地，不与 G1 混跑。
3. **真实 LB 能力带修正**：榜一 0.852 - seed luck 0.005 ≈ **0.847** 作为方法学峰值。本仓库 SOTA 0.84637 距离方法学峰值只有 **~0.001**，意味着**单方向（如 G1 DIN）只能吃掉一半差距，真正拉开需要 G1 + 工程加速 + 再做 1~2 次方向级探索的组合拳**。

### 原始快照位置

`docs/99-raw/community-snapshots/xhs-decem-2026-05-08/` （6 张截图 + 本节文字整理），不走 `diff_with_previous.sh` 流程，手工管理。
