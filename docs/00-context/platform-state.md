# Platform 实时状态 (Single Source of Truth)

> **每次 agent 推荐 platform 操作前必须 grep 此文件**。任何资源状态陈述 (slot / quota / in-flight) 凭此文件 · 不凭印象。
>
> **Last updated**: 2026-05-22 ET M151 public eval sync

> **Current Resource State · User Confirmed**: User reports **M148** public eval completed at **0.851365** and **M151** public eval completed at **0.850129**. These spend two eval quotas since the prior sync; remaining quota is **unknown in this file** and should be confirmed before recommending additional evals. Parsed training completions include M145-M153, M156, and M158. M154/M155/M157 are still pending by latest user report/context. Latest confirmed team public SOTA remains **M148 public 0.851365**, leaving **+0.001635** to target **0.853** and **+0.002125** to reported Top1 **0.85349**.

> **M151 Eval Result Update**: `M151-m123-vr02-mtl-seed13-auxselect-soup-gap0-4ep_eval_1779483206670` returned public **AUC 0.850129** (308.56s). This is **-0.001236 vs M148**, **-0.000882 vs M123**, **-0.000625 vs M119**, and **-0.000589 vs M124**. Interpretation: seed13+MTL was a validation false positive despite strong logloss (**0.221344**) and aux m91 (**0.857198**). This weakens trust in high-aux/high-logloss alone even inside the freshness family. Next eval priority, if quota remains and no new M154/M155/M157 result changes the picture, is **M147 best_model**; then **M146 best_model**. Do not evaluate M151 soup/aux side checkpoints.

> **M148 Eval Result Update**: `M148-m119-vr02-mtl-seed7-auxselect-soup-gap0-5ep_eval_1779482611683` returned public **AUC 0.851365**, a new team SOTA. Deltas: **+0.000354 vs M123** (0.851011), **+0.000611 vs M119** (0.850754), **+0.000647 vs M124** (0.850718), and **+0.002001 vs M106** (0.849364). Interpretation: the fresh `valid_ratio=0.02` + seed7 + MTL + 5ep direction transfers, but only modestly; it validates the freshness-family climb, not a 0.853-sized breakthrough. After M151's miss, next eval priority is **M147 best_model**, then **M146 best_model**, unless M154/M155/M157 return a materially stronger signal first. Do not jump to M148 soup/aux side checkpoint before trying the remaining normal best-model candidates unless specifically buying checkpoint-selection calibration.

> **M153/M156 Training Result Update**: M153 (`M119 no-MTL seed7 5ep`) is correct and healthy, but does not overtake the top freshness+MTL candidates. Its best primary is **0.870980** at **E4 / step65920**, logloss **0.221421**, aux m91 **0.854488**; E5 improves aux to **0.856338** but primary falls to **0.870553** and logloss worsens to **0.221897**. Aux/blend side checkpoint selects E5 because it stays within the primary tolerance; normal `best_model` is E4, and dense-only soup saved with 3 members. After M151's public miss, treat M153 as a backup/calibration behind **M147** and roughly comparable to M146, not a first-pass eval. M156 (`M123 MTL seed5 sparse_lr=0.07`) best is **0.870717** at E4, logloss **0.221962**, aux **0.854368**, soup 2 members; it is healthy but lower than M148/M147/M146, so hold/no default eval.

> **M145-M152/M158 Training Result Update**: The public-positive freshness family produced several legitimate eval candidates. Best pre-eval score-max signals were **M148** (`M119+MTL seed7 5ep`) best primary **0.871378** / logloss **0.221600** / aux m91 **0.857345**, **M151** (`M123 seed13+MTL`) **0.870969** / **0.221344** / **0.857198**, **M147** (`M123 seed5+MTL 5ep`) **0.870843** / **0.221623** / **0.857534**, and **M146** (`M119 seed7 no-MTL`) **0.871047** / **0.221636** / **0.855842**. Public eval later accepted M148 and rejected M151; current remaining normal-checkpoint priority is **M147 best_model**, then **M146 best_model**. Use normal `best_model` before dense-only soup for the first pass; soup checkpoints are side artifacts with no post-soup validation and should be evaluated only after a matching best_model public readout or if quota would otherwise expire. M145/M149/M150 are healthy but lower priority; M152 and M158 are hold/no-eval by default.

> **M137 Corrected Training Result Update**: User reports corrected `M137-m106-vr02-mtl-seed7-auxselect-gap0-4ep` completed. Config is now correct: `valid_ratio=0.02`, seed **7**, MTL, UE61-only, aux selector on, no soup. Best primary **0.870557** at E4 / step 65920, logloss **0.221938**, aux m91 **0.854349**. Aux and blend side checkpoints both point to E4 (`score=0.854349` and blend `0.862453`). This validates that seed7+MTL is healthy, but it is slightly weaker than M119 on primary/logloss/aux and essentially tied/slightly worse than M123 on primary/logloss, with only a small aux edge. Default: **hold / do not evaluate** unless quota would otherwise expire and no M145/M146/M157/M158 readout is available.

> **M141 Training Result Update**: User reports `M141-m106-vr02-mtl-seed7-auxselect-soup-gap0-4ep` completed. Config is correct: `valid_ratio=0.02`, seed **7**, MTL, UE61-only, aux selector on, dense-only weight soup on. Best primary **0.870376** at E4 / step 65920, logloss **0.222206**, aux m91 **0.854879**. Aux/blend side checkpoints both point to E4, and dense-only soup saved `global_step65920.layer=2.head=4.hidden=64.auc=0.870376.members=3.weight_soup` from E4/E3/E2 while preserving sparse embeddings from E4. Compared with M137, M141 has lower primary/logloss but higher aux; compared with M119/M123, it is not a score-max improvement. If forced to spend exactly one eval from M137/M141, M141 soup is the higher-information candidate, but the preferred action is to wait for M145/M146/M157/M158.

> **M142 Training Result Update**: User reports `M142-m123-vr015-mtl-seed5-auxselect-soup-gap0-4ep` completed. Config is correct: `valid_ratio=0.015`, seed **5**, MTL, UE61-only, aux selector on, dense-only weight soup on, and no TIN. Best primary **0.872461** at E4 / step 66256, logloss **0.221440**, aux m91 **0.853713**. Aux/blend side checkpoints both point to E4; dense-only soup saved `global_step66256.layer=2.head=4.hidden=64.auc=0.872461.members=2.weight_soup` from E4/E3 sources. Compared with M138, M142 is weaker on primary (**-0.000410**), logloss (**+0.000154**, worse), and aux (**-0.000313**). Since M138 public was **0.849451** and M144 soup public was **0.848183**, M142 is **logs-only / do not evaluate** by default.

> **Final Sprint Queue Prep**: User reports corrected M137 and M141 are still running and the training candidate queue is empty. Prepared focused M145-M150 helpers around the only public-positive family, M119/M123/M124: exact M123 + auxselect/soup, exact M119 + auxselect/soup, 5-epoch M123/M141-style extensions, and two MTL conversion-weight variants (1.5 / 3.0). Do not reopen `vr015`, TIN-lite, UE61/87 concat, time, history-CVR, or other rejected branches for score-max by default. Full record: [EXP-093](../40-experiments/EXP-093-final-sprint-m123-family-m145-m150-2026-05-21.md).

> **Travel Buffer Queue Prep**: User will have limited decision time on **2026-05-22** during Spain -> San Francisco travel. Prepared M151-M156 as lower-priority queue fillers after M145-M150: seed13+MTL on vr02, seed7+MTL on vr01, M119 no-MTL 5ep, M123 lower-LR 5ep, and sparse LR 0.03/0.07 variants. These are designed to keep the training pool busy while staying inside the public-positive M119/M123 family.

> **Last-Chance UE Separate-Token Prep**: User asked whether this is the right/last window to try the true separate-token interpretation of external `feat 61/87` intel. Added opt-in `--ue_split_separate_tokens` support: multiple UE dense fids can become independent NS tokens. Because direct 2-token UE would make RankMixer `T=17` at `d_model=64`, prepared M157/M158 with `--user_ns_tokens 4` so `4 user-int + 2 UE + 2 item-int = 8 NS`, preserving total `T=16`. M157 tests true separate 61/87 tokens without xitem; M158 adds mild `UE x item` with `alpha=0.3`.

> **M144 Eval Result Update**: M144 dense-only soup / TIN-lite (`vr015 + MTL seed5`) evaluated at public **0.848183**. This is **-0.002828 vs M123**, **-0.002571 vs M119**, **-0.001181 vs M106**, and **-0.001268 vs M138**. Interpretation: soup did not stabilize the `vr015` failure; TIN-lite did not transfer; and high aux/logloss improvement on M144 was misleading. Close TIN-lite in this form for score-max, close `vr015` primary-led/soup variants by default, and do not evaluate M142 unless there is an extraordinary reason.

> **M143 Training Result Update**: User reports `M143-m106-vr02-mtl-seed7-tin-auxselect-soup-gap0-4ep` completed. Config is correct: `valid_ratio=0.02`, seed **7**, MTL, UE61-only, TIN-lite on, aux selector on, and dense-only weight soup on. Best primary **0.869295** at E4 / step 65920, logloss **0.222664**, aux m91 **0.853440**. Aux/blend side checkpoints both point to E4; dense-only soup saved `global_step65920.layer=2.head=4.hidden=64.auc=0.869295.members=2.weight_soup` from E4/E3 sources. Compared with M119 seed7 freshness-only, M143 is **-0.001901 primary**, **+0.001206 logloss** (worse), and **-0.001640 aux**; compared with M123 seed5+MTL, it is **-0.001289 primary**, **+0.000780 logloss** (worse), and **-0.000505 aux**. Interpretation: in this seed7 `vr02` branch, TIN-lite hurts the public-positive backbone metrics. Do **not** spend the final quota on M143 by default; wait for M141 if possible, otherwise M144 soup remains the higher-information forced calibration.

> **M138 Eval Result Update**: M138 (`valid_ratio=0.015` + MTL seed5 + aux selector) evaluated at public **0.849451**. This is **-0.001560 vs M123**, **-0.001303 vs M119**, **-0.001267 vs M124**, and only **+0.000087 vs M106**, despite its very high tiny-window primary **0.872871** and aux m91 **0.854026**. Interpretation: the 1.5% validation window is now confirmed as overly optimistic for score selection. Downgrade M138/M142-style `vr015` primary-led selection; M144 is still an informative TIN/soup calibration candidate, but not a score-max favorite after this miss.

> **M144 Training Result Update**: User reports `M144-m123-vr015-mtl-seed5-tin-auxselect-soup-gap0-4ep` completed. Config is correct: `valid_ratio=0.015`, seed **5**, MTL, UE61-only, `enable_tin_interest=True`, `tin_time_alpha_init=1.0`, aux selector on, and dense-only weight soup on (`include_sparse=False`). Best primary **0.871954** at E4 / step 66256, logloss **0.220568**, aux m91 **0.855139**. Aux and blend side checkpoints both point to the same E4 checkpoint; dense-only soup successfully saved `global_step66256.layer=2.head=4.hidden=64.auc=0.871954.members=3.weight_soup` from E4/E3/E2 sources. Later public eval of the M144 soup checkpoint returned **0.848183**, so this is now a negative calibration: aux/logloss/TIN/soup gains on `vr015` do not transfer.

> **M138 Training Result Update**: User reports `M138-m123-vr015-mtl-seed5-auxselect-gap0-4ep` completed. Config is correct: `valid_ratio=0.015`, seed **5**, MTL, UE61-only (`ue_split_fids=''`, `ue_split_fid=61`), aux m91, and `enable_aux_ckpt_select=True`. Best primary **0.872871** at E4 / step 66256, logloss **0.221286**, aux m91 **0.854026**. Aux and blend side checkpoints were saved and both point to the same E4 checkpoint (`aux_m91_best` score 0.854026, `blend_m91_best` score 0.863449). This is **+0.002287 primary / +0.000081 aux vs M123**, **+0.004148 primary / -0.000354 aux vs M124**, and **+0.001494 primary / +0.000913 aux vs M135**. Later public eval returned only **0.849451**, so this strong tiny-window readout is now treated as an optimism warning, not a score-max signal.

> **M137 Config-Mismatch Result Update**: User reports `M137-m106-vr02-mtl-seed7-auxselect-gap0-4ep` completed, but the copied Args show **`seed=5`**, **`enable_aux_ckpt_select=False`**, and **`ue_split_fids='61,87'`** with resolved `fids=[61,87]`. The intended local helper `run_m137_m106_vr02_mtl_seed7_auxselect.sh` is correct (`seed 7`, aux selector on, legacy UE61 only), so the platform run likely used a stale/wrong `run.sh`. Actual result: best primary **0.868682** at E4 / step 65920, logloss **0.223456**, aux m91 **0.852891**. Treat this as an invalid M137 readout / M133-like duplicate, **not** as evidence about M119 seed7 + MTL. Do not evaluate it. For queued M141/M143, verify their start logs show `seed=7`, `enable_aux_ckpt_select=True`, `enable_weight_soup=True`, and no `ue_split_fids='61,87'` before trusting their results.

> **M135 Training Result Update**: User reports `M135-m123-vr02-mtl-ue61-xitem-seed5-auxm91-gap0-4ep` completed. Config confirms M123 recipe (`valid_ratio=0.02`, MTL, seed5) + legacy UE61 split + explicit UE x item (`alpha_init=1.0`), pre-aux-selector. Best primary **0.871377** at E4 / step 65920, logloss **0.221546**, aux m91 **0.853113**. This is **+0.000793 primary vs M123** but **-0.000832 aux vs M123**; also **+0.001156 primary vs M136** but **-0.003096 aux vs M136**. Interpretation: M135 removes the noisy `fid87` concatenation and has a much healthier primary curve than M134/M136, so it becomes a real high-information eval candidate. However, because it still contains explicit xitem and M136 public rejected the related branch, do not spend quota until comparing against M137/M138/M141/M142 unless quota is about to expire.

> **M133 Training Result Update**: User reports `M133-m123-vr02-mtl-ue6187-seed5-auxm91-gap0-4ep` completed. Config confirms M123 recipe (`valid_ratio=0.02`, MTL, seed5) + concatenated UE61/87 token, **without** explicit UE x item and without aux selector. Best primary **0.869387** at E4 / step 65920, logloss **0.222777**, aux m91 **0.853595**. This is **-0.001197 primary vs M123** and **-0.000350 aux vs M123**; compared with M135 it is **-0.001990 primary** but **+0.000482 aux**. Interpretation: the current concatenated UE61/87 token is not a positive transfer mechanism on the public-positive M123 recipe. It closes M133 as logs-only and strengthens the read that `fid87` concatenation is noisy; if revisiting external UE intel, prefer true separate-token/user-int-dense handling rather than the present concat token.

> **Strategy Shift · 0.853 Target**: The target gap is now nearly **+0.002** from team SOTA M123, which is too large for seed-only, longer training, or ordinary validation-selector tweaks. Public-positive evidence still concentrates in the **freshness + MTL M106/M119/M123 family**. M136 shows that very high aux m91 can be a false friend when clean-window and primary validation are unhealthy. The next viable path must either (a) find a cleaner freshness/label-window formulation inside the M123 family, (b) discover an actually transfer-positive representation mechanism that passes stronger validation gates, or (c) exploit checkpoint/ensemble-style averaging only if the platform/eval rules allow it.

> **0.853 Strategy Note**: Added [EXP-090](../40-experiments/EXP-090-strategy-to-0853-2026-05-21.md). Immediate priority: keep M137/M138 first; cancel/deprioritize M139/M140 after M136 miss if platform can reclaim the slots; move new mechanism work to a weight-soup/SWA-style side-checkpoint path for the M123 family; revisit external UE61/87 only as true separate tokens rather than the current concatenated implementation.

> **Weight-Soup Implementation Update**: Implemented opt-in post-train `--enable_weight_soup` in `trainer.py` and `train.py`. It averages eligible same-run `best_model + topk` checkpoints into one extra `*.weight_soup` side checkpoint after training summary time, without changing optimization, early stopping, primary `best_model`, top-k, or aux/blend side selection. After reviewing the epoch-boundary sparse cold-restart mechanism, the default soup is **dense-only**: sparse embedding tensors are preserved from the best source unless `--weight_soup_include_sparse` is explicitly set. Prepared M141/M142 templates: `M141-m106-vr02-mtl-seed7-auxselect-soup-gap0-4ep` and `M142-m123-vr015-mtl-seed5-auxselect-soup-gap0-4ep`. Full record: [EXP-091](../40-experiments/EXP-091-weight-soup-side-checkpoint-m141-m142-2026-05-21.md). Local verification: `python3 -m py_compile src/train.py src/trainer.py src/infer.py src/model.py` and `bash -n` for the two new templates pass; local smoke test could not import `torch`, so real averaging will first be exercised on platform logs.

> **TIN-lite Implementation Update**: Implemented opt-in `--enable_tin_interest` as a shape-compatible temporal upgrade to classifier-time DIN: each sequence token gets a zero-init target-conditioned time-bucket modulation before DIN attention, so the classifier width remains unchanged and the module starts close to ordinary DIN. Requires `--enable_din_interest`, raw DIN source, and time buckets. Prepared M143/M144 templates: `M143-m106-vr02-mtl-seed7-tin-auxselect-soup-gap0-4ep` and `M144-m123-vr015-mtl-seed5-tin-auxselect-soup-gap0-4ep`. Full record: [EXP-092](../40-experiments/EXP-092-tin-lite-temporal-interest-m143-m144-2026-05-21.md).

> **M136 Eval Result Update**: M136 (`M106/M119 seed7 freshness + MTL + UE61/87+xitem`) evaluated at public **0.849015** (179.61s). This is **-0.001996 vs M123**, **-0.001739 vs M119**, and **-0.000349 vs M106**, despite aux m91 **0.856209**. Interpretation: high aux m91 did **not** transfer here; the UE61/87+xitem branch is downgraded sharply. Do not evaluate M134/M132 by default. M135 may still be useful as UE61-only isolation, but after M136 its eval bar should be much higher: require very healthy primary validation, not merely high aux.

> **M137-M140 Prep Update**: Prepared the next queue fill because M136 was under eval and the training queue was empty. After M136 public miss, priority tightens to **M137 > M138 >> M139/M140**. M137 tests the missing core ablation `M119 seed7 freshness + MTL` without UE61/87/xitem confounds; M138 interpolates freshness at `valid_ratio=0.015` on the M123 seed5+MTL recipe. M139/M140 are still lower-risk than M134/M136 because they use UE61-only and `alpha_init=0.3`, but they should be treated as lower priority unless already queued. Full record: [EXP-089](../40-experiments/EXP-089-m123-family-auxselect-m137-m140-2026-05-21.md).

> **M127-M136 Training Result Update**: Current copied log has no crash/OOM. M127 clean MultiEmbK2 is red (**0.763918**, vs M91 **-0.005595**), and M128 full MultiEmbK2 is below accepted baselines (main **0.870263**, aux m91 **0.851802**), so close the MultiEmb forensic branch. M129 clean UE61/87 is red (**0.762276**, vs M91 **-0.007237**), and M131 clean UE61/87+xitem is also red (**0.757423**, vs M91 **-0.012090**). Full M130/M132 do not beat the standard M106/M119/M123 primary baselines, but M132 has a high aux m91 (**0.856140**). M133 primary **0.869387** / aux **0.853595** confirms the concatenated UE61/87 token alone is not enough on the M123 recipe. M134 and M136 were pre-selector runs; M134 primary **0.869284** / aux **0.856402**, M136 primary **0.870221** / aux **0.856209**. Later M136 public **0.849015** rejected the branch, so treat these high-aux UE61/87+xitem signals as non-transferable unless a future variant has much healthier primary and clean-window behavior.

> **M123/M124/M116 Eval Result Update**: M123 (`valid_ratio=0.02` + MTL, seed5) public **0.851011**, new SOTA. M124 (`valid_ratio=0.01` + MTL, seed5) public **0.850718**, essentially tied with M119 but slightly lower. M116 (standard 5% valid seed17) public **0.847886**, far below M106/M119/M123 and below M84, closing the standard-valid seed upper-tail branch for score-max. Interpretation: public ranking now strongly favors **freshness + MTL** over standard-valid seed search; internal standard-valid AUC is a poor selector for this late-stage public objective.

> **M119 Eval Result Update**: M119 (`M106 seed7`, `valid_ratio=0.02`, freshness-only) public **0.850754** (703.23s). Interpretation: reduced-valid-ratio / fresher-training is not just validation noise; it transferred strongly to public and is now the clearest score-max lever so far. Remaining gap to target **0.852** is **+0.001246**; remaining gap to reported Top1 **0.85344** is **+0.002686**. This result raises the priority of freshness-family calibration candidates (M123/M124) and future M129-M132 stacks over older standard-valid seed attempts.

> **Eval-Speed Hotfix**: `infer.py` test-set EDA blob is now **default off** (`EDA_BLOB` defaults to `0`; set `EDA_BLOB=1` only for deliberate diagnostic evaluations). This does not change predictions or checkpoint compatibility; it only removes per-batch CPU/logging overhead from score-max evals.

> **External Intel Update · UE 61/87 + UE x item**: 用户贴出社区截图：one external LB row shows **auc 0.852746** with **152.3s** inference time; user also reports current LB Top1 has reached **0.85344**. Another post asks about "UE 分离" and "user int-dense"; comments explicitly say **"feat 61/87"** and "作为一个单独的 token，然后做显式交互 **UE x item**". This is high-value because current implementation only supports `--ue_split_fid` as a **single** fid and all accepted M49/M75/M84/M106 path-B runs split **fid61 only**; `fid87` remains in `other_dense`/dense_bypass, not a separate UE token. Current code also has UE split, dense_bypass, UE x user_int gating, item-conditioned query, DIN, and user_int x item_int DCN, but **does not implement explicit dense UE x item interaction**. New top mechanism hypothesis: split or jointly tokenise **fid61+fid87**, then add an OOV-friendly explicit `UE x target item` interaction bypass; treat this as a stronger 0.852 path than more seed/time/validation tuning.

> **M129-M132 Prep Update**: Implemented opt-in `--ue_split_fids "61,87"` and `--enable_ue_item_interaction`. Prepared four helpers under `platform_run_templates/`: M129 clean 61+87 UE token, M130 full 61+87 UE token, M131 clean 61+87 + explicit UE x item, M132 full 61+87 + explicit UE x item. Upload selected helper as platform `run.sh`; include updated `train.py`, `model.py`, `infer.py`, and usual shared files. Full record: [EXP-086](../40-experiments/EXP-086-ue6187-xitem-m129-m132-2026-05-20.md).

> **M129-M132 Completion Update**: 用户回报 M129-M132 have completed. Clean diagnostics M129/M131 are both red; full runs M130/M132 are logs-only by default, with M132 notable only for high aux m91 **0.856140**.

> **M133-M136 Prep Update**: Prepared four helpers that stack the external UE mechanism on the public-positive M123 recipe: M133 (`vr02+MTL+UE61/87`), M134 (`vr02+MTL+UE61/87+xitem`, main 0.852 shot), M135 (`vr02+MTL+UE61+xitem`, low-risk isolation), and M136 (`vr02+MTL+UE61/87+xitem seed7`, high-aux seed variant). Queue priority if constrained: **M134 > M135 > M136 > M133**. Upload the selected helper as platform `run.sh`; include updated `train.py`, `model.py`, `infer.py`, and usual shared files. Full record: [EXP-087](../40-experiments/EXP-087-m123-ue-stack-m133-m136-2026-05-20.md).

> **M133-M136 Queue Clarification**: 用户确认 M133-M136 have already been submitted to the platform queue and will not be paused/restarted for the aux checkpoint selector change. Therefore, treat the currently queued M133-M136 as primary-valid/topk-selected jobs. The local helper templates may include selector flags for future reruns, but those flags are **not assumed active** for the already queued jobs.

> **Aux Checkpoint Selector Update**: Implemented opt-in `--enable_aux_ckpt_select` for future aux-window runs. It preserves primary-valid `best_model`, `topk`, early stopping, and training updates, but can additionally save `*.aux_m91_best` and `*.blend_m91_best` side checkpoints when `aux_valid/m91` is present and primary AUC is within the configured tolerance of the current primary best. Use this for new jobs after M133-M136; do not assume it is active in already-submitted platform jobs unless the copied log Args line confirms it. Full record: [EXP-088](../40-experiments/EXP-088-aux-ckpt-side-selector-2026-05-20.md).

> **M124 Result Update**: 用户回报 `M124-m106-vr01-mtl-seed5-auxm91-gap0-4ep` completed and copied logs to `~/Downloads/train_log.txt`. Config confirmed `valid_ratio=0.01`, `multi_task_loss=True`, `action_num=2`, seed **5**, M106 corrected-T38 backbone, and aux m91 logging. Main valid curve: E1 **0.863527**, E2 **0.866578**, E3 **0.868557**, E4 **0.868723** (`global_step66592...best_model`, logloss **0.225034**). Aux m91 curve: E1 **0.845686**, E2 **0.847602**, E3 **0.849584**, E4 **0.854380**. Interpretation: high aux but weak aggressive-1% main; hold, do **not** evaluate by default.

> **M119/M126 Result Update**: 用户回报 M119 and M126 completed and copied logs to `~/Downloads/train_log.txt`. M119 (`M106 seed7`, `valid_ratio=0.02`, freshness-only) best **0.871196** at E4 / step 65920, logloss **0.221458**, aux m91 **0.855080**. It beats M120/M117/M123 on the 2% freshness family and has the highest sibling aux seen so far. M126 (`M106 seed5 + is_missing`, standard 5% valid) best **0.870945** at E4 / step 63764, logloss **0.221435**, aux m91 **0.852607**; config flags were correct, but M125 clean gate was red, so M126 remains logs-only by default.

> **M123 Result Update**: 用户回报 `M123-m106-vr02-mtl-seed5-auxm91-gap0-4ep` completed. Config confirmed `valid_ratio=0.02`, `multi_task_loss=True`, `action_num=2`, seed **5**, M106 corrected-T38 backbone, and aux m91 logging. Main valid curve: E1 **0.865981**, E2 **0.869054**, E3 **0.869818**, E4 **0.870584** (`global_step65920...best_model`, logloss **0.221884**). Aux m91: E1 **0.845028**, E2 **0.848757**, E3 **0.854422**, E4 **0.853945**. Compared with M117 seed5 freshness-only, M123 gains about **+0.000550 main** and **+0.001600 aux** at E4; compared with M120 seed13 freshness-only, M123 is slightly lower on 2% main but much higher on aux. Later M119 surpassed M123 on both 2% valid and aux, so M123 becomes a secondary freshness+MTL calibration candidate rather than the top upside pick.

> **M120/M125 Result Update**: 用户回报 `M120-m106-vr02-seed13-auxm91-gap0-4ep` and `M125-cleanwin0303-m84-ismiss-useritem-gap0-seed42-4ep` completed. M120 (`valid_ratio=0.02`, seed13 freshness) best **0.870606** at E4 / step 65920, logloss **0.221836**, aux m91 **0.852176** at E4 (peak **0.852588** at E3). This is below M116 main/aux and does not rescue freshness-only. M125 clean-window is-missing best **0.765837** at E4 / step 10136, logloss **0.217792**; vs M91 clean baseline **0.769513**, delta **-0.003676**, red by the pre-registered gate. Decision: **M125 vetoes M126 public eval by default**.

> **M116/M117 Result Update**: 用户回报 M116 and M117 completed and copied logs to `~/Downloads/train_log.txt`. M116 (`M84 corrected-T38 seed17`, standard `valid_ratio=0.05`) best main valid **0.870697** at E4 / step 63764, logloss **0.222083**, aux m91 **0.853761**. This is **+0.000006 vs M121**, **+0.000388 vs M115**, but **-0.000478 vs M106 E4**. M117 (`M106 seed5`, `valid_ratio=0.02`) best tiny-valid **0.870034** at E4 / step 65920, logloss **0.222221**, aux m91 **0.852345**. Interpretation: M116 becomes the least-bad forced-choice candidate from the completed overnight queue; M117 confirms that simple 2% freshness alone is not enough.

> **M115 Result Update**: 用户回报 `M115-m84-t38fix-seed13-auxm91-gap0-4ep` completed and copied logs to `~/Downloads/train_log.txt`. Config confirmed M84 corrected-T38 seed **13**, standard `valid_ratio=0.05`, train **1,990,981** rows / valid **108,975** rows, and aux `m91:1772502591:1772520547` (`status=leaky`). Main valid curve: E1 **0.866463**, E2 **0.867895**, E3 **0.870068**, E4 **0.870309** (`global_step63764...best_model`, logloss **0.223152**). Aux m91 curve: E1 **0.843090**, E2 **0.848957**, E3 **0.850971**, E4 **0.853551**. Interpretation: aux is interesting and slightly above M113, but main valid is **-0.000866 vs M106 E4** and **-0.000382 vs M121 E4**; **do not evaluate M115 by default**.

> **M118 Result Update**: 用户回报 `M118-m106-vr01-seed5-auxm91-gap0-4ep` completed and copied logs to `~/Downloads/train_log.txt`. Config confirmed M106 seed5 backbone with `valid_ratio=0.01`, train **2,079,013** rows / valid **20,943** rows, and aux `m91:1772502591:1772520547`. Main valid curve: E1 **0.863552**, E2 **0.866750**, E3 **0.868792**, E4 **0.869640** (`global_step66592...best_model`). Copied log ends immediately after E4 checkpoint save, so E4 aux/final summary are unavailable; E1-E3 aux m91 peaks at **0.849728**. Interpretation: `vr01` freshness alone is not eval-worthy by default; tiny-valid AUC is not comparable to 5% valid, and sibling aux is weaker than M106/M113/M121/M122.

> **M125-M128 Forensic Whitelist Submission Update**: 用户同意复盘“实现瑕疵/旧验证体系是否错过好方案”, and later confirmed all four jobs were submitted. These are clean/full pairs for the only unresolved old-branch candidates worth queueing: `M125/M126` test T31 is-missing on M84/M106, and `M127/M128` retry Multi-Emb K=2 after the historical M38 silent-copy accident. Do **not** reopen Q-init, UE-bilinear, DCN, time stacks, history-CVR, seqstats, mature-negative, idmask, or AUC/BPR by default. Full record: [EXP-085](../40-experiments/EXP-085-forensic-whitelist-m125-m128-2026-05-20.md).

> **M123/M124 Freshness+MTL Prep Update**: 用户同意尽快启动新机制. Prepared `M123-m106-vr02-mtl-seed5-auxm91-gap0-4ep` and `M124-m106-vr01-mtl-seed5-auxm91-gap0-4ep` under `platform_run_templates/`. These combine the M106 seed5 corrected-T38 backbone, M117/M118 reduced-valid-ratio freshness, and M121 MTL supervision. Upload the selected helper as platform `run.sh`; templates remain ignored by git. Full record: [EXP-084](../40-experiments/EXP-084-m106-freshness-mtl-m123-m124-2026-05-20.md).

> **M113/M114/M121/M122 Result Update**: 用户回报 M113, M114, M121, and M122 completed and copied logs to `~/Downloads/train_log.txt`. M113 seed7 best **0.870344** at E4 / step 63764 with aux m91 **0.853309**; M114 seed11 best **0.870128** at E3 / step 47823; M121 MTL seed5 best **0.870691** at E4 / step 63764; M122 MTL seed7 best **0.870176** at E3 / step 47823, with high E4 aux m91 **0.853627** but lower main valid. Later M116 slightly surpassed M121 on both main valid and aux; see the M116/M117 update above.

> **M106 E3 Eval + Queue Update**: 用户回报 all ten M113-M122 jobs have been started or entered the platform queue; M110/M112 will be allowed to finish. M106 E3 public eval completed: `M106-m84-t38fix-seed5-auxm91-gap0-4ep-E3_eval_1779222359418`, public **0.848760** (744.33s), which is **-0.000604 vs M106 E4** and **-0.000094 vs M84**. Interpretation: M106's public gain is checkpoint-specific to E4, not a general early-checkpoint improvement; avoid spending further quota on M106 internal checkpoint mining unless a new log signal appears.

> **M110/M112 Partial Running Update**: 用户回报 M110/M112 are still running and have E2 checkpoints. M110 reached **0.868925** at step 31882; M112 reached **0.868436** at step 31882. Because their clean companions are already red (M109 **0.767038**, M111 **0.767948** vs M91 **0.769513**) and the partial full-data dev is still below M106 E4 **0.871175**, recommendation is **stop/cancel M110 and M112 if platform can reclaim slots**. If stopping is impossible or nearly free to let finish, allow completion but do not promote/evaluate unless a later checkpoint produces an exceptional, cross-validated signal.

> **M106 E3 Eval + M113-M122 Prep Update**: 用户确认 M106 E3 checkpoint was evaluable and evaluated; result later returned public **0.848760**, below M106 E4 **0.849364**. Prepared ten overnight training templates under `platform_run_templates/`: M113-M116 seed upper-tail, M117-M120 reduced valid-ratio freshness variants, M121-M122 multi-task click+conversion variants; user later confirmed all ten have been started or queued. Full record: [EXP-083](../40-experiments/EXP-083-m106-overnight-upper-tail-m113-m122-2026-05-19.md).

> **M106 Eval Result Update**: 用户回报 M106 evaluation completed: `M106-m84-t38fix-seed5-auxm91-gap0-4ep_eval_1779219154210`, public **0.849364** (858.21s). This is a new LB SOTA, **+0.000510 vs M84** (0.848854) and **+0.000576 vs M75** (0.848788). Against target **0.852**, remaining gap is **+0.002636**. Interpretation: M84-family seed upper-tail can transfer, and M106 captured a real same-family public gain; however this is still a seed/score-max gain, not a mechanism-level jump. Remaining eval quota should be reserved for a genuinely fresh signal, not M110/M112 unless explicitly buying calibration.

> **M111 Result Update**: 用户回报 M111 completed and copied logs to `~/Downloads/train_log.txt`. Config confirmed clean-window `M111-cleanwin0303-m84-edindin-perdomain-gap0-seed42-4ep`: `din_interest_source=encoded`, `din_interest_merge=per_domain`, fixed valid `[1772502591,1772520547)`. Best clean AUC **0.767948** at E2 / step 5068, logloss **0.220206**. Against M91 clean baseline **0.769513**, delta is **-0.001565**, red by the pre-registered gate. Decision: **per-domain encoded-DIN also fails; do not evaluate M112 by default**. Since M109 and M111 are both red, close DIN 2.0 in this encoded-late-bypass form unless deliberately buying a public calibration burn.

> **M109 Result Update**: 用户回报 M109 completed and copied logs to `~/Downloads/train_log.txt`, with manual checkpoint table. Config confirmed clean-window `M109-cleanwin0303-m84-edindin-compact-gap0-seed42-4ep`: `din_interest_source=encoded`, `din_interest_merge=compact`, fixed valid `[1772502591,1772520547)`. Best clean AUC **0.767038** at E3 / step 7602, logloss **0.218418**; E4 checkpoint **0.765129**. Against M91 clean baseline **0.769513**, delta is **-0.002475**, red by the pre-registered gate. Decision: **compact encoded-DIN fails; do not evaluate M110 by default**. M111 later returned red as well, so the per-domain branch is also closed in this encoded-late-bypass form.

> **M109-M112 Restart Update**: 用户回报 M109-M112 were restarted after the hotfix; M109 and M111 have now completed, and M110/M112 remain running by last confirmed state. Treat the previous `ValueError: not enough values to unpack (expected 5, got 4)` as a resolved launch issue unless the new logs repeat it. No public eval should be spent on M110/M112 after both clean companions returned red.

> **M106 Result Update**: 用户回报 M106 completed and copied logs to `~/Downloads/train_log.txt`. Parsed result: M106 (`M84 corrected-T38 seed=5 + aux m91`) best normal dev **0.871175** at E4 / step 63764, logloss **0.222041**. This is **+0.000685 vs M84**, **+0.000319 vs M75**, and **+0.000377 vs M105**. Aux m91 was configured as `status=leaky`; copied log contains E1-E3 aux only, with E3 **0.852261**, stronger than M105's aux peak **0.850865** by **+0.001396**. Decision before public eval: **M106 is the cross-seed batch winner and an eval-worthy fallback**. It later evaluated at public **0.849364**, becoming new SOTA. This is still a small seed/score-max candidate, not a mechanism-level 0.852 breakthrough.

> **M107/M108 Second-Oracle Result Update**: 用户回报 M107 completed and copied logs to `~/Downloads/train_log.txt`. M107 (`clk0302` M84 baseline) best AUC **0.821260** at E1 / step 1688, then **0.819377**, **0.733079**, **0.752655**, stopped by `logloss_rebound_epoch_4`. M108 (`clk0302` M104 timestamp-cutoff history-CVR) best AUC **0.876195** at E1 / step 1688. The paired delta is **M108 - M107 = +0.054935**, massively green by the pre-registered gate. However M104 public already returned **0.847560**, **-0.001294 vs M84**, so this is a validation-system calibration result: `clk0302` would have falsely endorsed history-CVR, while `clk0303_m91`/M103 was the better warning. Do **not** reopen history-CVR for score-max; treat second-oracle success as mechanism/window mismatch evidence.

> **M108 Result Update**: 用户回报 M108 completed and copied logs to `~/Downloads/train_log.txt`. Parsed result: M108 (`clk0302` fixed clean window + M104 timestamp-cutoff history-CVR) built a history-CVR cache from **164,429** pre-window rows (`global_prior=0.028547`, `values=8099`, `entries=46311`) and validated on `1772416191:1772434147` (EDA73 `clk0302`, 26,403 effective rows). Best AUC is **0.876195** at **E1 / step 1688**, with logloss **0.386351**; then AUC collapses to **0.820024** at E2 and **0.780074** at E3, triggering `logloss_rebound_epoch_3`. Later M107 returned and made the paired delta green, but the M104 public miss means this is not a score-max reopen signal.

> **M109-M112 Launch Failure Update**: 用户贴出 M109 clean-window launch log. The job reached dataset construction, then failed at `train.py:907` while unpacking `get_pcvr_data`: expected 5 values `(train, valid, holdout, aux_valid_loaders, dataset)` but got 4. Root cause is a mixed-version upload / interface mismatch from the aux-validation framework: updated `train.py` expected `aux_valid_loaders`, while the platform-side `dataset.py` path returned the old 4-tuple. This is **not** encoded-DIN model failure. Local hotfix: `train.py` now accepts either 4 or 5 return values and warns if aux windows cannot be logged. For full aux jobs, still upload the updated `dataset.py` so M110/M112 keep aux metrics.

> **M105 Result Update**: 用户回报 M105 completed and copied logs to `~/Downloads/train_log.txt`. Parsed result: M105 (`M84 corrected-T38 seed=2 + aux m91`) best normal dev **0.870798** at E4 / step 63764, logloss **0.221729**. This is **+0.000308 vs M84**, but **-0.000058 vs M75** and in the same weak seed-variance band rejected by M88's public miss. Aux m91 was configured as `status=leaky`, with AUC **0.850865** at E3 and **0.850576** at E4; use it only against M106, not against clean M91. Decision: **do not evaluate M105 by default**; M106 already won this seed pair and converted to public SOTA.

> **DIN 2.0 Prep Update**: 用户同意回到 0.852 主线，降低 seed 尝试优先级。Added opt-in `--din_interest_source {raw,encoded}` and `--din_interest_merge {compact,per_domain}`. Prepared M109-M112 as M84 + late classifier-time encoded DIN clean/full pairs. Full record: [EXP-082](../40-experiments/EXP-082-din2-late-encoded-bypass-m109-m112-2026-05-19.md).

> **M104 Eval Result Update**: 用户回报 M104 evaluation completed: `M104-m84-hcvr-ts24h-gap0-seed42-4ep_eval_1779199262757`, public **0.847560** (644.81s). This is **-0.001294 vs then-SOTA M84 0.848854** and **-0.000090 vs M89 0.847650**, despite M104 normal dev **0.871796**. Conclusion: timestamp-cutoff history-CVR does **not** transfer as a score-max branch. The mildly negative M103 clean signal was directionally correct. Do **not** evaluate M102 by default with scarce quota.

> **M107/M108 Prep Update**: 用户同意进入 M104 public eval + `clk0302` second-oracle diagnostics. Prepared local helpers `platform_run_templates/run_m107_cleanwin0302_m84_t38_fix.sh` and `platform_run_templates/run_m108_cleanwin0302_m84_hcvr_ts24h.sh`. After M104 public miss, these are downgraded to optional diagnostics unless already submitted: useful for explaining the miss, but not urgent score-max work. Full record: [EXP-081](../40-experiments/EXP-081-clk0302-second-oracle-m107-m108-2026-05-19.md).

> **M104 Eval Readout**: The recommended information eval returned negative. Treat this as a useful calibration burn: normal-dev history-CVR gains are not trustworthy without green clean-window support. Later M106 used one of the remaining quotas and became new SOTA, leaving **1** quota for a genuinely fresh signal.

> **M104 Full-Data Result Update**: 用户回报 M104 completed and copied logs to `~/Downloads/train_log.txt`. Parsed result: M104 (`M84 + timestamp-cutoff 24h history CVR`) best dev **0.871796** at E4 / step 63764, logloss **0.221065**. This is about **+0.001306 vs M84**, **+0.000338 vs M102**, and pairs with M103 clean-window **0.769341** (**-0.000172 vs M91**). Classification: **yellow-plus / information-eval candidate**. If spending one history-CVR quota now, prefer **M104 over M102** on full-data AUC/logloss, but the clean companion is weaker, so this is still not a green score-max candidate. If evaluated, use `global_step63764.layer=2.head=4.hidden=64.auc=0.871796.best_model`.

> **M102 Full-Data Result Update**: 用户回报 M102 completed and copied logs to `~/Downloads/train_log.txt`. Parsed result: M102 (`M84 + available-time history CVR`) best dev **0.871458** at E4 / step 63764, logloss **0.221240**. This is about **+0.000968 vs M84** normal dev and pairs with M101 clean-window **0.769523** (**+0.000010 vs M91**). Classification: **yellow-plus / information-eval candidate**, not a confirmed score-max candidate. Because EDA73 showed M91 should not be the sole oracle, M101's neutral clean result should not automatically veto M102. If evaluated, use `global_step63764.layer=2.head=4.hidden=64.auc=0.871458.best_model`; if waiting is possible, compare against M104/M105/M106 first.

> **EDA73 Result Update**: 用户回报 `EDA73-validation-oracle-audit-cappedseq` completed and copied logs to `~/Downloads/train_log.txt`. With sequence lengths capped to model input limits, the ranking is stable and now quantitatively cleaner: `clk0302` is best overall (`oracle_dist=0.11092`, rows 26,403, CVR 0.148809), `pre0304_all` is second (`oracle_dist=0.12854`, rows 1,551,840, CVR 0.110854), and current `clk0303_m91` is third (`oracle_dist=0.13369`, rows 29,962, CVR 0.069088). Recommendation remains `recommend_multi_window_gate=true`: use **M91 + clk0302** as two same-clock negative filters, and use **pre0304_all** as same-day full-data sanity, not as a clean label oracle.

> **EDA72 Result Update**: 用户回报 `EDA72-validation-oracle-audit` completed and copied logs to `~/Downloads/train_log.txt`. Result: `viable_window_count=7`, `best_same_clock=clk0302`, `best_same_day_pre_public=pre0304_all`, `recommend_multi_window_gate=true`. `clk0302` has 26,403 rows, CVR 0.148809, time-shape distance 0.06512; current `clk0303_m91` has 29,962 rows, CVR 0.069088, time-shape distance 0.07189; `pre0304_all` has 1,551,840 rows and CVR 0.110854. Caveat: the logged `seq_*_l1` and aggregate `oracle_dist` used raw train sequence lengths against capped public sequence lengths, so use them qualitatively only; the reliable readout is that M91 is useful but should become one member of a multi-window gate, not the sole oracle.

> **EDA73 Completion Update**: `EDA73-validation-oracle-audit-cappedseq` completed. Full record: [EXP-080](../40-experiments/EXP-080-eda73-validation-oracle-cappedseq-2026-05-19.md).

> **M101/M103 Clean Result Update**: 用户回报 M101/M103 completed and copied logs to `~/Downloads/train_log.txt`. Parsed results: M101 available-time history CVR clean best **0.769522861** at E4, logloss **0.215036**, delta vs M91 **+0.000010**; M103 timestamp-cutoff clean best **0.769340533** at E4, logloss **0.214394**, delta vs M91 **-0.000172**. Both are below the green promotion threshold **0.769913** but not red; classify as **yellow / neutral**.

> **Run Template Hygiene Update**: Local platform helper scripts have been moved from `src/run_*.sh` to `platform_run_templates/run_*.sh`. Keep `src/run.sh` as the only canonical source-side platform entry. The template files remain ignored by git; upload a selected helper as platform `run.sh` when needed.

> **M105/M106 Submission Update**: 用户回报 M105/M106 have both been submitted. They are low-regret M84 corrected-T38 cross-seed upper-tail jobs with auxiliary M91-window logging: `platform_run_templates/run_m105_m84_t38fix_seed2_auxm91.sh` and `platform_run_templates/run_m106_m84_t38fix_seed5_auxm91.sh`. Full record: [EXP-078](../40-experiments/EXP-078-m84-cross-seed-aux-m105-m106-2026-05-19.md).

> **EDA72 Prep Update**: Prepared `EDA72-validation-oracle-audit` to re-examine validation oracle design and public/test distribution assumptions. Upload `platform_run_templates/run_eda72_validation_oracle_audit.sh` as `run.sh` plus `src/eda_validation_oracle_audit.py`. This is a log-only EDA job and does not consume eval quota. Full record: [EXP-079](../40-experiments/EXP-079-eda72-validation-oracle-audit-2026-05-19.md).

> **M89 Eval Calibration Update**: M89 (`M84 + Beijing time v1+v2, no UTC`) evaluated at **public AUC 0.847650** (746.62s), job `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep_eval_1779176001150`. This is **-0.001204 vs then-SOTA M84 0.848854** despite normal dev **0.871864**, while its clean-window proxy M99 had failed vs M91 by **-0.004598**. Conclusion: clean-window was not over-rejecting this time branch; it correctly warned that the normal-dev gain would not transfer. Time-feature branch stays closed for score-max quota.

> **M101-M104 Submission Update**: 用户回报 the four history-CVR jobs have been submitted. M101/M103 are clean-window diagnostics against M91, M102/M104 are the matching full-data candidates. Do **not** spend public eval quota on this branch until the clean diagnostics return; see the clean-window caveat below and [EXP-076](../40-experiments/EXP-076-history-cvr-available-m101-m104-2026-05-19.md).

> **Clean-Window Caveat**: Current single clean-window gate is a strong guardrail against contaminated normal-dev signals, not a perfect public oracle. M89's public miss after M99's clean-window failure strengthens confidence in this gate for time-feature branches. Still use it as a tiered decision signal, not an absolute veto, for non-time mechanisms with large full-data dev gain and strong rationale.

> **Validation Framework Update**: Added optional `--aux_valid_windows` support. Future full-data jobs can log auxiliary timestamp-window AUC/LogLoss every epoch without changing best-model selection, top-k, early stopping, or training. Full record: [EXP-077](../40-experiments/EXP-077-aux-validation-framework-2026-05-19.md).

> **M93-M100 Gate Readout**: 用户回报 M93-M100 已跑完，日志在 `~/Downloads/train_log.txt`，部分日志回收不完整。Parsed results: **M93 clean delayed-feedback** best **0.762562**, below M91 **0.769513** by **-0.006951**; Gate 2 failed, so **do not evaluate M94**. M94 copied log only shows E1-E3, best **0.857370**, already far below M84 normal dev **0.870490**. **M95 seqstats** has normal dev **0.871160** (+0.000670 vs M84), but its clean diagnostic **M97** best is only **0.760092** (-0.009421 vs M91), so **do not evaluate M95 by default**. **M96 interaction** normal dev **0.859312** and clean **M98 0.756595** both fail. Time-feature closure also fails: **M99 0.764915** and **M100 0.763198**, both below M91. Conclusion: this entire overnight mechanism batch does not produce an eval-worthy score-max candidate; ordinary dev remains misleading for these directions.

> **Hotfix Update · M75 Result**: M75 evaluation completed with **public AUC 0.848788** (731.79s), replacing M49 0.848162 as LB SOTA. Attribution caveat: M75 should **not** be counted as true T38/per-domain bucket success because its training log showed per-domain fit `n_samples=0`; treat it as a stronger path B / UE split family realization. Full record: [EXP-071](../40-experiments/EXP-071-m75-lb-new-sota-attribution-2026-05-18.md).

> **Morning Batch Update**: 用户回报 **M76 / M82 / M83 / M84 / M75(E6)** 均训练完成. Best dev: M75(E6)=0.870651, M84=0.870490, M76=0.870247, M82=0.869709, M83=0.867486. Full record: [EXP-072](../40-experiments/EXP-072-may18-morning-batch-results.md).

> **M84/M82 Eval Update**: M84 public **0.848854** (695.58s) is a tiny new SOTA over M75 by +0.000066 and validates true T38 as small public-positive. M82 public **0.846197** (684.43s) rejects the alpha+hour daypart reweight path.

> **XHS Intel Update**: external post says baseline + rewritten **time** features can reach ~0.849 while TokenFormer is not clearly superior. We added opt-in `--enable_beijing_time_features` and prepared M85-M88. Full record: [EXP-073](../40-experiments/EXP-073-xhs-time-intel-m85-m88-plan-2026-05-18.md).

> **M85/M86 Train Update**: M85 best dev **0.870887** and M86 best dev **0.871415**. Both used corrected T38 nonzero per-domain samples. M86 was the strongest dev candidate, but its later public result rejected naive UTC+Beijing time stacking.

> **M89/M90 Training Update**: 用户回报 M89 and M90 both started. M89 and M90 have both completed.

> **M89 Completion Update**: M89 completed with best dev **0.871864** at E4, logloss **0.221121**. Config confirmed Beijing time v1+v2, no UTC `time_of_day_features`, corrected T38 nonzero samples. This is the strongest normal-dev time candidate so far, but M86 public miss means M89 evaluation still depends on M91/M92 clean-window gate unless quota is being used specifically for high-information risk.

> **M90 Completion Update**: M90 (`M86 exact UTC+Beijing time stack, 6ep`) completed. Best dev **0.871206** at E4; user-supplied checkpoint table shows E5 **0.870696** and E6 **0.870880**, so longer training did not improve beyond E4 and did not beat M86. Combined with M91/M92 Gate 1 failure, keep M90 on hold; do not evaluate.

> **M91/M92 Training Update**: 用户回报 M91 and M92 both submitted. Platform has a queue mechanism; submitted jobs may wait and then auto-start when a slot frees. Treat M91/M92 as **submitted / platform-scheduled**, and wait for completion logs to know actual run status. They are clean-window diagnostics, not direct LB submissions: use M91 vs M92 pseudo-public dev delta to judge whether time features transfer.

> **M91/M92 Clean-Window Result**: 用户回报 M91 / M92 均已完成. M91 clean-window M84 baseline best **0.769513**; M92 clean-window M86 UTC+Beijing time stack best **0.766748** (user supplied checkpoint table for incomplete M92 log). Delta **M92-M91 = -0.002765**, so Gate 1 fails hard. Treat M86 public miss as structurally reproduced by pseudo-public validation. Do **not** evaluate M89/M90 based on normal dev alone.

> **Breakthrough Protocol Update**: 用户回报 M93-M96 have been submitted to the platform training queue. They will auto-start as earlier queued/running jobs finish. M93 = clean-window delayed-feedback diagnostic, M94 = full delayed-feedback candidate, M95 = seq-stats candidate, M96 = delayed-feedback + seq-stats interaction. Full gates are in [EXP-075](../40-experiments/EXP-075-breakthrough-protocol-m93-m96-2026-05-18.md).

> **M87 Update**: M87 (`M84 + temporal_weight_alpha=2.5`) completed with best dev **0.869053**. It used corrected T38 nonzero samples, but is **-0.001438 vs M84** and far below M85/M86. Do **not** evaluate unless quota becomes abundant.

> **M88 Update**: M88 (`M84 6ep`) completed with best dev **0.870910** at E6. Longer training helped normal dev by **+0.000420 vs M84**, but the later public result rejected it.

> **M88 Eval Submission Update**: 用户回报 **M88 E6 已提交评估** as the final 5/18 quota; the result is now recorded below.

> **M88 Eval Result Update**: M88 E6 evaluation completed: public **0.846209** (722.84s), job `M88-m84-t38fix-6ep-gap0-seed42_eval_1779130091934_copy_1779131732501_copy_1779131958877_copy_1779132760150`. Eval log confirms checkpoint `global_step95646...auc=0.870910.best_model` and intended M84 6ep config (`time_of_day=True`, no Beijing/seqstats/mature weighting). Reject longer-training M84 E6; public is **-0.002645 vs M84 E4**.

> **M86 Eval Update**: M86 public **0.847106** (472.2s) is **-0.001748 vs M84** despite best dev 0.871415. Eval log confirms the intended `time_of_day_features=True` + `beijing_time_features=True` config and correct checkpoint, so treat this as a dev-public mismatch / time-stack overfit signal, not an infer config failure.

> **M86 Eval EDA Update**: Decoded the reversed Logs-UI base64/gzip blob from `eval_log.txt`. Test set has **523,244 rows**, public window **BJ 2026-03-04 09:49:51~14:49:07**, median **12:57:28**, and user IDs are singleton-like in the 200k cap. This supports EDA-71's time-window backtest focus.

> **EDA-71 Update**: `EDA71-time-window-backtest-audit` completed. Train ends at **BJ 2026-03-04 09:49:50** and public starts **1 second later** at **09:49:51**; exact public-window train rows = **0**. Current RG time split has **341,086s (~3.95d)** train/valid overlap, so normal dev is contaminated for time-feature decisions. Next time branch should use clean fixed-window backtests, not more absolute-time dense features by dev score.

> **M93-M96 Running Update**: 用户回报 **M93/M94/M95/M96 都还在训练中**. Treat training pool as occupied by these four mechanism jobs until completion logs arrive.

> **M97/M98 Diagnostic Prep**: Prepared clean-window seqstats diagnostics for the overnight queue. M97 = M84 + seqstats on the M91 clean pseudo-public window; M98 = M84 + mature-negative weighting + seqstats on the same window. These protect M95/M96 eval decisions from normal-dev contamination.

> **M99/M100 Diagnostic Prep**: Prepared clean-window time-feature closure diagnostics. M99 = M89-style Beijing time v1+v2 without UTC on the M91 clean window; M100 = M85-style Beijing time v1 only on the same window. These decide whether M89/M85 deserve any remaining information quota despite the M92 failure.

> **M97-M100 Queue Update**: 用户回报 **M97/M98/M99/M100 均已 submitted**. They are queued behind the current running jobs and will auto-start when prior tasks finish.

> **Staleness Warning**: M89-M138 plus M141-M153/M156/M158 training results and M84/M82/M86/M88/M89/M104/M106/M116/M119/M123/M124/M136/M138/M144/M148/M151 eval results are synced from copied logs and user-reported leaderboard rows. The first M137 copied Args did not match its intended recipe and is retained as a config-mismatch note; the corrected M137 rerun has now completed and is the valid seed7+MTL readout. M154/M155/M157 are pending by the latest user report/context. M94 final E4 is missing from the copied log, but the visible E1-E3 curve is already far below the gate, so this does not affect the no-eval decision. M106 copied train log stops after E4 aux evaluation starts, so E4 aux is unavailable, but E4 main best_model and public eval are confirmed. M118 copied train log stops after E4 checkpoint save, so E4 aux/final summary are unavailable. M106 E3 eval completed at public **0.848760**. M109 E4 logloss/final summary are missing from copied logs, but user-supplied checkpoint table confirms E4 AUC. Current resource state: M154/M155/M157 pending; M148 public **0.851365** is current SOTA; post-refresh eval quota remaining is unknown unless user says otherwise.

---

## Maintenance Protocol (硬流程)

**用户每次以下事件后 · agent 必须先 update 本文件再做后续推荐**:
1. 贴 platform job 启动 log (Args dict) → update In-Flight Jobs
2. 贴 platform job 完成 log (Training Summary) → update Last Job Status + free slot count
3. 贴 evaluation LB 结果 → update quota used + LB results
4. 反问 "现在还有几个 slot / quota" → agent 必须 grep 本文件回答 · 不凭印象

**如本文件 last updated > 30 min 或 agent 不确定准确性**:
- agent 必须 ask 用户确认 · **不能凭印象推荐 platform 操作**
- 用户回答后 agent 立即 update 本文件 + 改动 commit

---

## Training Slots / Queue (4 workers · platform queue auto-schedules)

> 来源: team-memory line 175 ("训练 slot 与 evaluation quota 是两个完全独立的资源池")
>
> **Queue correction (2026-05-18)**: 平台训练池有队列机制。提交到队列的 job 不一定已经占用 worker；当前面 job 训练完成释放 slot 时，平台会自动从队列取下一个 job 跑。因此后续资源判断要区分 **running/in-flight** 和 **queued/submitted**；不要把 queued job 误当成已完成或手动 free slot。

### Current Running / Platform-Scheduled (latest sync · 2026-05-21)

| Worker/Order | Status | Job | Notes |
|---|---|---|---|
| completed | ✅ done | **M90-m86-utc-bjtime-6ep-gap0-seed42** | best dev 0.871206 at E4; E5/E6 did not improve; no eval |
| completed | ✅ done | **M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep** | best dev 0.871864; public 0.847650; clean-window warning validated |
| completed | ✅ done | **M91-cleanwin0303-m84-t38fix-gap0-seed42-4ep** | clean-window M84 baseline best 0.769513 |
| completed | ✅ done | **M92-cleanwin0303-m86-utc-bjtime-gap0-seed42-4ep** | clean-window M86 UTC+Beijing stack best 0.766748; Gate 1 failed vs M91 by -0.002765 |
| completed | ✅ done | **M93-cleanwin0303-m84-mnw6h-w03-gap0-seed42-4ep** | best clean 0.762562; Gate 2 failed vs M91 by -0.006951 |
| completed | ✅ done | **M94-m84-mnw6h-w03-gap0-seed42-4ep** | copied log shows E1-E3 best 0.857370; far below M84 dev; no eval |
| completed | ✅ done | **M95-m84-seqstats-gap0-seed42-4ep** | normal dev best 0.871160, but gated by M97 failure |
| completed | ✅ done | **M96-m84-mnw6h-seqstats-gap0-seed42-4ep** | normal dev best 0.859312; failed |
| completed | ✅ done | **M97-cleanwin0303-m84-seqstats-gap0-seed42-4ep** | clean seqstats best 0.760092; fails vs M91 by -0.009421 |
| completed | ✅ done | **M98-cleanwin0303-m84-mnw6h-seqstats-gap0-seed42-4ep** | clean interaction best 0.756595; fails vs M91/M93/M97 |
| completed | ✅ done | **M99-cleanwin0303-m84-bjtime-v2-gap0-seed42-4ep** | clean Beijing v1+v2 best 0.764915; fails vs M91 by -0.004598 |
| completed | ✅ done | **M100-cleanwin0303-m84-bjtime-gap0-seed42-4ep** | clean Beijing-only best 0.763198; fails vs M91 by -0.006315 |
| completed | ✅ done | **M101-cleanwin0303-m84-hcvr-avail-gap0-seed42-4ep** | clean-window available-time history CVR best 0.769523; vs M91 +0.000010; yellow |
| completed | ✅ done | **M102-m84-hcvr-avail-gap0-seed42-4ep** | best dev 0.871458; deprioritized after M104 public miss |
| completed | ✅ done | **M103-cleanwin0303-m84-hcvr-ts24h-gap0-seed42-4ep** | clean-window timestamp-cutoff history CVR best 0.769341; vs M91 -0.000172; yellow |
| completed | ✅ evaluated | **M104-m84-hcvr-ts24h-gap0-seed42-4ep** | best dev 0.871796; public 0.847560, below M84; close score-max branch |
| completed | ✅ done | **M105-m84-t38fix-seed2-auxm91-gap0-4ep** | best dev 0.870798 at E4; aux m91 leaky 0.850576; hold eval by default |
| completed | ✅ evaluated | **M106-m84-t38fix-seed5-auxm91-gap0-4ep** | best dev 0.871175 at E4; public 0.849364, new SOTA; same-family seed upper-tail transferred |
| completed | ✅ done | **M107-cleanwin0302-m84-t38fix-gap0-seed42-4ep** | `clk0302` M84 baseline best 0.821260 at E1; unstable |
| completed | ✅ done | **M108-cleanwin0302-m84-hcvr-ts24h-gap0-seed42-4ep** | best clk0302 AUC 0.876195 at E1; delta vs M107 +0.054935 but public contradicted |
| completed | 🔴 gate failed | **M109-cleanwin0303-m84-edindin-compact-gap0-seed42-4ep** | best clean 0.767038; vs M91 -0.002475; compact encoded-DIN fails |
| partial recorded | 🔴 logs-only | **M110-m84-edindin-compact-auxm91-gap0-seed42-4ep** | only E2 partial recorded here (0.868925); M109 clean red and below M106 dev |
| completed | 🔴 gate failed | **M111-cleanwin0303-m84-edindin-perdomain-gap0-seed42-4ep** | best clean 0.767948; vs M91 -0.001565; per-domain encoded-DIN fails |
| partial recorded | 🔴 logs-only | **M112-m84-edindin-perdomain-auxm91-gap0-seed42-4ep** | only E2 partial recorded here (0.868436); M111 clean red and below M106 dev |
| completed | 🟡 hold | **M113-m84-t38fix-seed7-auxm91-gap0-4ep** | best dev 0.870344 E4; aux m91 0.853309; below M106/M116 main dev |
| completed | 🔴 hold | **M114-m84-t38fix-seed11-auxm91-gap0-4ep** | best dev 0.870128 E3, E4 drops; no default eval |
| completed | 🟡 aux-interest / hold | **M115-m84-t38fix-seed13-auxm91-gap0-4ep** | best dev 0.870309 E4; aux m91 0.853551; main below M106/M121, no default eval |
| completed | ✅ evaluated / rejected | **M116-m84-t38fix-seed17-auxm91-gap0-4ep** | best dev 0.870697 E4; public 0.847886; standard-valid seed upper-tail failed |
| completed | 🔴 hold | **M117-m106-vr02-seed5-auxm91-gap0-4ep** | best 2% valid 0.870034 E4; aux m91 0.852345; freshness alone weak |
| completed | 🔴 hold | **M118-m106-vr01-seed5-auxm91-gap0-4ep** | best tiny-valid 0.869640 E4; aux m91 through E3 peaks 0.849728; no default eval |
| completed | ✅ evaluated / SOTA | **M119-m106-vr02-seed7-auxm91-gap0-4ep** | best 2% valid 0.871196 E4; public 0.850754, new SOTA; freshness transferred |
| completed | 🔴 hold | **M120-m106-vr02-seed13-auxm91-gap0-4ep** | best 2% valid 0.870606 E4; aux m91 peak 0.852588; below M116, freshness weak |
| completed | 🟡 hold | **M121-m106-mtl-seed5-auxm91-gap0-4ep** | best dev 0.870691 E4; essentially tied with M116 but lower aux; below M106 |
| completed | 🔴 hold | **M122-m106-mtl-seed7-auxm91-gap0-4ep** | best dev 0.870176 E3; E4 aux 0.853627 but main valid lower |
| completed | ✅ evaluated / SOTA | **M123-m106-vr02-mtl-seed5-auxm91-gap0-4ep** | best 2% valid 0.870584 E4; public 0.851011, new SOTA; freshness+MTL transferred |
| completed | ✅ evaluated / near-SOTA | **M124-m106-vr01-mtl-seed5-auxm91-gap0-4ep** | best 1% valid 0.868723 E4; public 0.850718, near M119; aggressive freshness not harmful |
| completed | 🔴 gate failed | **M125-cleanwin0303-m84-ismiss-useritem-gap0-seed42-4ep** | clean is-missing best 0.765837; vs M91 -0.003676; veto M126 eval |
| completed | 🔴 logs-only | **M126-m106-ismiss-useritem-seed5-auxm91-gap0-4ep** | best dev 0.870945 E4; aux 0.852607; M125 clean red, do not evaluate by default |
| completed | 🔴 gate failed | **M127-cleanwin0303-m84-multiembk2-b96-gap0-seed42-4ep** | clean MultiEmbK2 best 0.763918; vs M91 -0.005595; veto M128 eval by default |
| completed | 🔴 hold | **M128-m106-multiembk2-b96-seed5-auxm91-gap0-4ep** | best dev 0.870263 E4; aux m91 0.851802; below M106/M119/M123 anchors |
| completed | 🔴 gate failed | **M129-cleanwin0303-m84-ue6187-gap0-seed42-4ep** | clean UE61/87 best 0.762276; vs M91 -0.007237 |
| completed | 🔴 hold | **M130-m106-ue6187-seed5-auxm91-gap0-4ep** | best dev 0.870131 E4; aux m91 peak 0.853780; not eval-worthy by default |
| completed | 🔴 gate failed | **M131-cleanwin0303-m84-ue6187-xitem-gap0-seed42-4ep** | clean UE61/87+xitem best 0.757423; vs M91 -0.012090 |
| completed | 🟡 high-aux / hold | **M132-m106-ue6187-xitem-seed5-auxm91-gap0-4ep** | best dev 0.869551 E4; aux m91 0.856140; high aux but primary/clean weak |
| completed | 🔴 logs-only | **M133-m123-vr02-mtl-ue6187-seed5-auxm91-gap0-4ep** | best 2% valid 0.869387 E4; aux m91 0.853595; concat UE61/87 alone below M123/M135, do not evaluate |
| completed | 🔴 hold | **M134-m123-vr02-mtl-ue6187-xitem-seed5-auxm91-gap0-4ep** | best 2% valid 0.869284 E4; aux m91 0.856402; do not evaluate by default after M136 public miss |
| completed | 🟡 high-information / hold | **M135-m123-vr02-mtl-ue61-xitem-seed5-auxm91-gap0-4ep** | best 2% valid 0.871377 E4; aux m91 0.853113; healthier than 61/87 branches, hold until M137/M138/M141/M142 |
| completed | 🔴 evaluated / rejected | **M136-m106-vr02-mtl-ue6187-xitem-seed7-auxm91-gap0-4ep** | public 0.849015; high aux 0.856209 did not transfer; downgrade UE61/87+xitem |
| completed | 🟡 corrected / hold | **M137-m106-vr02-mtl-seed7-auxselect-gap0-4ep** | corrected rerun: best 0.870557 E4, logloss 0.221938, aux 0.854349; healthy but not above M119/M123 enough; earlier upload was config-mismatched |
| completed | 🔴 evaluated / rejected | **M138-m123-vr015-mtl-seed5-auxselect-gap0-4ep** | public 0.849451; best 1.5% valid 0.872871 did not transfer; downgrade vr015 primary-led selection and M142-like candidates |
| prepared | 🟡 lower-priority queue | **M139-m123-vr02-mtl-ue61-xitem-a03-seed5-auxselect-gap0-4ep** | low-alpha UE61-only xitem, seed5; run only after M137/M138 unless already queued |
| prepared | 🟡 lower-priority queue | **M140-m106-vr02-mtl-ue61-xitem-a03-seed7-auxselect-gap0-4ep** | low-alpha UE61-only xitem, seed7; run only after M137/M138 unless already queued |
| completed | 🟡 soup saved / hold | **M141-m106-vr02-mtl-seed7-auxselect-soup-gap0-4ep** | correct seed7 vr02+MTL+soup; best 0.870376 E4, logloss 0.222206, aux 0.854879, soup members=3; higher-information than M137 if forced, but wait for M145/M146/M157/M158 |
| completed | 🔴 logs-only | **M142-m123-vr015-mtl-seed5-auxselect-soup-gap0-4ep** | correct vr015+MTL+soup; best 0.872461 E4, aux 0.853713, soup members=2; weaker than M138 and vr015 already rejected, no eval |
| completed | 🔴 hold / no default eval | **M143-m106-vr02-mtl-seed7-tin-auxselect-soup-gap0-4ep** | correct seed7 vr02+MTL+TIN; best 0.869295 E4, aux 0.853440, soup members=2; worse than M119/M123 anchors, do not eval by default |
| completed | 🔴 evaluated / rejected | **M144-m123-vr015-mtl-seed5-tin-auxselect-soup-gap0-4ep** | public 0.848183; correct TIN-lite+soup but did not transfer; close TIN-lite/vr015 soup for score-max |
| completed | 🟡 hold | **M145-m123-vr02-mtl-seed5-auxselect-soup-gap0-4ep** | exact M123 rerun + selector/soup; best 0.870540 E4, logloss 0.221967, aux 0.853691, soup members=2; essentially tied/slightly below M123, not first eval |
| completed | ✅ eval candidate | **M146-m119-vr02-seed7-auxselect-soup-gap0-4ep** | M119 no-MTL + selector; best 0.871047 E4, logloss 0.221636, aux 0.855842; soup skipped; eval-worthy as low-risk M119 calibration |
| completed | ✅ eval candidate | **M147-m123-vr02-mtl-seed5-auxselect-soup-gap0-5ep** | M123 5ep; best 0.870843 E5, logloss 0.221623, aux 0.857534, soup members=2; strong public-positive-family candidate |
| completed | ✅ evaluated / SOTA | **M148-m119-vr02-mtl-seed7-auxselect-soup-gap0-5ep** | public 0.851365, new SOTA; best train readout 0.871378 E5, logloss 0.221600, aux 0.857345, soup members=2; validates freshness+MTL+5ep but gain is modest |
| completed | 🟡 hold | **M149-m123-vr02-mtl-cw15-seed5-auxselect-soup-gap0-4ep** | M123 with conversion weight 1.5; best 0.870833 E4, logloss 0.221880, aux 0.853163, soup members=2; primary/logloss ok but aux weaker |
| completed | 🟡 hold | **M150-m123-vr02-mtl-cw30-seed5-auxselect-soup-gap0-4ep** | M123 with conversion weight 3.0; best 0.870717 E4, logloss 0.222292, aux 0.854515, soup members=2; not ahead of top candidates |
| completed | 🔴 evaluated / rejected | **M151-m123-vr02-mtl-seed13-auxselect-soup-gap0-4ep** | public 0.850129; seed13+MTL failed despite best 0.870969 E4, logloss 0.221344, aux 0.857198, soup members=2; high aux/logloss false positive |
| completed | 🔴 hold | **M152-m123-vr01-mtl-seed7-auxselect-soup-gap0-4ep** | aggressive vr01 seed7; best 0.868962 E4, logloss 0.226037, aux 0.853201, soup members=2; too weak, no default eval |
| completed | 🟡 hold / backup eval | **M153-m119-vr02-seed7-auxselect-soup-gap0-5ep** | M119 no-MTL 5ep; best primary 0.870980 at E4, logloss 0.221421, aux 0.854488; E5 aux 0.856338 but primary drops; soup members=3; backup behind M147, not clearly above M146 |
| pending | ⏳ waiting | **M154-m123-vr02-mtl-seed5-lr7e5-auxselect-soup-gap0-5ep** | pending |
| pending | ⏳ waiting | **M155-m123-vr02-mtl-seed5-sparse003-auxselect-soup-gap0-4ep** | pending |
| completed | 🟡 hold | **M156-m123-vr02-mtl-seed5-sparse007-auxselect-soup-gap0-4ep** | M123 sparse_lr=0.07; best 0.870717 E4, logloss 0.221962, aux 0.854368, soup members=2; healthy but not ahead of top candidates |
| pending | ⏳ waiting | **M157-m123-vr02-mtl-ue6187-septok-u4-seed5-auxselect-soup-gap0-4ep** | pure separate-token UE61/87 test pending; compare carefully against M158 before eval |
| completed | 🔴 hold | **M158-m123-vr02-mtl-ue6187-septok-xitem-a03-u4-seed5-auxselect-soup-gap0-4ep** | separate UE61/87 tokens + mild xitem; best 0.870462 E4, logloss 0.222107, aux 0.853288; soup skipped; not eval-worthy before M157 |
| completed | ✅ done | **EDA72-validation-oracle-audit** | recommends multi-window gate; best same-clock window `clk0302`, best same-day sanity `pre0304_all` |
| completed | ✅ done | **EDA73-validation-oracle-audit-cappedseq** | capped-seq rerun confirms `clk0302` > `pre0304_all` > `clk0303_m91`; multi-window gate |

### Submitted (M101-M104 · history-CVR available-time revisit)

| Priority | Job | Run script | Purpose |
|---|---|---|---|
| 1 | `M101-cleanwin0303-m84-hcvr-avail-gap0-seed42-4ep` | `platform_run_templates/run_m101_cleanwin0303_m84_hcvr_avail.sh` | Clean-window diagnostic: M84 + available-time history CVR; compare vs M91 |
| 2 | `M102-m84-hcvr-avail-gap0-seed42-4ep` | `platform_run_templates/run_m102_m84_hcvr_avail.sh` | Full-data candidate if M101 clean gate passes |
| 3 | `M103-cleanwin0303-m84-hcvr-ts24h-gap0-seed42-4ep` | `platform_run_templates/run_m103_cleanwin0303_m84_hcvr_ts24h.sh` | Timestamp-cutoff control under the same clean window |
| 4 | `M104-m84-hcvr-ts24h-gap0-seed42-4ep` | `platform_run_templates/run_m104_m84_hcvr_ts24h.sh` | Full-data candidate if M103 clean gate passes |

Gate: M91 clean baseline is **0.769513**; M101/M103 should reach at least **0.769913** before their full-data partner receives public eval quota.

Result: M101 **0.769523** and M103 **0.769341** both miss the green threshold. M102 full-data returned **0.871458**, about **+0.000968 vs M84**. M104 full-data returned **0.871796**, about **+0.001306 vs M84**, but M104 public returned only **0.847560**. History-CVR is downgraded after this miss; do not evaluate M102 by default.

### Completed (M105-M106 · M84 cross-seed upper-tail search)

| Priority | Job | Run script helper | Purpose |
|---|---|---|---|
| 1 | `M105-m84-t38fix-seed2-auxm91-gap0-4ep` | `platform_run_templates/run_m105_m84_t38fix_seed2_auxm91.sh` | M84 corrected-T38 seed=2 upper-tail search; logs M91 aux window |
| 2 | `M106-m84-t38fix-seed5-auxm91-gap0-4ep` | `platform_run_templates/run_m106_m84_t38fix_seed5_auxm91.sh` | M84 corrected-T38 seed=5 upper-tail search; logs M91 aux window |

M105 result: best dev **0.870798** at E4, logloss **0.221729**; aux m91 is **leaky** and should only be compared to M106's aux readout. This is a yellow/fallback seed result, not an immediate eval candidate.

M106 result: best dev **0.871175** at E4, logloss **0.222041**; copied log has aux m91 through E3 only, where M106 reaches **0.852261** vs M105 peak **0.850865**. M106 is the cross-seed batch winner.

Public eval result: M106 E4 best_model returned **0.849364**, new SOTA and **+0.000510 vs M84**. Default eval policy after this result: do not evaluate M105, and reserve the final quota for a fresher mechanism signal.

### Ready To Launch (M107-M108 · clk0302 second oracle)

| Priority | Job | Run script helper | Purpose |
|---|---|---|---|
| 1 | `M107-cleanwin0302-m84-t38fix-gap0-seed42-4ep` | `platform_run_templates/run_m107_cleanwin0302_m84_t38_fix.sh` | Baseline M84 on EDA73 best same-clock proxy window |
| 2 | `M108-cleanwin0302-m84-hcvr-ts24h-gap0-seed42-4ep` | `platform_run_templates/run_m108_cleanwin0302_m84_hcvr_ts24h.sh` | M104 timestamp-cutoff history-CVR on the same `clk0302` window |

Gate: compare **M108 - M107**. Green >= +0.0004, yellow in [-0.0003, +0.0004), red < -0.0003. After M104 public miss, this pair is optional unless already submitted.

M107/M108 result: M107 best **0.821260** at E1/step1688; M108 best **0.876195** at E1/step1688. Delta **+0.054935** is green by the original gate, but M104 public already returned **0.847560**. Therefore `clk0302` is **not** a reliable positive oracle for history-CVR; it is useful as a negative/diagnostic window only after mechanism-specific calibration.

### Running / Partial Results (M109-M112 · DIN 2.0 late encoded bypass)

| Priority | Job | Run script helper | Purpose |
|---|---|---|---|
| 1 | `M109-cleanwin0303-m84-edindin-compact-gap0-seed42-4ep` | `platform_run_templates/run_m109_cleanwin0303_m84_encoded_din_compact.sh` | Clean-window diagnostic: M84 + final encoded sequence states for DIN, compact merge |
| 2 | `M110-m84-edindin-compact-auxm91-gap0-seed42-4ep` | `platform_run_templates/run_m110_m84_encoded_din_compact_auxm91.sh` | Full-data compact candidate with M91 aux logging |
| 3 | `M111-cleanwin0303-m84-edindin-perdomain-gap0-seed42-4ep` | `platform_run_templates/run_m111_cleanwin0303_m84_encoded_din_perdomain.sh` | Clean-window diagnostic: keep per-domain DIN interests into classifier |
| 4 | `M112-m84-edindin-perdomain-auxm91-gap0-seed42-4ep` | `platform_run_templates/run_m112_m84_encoded_din_perdomain_auxm91.sh` | Full-data per-domain candidate with M91 aux logging |

Gate: M91 clean baseline is **0.769513**. Green >= +0.0004 over M91; yellow in [-0.0003, +0.0004) plus strong full-data dev/logloss; red < -0.0003. This is the current 0.852 mainline because it upgrades the only historically large validated mechanism (target-aware DIN) while keeping it as a late classifier bypass.

M109 result: compact encoded-DIN clean diagnostic best **0.767038**, delta vs M91 **-0.002475**. This is red. Do **not** evaluate M110 by default, regardless of ordinary dev, unless explicitly spending a high-risk calibration quota. M111 later returned red as well, so the per-domain encoded-DIN branch is also closed.

M111 result: per-domain encoded-DIN clean diagnostic best **0.767948**, delta vs M91 **-0.001565**. This is also red. Do **not** evaluate M112 by default. M109+M111 jointly close DIN 2.0 in this encoded-late-bypass form.

### Completed EDA (validation oracle audit)

| Job | Platform entry | Purpose |
|---|---|---|
| `EDA72-validation-oracle-audit` | uploaded `platform_run_templates/run_eda72_validation_oracle_audit.sh` as `run.sh`; included `src/eda_validation_oracle_audit.py` | Result: multi-window gate recommended; `clk0302` best same-clock, `pre0304_all` best same-day sanity |

### Ready To Launch (M85-M88 · XHS time-feature response)

| Priority | Job | Run script | Purpose |
|---|---|---|---|
| 1 | `M85-m84-t38fix-bjtime-gap0-seed42-4ep` | `platform_run_templates/run_m85_t38_fix_bjtime.sh` | M84 stack; replace UTC time with Beijing-rich time |
| 2 | `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep` | `platform_run_templates/run_m86_t38_fix_utc_bjtime.sh` | M84 stack; append Beijing-rich time to current UTC time |
| 3 | `M87-m84-t38fix-tempweight25-gap0-seed42-4ep` | `platform_run_templates/run_m87_t38_fix_tempweight25.sh` | M84 stack + M41-style soft temporal reweight |
| 4 | `M88-m84-t38fix-6ep-gap0-seed42` | `platform_run_templates/run_m88_t38_fix_6ep.sh` | M84 stack longer training |

Started after M85/M86回流:

| Job | Run script | Purpose |
|---|---|---|
| `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep` | `platform_run_templates/run_m89_t38_fix_bjtime_v2.sh` | Beijing time v1 + distribution time v2; no sample reweight |
| `M90-m86-utc-bjtime-6ep-gap0-seed42` | `platform_run_templates/run_m90_t38_fix_utc_bjtime_6ep.sh` | M86 exact config extended to 6 epochs |

EDA / audit completed:

| Job | Platform entry | Purpose |
|---|---|---|
| `EDA71-time-window-backtest-audit` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Scan train same-clock windows for the public BJ 09:49~14:49 window and quantify RG time-split overlap |

Clean-window backtest completed:

| Job | Platform entry | Purpose |
|---|---|---|
| `M91-cleanwin0303-m84-t38fix-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | M84 stack, train rows `< 2026-03-03 09:49:51 BJ`, valid `2026-03-03 09:49:51~14:49:07 BJ`; best 0.769513 |
| `M92-cleanwin0303-m86-utc-bjtime-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | M86 UTC+Beijing time stack under the same clean pseudo-public window; best 0.766748; loses to M91 by -0.002765 |

Next breakthrough batch submitted to queue:

| Job | Platform entry | Purpose |
|---|---|---|
| `M93-cleanwin0303-m84-mnw6h-w03-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Clean-window delayed-feedback diagnostic; compare vs M91 |
| `M94-m84-mnw6h-w03-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Full-data delayed-feedback public candidate if M93 passes |
| `M95-m84-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Full-data sequence length / timestamp-span feature candidate |
| `M96-m84-mnw6h-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Interaction candidate: delayed feedback + seq stats |

Clean-window seqstats diagnostics submitted to queue:

| Job | Platform entry | Purpose |
|---|---|---|
| `M97-cleanwin0303-m84-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Clean-window seqstats diagnostic; compare vs M91 before evaluating M95 |
| `M98-cleanwin0303-m84-mnw6h-seqstats-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | Clean-window interaction diagnostic; compare vs M91/M93/M97 before evaluating M96 |

Clean-window time-feature closure diagnostics submitted to queue:

| Job | Platform entry | Purpose |
|---|---|---|
| `M99-cleanwin0303-m84-bjtime-v2-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | M89-style Beijing v1+v2 clean-window diagnostic; decide whether M89 deserves high-risk info quota |
| `M100-cleanwin0303-m84-bjtime-gap0-seed42-4ep` | uploaded as `run.sh`; local `run_*.sh` template intentionally untracked | M85-style Beijing-only clean-window diagnostic; closes the XHS time-feature path more cleanly than M92 alone |

### Last Completed Jobs (按时间倒序)

EDA / diagnostics:

| Job | Result | Decision |
|---|---|---|
| **M93-M100** | M93 clean 0.762562; M97 clean 0.760092; M98 clean 0.756595; M99 clean 0.764915; M100 clean 0.763198; all below M91 0.769513 | Overnight gates failed; do not evaluate M94/M95/M96/M89/M85 by default |
| **M91/M92** | M91 clean M84 **0.769513**; M92 clean UTC+Beijing **0.766748**; delta **-0.002765** | Gate 1 failed; hold M89/M90 unless deliberately spending high-risk information quota |
| **EDA71** | exact public-window train rows **0**; train max to public start **1s**; RG time split overlap **341,086s (~3.95d)** | `recommend_clean_window_backtest=true`; `recommend_more_absolute_time_dense=false` |

5/19 breakthrough batch:

| Job | Config | Best dev / clean AUC | Best epoch | LB | Notes |
|---|---|---:|---:|---|---|
| **M93** | clean M84 + mature-negative weight 6h/w0.3 | **0.762562** | E2 | no eval | Delayed-feedback gate failed vs M91 by -0.006951 |
| **M94** | full M84 + mature-negative weight 6h/w0.3 | **0.857370** | E2 visible | no eval | Copied log misses final E4, but visible curve is far below M84 dev |
| **M95** | full M84 + seqstats | **0.871160** | E4 | gated/hold | Normal dev positive, but M97 clean diagnostic failed hard |
| **M96** | full M84 + mature-negative + seqstats | **0.859312** | E4 | no eval | Interaction collapsed on normal dev and clean gate |
| **M97** | clean M84 + seqstats | **0.760092** | E4 | diagnostic | Seqstats transfer gate failed vs M91 by -0.009421 |
| **M98** | clean M84 + mature-negative + seqstats | **0.756595** | E2 | diagnostic | Worse than M93 and M97; interaction rejected |
| **M99** | clean M89-style Beijing v1+v2 | **0.764915** | E3 | diagnostic | Time closure failed vs M91 by -0.004598 |
| **M100** | clean M85-style Beijing-only | **0.763198** | E4 | diagnostic | Time closure failed vs M91 by -0.006315 |

5/18 XHS time-feature batch:

| Job | Config | Best dev | Best epoch | LB | Notes |
|---|---|---:|---:|---|---|
| **M89** | M84 + Beijing time v1+v2, no UTC time | **0.871864** | E4 | **0.847650** ❌ | Normal-dev leader, but public -0.001204 vs M84; clean-window warning validated |
| **M86** | M84 + UTC time + Beijing time | **0.871415** | E4 | **0.847106** ❌ | Dev leader but public -0.001748 vs M84; time-stack did not transfer |
| **M90** | M86 exact config extended to 6 epochs | 0.871206 | E4 | 不优先 | 6ep does not beat M86; E5/E6 lower; same public-negative mechanism |
| **M85** | M84 + Beijing time only | **0.870887** | E4 | 待评 | Slightly above M75 dev; mechanism-confirm candidate |
| **M87** | M84 + temporal_weight_alpha=2.5 | 0.869053 | E4 | 不优先 | Temporal reweight does not stack with corrected T38/path B |
| **M88** | M84 extended to 6 epochs | 0.870910 | E6 | **0.846209** ❌ | Reject longer training; dev +0.000420 vs M84 but public -0.002645 |

5/18 morning batch:

| Job | Config | Best dev | Best epoch | LB | Notes |
|---|---|---:|---:|---|---|
| **M75(E6)** | M75 family extended to 6 epochs | **0.870651** | E6 | 待评 | E4/E5/E6 几乎持平；仍是 T38 `n_samples=0` default path |
| **M84** | path B + corrected per-domain T38 | 0.870490 | E4 | **0.848854** · previous SOTA | 有 schema timestamp resolve log；true T38 小正向 |
| **M76** | path B + DCN cross | 0.870247 | E4 | 待评 | DCN stack dev 未超过 M75 |
| **M82** | path B + alpha2.5 + hour 9-14 x5 | 0.869709 | E4 | **0.846197** ❌ | hour-aware reweight public 负向 |
| **M83** | target-day/hour x25 global weighting | 0.867486 | E4 | 不优先 | 明确负向；reject |

5/17 早起 4 jobs:

5/17 早起 4 jobs 全部完成:

| Job | Config | dev (E4 best) | E3 dev | LB | Notes |
|---|---|---|---|---|---|
| **M68** | path B + Q_init + DIN bypass (3 stack) | 0.870376 | 0.869659 | **0.846468** ❌ | Stack net −0.001694 LB · Q_init in path B 失败 |
| **M65** | path A + per-domain T38 v2 + DIN bypass | 0.868973 | 0.868623 | 待评 | T38 v2 fix 平台首验 ✅ · 弱真信号 |
| **M68 v2** | M68 same config rerun | 0.870355 | 0.869836 | 待评 | M68 dev rerun band ±0.0001 内 |
| **M71** | path B + DIN × s11 (cross-seed) | 0.870344 | 0.869952 | **0.846641** ⚠️ | path B cross-seed 第 2 数据点 · 下沿 |

### Training Summary 关键 ckpt (`keep_topk_ckpt=4` · top4 全保留)

5/17 早 4 jobs 各 4 ckpt (E1/E2/E3/E4) 都保留 · ckpt selection 战略 ready (但首测未做)。

5/16 晚 4 jobs (M51/M52/M53/M54): training_summary 没在 docs 详细记录 · 但 ckpt 5/18 04:22 24h 过期红线 · 需 5/18 早 quota refresh 后立即 evaluate (4 quota 全用 + 部分 ckpt selection)。

---

## Evaluation Quota (4/day · 北京 12:00 = CEST 06:00 重置 · 不累积)

> 来源: team-memory line 9

### 5/19 (today · refreshed @ CEST 06:00 / 北京 12:00)

| Used | Remaining | Used by |
|---|---|---|
| **4 used** | **0 uncommitted** | M89 LB 0.847650 + M104 LB 0.847560 + M106 E4 LB 0.849364 + M106 E3 LB 0.848760 |

User-confirmed current state:

- Evaluation quota: **0 uncommitted remaining** after M89, M104, M106 E4, and completed M106 E3 evaluation.
- Training pool: M101/M102/M103/M104/M105/M106/M107/M108/M109/M111/M113/M114/M115/M116/M117/M118/M119/M120/M121/M122/M123/M124/M125/M126, EDA72, and EDA73 completed; M110/M112 have only E2 partial logs recorded and remain logs-only by default; M127-M128 are submitted/queued by latest user update.

### 5/18 (closed)

| Used | Remaining | Used by |
|---|---|---|
| **4** | **0** | M84 LB 0.848854 + M82 LB 0.846197 + M86 LB 0.847106 + M88 LB 0.846209 |

**Future allocation policy (user decision)**:

- Spain morning: use **2-3** eval quotas.
- Spain evening: use **1-2** eval quotas.
- Keep the last quota flexible when training pool has unresolved high-upside candidates.

**User decision (2026-05-18 morning)**:

- Evaluate **M84 E4** now.
- Evaluate **M82 E4** now.
- Reserve the other **2 quotas** for evening decision after more information.
- Expected remaining after these two submissions: **2/4**.

**Evaluation failure note (2026-05-18 14:31 Beijing log)**:

- **M82 E4 evaluation failed before inference** with `dcn_cross_user_fids contains unknown fids [0, 5]`.
- Root cause: `infer.py` parsed residual DCN fields even when `enable_dcn_cross=False`; M82 should ignore those fields.
- Fix applied in `src/infer.py`: skip DCN fid parsing unless DCN is enabled; for true DCN ckpts, accept both raw-fid strings and already-mapped index lists.
- Retry M82 (and M84 if not completed) with patched custom `infer.py`. Failure should not consume quota under the known platform rule, but confirm UI remaining count before evening decisions.

**Evaluation results (2026-05-18 later update)**:

| Candidate | Eval job | Public AUC | Time | Decision |
|---|---|---:|---:|---|
| **M84 E4** | `M84-m22-uesplit-perdomain-t38fix-gap0-seed42-4ep_eval_1779085777310` | **0.848854** | 695.58s | Tiny new SOTA; true T38 small positive |
| **M82 E4** | `M82-m22-uesplit-alpha25-hour9to14x5-gap0-seed42-4ep_eval_1779085835759_copy_1779086114784` | **0.846197** | 684.43s | Reject alpha+hour reweight |
| **M86 E4** | `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep_eval_1779111177317` | **0.847106** | 472.2s | Reject naive UTC+Beijing time stacking for public score |
| **M88 E6** | `M88-m84-t38fix-6ep-gap0-seed42_eval_1779130091934_copy_1779131732501_copy_1779131958877_copy_1779132760150` | **0.846209** | 722.84s | Reject M84 longer training / E6 checkpoint |
| **M89 E4** | `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep_eval_1779176001150` | **0.847650** | 746.62s | Reject Beijing v1+v2 time stack |
| **M104 E4** | `M104-m84-hcvr-ts24h-gap0-seed42-4ep_eval_1779199262757` | **0.847560** | 644.81s | Reject timestamp-cutoff history-CVR |
| **M106 E4** | `M106-m84-t38fix-seed5-auxm91-gap0-4ep_eval_1779219154210` | **0.849364** | 858.21s | New SOTA; M84-family seed upper-tail transferred |
| **M106 E3** | `M106-m84-t38fix-seed5-auxm91-gap0-4ep-E3_eval_1779222359418` | **0.848760** | 744.33s | Below E4 by -0.000604; do not chase M106 checkpoint mining |

**Current recommendation after M93-M100 gate readout**:

| Priority | Candidate | Why |
|---|---|---|
| score-max | **No M93-M100 candidate** | All clean-window gates failed; normal-dev-only gains are not trustworthy under EDA71 |
| possible fallback | **M75(E6)** | Same closest path-B family as SOTA; still low-upside and not a new mechanism |
| high-risk info only | **M95 E4 only if explicitly buying calibration** | M89 calibration failed publicly; M95 clean-window failure is even larger, so default is no eval |
| hold | **M94/M96/M85/M90** | Gate or normal-dev evidence is weak/negative |
| skip | **M88 E6 / longer-training M84** | Public 0.846209 despite dev 0.870910; longer training overfits public |
| skip | **M87 E4** | 0.869053; temporal_weight stack is below M84 by -0.001438 |

**Current action recommendation after 5/19 resource refresh**:

| Resource | Recommendation | Rationale |
|---|---|---|
| training pool | **Wait for M127 if imminent; otherwise M119 is the best returned eval candidate** | M124 returned weaker than M119; M116 is safer but lower-upside; M126 is vetoed by M125 |
| 5/20 eval quota | **Confirm remaining quota before spending** | If forced to spend now: M119 E4 for upside/freshness calibration, M116 E4 for safer same-family seed calibration |
| skip by default | M102 | Weaker full-dev than M104; do not spend scarce quota after M104 miss |

**Reference 4-quota recommendation (post morning-batch)**:

| Priority | Candidate | Why | Mode |
|---|---|---|---|
| 1 | **M75(E6) E6** | Closest same-family score candidate; best dev 0.870651 | score-max |
| 2 | **M84 E4** | Corrected true-T38 path; highest information among new jobs | mechanism + possible score |
| 3 | **M82 E4** | Direct test daypart/reweight hypothesis; dev negative but public-shift question worth one quota | distribution-shift calibration |
| 4 | **M51** if publish-ready; otherwise **M76 E4** | M51 is softer temporal reweight with prior M41 public-positive hint; M76 only if limiting to new-batch jobs | fallback / closure |

Do **not** evaluate M83 with scarce quota.

### 5/17 (closed)

| Used | Remaining | Used by |
|---|---|---|
| **3** | **1** | M55 LB 0.844343 + M71 LB 0.846641 + M68 LB 0.846468 |

**M50 LB · 今晚剩 1 quota** (待 M50 重跑完成 ~22:30 后立即 evaluate · 用户已决)。

**🟢 失败 evaluation 不消耗 quota** (5/17 用户告知铁律): M68 第 1 次 evaluation state_dict mismatch 失败 (q_init_item key bug · commit `cfd15f7` 修复) · 重提成功。

### 5/18 (historical pre-refresh plan) · 4 quota refresh @ CEST 06:00

**Latest note (2026-05-18 morning)**: superseded by current 4-quota recommendation above.

**5/18 早 4 quota 候选** (优先级修订 · 5/17 18:35):

| 优先级 | 候选 | 触发条件 | 期望 LB |
|---|---|---|---|
| ⭐⭐⭐ | **M53 LB** (5/16 晚 path B + NLIR · ckpt 5/18 04:22 过期 风险!) | publish-ready · 必须明早评 | 0.846~0.848 |
| ⭐⭐⭐ | **M51 LB** (5/16 晚 path B + temporal_weight · ckpt 5/18 04:22 过期 风险!) | publish-ready · 必须明早评 | 0.846~0.848 |
| ⭐⭐⭐ | **M75 LB** (今晚 path B + per-domain T38 v2 stack) | training_summary 出 + dev verify | 0.847~0.849 |
| ⭐⭐ | **M65 LB** (path A + T38 v2) | training_summary 已出 dev 0.868973 | 0.846~0.847 |
| ⭐⭐ | **M68 v2 LB** (M68 rerun · LB rerun band confirm) | dev 已出 | 0.846~0.847 |
| ⭐⭐ | **M68 E3 ckpt LB** (ckpt selection 战略首测 · E3 dev 0.869659) | E3 ckpt verify | 0.846~0.848 |
| ⭐ | **M73/M74 LB** (path B cross-seed s2/s5) | dev 出 | 0.845~0.848 |
| ⭐ | **M76 LB** (path B + DCN) | training 完成 ~03:30 5/18 | 0.847~0.848 |

**5/18 早 4 quota 推荐**: M53 + M51 (ckpt 24h 过期) + M75 (T38 v2 + UE 真测) + M68 E3 ckpt (ckpt selection 首测) **或** M76 (T34 DCN 首测)。

### 历史 quota 使用

| Day | Used | Remaining | Used by |
|---|---|---|---|
| 5/17 (today) | 3 | 1 | M55 0.844343 + M71 0.846641 + M68 0.846468 |
| 5/16 (closed) | 4 | 0 | M40 LB 0.846931 + M41 LB 0.847846 + M23 retry LB 0.836631 + **M49 LB 0.848162 ⭐ SOTA** |
| 5/15 (closed) | 4 | 0 | M28 LB 0.846514 + M29 LB 0.845312 + M22-rerun LB 0.847324 + M33 LB 0.844710 |
| 5/14 (closed) | 1 | (overflow ❌) | M22 LB 0.847592 (前 SOTA · 现 #2) |

**Next reset**: CEST 06:00 5/18 = 北京 12:00 5/18

---

## Publish-Ready Ckpt 状态 (5/17 CEST 18:35 audit)

| Job | Dev | Publish | 24h 过期 (CEST) | LB |
|---|---|---|---|---|
| **M22** (vocabtier × s42) | 0.868576 | ✅ historical | safe | **0.847592** ⭐ #2 |
| **M22-rerun** | 0.868582 | ✅ | safe | 0.847324 |
| **M49** ⭐⭐⭐ (UE split path B) | 0.870741 | ✅ | safe | **0.848162** · previous SOTA |
| **M75** ⭐⭐⭐ (path B family · nominal per-domain) | 0.870856 | ✅ evaluated | safe | **0.848788** · previous SOTA · T38 not validated (`n_samples=0`) |
| **M75(E6)** (same family extended) | 0.870651 | TBD | 5/19 ~11:22 | 待评 · score-max candidate only if quota allows |
| **M84** (corrected T38 + path B) | 0.870490 | ✅ evaluated | safe | **0.848854** · previous SOTA · true T38 small positive |
| **M106** (M84 seed5 + aux m91 logging) | 0.871175 | ✅ evaluated | safe | **0.849364** ⭐ SOTA · same-family seed upper-tail transferred |
| **M86** (M84 + UTC time + Beijing time) | 0.871415 | ✅ evaluated | safe | **0.847106** ❌ · dev-public mismatch |
| **M89** (M84 + Beijing time v1+v2) | 0.871864 | ✅ evaluated | safe | **0.847650** ❌ · normal-dev leader did not transfer |
| **M88** (M84 6ep) | 0.870910 | ✅ evaluated | safe | **0.846209** ❌ · reject longer training / E6 |
| **M85** (M84 + Beijing time only) | 0.870887 | TBD | TBD | 待评低优先 · mechanism only after M86 miss |
| **M76** (path B + DCN) | 0.870247 | TBD | 5/19 ~07:37 | 待评低优先 |
| **M82** (path B + alpha/hour reweight) | 0.869709 | ✅ evaluated | safe | **0.846197** ❌ |
| **M83** (exact day/hour global weighting) | 0.867486 | TBD | 5/19 ~10:22 | 不评 |
| **M51** (path B + temporal_weight) | TBD | ✅ **published 5/17 19:51** | safe (publish 锁) | 待评 |
| **M52** (path B + per-domain · M52' silent skip) | TBD | ⚠️ TBD | 5/18 04:22 ⚠️ | 不评 (silent skip) |
| **M53** (path B + NLIR) | TBD | ✅ **published 5/17 19:51** | safe (publish 锁) | 待评 |
| **M54** (per-domain alone · M54 silent skip) | TBD | ⚠️ TBD | 5/18 04:22 ⚠️ | 不评 (silent skip) |
| **M50 (5/16 训)** | 0.861974 | ❌ **已过期** (5/17 15:53) | gone | LB 永远评不到 |
| **M50 重跑** (今晚 ~22:30 完) | TBD | TBD | 5/18 22:30 OK | 今晚剩 1 quota 评 |
| **M68 / M65 / M68 v2 / M71** (今早) | (见 above) | 待 publish | 5/18 19:00~21:30 | 5/18+ 评 |
| **M30** (Trick 6) | 0.868438 | ✅ | safe | 待评 (LB 决 Q_init 路径) |
| **M37** (NLIR) | 0.868311 | ✅ | safe | 待评 |
| **M38 v2** (Multi-Emb K=2) | 0.868403 | ✅ | safe | 待评 |
| **M39 v2** (Global Token) | 0.867963 | ✅ | safe | 待评 |
| **M27 v2** (T31+user_15) | 0.868166 | ✅ | safe | 待评 |
| **M17** (seq_d=768) | 0.867794 | ✅ | safe | 待评 |
| ~~M34~~ / ~~M45~~ / ~~M48~~ | — | ❌ 已丢失 / dev 灾难 | gone | 不评 |

---

## Strategic Status (5/17 CEST 18:35 修订 · 5/20 ddl 战略约束下)

**LB SOTA**: **M106 (M84 seed5 + corrected true T38) = 0.849364** ⭐
**前 SOTA**: M84 (path B + corrected true T38) = 0.848854
**距 0.852 (用户硬 deadline)**: 还差 +0.002636
**距 0.85 (Top 20)**: 还差 +0.000636

**用户硬 deadline (5/17 CEST 16:14 立)**: **0.852 by 5/20** (推前 5/22 → 5/20)

**5/17~5/20 战略 (team-memory line 195 新铁律)**:
- ✅ **cross-seed 主线开放** (5/22 lock 已 ARCHIVED)
- ✅ **path B cross-seed 优先** (M22 已 5 数据点 · path B 仅 2 + 今晚 +2 = 4 接近饱和)
- ✅ **stack accumulation** (T38 v2 + UE / DCN + UE 今晚)
- ✅ **ckpt selection 战略激活** (M68 E3 ckpt 首测 待 5/18 早)

**5/20 前到 0.852 概率**: ~20~30% (M68 stack 失败后下调 · 仍可行)

**5/17 全天 LB 数据点 (今天新)**:
- M55 LB **0.844343** (Bilinear gating · 闭环失败)
- M71 LB **0.846641** (path B × s11 cross-seed · 第 2 数据点)
- M68 LB **0.846468** (path B + Q_init stack · 净损害 −0.001694)
- → **path B cross-seed 跨度小** (s42 0.848 / s11 0.846 · 0.0015) · stack 加性失败 (Q_init 在 path B 上失败)

**今晚 4 jobs 设计逻辑** (EXP-068 落档):
- 2 cross-seed (M73 s2 / M74 s5) → path B 路径 4 数据点 · 触饱和阈值
- 1 stack (M75 T38 v2 + UE) → T38 v2 fix 真测 + path B
- 1 救活 (M50 重跑) → 防 ckpt 过期 + LB 评估
- M76 (DCN + UE) 接力 M50 重跑

**Post-update caveat**:
- M75 LB 已回流为 **0.848788 SOTA**。
- 但 M75 训练日志里的 T38 fit `n_samples=0`，所以它不是 T38 真测成功；后续 T38 仍需依赖 corrected nonzero-sample run (如 M84b) 判断。
- 5/18 morning batch 未产生 dev 超越 M75-4ep 的新模型：M75(E6) 0.870651 / M84 0.870490 / M76 0.870247 / M82 0.869709 / M83 0.867486。
- 单 quota 下：若冲分优先，M75(E6) 是最接近 SOTA family 的候选；若机制学习优先，M84 是 true-T38 高信息候选。M83 不评。
- M84 public 0.848854 证明 true T38 小幅 public-positive，但增益只有 **+0.000066 vs M75**，不是足以单独冲 0.852 的大机制。
- M82 public 0.846197 证明 alpha2.5 + hour9~14 x5 reweight 负向；暂停强 daypart reweight variants。
- M86 public 0.847106 证明 UTC+Beijing dense time stacking 出现明显 dev-public mismatch；虽然 dev 比 M84 高 **+0.000925**，public 反而低 **-0.001748**。时间方向仍可研究，但不能再按 dev 绝对值直接消耗 quota。
- M88 public **0.846209** rejects longer M84 training / E6 checkpoint selection. It had dev **+0.000420 vs M84**, but public **-0.002645 vs M84**. Do not spend quota on M84-family longer-training variants by dev alone.
- M106 public **0.849364** becomes new SOTA, **+0.000510 vs M84**. Its normal-dev gain over M84 was **+0.000685**, so this same-family seed upper-tail transferred meaningfully, unlike longer training or dense time features.
- M90 (`M86 6ep`) completed with best dev **0.871206** at E4; E5 **0.870696** and E6 **0.870880** did not improve. Longer training does not rescue the M86 UTC+Beijing stack.
- M91/M92 clean-window Gate 1 failed hard: M91 **0.769513** vs M92 **0.766748**, delta **-0.002765**. This reproduces the M86 public miss in a pseudo-public window and rejects UTC+Beijing dense time stacking as an eval-priority path.
- M89 normal dev **0.871864** was very strong (**+0.001374 vs M84**, **+0.000449 vs M86**), but public returned **0.847650**, **-0.001204 vs M84**. This validates M99 clean-window's warning for the Beijing v1+v2 branch and closes time-feature score-max allocation.
- 5/19 evaluation quota has **0 uncommitted remaining** after M89, M104, M106 E4, and completed M106 E3 eval.
- Training pool current state: M113-M126 completed; M127-M128 submitted/queued/running by last known state. M110/M112 have only partial E2 logs recorded and remain logs-only by default because M109/M111 clean companions failed.
- Current history-CVR readout: available-time (M101/M102) and timestamp-cutoff 24h (M103/M104) produced normal-dev gains but failed public validation through M104. Downgrade the branch; M102 is not default eval-worthy.
- M107/M108 tested M104 on EDA73's best same-clock `clk0302` oracle. The pair strongly favored history-CVR (**+0.054935**), but M104 public missed; `clk0302` is therefore not a reliable positive oracle for this mechanism.
- Clean-window gate should be treated as a guardrail, not a perfect oracle: it reproduced M86's public miss and correctly warned against M89. For large full-data dev gains with strong non-time mechanism rationale, one calibration quota can still be justified, but time-feature normal-dev gains are now strongly suspect.
- Code now supports `--aux_valid_windows "m91:1772502591:1772520547"` for future jobs. Aux metrics are diagnostic only; they do not select checkpoints. Full-data aux windows may be leaky, so separate fixed-window diagnostics remain the stronger gate when quota/slots allow.
- M93-M100 have completed from copied logs. Every clean-window gate failed vs M91, so there is **no default eval-worthy candidate** from this batch.
- M95 normal dev **0.871160** is tempting, but M97 clean **0.760092** is **-0.009421 vs M91**. This is the clearest warning that seqstats learned the contaminated split, not the future public window.
- M99/M100 close the XHS time-feature path for quota allocation: even Beijing v1+v2 without UTC fails the pseudo-public clean window.
- EDA71 证明 public 窗口是 train max 后 **1 秒**的未来窗口，且当前 RG time split valid 与 train 时间范围重叠 **~3.95 天**；因此时间特征下一步必须走 clean-window backtest (M91/M92)，不能继续堆 dense absolute time 后直接看 normal dev。
- M93-M96 实践协议已准备: 以 M91 clean-window 为尺子，优先检验 delayed feedback 和 sequence activity，而不是继续堆日历特征。Eval gate: M93-M91 >= +0.0004 才考虑 M94；M95 dev 至少 M84 +0.0003 且 logloss 不坏才考虑。
- 重新检查旧证据后，**不要启动 M91 hour_only**：M40 已测 hour_only，LB **0.846931**，比当前 SOTA 明显低；EDA-056 也指出 DOW/weekend 不是纯噪声。时间方向下一步应先做 EDA-71 / clean window backtest。

**关键 deprecated 方向 (不再投资源)**:
- ❌ Q_init in path B (M68 LB 净损害 −0.001694 · stack 失败)
- ❌ Bilinear gating (M55 LB −0.004 · user-side gating 路径死)
- ❌ daypart filter (M45/M48 dev 灾难 · M50 双 stack 净损害)
- ❌ hour_only replacing time_of_day (M40 LB 0.846931; DOW/weekend removal semi-rejected)
- ❌ naive UTC+Beijing time-feature stacking by dev score alone (M86 dev leader but LB -0.001748 vs M84)
- ❌ Beijing v1+v2 time-feature stack by dev score alone (M89 normal-dev leader but LB 0.847650, -0.001204 vs M84)
- ❌ current clean-window delayed-feedback 6h/w0.3 and seqstats variants (M93/M97/M98 all below M91 by large margins)
- ❌ DCN cross alone (M23 LB −0.011 · path A DCN 死透 · 但 path B + DCN 是 M76 重新探索)
- ❌ 5-cluster ns_groups / idmask / NLIR alone / FaFE / per-token FFN
- ⚠️ Multi-Emb K=2 / Global Token (M38v2/M39v2 dev 弱负 · 待 LB 闭环 · 不优先 stack)

---

## 历史 Lessons (5+ 次资源混乱事故 · 5/9 ~ 5/17)

1. 5/9 22:10: agent 凭印象 "evaluation 截止 CEST 14:00" → 用户纠正
2. 5/10 08:40: num_epochs=3 凭印象 → 改 default 4
3. 5/10 16:21: agent 把 training slot 当 evaluation quota → line 175 铁律
4. 5/14 12:39: agent 又把 slot 当 quota · "等凌晨" 伪时序 → 用户反问
5. 5/16 08:35: agent 凭印象 "2 free slot" 实际 4 free → 用户反问 "彻底解决问题"
6. **5/17 16:42**: agent 凭印象 "5/16 晚 4 jobs (M60/M55/M54'/M52')" 实际 5/17 早起 4 jobs (M68/M65/M68 v2/M71) · 用户贴 log 后才修正 (4 jobs 全带 DIN bypass · 5/16 晚 EXP-065 4 jobs 实际不在今天 log 里)

**Root cause** (累计):
- 静态规则铁律已立 (4/day quota / 4 slots)
- **动态状态 (当前 free slot 数 / 剩余 quota / 哪些 jobs 在跑)** 由本文件单一 source of truth 维护
- agent 任何时候陈述资源状态前**必须 grep 本文件** · 不凭印象

---

## Update Log

- **2026-05-20 CEST**: 用户回报 final three 5/20 evals completed. `M123-m106-vr02-mtl-seed5-auxm91-gap0-4ep_eval_1779309183122` public **0.851011** (709.88s), new SOTA; `M124-m106-vr01-mtl-seed5-auxm91-gap0-4ep_eval_1779309198758` public **0.850718** (626.69s), near M119; `M116-m84-t38fix-seed17-auxm91-gap0-4ep_eval_1779309435749` public **0.847886** (532.77s), rejected. Current 5/20 quota is **4/4 used** until the 06:00 CEST refresh.
- **2026-05-20 CEST**: Eval-speed hotfix: changed `infer.py` test EDA default from on to off. Use `EDA_BLOB=1` explicitly only when a diagnostic test-set EDA blob is needed. This should reduce score-max eval wall time without affecting predictions.
- **2026-05-20 CEST**: 用户确认 **M129-M132 have all been submitted / queued** on platform. Treat them as platform-scheduled until completion logs return.
- **2026-05-20 CEST**: 用户回报 **M119 evaluation completed**: `M119-m106-vr02-seed7-auxm91-gap0-4ep_eval_1779306323511`, public **0.850754** (703.23s), new LB SOTA. This is **+0.001390 vs M106 E4** and validates the freshness/reduced-valid-ratio path as public-positive. Current 5/20 quota is **1 used / 3 remaining** before the 06:00 CEST refresh.
- **2026-05-20 CEST**: Implemented and prepared **M129-M132** after the UE 61/87 external intel. Code changes: `--ue_split_fids` supports multi-fid UE tokenisation while preserving legacy `--ue_split_fid`; `--enable_ue_item_interaction` appends explicit `[UE, item_repr, UE*item_repr]` bypass to the classifier. Prepared helpers under `platform_run_templates/`; full record: [EXP-086](../40-experiments/EXP-086-ue6187-xitem-m129-m132-2026-05-20.md).
- **2026-05-20 CEST**: 用户贴出新的社区情报截图：external row **auc 0.852746 / 152.3s**, plus user-reported LB Top1 **0.85344**. Community comments point to **feat 61/87** and explicit **UE x item** interaction. Recorded as a high-priority mechanism hypothesis because current accepted UE split path only tokenises fid61, leaves fid87 in dense_bypass, and lacks dense UE x target-item interaction.
- **2026-05-20 CEST**: 用户回报 **M119 has been submitted for evaluation**. Treat M119 E4 (`global_step65920.layer=2.head=4.hidden=64.auc=0.871196.best_model`) as pending active eval. If accepted, current 5/20 quota becomes **1 pending/used, 3 remaining**. M106 E4 remains recorded LB SOTA **0.849364** until M119 eval result returns.
- **2026-05-20 CEST**: 用户确认 today evaluation quota has **not been used yet**; treat current eval quota as **0 used / 4 remaining** unless the platform UI later shows otherwise. Since M127/M128 are still 3-4 hours away and M119 is already the strongest returned upside candidate, the low-regret action is to spend exactly one quota on **M119 E4**, then reserve the remaining quota for M127/M128 readout and one safer fallback if needed.
- **2026-05-20 CEST / 2026-05-21 BJT**: 用户回报 **M124 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M124-m106-vr01-mtl-seed5-auxm91-gap0-4ep`: `valid_ratio=0.01`, `multi_task_loss=True`, `action_num=2`, seed **5**, and M106 corrected-T38 backbone. Best 1% valid **0.868723** at E4 / step 66592, logloss **0.225034**; aux m91 rises to **0.854380** at E4. Interpretation: aux is interesting, but aggressive 1% valid main is weak and below M118's 1% non-MTL main; M124 does not overtake M119 and should not be evaluated by default.
- **2026-05-20 CEST**: 用户回报 **M119/M126 completed**, logs copied to `~/Downloads/train_log.txt`. M119 (`M106 seed7`, `valid_ratio=0.02`) best 2% valid **0.871196** at E4 / step 65920, logloss **0.221458**, aux m91 **0.855080**; this is now the strongest returned upside / calibration candidate, while noting the split is non-comparable to standard 5% valid. M126 (`M106 seed5 + is_missing`) best standard valid **0.870945** at E4 / step 63764, logloss **0.221435**, aux m91 **0.852607**; training is sane, but M125 clean gate was red, so keep M126 logs-only by default.
- **2026-05-20 CEST**: 用户回报 **M123 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M123-m106-vr02-mtl-seed5-auxm91-gap0-4ep`: `valid_ratio=0.02`, `multi_task_loss=True`, `action_num=2`, seed **5**, and M106 corrected-T38 backbone. Best 2% valid **0.870584** at E4 / step 65920, logloss **0.221884**; aux m91 is **0.853945** at E4 and peaks at **0.854422** at E3. This was a high-upside information candidate before M119 returned, but M119 later surpassed it on both 2% valid and aux m91.
- **2026-05-20 CEST**: 用户回报 **M120/M125 completed**, logs copied to `~/Downloads/train_log.txt`. M120 (`M106 seed13`, `valid_ratio=0.02`) best **0.870606** at E4 / step 65920, logloss **0.221836**, aux m91 **0.852176** at E4 and peak **0.852588** at E3; below M116 and does not rescue freshness-only. M125 clean-window is-missing diagnostic best **0.765837** at E4 / step 10136, logloss **0.217792**; vs M91 clean baseline **0.769513**, delta **-0.003676**, red. Decision: do not evaluate M126 by default.
- **2026-05-20 CEST**: 用户回报 **M116/M117 completed**, logs copied to `~/Downloads/train_log.txt`. M116 (`M84 corrected-T38 seed17`, standard 5% valid) best main valid **0.870697** at E4 / step 63764, logloss **0.222083**, aux m91 **0.853761**. M117 (`M106 seed5`, `valid_ratio=0.02`) best **0.870034** at E4 / step 65920, logloss **0.222221**, aux m91 **0.852345**. Interpretation: M116 is now the best returned forced-choice information candidate, narrowly above M121 and with the strongest aux, but still below M106; M117 does not rescue the freshness-only path.
- **2026-05-20 CEST**: 用户回报 **M115 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M115-m84-t38fix-seed13-auxm91-gap0-4ep`: M84 corrected-T38 stack, seed **13**, standard `valid_ratio=0.05`, and aux m91 logging. Best main valid **0.870309** at E4 / step 63764, logloss **0.223152**; aux m91 rises to **0.853551** at E4. Main dev is still below M106 E4 **0.871175** and M121 E4 **0.870691**, so M115 is an aux-interest hold, not a default public eval candidate.
- **2026-05-20 CEST**: 用户回报 **M118 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M118-m106-vr01-seed5-auxm91-gap0-4ep`: M106 seed5 backbone, `valid_ratio=0.01`, train **2,079,013** rows / valid **20,943** rows, aux m91 window configured. Main valid improves through E4 to **0.869640** at `global_step66592...best_model`, but the copied log ends immediately after E4 checkpoint save, so E4 aux/final summary are unavailable. E1-E3 aux m91 peaks at only **0.849728**. Decision: hold M118; do not spend public quota by default.
- **2026-05-20 CEST**: 用户回报 **M125-M128 have all been submitted** after the forensic whitelist recommendation. Treat M125/M127 as clean diagnostics and M126/M128 as matching full candidates; keep promotion gate-controlled.
- **2026-05-20 CEST**: Prepared **M125-M128** forensic whitelist retry after the user raised implementation/validation-miss concerns. M125/M126 are T31 is-missing clean/full pair; M127/M128 are Multi-Emb K=2 clean/full pair. Q-init and UE-bilinear are explicitly not reopened because their fixed / intended evaluations were public-negative. Full record: [EXP-085](../40-experiments/EXP-085-forensic-whitelist-m125-m128-2026-05-20.md).
- **2026-05-20 CEST**: Prepared **M123/M124** as narrow `freshness + MTL` queue hedges after the user agreed to start a new mechanism before all M115-M120 results return. M123 uses M106 seed5 + `valid_ratio=0.02` + MTL; M124 uses M106 seed5 + `valid_ratio=0.01` + MTL. Upload helpers from `platform_run_templates/` as platform `run.sh`; full record: [EXP-084](../40-experiments/EXP-084-m106-freshness-mtl-m123-m124-2026-05-20.md).
- **2026-05-20 CEST**: 用户回报 **M113/M114/M121/M122 completed**, logs copied to `~/Downloads/train_log.txt`. Parsed results: M113 seed7 best **0.870344** at E4 / step 63764, logloss **0.221712**, aux m91 **0.853309**; M114 seed11 best **0.870128** at E3 / step 47823, E4 drops to **0.869530**; M121 MTL seed5 best **0.870691** at E4 / step 63764, logloss **0.221751**, aux m91 **0.852123** at E4; M122 MTL seed7 best **0.870176** at E3 / step 47823, E4 **0.870031** but aux m91 **0.853627**. All are below M106 E4 normal dev **0.871175**. Later M116 slightly surpassed M121; see the newer update.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **all ten M113-M122 jobs have been started or entered the platform queue**. User also decided to let **M110/M112** continue to completion despite the earlier stop recommendation; keep them as logs-only by default. **M106 E3 evaluation completed**: job `M106-m84-t38fix-seed5-auxm91-gap0-4ep-E3_eval_1779222359418`, public **0.848760** (744.33s), which is **-0.000604 vs M106 E4** and slightly below M84. Interpretation: M106 E4 remains SOTA; checkpoint-selection mining within M106 is not validated.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M110/M112 still running** with partial checkpoint tables. M110 E2 / step 31882 is **0.868925**; M112 E2 / step 31882 is **0.868436**. Since M109 and M111 clean-window companions are both red, and these partial full-data dev scores are below M106 E4 **0.871175**, recommendation is to **stop/cancel M110 and M112 if platform can reclaim slots**. If stopping is impossible or operationally not worth it, let them finish but do not evaluate/promote by default.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户确认 **M106 E3 checkpoint was still evaluable and was submitted for public evaluation**. It later returned public **0.848760** as recorded above. Prepared overnight queue templates **M113-M122** under `platform_run_templates/`: M113-M116 M106/M84-family seed upper-tail, M117-M120 reduced-valid-ratio freshness variants, M121-M122 multi-task click+conversion variants. Full record: [EXP-083](../40-experiments/EXP-083-m106-overnight-upper-tail-m113-m122-2026-05-19.md).
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M106 evaluation completed**: job `M106-m84-t38fix-seed5-auxm91-gap0-4ep_eval_1779219154210`, public **0.849364** (858.21s). This is new LB SOTA, **+0.000510 vs M84** and **+0.000576 vs M75**. Remaining gap to **0.852** is **+0.002636**. Interpretation: seed5 on the M84 corrected-T38 family is a real transferable upper-tail result; it supports limited cross-seed score-maxing but does not solve the missing mechanism-level jump. At that point quota was **1/4 remaining**; later M106 E3 committed the final quota.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M111 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed clean-window per-domain encoded-DIN: `din_interest_source=encoded`, `din_interest_merge=per_domain`, fixed valid `[1772502591,1772520547)`. Result E1 **0.750970**, E2 **0.767948**, E3 **0.756221**, E4 **0.762607**; best checkpoint `global_step5068.layer=2.head=4.hidden=64.auc=0.767948.best_model`, best logloss **0.220206**. Compared with M91 clean baseline **0.769513**, delta is **-0.001565**, red. Decision: per-domain encoded-DIN fails; do not evaluate M112 by default. Together with M109 red, close DIN 2.0 encoded-late-bypass in this form.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M109 completed**, logs copied to `~/Downloads/train_log.txt`, and supplied checkpoint table. Config confirmed clean-window compact encoded-DIN: `din_interest_source=encoded`, `din_interest_merge=compact`, fixed valid `[1772502591,1772520547)`. Visible log has E1 **0.750343**, E2 **0.764080**, E3 **0.767038** with logloss **0.218418**; user checkpoint table supplies E4 **0.765129** and confirms E3 best_model `global_step7602.layer=2.head=4.hidden=64.auc=0.767038.best_model`. Compared with M91 clean baseline **0.769513**, delta is **-0.002475**, red. Decision: compact encoded-DIN fails; do not evaluate M110 by default. M111 later returned red too, closing the per-domain branch in this encoded-late-bypass form.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M109-M112 have been restarted and are running** after the local compatibility hotfix. Treat the previous 4/5 return mismatch as a resolved launch issue unless the new logs repeat it. Later updates confirmed M109/M111 clean-window diagnostics both red, so M110/M112 should not be promoted by default.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M106 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M106-m84-t38fix-seed5-auxm91-gap0-4ep`: M84 corrected-T38 stack, seed **5**, and `aux_valid_windows="m91:1772502591:1772520547"`. Best normal dev **0.871175** at E4 / step 63764, logloss **0.222041**; confirmed checkpoint `global_step63764.layer=2.head=4.hidden=64.auc=0.871175.best_model`. Copied log stops after E4 aux evaluation starts, so E4 aux/final summary are unavailable; E1-E3 aux m91 peaks at **0.852261**, stronger than M105's aux peak **0.850865**. Decision: M106 is the cross-seed batch winner and an eval-worthy fallback, but still a small seed-level score-max candidate rather than a 0.852 mechanism signal.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M107 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M107-cleanwin0302-m84-t38fix-gap0-seed42-4ep`: fixed valid `1772416191:1772434147`, train `<1772416191`, no history-CVR. Best clk0302 baseline AUC **0.821260** at E1 / step 1688, logloss **0.433156**; E2 **0.819377**, E3 **0.733079**, E4 **0.752655**, stopped by `logloss_rebound_epoch_4`. Paired with M108 best **0.876195**, delta **+0.054935**, originally green. Since M104 public was **0.847560** and below M84, this proves `clk0302` would have falsely promoted history-CVR; keep history-CVR closed for score-max and treat this as validation-system calibration.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户贴出新启动 **M109-M112** repeatedly failing, sample log from M109 at platform time **2026-05-20 00:23:54**. Failure occurs after fixed-window dataset construction, before model training: `ValueError: not enough values to unpack (expected 5, got 4)` at `train.py:907`. Diagnosis: aux-validation interface mismatch / mixed source bundle, not DIN 2.0 algorithm failure. Local fix applied in `src/train.py` to accept both 4- and 5-value `get_pcvr_data` returns; retry should upload matching updated `train.py` and `dataset.py` plus the selected `run.sh`.
- **2026-05-19 CEST / 2026-05-20 BJT**: 用户回报 **M108 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `M108-cleanwin0302-m84-hcvr-ts24h-gap0-seed42-4ep`: fixed valid `1772416191:1772434147`, train `<1772416191`, `enable_history_cvr_features=True`, `history_cvr_time_mode=timestamp_cutoff`, `history_cvr_cutoff_sec=86400`. History-CVR cache used **164,429** source rows, `global_prior=0.028547`. Best clk0302 AUC **0.876195** at E1 / step 1688, logloss **0.386351**; E2 **0.820024**, E3 **0.780074**, stopped by `logloss_rebound_epoch_3`. Interpretation: strong but unstable second-window signal; compare against M107 before using diagnostically, and do not reopen M104 as score-max after public **0.847560**.
- **2026-05-19 evening**: 用户回报 **M105 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed M84 corrected-T38 stack with seed **2** and `aux_valid_windows="m91:1772502591:1772520547"`. Best normal dev **0.870798** at E4 / step 63764, logloss **0.221729**. This is **+0.000308 vs M84**, but **-0.000058 vs M75**; aux m91 was logged as **leaky** and should only be compared against M106. Decision: hold M105 by default; if later used as fallback, evaluate `global_step63764.layer=2.head=4.hidden=64.auc=0.870798.best_model`.
- **2026-05-19 evening**: 用户回报 **M104 evaluation completed**: job `M104-m84-hcvr-ts24h-gap0-seed42-4ep_eval_1779199262757`, public **0.847560** (644.81s). This is **-0.001294 vs M84** despite normal dev **0.871796**. The M103 clean-window yellow/negative signal was directionally correct. At that point quota became **2/4 remaining** after M89 + M104; later M106 used one more quota and became new SOTA. Downgrade history-CVR branch; do not evaluate M102 by default.
- **2026-05-19 evening**: 用户同意 deep-think recommendation. Prepared **M107/M108 `clk0302` second-oracle diagnostics**. M107 = `M107-cleanwin0302-m84-t38fix-gap0-seed42-4ep`, baseline M84 on `1772416191:1772434147`. M108 = `M108-cleanwin0302-m84-hcvr-ts24h-gap0-seed42-4ep`, M104 timestamp-cutoff history-CVR on the same window. Also recommended spending one current eval quota on **M104 E4 best_model** (`global_step63764...auc=0.871796.best_model`) while M105/M106 continue running.
- **2026-05-19 evening**: 用户回报 **M104 completed**, logs copied to `~/Downloads/train_log.txt`. Config confirmed `history_cvr_time_mode=timestamp_cutoff`, `history_cvr_cutoff_sec=86400`, M84 corrected-T38 stack. Best dev **0.871796** at E4 / step 63764, logloss **0.221065**. This is **+0.001306 vs M84**, **+0.000338 vs M102**, but paired M103 clean was **0.769341** (**-0.000172 vs M91**). Classification: **yellow-plus**. If evaluating one history-CVR candidate now, prefer **M104** over M102; if M105/M106 return soon, compare before spending quota.
- **2026-05-19 morning**: Implemented optional auxiliary validation windows. New CLI `--aux_valid_windows "name:start:end"` builds read-only timestamp-window validation loaders and logs `[aux_valid/name]` AUC/LogLoss per epoch after primary valid. It does not affect best_model, top-k, early stopping, logloss guard, or training updates. Syntax checks passed. See EXP-077.
- **2026-05-19 morning**: 用户回报 **M89 evaluation completed**: job `M89-m84-t38fix-bjtime-v2-gap0-seed42-4ep_eval_1779176001150`, public **0.847650** (746.62s). This is **-0.001204 vs M84** despite M89 normal dev **0.871864**. Since M99 clean-window diagnostic for the same Beijing v1+v2 time stack failed vs M91 by **-0.004598**, this validates clean-window's warning for the time-feature branch. 5/19 quota now **1 used / 3 remaining**. Close M89/M85/M90 score-max eval path.
- **2026-05-19 morning**: 用户回报 **M101-M104 均已提交**. At submission time, eval quota remained **4/4**. Treat training pool as occupied/queued by M101-M104. Added clean-window caveat: current single-window gate is useful but not a perfect public oracle; use tiered decisions for high-dev/high-rationale candidates.
- **2026-05-19 morning**: 用户确认 current resources: evaluation quota refreshed with **4 remaining**, and training pool is **empty** with no queued/running jobs. Updated state file before recommending next platform actions.
- **2026-05-19 morning**: 用户回报 **M93-M100 已跑完**，logs copied to `~/Downloads/train_log.txt` with partial log recovery gaps. Parsed final gate readout: M93 clean delayed-feedback best **0.762562** (E2), below M91 **0.769513** by **-0.006951**; M94 visible full-data best **0.857370** (E2) and copied log lacks final E4, but this does not affect the no-eval decision. M95 full seqstats normal dev **0.871160** (E4) is the only ordinary-dev bright spot, but M97 clean seqstats best **0.760092** fails vs M91 by **-0.009421**. M96 full interaction best **0.859312**, M98 clean interaction **0.756595**, both rejected. M99 clean Beijing v1+v2 **0.764915** and M100 clean Beijing-only **0.763198** both fail vs M91, closing the XHS time-feature path for default quota allocation. Conclusion: no M93-M100 candidate should receive scarce eval quota by default; consider M95/M89 only as deliberate high-risk LB calibration, not score-max.
- **2026-05-19 morning**: 用户确认 5/19 evaluation quota refreshed with **4 remaining**, and training pool is **empty** with no queued/running jobs. Prepared M101-M104 launch plan and local run templates: M101/M102 = M84 + `history_cvr_time_mode=available`; M103/M104 = M84 + timestamp-cutoff 24h control. M101/M103 use the M91 clean pseudo-public window and gate full-data partner evals.
- **2026-05-19 morning**: 用户回报 **M93 and M94 have reached epoch 3**, logs copied to `~/Downloads/train_log.txt`. Parsed M93 clean-window config: train `timestamp_max=1772502591`, valid `timestamp_min=1772502591`, `timestamp_max=1772520547`, `mature_negative_weighting=True` on train only. Confirmed M93 E1 **0.752438**, E2 **0.762562**. 用户随后补充前三个 epoch checkpoint 表，文字标为 M91，但数值/step/时间与当前 M93 job 对齐；暂按 M93 记录 E3 **0.760782**, best-so-far **0.762562**. Compared with M91 clean-window final best **0.769513**, M93 is **-0.006951**; it would need E4 to jump to at least **0.769913** to clear Gate 2. If the supplied table truly belongs to M91, then M93/M91 identical early scores are a red flag and require explicit mature-negative sample-weight auditing. Parsed M94 full-data config: normal RG split 1,990,981 train / 108,975 valid, mature-negative train on, valid off. M94 E1 **0.854777**, E2 **0.857370**, E3 **0.857065** with best at E2; far below M84 normal dev **0.870490**. Do **not** evaluate M94 unless a later complete M93 clean-window result unexpectedly clears the gate.
- **2026-05-18 later**: 用户回报 **M97/M98/M99/M100 均已 submitted**. They are queued behind the current running tasks and will auto-start when prior jobs finish. Update status from prepared to submitted/scheduled.
- **2026-05-18 later**: 用户回报 **M88 E6 eval completed**: job `M88-m84-t38fix-6ep-gap0-seed42_eval_1779130091934_copy_1779131732501_copy_1779131958877_copy_1779132760150`, public **0.846209** (722.84s). Eval log confirms intended checkpoint `global_step95646.layer=2.head=4.hidden=64.auc=0.870910.best_model` and config (`time_of_day_features=True`, `beijing_time_features=False`, `seq_stats_features=False`, `mature_negative_weighting=False`). Conclusion: reject M84 longer training / E6; dev +0.000420 vs M84 did not transfer and public is -0.002645 vs M84.
- **2026-05-18 later**: Prepared **M99/M100 clean-window time-feature closure diagnostics** for the overnight queue. M99 = `M99-cleanwin0303-m84-bjtime-v2-gap0-seed42-4ep`, using M89-style Beijing v1+v2 and no UTC on the M91 clean window. M100 = `M100-cleanwin0303-m84-bjtime-gap0-seed42-4ep`, using M85-style Beijing v1 only on the same clean window. Purpose: decide whether M89/M85 deserve any high-risk information quota after M92 failed.
- **2026-05-18 later**: Prepared **M97/M98 clean-window seqstats diagnostics** for the overnight queue. M97 = `M97-cleanwin0303-m84-seqstats-gap0-seed42-4ep`, using M95 seqstats on the same clean BJ 2026-03-03 09:49:51~14:49:07 window as M91. M98 = `M98-cleanwin0303-m84-mnw6h-seqstats-gap0-seed42-4ep`, using M96 interaction on the same clean window. Purpose: gate M95/M96 eval decisions without spending public quota.
- **2026-05-18 later**: 用户回报 **M93/M94/M95/M96 都还在训练中**，且 **M88 E6 已提交评估**。At submission time, treated 5/18 quota as **4/4 committed**. Training pool currently occupied by M93-M96 mechanism jobs; await logs/results before next action.
- **2026-05-18 later**: 用户回报 **M90 completed**, logs in `~/Downloads/train_log.txt`, and manually supplied checkpoint table for incomplete log copy. Config is M86 exact UTC+Beijing time stack extended to **6ep**. Best dev **0.871206** at E4/global_step63764; E5 **0.870696**, E6 **0.870880**. Conclusion: longer training does not beat M86 or rescue the public-negative time-stack mechanism; keep M90 on hold.
- **2026-05-18 later**: 用户回报 **M91 / M92 均已完成**，M92 日志不完整但用户补充 checkpoint 表。M91 clean-window M84 best **0.769513**; M92 clean-window M86 UTC+Beijing time stack best **0.766748**. Delta **M92-M91 = -0.002765**, so Gate 1 failed hard. Conclusion: M86 public miss is structurally reproduced by clean pseudo-public validation; do **not** evaluate M89/M90 based on normal dev alone.
- **2026-05-18 later**: 用户回报 **M89 completed**, logs in `~/Downloads/train_log.txt`. Config: `enable_time_of_day_features=False`, `enable_beijing_time_features=True`, `enable_beijing_time_v2_features=True`, corrected T38 nonzero samples. Dev curve E1 0.867292 / E2 0.869521 / E3 0.870616 / E4 **0.871864**, best logloss **0.221121**. Conclusion: strongest normal-dev time candidate, but still gated by clean-window evidence after M86 public miss.
- **2026-05-18 later**: 用户补充平台训练池有 **queue mechanism**: jobs submitted to queue auto-start when earlier jobs finish and a worker frees. 用户回报 **M93-M96 已提交平台训练队列**。State file corrected to separate **started/running** from **queued/submitted**. Current queued/scheduled set includes M91-M96 after M89/M90; do not infer manual free slots while queue is non-empty.
- **2026-05-18 later**: Prepared **M93-M96 breakthrough protocol**. M93 = clean-window mature-negative weighting 6h/w0.3 diagnostic; M94 = full-data mature-negative candidate; M95 = sequence stats candidate; M96 = mature-negative + sequence-stats interaction. Evaluation is gate-controlled; see EXP-075.
- **2026-05-18 later**: 用户回报 **M91 / M92 均已提交**。Initially treated as 4 in-flight / 0 free, later corrected after queue-mechanism clarification: M91/M92 are submitted/platform-scheduled clean-window diagnostics on BJ **2026-03-03 09:49:51~14:49:07**; use M91 vs M92 delta to decide whether M89/M90 deserve eval quota despite M86 public miss.
- **2026-05-18 later**: 用户回报 **EDA71-time-window-backtest-audit completed**。日志显示 train range BJ **2026-02-28 08:35:09~2026-03-04 09:49:50**，public window BJ **2026-03-04 09:49:51~14:49:07**，`TRAIN_MAX_TO_PUBLIC_START_SEC=1`，`EXACT_PUBLIC_WINDOW_ROWS_IN_TRAIN=0`，normal RG time split overlap **341086s (~3.95d)**。结论: `recommend_clean_window_backtest=true` / `recommend_more_absolute_time_dense=false`。训练池状态更新为 **2 in-flight / 2 free** (M89, M90 running). Added fixed-window backtest scripts **M91/M92**.
- **2026-05-18 later**: 用户已启动 **EDA71-time-window-backtest-audit**。训练池状态改为 3 in-flight / 1 free (M89, M90, EDA71)。Also decoded M86 eval EDA blob: test rows **523,244**, public BJ window **2026-03-04 09:49:51~14:49:07**, median **12:57:28**, user IDs singleton-like under 200k cap.
- **2026-05-18 later**: 用户同意继续深入时间方向。Re-audit found old **M40 hour_only** evidence: LB **0.846931**, and EDA-056 argued DOW/weekend is not pure noise. Therefore do **not** launch M91 hour_only. Added `src/eda_time_window_backtest.py` for log-only clean time-window audit around public BJ 2026-03-04 09:49:51~14:49:07. Platform entry was uploaded as `run.sh`; local `run_*.sh` templates are intentionally untracked.
- **2026-05-18 later**: 用户回报 **M86 eval completed**: job `M86-m84-t38fix-utc-bjtime-gap0-seed42-4ep_eval_1779111177317`, public **0.847106** (472.2s). Eval log confirms intended checkpoint `auc=0.871415.best_model` and dataset flags `time_of_day_features=True`, `beijing_time_features=True`, `beijing_time_v2_features=False`. Conclusion: M86 is a real dev-public mismatch, not an infer config miss. 今日 quota 用 **3/4**，剩 **1/4**。Next eval priority shifts to **M88 E6** if using the last quota.
- **2026-05-18 later**: 用户回报 **M88** completed. Config confirmed `M84 6ep`: `num_epochs=6`, `enable_time_of_day_features=True`, `enable_beijing_time_features=False`, corrected T38 nonzero sample fit. Best dev **0.870910** at E6, history E1 0.865492 / E2 0.868755 / E3 0.869584 / E4 0.870313 / E5 0.870022 / E6 0.870910. Initial conclusion: longer training helps M84 by +0.000420 dev; later public eval rejected this path.
- **2026-05-18 later**: 用户回报 **M87** completed. Config confirmed `temporal_weight_alpha=2.5`, `enable_time_of_day_features=True`, `enable_beijing_time_features=False`, corrected T38 nonzero sample fit. Best dev **0.869053** (E4), history E1 0.864877 / E2 0.867883 / E3 0.868288 / E4 0.869053. Conclusion: M41-style temporal reweight does **not** stack with corrected T38/path B; do not evaluate with scarce quota.
- **2026-05-18 14:13 CEST**: 用户回报 **M89 / M90 均已启动训练任务**。训练池状态改为 2 in-flight / 2 free。M89 = Beijing time v1+v2 feature-only; M90 = M86 exact config extended to 6 epochs.
- **2026-05-18 CEST later**: M84/M82 eval 回流。M84 public **0.848854** (695.58s) 成为 tiny new SOTA, vs M75 +0.000066；M82 public **0.846197** (684.43s), vs M75 -0.002591。结论: true T38 小正向，hour-aware reweight 负向。今日 quota 用 2/4，剩 2/4。
- **2026-05-18 CEST morning**: 用户回报 M76/M82/M83/M84/M75(E6) 全部训练完成并贴 `train_log.txt`。Best dev: M75(E6)=0.870651, M84=0.870490, M76=0.870247, M82=0.869709, M83=0.867486。结论: 无新 dev SOTA；M75(E6) 是弱冲分候选，M84 是高信息候选，M83 reject。详见 EXP-072。
- **2026-05-17 CEST 19:51**: 用户 publish M51 / M53 ckpt · 5/18 04:22 24h 过期风险解除 · 5/18 早 quota 可正常评 (无 ckpt 时间窗约束) · M52 / M54 未 publish (silent skip 失败 job · 不值得评)
- **2026-05-17 CEST 18:35**: 用户启动 M73/M74/M75 + M50 重跑 (M76 接力 M50 重跑) · M50 ckpt 5/17 15:53 已过期 · 重训挽救 LB 评估机会 · 4 slots 全占 · today quota used 3/4 (M55+M71+M68 LB) · 今晚剩 1 quota = M50 LB · 5/20 ddl 战略下 cross-seed heavy + stack accumulation + ckpt selection 三轨并进
- **2026-05-17 CEST 16:14**: 用户立新硬 deadline = 0.852 by 5/20 (推前 5/22 → 5/20) · 5/22 cross-seed lock ARCHIVED · 立新铁律 team-memory line 195 (cross-seed 主线开放) · commit `541ef12`
- **2026-05-17 CEST 14:15~15:38**: 用户启 M68 / M65 / M68 v2 / M71 4 jobs 5/17 早 (overnight train) · 全部 19:00~21:30 完成 dev
- **2026-05-17 CEST 15:30~16:30**: M55/M71/M68 LB 回流 · M55 LB 0.844343 (Bilinear 死) / M71 LB 0.846641 (path B s11) / M68 LB 0.846468 (path B + Q_init stack 净损害)
- **2026-05-17 CEST 13:50~14:30**: EDA-67 v2 evaluation submitted (G2/A2/F3 train-side) · 22:26 完成 · 18.9MB blob 写到平台 storage · 用户决今晚不下载 · 5/18 后再分析
- **2026-05-17 CEST 13:44**: M68 第 1 次 evaluation state_dict mismatch 失败 (q_init_item key bug) · commit `cfd15f7` 修复 + 用户重提成功 · **失败 evaluation 不消耗 quota 铁律告知** (team-memory line 109)
- **2026-05-16 CEST 21:15**: 🎉 **M49 LB 0.848162** = 项目首个超 SOTA · UE split (T37 ADR-014) 弱真信号 · Δ+0.000570 vs M22
- **2026-05-16 CEST 16:58**: M51-M54 4 jobs 启动 (T38 NameError fix commit `2638e1b` 后) · 全部 in-flight 预计 ~CEST 22:00 完成 dev
- **2026-05-16 CEST 16:25**: publish 状态全 audit · 5/17 早 quota matrix 修订
- **2026-05-16 CEST 14:43**: M23 retry LB 0.836631 + M45/M48/M49 dev 回流 · M49 dev 0.870741 (项目首个超 M22 真信号)
- **2026-05-16 CEST 08:43**: 文件创建 (用户反问 "彻底解决问题" 触发) · single source of truth
