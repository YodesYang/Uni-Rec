# EXP-079 · EDA72 validation-oracle audit · 2026-05-19

> **Status**: completed on platform.
> **Purpose**: Re-examine whether M91 should remain the only clean-window gate, and build a multi-window validation policy for public/test distribution assumptions.

## Trigger

M86 and M89 both improved normal dev but lost public. M91-style clean-window
diagnostics correctly warned on those time-feature branches, but M91 is still a
single pseudo-public window. If M101/M103 also lose heavily to M91, we need to
know whether that means the mechanisms are bad or whether our validation oracle
is too narrow.

## Job

Job name:

```text
EDA72-validation-oracle-audit
```

Upload files:

- upload `platform_run_templates/run_eda72_validation_oracle_audit.sh` as platform `run.sh`
- upload `src/eda_validation_oracle_audit.py`

No model files are needed. This is a log-only EDA job and does not consume
evaluation quota.

## Candidate Windows

The script compares each candidate window against public timestamp and sequence
shape decoded from eval logs.

| Window | Purpose |
|---|---|
| `clk0303_m91` | Current M91 clean pseudo-public window |
| `clk0302` | Same-clock stability candidate with enough prior-day rows |
| `clk0301` | Same-clock low-row stress candidate |
| `p5p95_0303` | Same M91 date, public p05-p95 clock mass only |
| `p25p75_0303` | Same M91 date, public interquartile clock mass only |
| `pre0304_all` | Same target day rows before public starts |
| `pre0304_last60m` | Closest same-day hour before public starts |
| `pre0304_last30m` | Closest same-day 30 minutes before public starts |

## Output

The key log tables are:

- `PUBLIC_REFERENCE`: public time-normalized quantiles and sequence summaries.
- `WINDOW_SCOREBOARD`: rows, CVR, time-shape distance, sequence-length distance,
  user singleton fraction, item coverage, and aggregate oracle distance.
- `WINDOW_TIMESTAMP_ARGS`: exact epoch ranges to reuse in future
  `--fixed_valid_timestamp_*` or `--aux_valid_windows` jobs.
- `FINAL_RECOMMENDATION`: whether a multi-window gate is viable and which
  same-clock / same-day windows look most useful.

## Decision Policy

Expected interpretation:

- Same-clock windows are label-bearing negative filters.
- Same-day pre-public windows are distribution sanity checks, not label oracles.
- If `clk0303_m91` is far from public shape while `clk0302` or same-day windows
  look better, upgrade future gates from single M91 to a composite view.
- If M91 remains closest and clean-failing candidates also fail public, keep M91
  as the primary scarce-quota guardrail.

## Platform Result

Source: user copied `~/Downloads/train_log.txt` after the platform job
completed.

Overall:

- Train rows: **2,099,956**.
- Train CVR: **213,518 / 2,099,956 = 0.101677**.
- Train range BJ: **2026-02-28 08:35:09 -> 2026-03-04 09:49:50**.
- Public starts one second later: **2026-03-04 09:49:51**.
- `recommend_multi_window_gate=true`.

Window scoreboard excerpt:

| Window | Rows | CVR | time_q_l1 | user singleton | Item unique | Readout |
|---|---:|---:|---:|---:|---:|---|
| `clk0302` | 26,403 | 0.148809 | **0.06512** | 1.000000 | 7,460 | best same-clock by time shape |
| `pre0304_all` | 1,551,840 | 0.110854 | 0.08971 | 1.000000 | 172,219 | best same-day pre-public sanity window |
| `clk0303_m91` | 29,962 | 0.069088 | 0.07189 | 1.000000 | 9,193 | current M91 window; still viable |
| `p25p75_0303` | 11,915 | 0.075367 | 0.11070 | 1.000000 | 4,525 | narrower M91-day clock mass |
| `p5p95_0303` | 21,744 | 0.070732 | 0.09044 | 1.000000 | 7,066 | public p05-p95 clock mass on M91 date |
| `pre0304_last60m` | 249,160 | 0.123387 | 0.17139 | 1.000000 | 62,371 | closest same-day hour, but time shape less similar |
| `pre0304_last30m` | 95,459 | 0.128003 | 0.18849 | 1.000000 | 34,280 | closest same-day half-hour, but time shape less similar |

Timestamp args:

| Name | Start | End | BJ window |
|---|---:|---:|---|
| `clk0303_m91` | 1772502591 | 1772520547 | 2026-03-03 09:49:51 -> 14:49:07 |
| `clk0302` | 1772416191 | 1772434147 | 2026-03-02 09:49:51 -> 14:49:07 |
| `p5p95_0303` | 1772508558 | 1772519296 | 2026-03-03 11:29:18 -> 14:28:16 |
| `p25p75_0303` | 1772511299 | 1772516621 | 2026-03-03 12:14:59 -> 13:43:41 |
| `pre0304_all` | 1772553600 | 1772588991 | 2026-03-04 00:00:00 -> 09:49:51 |
| `pre0304_last60m` | 1772585391 | 1772588991 | 2026-03-04 08:49:51 -> 09:49:51 |
| `pre0304_last30m` | 1772587191 | 1772588991 | 2026-03-04 09:19:51 -> 09:49:51 |

## Interpretation

M91 should remain useful as a negative filter, but it should no longer be the
only validation oracle. The best same-clock candidate by time-shape distance is
`clk0302`, while the best same-day distribution sanity window is
`pre0304_all`.

Important caveat: the logged `seq_mean_l1`, `seq_p50_l1`, and aggregate
`oracle_dist` compare raw training list lengths with public sequence lengths
observed after model truncation. Therefore, do not use the aggregate distance
as a strict ranking signal from this run. The reliable signals are row count,
timestamp shape, user singleton behavior, and item coverage.

Local follow-up: after reading this result, `src/eda_validation_oracle_audit.py`
was patched so future reruns cap sequence lengths to the model input limits
(`seq_a/b=256`, `seq_c/d=512`) before computing sequence-distance terms.

Updated policy:

- Use `clk0303_m91` and `clk0302` as same-clock negative filters.
- Use `pre0304_all` as a same-day distribution sanity check in full-data jobs,
  not as a clean label oracle.
- A candidate that is neutral on M91, like M101, should not be auto-rejected;
  require full-data strength or a second-window diagnostic before spending
  scarce public quota.
