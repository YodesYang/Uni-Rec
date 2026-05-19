# EXP-080 · EDA73 validation-oracle capped-sequence rerun · 2026-05-19

> **Status**: completed on platform.
> **Purpose**: Rerun EDA72 after fixing the sequence-length comparison to use model-capped lengths.

## Why Rerun

EDA72 already established the important qualitative result:

- `recommend_multi_window_gate=true`
- best same-clock candidate: `clk0302`
- best same-day pre-public sanity window: `pre0304_all`

However, the EDA72 `seq_mean_l1`, `seq_p50_l1`, and aggregate `oracle_dist`
compared raw training list lengths against public sequence lengths observed
after model truncation. That made the sequence-distance component too large and
not directly comparable.

The local script has now been patched to cap lengths before computing sequence
distance:

- `seq_a`, `seq_b`: 256
- `seq_c`, `seq_d`: 512

## Job

Job name:

```text
EDA73-validation-oracle-audit-cappedseq
```

Upload files:

- upload `platform_run_templates/run_eda73_validation_oracle_audit_cappedseq.sh`
  as platform `run.sh`
- upload `src/eda_validation_oracle_audit.py`

No model files are needed. This is a log-only EDA job and does not consume
evaluation quota.

## Expected Use

Use the rerun to decide whether the multi-window validation gate should be:

- M91 + `clk0302` as two same-clock negative filters;
- plus `pre0304_all` as same-day distribution sanity;
- or a different combination if the capped sequence-distance ranking changes.

## Platform Result

Source: user copied `~/Downloads/train_log.txt` after the platform job
completed.

The capped-sequence rerun confirms the EDA72 qualitative conclusion and makes
the aggregate ranking usable:

| Window | Rows | CVR | time_q_l1 | seq_mean_l1 | seq_p50_l1 | oracle_dist | Readout |
|---|---:|---:|---:|---:|---:|---:|---|
| `clk0302` | 26,403 | 0.148809 | **0.06512** | 0.03642 | 0.03753 | **0.11092** | best same-clock and best overall |
| `pre0304_all` | 1,551,840 | 0.110854 | 0.08971 | **0.03565** | 0.01273 | **0.12854** | best same-day pre-public sanity |
| `clk0303_m91` | 29,962 | 0.069088 | 0.07189 | 0.05376 | 0.03217 | **0.13369** | current gate; viable but not best |
| `p5p95_0303` | 21,744 | 0.070732 | 0.09044 | 0.05656 | 0.03485 | 0.15571 | weaker M91-day alternative |
| `p25p75_0303` | 11,915 | 0.075367 | 0.11070 | 0.05405 | 0.03954 | 0.17463 | too narrow / weaker |
| `pre0304_last60m` | 249,160 | 0.123387 | 0.17139 | 0.03418 | 0.00402 | 0.20658 | closest in time, but poor time-shape match |
| `pre0304_last30m` | 95,459 | 0.128003 | 0.18849 | **0.03100** | **0.00067** | 0.21966 | sequence-like, time-shape poor |

Final recommendation from log:

- `viable_window_count=7`
- `best_same_clock=clk0302`
- `best_same_day_pre_public=pre0304_all`
- `recommend_multi_window_gate=true`

## Decision

Adopt a multi-window validation policy:

1. `clk0303_m91`: keep as proven negative filter because it predicted M86/M89
   public failures.
2. `clk0302`: add as the second same-clock clean filter; it is closer to public
   by the capped-seq oracle score.
3. `pre0304_all`: add as full-data/same-day distribution sanity only. It has
   labels, but using it as a clean oracle would train on a different clock
   segment from public; it is better used as an auxiliary readout.

Practical impact for M101/M103:

M101 being neutral on M91 should not be treated as a rejection. The next
high-information diagnostic would be to run the same candidate on `clk0302`, or
to wait for M102 full-data plus auxiliary readouts before spending public quota.
