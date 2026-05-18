"""Log-only hour-aware reweight audit for M82.

This script is designed for the Tencent platform Logs UI constraint: do not
write large artifacts, print compact human-readable tables only. It scans
``timestamp`` and ``label_type`` from train parquet files and estimates how
M82's ``temporal_weight_alpha + hour_weight`` changes the effective training
distribution.
"""

from __future__ import annotations

import argparse
import functools
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
LOG = logging.getLogger("eda_hour_weight_audit")
print = functools.partial(print, flush=True)


Stat = List[float]  # rows, pos, weight_sum, weighted_pos, weight_sq_sum


def _find_parquet_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return files


def _scalar_to_numpy(col) -> np.ndarray:
    return col.to_numpy(zero_copy_only=False)


def _beijing_date_hour(ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Beijing time = UTC + 8h. Use integer days/hours for vectorized grouping.
    bj = ts.astype(np.int64) + 8 * 3600
    day = bj // 86400
    hour = (bj // 3600) % 24
    return day.astype(np.int64), hour.astype(np.int64)


def _day_to_date(day: int) -> str:
    dt = datetime.fromtimestamp(int(day) * 86400, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _weekday(day: int) -> int:
    dt = datetime.fromtimestamp(int(day) * 86400, tz=timezone.utc)
    return dt.weekday()  # Monday=0, Sunday=6


def _scan_ts_range(files: Iterable[Path], max_row_groups: int) -> Tuple[int, int, int]:
    ts_min: int | None = None
    ts_max: int | None = None
    rows = 0
    rg_seen = 0
    for fp in files:
        pf = pq.ParquetFile(str(fp))
        for rg_idx in range(pf.num_row_groups):
            if max_row_groups and rg_seen >= max_row_groups:
                break
            tbl = pf.read_row_group(rg_idx, columns=["timestamp"])
            ts = _scalar_to_numpy(tbl.column("timestamp")).astype(np.int64)
            if ts.size:
                mn = int(ts.min())
                mx = int(ts.max())
                ts_min = mn if ts_min is None else min(ts_min, mn)
                ts_max = mx if ts_max is None else max(ts_max, mx)
                rows += int(ts.size)
            rg_seen += 1
        if max_row_groups and rg_seen >= max_row_groups:
            break
    if ts_min is None or ts_max is None:
        raise RuntimeError("No timestamps scanned")
    return ts_min, ts_max, rows


def _add_stat(stats: Dict[Tuple, Stat], key: Tuple, labels: np.ndarray, weights: np.ndarray) -> None:
    st = stats[key]
    st[0] += float(labels.size)
    st[1] += float(labels.sum())
    st[2] += float(weights.sum())
    st[3] += float((weights * labels).sum())
    st[4] += float((weights * weights).sum())


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:6.2f}%"


def _print_table(title: str, rows: List[Tuple], header: str, limit: int = 200) -> None:
    print("")
    print(title)
    print(header)
    for row in rows[:limit]:
        print(" ".join(str(x) for x in row))


def main() -> None:
    ap = argparse.ArgumentParser(description="Log-only M82 hour-weight audit")
    ap.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH", "/data_ams/industrial_training_data"))
    ap.add_argument("--max_row_groups", type=int, default=0, help="0 means full scan")
    ap.add_argument("--temporal_weight_alpha", type=float, default=2.5)
    ap.add_argument("--hour_weight_min", type=int, default=9)
    ap.add_argument("--hour_weight_max", type=int, default=14)
    ap.add_argument("--hour_weight_multiplier", type=float, default=5.0)
    ap.add_argument(
        "--target_date",
        default="",
        help="Beijing date YYYY-MM-DD. Empty means latest date observed.",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = _find_parquet_files(data_dir)
    LOG.info("[plan] files=%d data_dir=%s max_row_groups=%d", len(files), data_dir, args.max_row_groups)

    t0 = time.time()
    ts_min, ts_max, rows_for_range = _scan_ts_range(files, args.max_row_groups)
    LOG.info(
        "[pass1] rows=%d timestamp_range=[%d,%d] wall=%.1fs",
        rows_for_range,
        ts_min,
        ts_max,
        time.time() - t0,
    )
    latest_day, _ = _beijing_date_hour(np.asarray([ts_max], dtype=np.int64))
    target_day = int(latest_day[0])
    if args.target_date:
        target_day = int(
            datetime.strptime(args.target_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            // 86400
        )

    by_day: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    by_hour: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    by_day_hour: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    by_bucket: Dict[Tuple, Stat] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    raw_weights_for_quantiles: List[np.ndarray] = []
    total_rows = 0
    total_pos = 0
    total_w = 0.0
    total_w_pos = 0.0
    total_w2 = 0.0
    rg_seen = 0
    t1 = time.time()
    ts_range = max(float(ts_max - ts_min), 1.0)

    for fp_i, fp in enumerate(files, start=1):
        pf = pq.ParquetFile(str(fp))
        for rg_idx in range(pf.num_row_groups):
            if args.max_row_groups and rg_seen >= args.max_row_groups:
                break
            tbl = pf.read_row_group(rg_idx, columns=["timestamp", "label_type"])
            ts = _scalar_to_numpy(tbl.column("timestamp")).astype(np.int64)
            labels = (_scalar_to_numpy(tbl.column("label_type")).astype(np.int64) == 2).astype(np.float32)
            if ts.size == 0:
                rg_seen += 1
                continue

            day, hour = _beijing_date_hour(ts)

            norm_ts = np.clip((ts.astype(np.float32) - float(ts_min)) / ts_range, 0.0, 1.0)
            temporal_w = np.exp(float(args.temporal_weight_alpha) * norm_ts).astype(np.float32)
            in_hour = (hour >= int(args.hour_weight_min)) & (hour <= int(args.hour_weight_max))
            hour_w = np.ones(ts.size, dtype=np.float32)
            hour_w[in_hour] = float(args.hour_weight_multiplier)
            hour_w = hour_w / max(float(hour_w.mean()), 1e-6)
            weights = temporal_w * hour_w

            if len(raw_weights_for_quantiles) < 256:
                raw_weights_for_quantiles.append(weights[: min(weights.size, 4096)].copy())
            else:
                raw_weights_for_quantiles[rg_seen % 256] = weights[: min(weights.size, 4096)].copy()

            total_rows += int(ts.size)
            total_pos += int(labels.sum())
            total_w += float(weights.sum())
            total_w_pos += float((weights * labels).sum())
            total_w2 += float((weights * weights).sum())

            for d in np.unique(day):
                mask = day == d
                _add_stat(by_day, (int(d),), labels[mask], weights[mask])
            for h in range(24):
                mask = hour == h
                if mask.any():
                    _add_stat(by_hour, (h,), labels[mask], weights[mask])
            for d in np.unique(day):
                day_mask = day == d
                for h in np.unique(hour[day_mask]):
                    mask = day_mask & (hour == h)
                    _add_stat(by_day_hour, (int(d), int(h)), labels[mask], weights[mask])

            is_target_day = day == target_day
            is_weekend = np.isin(day, [d for d in np.unique(day) if _weekday(int(d)) >= 5])
            buckets = {
                "target_day_09_14": is_target_day & in_hour,
                "target_day_other": is_target_day & (~in_hour),
                "prev_days_09_14": (~is_target_day) & (~is_weekend) & in_hour,
                "prev_days_other": (~is_target_day) & (~is_weekend) & (~in_hour),
                "weekend_all": is_weekend,
            }
            for name, mask in buckets.items():
                if mask.any():
                    _add_stat(by_bucket, (name,), labels[mask], weights[mask])

            rg_seen += 1
        if fp_i % 50 == 0 or fp_i == len(files) or (args.max_row_groups and rg_seen >= args.max_row_groups):
            LOG.info("[pass2] files=%d/%d rg=%d rows=%d wall=%.1fs", fp_i, len(files), rg_seen, total_rows, time.time() - t1)
        if args.max_row_groups and rg_seen >= args.max_row_groups:
            break

    def row_from_stat(label: str, st: Stat) -> Tuple:
        rows, pos, wsum, wpos, w2 = st
        cvr = pos / rows if rows else 0.0
        wpct = wsum / total_w if total_w else 0.0
        rpct = rows / total_rows if total_rows else 0.0
        wcvr = wpos / wsum if wsum else 0.0
        avgw = wsum / rows if rows else 0.0
        return (
            f"{label:>18}",
            f"{int(rows):>9}",
            _fmt_pct(rpct),
            f"{cvr:8.5f}",
            _fmt_pct(wpct),
            f"{wcvr:8.5f}",
            f"{avgw:8.3f}",
        )

    print("")
    print("========== EDA-70 · M82 HOUR-WEIGHT AUDIT ==========")
    print(f"TOTAL_ROWS={total_rows:,} TOTAL_POS={total_pos:,} CVR={total_pos / max(total_rows, 1):.6f}")
    print(f"TIMESTAMP_RANGE_UTC=[{ts_min},{ts_max}]")
    print(f"TARGET_DATE_BJ={_day_to_date(target_day)} HOUR_WINDOW_BJ=[{args.hour_weight_min},{args.hour_weight_max}]")
    print(f"TEMPORAL_ALPHA={args.temporal_weight_alpha} HOUR_MULTIPLIER={args.hour_weight_multiplier}")

    weights_all = np.concatenate(raw_weights_for_quantiles) if raw_weights_for_quantiles else np.ones(1, dtype=np.float32)
    q = np.quantile(weights_all, [0, 0.5, 0.9, 0.95, 0.99, 1.0])
    ess = (total_w * total_w) / max(total_w2, 1e-9)
    print("")
    print("WEIGHT_HEALTH")
    print(f"weight_quantiles_sample min={q[0]:.4f} p50={q[1]:.4f} p90={q[2]:.4f} p95={q[3]:.4f} p99={q[4]:.4f} max={q[5]:.4f}")
    print(f"effective_sample_size={ess:,.0f} ess_ratio={ess / max(total_rows, 1):.4f}")
    print(f"weighted_cvr_global={total_w_pos / max(total_w, 1e-9):.6f}")

    header = "              bucket      rows row_pct      cvr weight_pct weighted_cvr  avg_w"
    bucket_rows = [row_from_stat(k[0], st) for k, st in sorted(by_bucket.items())]
    _print_table("BUCKET_SUMMARY", bucket_rows, header)

    day_rows = [
        row_from_stat(f"{_day_to_date(k[0])}", st)
        for k, st in sorted(by_day.items())
    ]
    _print_table("DATE_SUMMARY_BJ", day_rows, header)

    hour_rows = [
        row_from_stat(f"hour_{k[0]:02d}", st)
        for k, st in sorted(by_hour.items())
    ]
    _print_table("HOUR_SUMMARY_BJ", hour_rows, header)

    dh_rows = [
        row_from_stat(f"{_day_to_date(k[0])}_{k[1]:02d}", st)
        for k, st in sorted(by_day_hour.items(), key=lambda kv: kv[1][2], reverse=True)
    ]
    _print_table("TOP_DATE_HOUR_BY_WEIGHT_MASS", dh_rows, header, limit=40)

    target_st = by_bucket.get(("target_day_09_14",), [0.0, 0.0, 0.0, 0.0, 0.0])
    target_row_pct = target_st[0] / max(total_rows, 1)
    target_weight_pct = target_st[2] / max(total_w, 1e-9)
    ok = (
        target_st[0] >= 10_000
        and 0.08 <= target_weight_pct <= 0.45
        and ess / max(total_rows, 1) >= 0.25
    )
    if target_weight_pct > 0.45 or ess / max(total_rows, 1) < 0.25:
        rec_mul = max(2.0, float(args.hour_weight_multiplier) * 0.6)
    elif target_weight_pct < 0.15:
        rec_mul = min(10.0, float(args.hour_weight_multiplier) * 1.5)
    else:
        rec_mul = float(args.hour_weight_multiplier)

    print("")
    print("FINAL_DECISION")
    print(f"M82_REWEIGHT_OK={str(ok).lower()}")
    print(f"target_day_09_14_row_pct={target_row_pct:.6f}")
    print(f"target_day_09_14_weight_pct={target_weight_pct:.6f}")
    print(f"recommended_temporal_weight_alpha={args.temporal_weight_alpha:.3f}")
    print(f"recommended_hour_weight_multiplier={rec_mul:.3f}")
    print("<<<EDA_70_EOF>>>")


if __name__ == "__main__":
    main()
