"""Platform-ready EDA: 验证 5 个关键假设 (H1~H5)

Usage:
    python baseline/eda_platform.py --data_dir ./data/demo
    python baseline/eda_platform.py \\
        --data_dir /data_ams/industrial_training_data \\
        --max_files 50 --rows_per_file 2000

Design:
    平台 1000 个 parquet × 每文件 ~1900 行 = 1.9M 总行 + 每行含 list<int64>
    seq 列 → 全量 concat 到内存会 OOM（已踩坑: job exit 137 @ 2026-05-10）。
    解决方案: **file-level subsample + per-file row cap**:
      - --max_files=100 抽样（files 按名字 sorted, 假设暗含时间顺序, step 均匀采）
      - --rows_per_file=1000 每文件最多 1000 行
      - 默认 100 × 1000 = 100k rows → concat 后 ~5GB peak, 容器限 15GB+ 安全
    100k 行样本 × 每行 seq 总 tokens ~2800 → H5 获 ~2.8 亿 delta 点, 统计极显著。
    用户实测 training job 活跃内存 17GB RAM + 9.8GB GPU, EDA 无 GPU 用 ~15GB RAM。

Outputs:
    - prints human-readable table to stdout
    - writes eda_report.json to --output_dir (default: /tmp/eda/)

Hypotheses:
    H1: label_time - timestamp 正样本分布 (是否集中 <15min → 短时 intent)
    H2: seq_a/d 是否真的不是 item_id sequence
         (unique_count / zero_frac / top3_values per fid)
    H3: seq_c_47 是否唯一真正的 long user-behavior
    H4: user_id 重复性 (demo 100% unique, 真实可能不同)
    H5: seq_ts - sample_ts 距离分布 (验证 "近 5 分钟" 价值)

Dependencies: stdlib + pyarrow + numpy only.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

print = functools.partial(print, flush=True)


# --------------------------------------------------------------------------
# 配置常量
# --------------------------------------------------------------------------

# 4 个 domain 的列名前缀 + 预期的 ts_fid (基于 data/demo/schema.json)
DOMAIN_CONFIG: dict[str, dict[str, Any]] = {
    "seq_a": {"prefix": "domain_a_seq", "ts_fid": 39, "fids": [38, 39, 40, 41, 42, 43, 44, 45, 46]},
    "seq_b": {"prefix": "domain_b_seq", "ts_fid": 67, "fids": [67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 88]},
    "seq_c": {"prefix": "domain_c_seq", "ts_fid": 27, "fids": [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 47]},
    "seq_d": {"prefix": "domain_d_seq", "ts_fid": 26, "fids": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]},
}


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------


def _find_parquet_files(data_dir: Path) -> list[Path]:
    """Find all .parquet files recursively (sorted for determinism)."""
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found under {data_dir}")
    return files


def _concat_list_col(col_values: list) -> np.ndarray | None:
    """Concat list<int64> column across rows into 1D ndarray (None→[]).

    Returns None if all None/empty.
    """
    parts: list[np.ndarray] = []
    for x in col_values:
        if x is None:
            continue
        if len(x) == 0:
            continue
        parts.append(np.asarray(x, dtype=np.int64))
    if not parts:
        return None
    return np.concatenate(parts)


def _percentiles(arr: np.ndarray, pcts: tuple[float, ...] = (5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if len(arr) == 0:
        return {f"p{int(p)}": float("nan") for p in pcts}
    return {f"p{int(p)}": float(np.percentile(arr, p)) for p in pcts}


def _summary_stats(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(len(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _top_k_values(arr: np.ndarray, k: int = 3) -> list[tuple[int, int]]:
    """Return top-k (value, count) pairs. arr must be int-like."""
    if len(arr) == 0:
        return []
    uniq, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)[:k]
    return [(int(uniq[i]), int(counts[i])) for i in order]


# --------------------------------------------------------------------------
# 5 个 hypothesis check
# --------------------------------------------------------------------------


class H1Accumulator:
    """H1: label_time - timestamp 分布 (正样本 label_type==2 only).

    Streaming: update() 每次吞一个 arrow table, state 只保留正样本 delta list
    (预期 <1% × 100k = <1k samples, 几 KB 内存) + label_type counter."""

    def __init__(self) -> None:
        self.n_total = 0
        self.n_pos = 0
        self.label_type_counts: dict[int, int] = {}
        self.lt_zero_count = 0
        self.delta_pos: list[int] = []

    def update(self, tbl) -> None:
        lt = tbl.column("label_time").to_numpy(zero_copy_only=False).astype(np.int64)
        ts = tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        lab = tbl.column("label_type").to_numpy(zero_copy_only=False).astype(np.int64)

        self.n_total += len(lab)
        uniq, counts = np.unique(lab, return_counts=True)
        for u, c in zip(uniq.tolist(), counts.tolist()):
            self.label_type_counts[int(u)] = self.label_type_counts.get(int(u), 0) + int(c)

        self.lt_zero_count += int((lt == 0).sum())

        pos_mask = lab == 2
        n_pos = int(pos_mask.sum())
        self.n_pos += n_pos
        if n_pos > 0:
            delta = (lt[pos_mask] - ts[pos_mask]).astype(np.int64)
            self.delta_pos.extend(delta.tolist())

    def finalize(self) -> dict[str, Any]:
        if self.n_pos == 0:
            return {"hypothesis": "H1", "note": "no positive samples", "n_pos": 0,
                    "label_type_dist": self.label_type_counts}
        delta = np.asarray(self.delta_pos, dtype=np.int64)
        return {
            "hypothesis": "H1",
            "label_type_dist": self.label_type_counts,
            "n_pos": self.n_pos,
            "pos_rate": self.n_pos / self.n_total if self.n_total > 0 else 0.0,
            "label_time_zero_frac": self.lt_zero_count / self.n_total if self.n_total > 0 else 0.0,
            "delta_stats_sec": _summary_stats(delta),
            "delta_percentiles_sec": _percentiles(delta),
            "delta_within_15min_frac": float((delta <= 900).mean()),
            "delta_within_1h_frac": float((delta <= 3600).mean()),
            "delta_within_24h_frac": float((delta <= 86400).mean()),
        }


def check_h1_label_time_delta(tbl) -> dict[str, Any]:
    """One-shot wrapper (for backward-compat with smoke tests)."""
    acc = H1Accumulator()
    acc.update(tbl)
    return acc.finalize()


class _FidState:
    """Per-fid streaming state for H2H3."""
    __slots__ = ("n_vals", "zero_count", "neg_count", "min_v", "max_v",
                 "unique_set", "unique_saturated", "value_counter",
                 "UNIQUE_CAP", "COUNTER_CAP", "COUNTER_TRIM")

    def __init__(self, unique_cap: int = 200000, counter_cap: int = 30000,
                 counter_trim: int = 5000) -> None:
        self.n_vals = 0
        self.zero_count = 0
        self.neg_count = 0
        self.min_v: int | None = None
        self.max_v: int | None = None
        self.unique_set: set[int] = set()
        self.unique_saturated = False
        # Counter (value -> count). Periodic prune if size > COUNTER_CAP, keep top
        # COUNTER_TRIM. After prune, top-3 at finalize is still approximate but
        # very accurate since top values almost never get evicted under Zipf dist.
        self.value_counter: dict[int, int] = {}
        self.UNIQUE_CAP = unique_cap
        self.COUNTER_CAP = counter_cap
        self.COUNTER_TRIM = counter_trim

    def update(self, arr: np.ndarray) -> None:
        self.n_vals += len(arr)
        self.zero_count += int((arr == 0).sum())
        self.neg_count += int((arr < 0).sum())
        mn = int(arr.min())
        mx = int(arr.max())
        self.min_v = mn if self.min_v is None else min(self.min_v, mn)
        self.max_v = mx if self.max_v is None else max(self.max_v, mx)

        # Unique set (cap-protected)
        if not self.unique_saturated:
            uniq_vals = np.unique(arr)
            for v in uniq_vals.tolist():
                self.unique_set.add(int(v))
                if len(self.unique_set) > self.UNIQUE_CAP:
                    self.unique_saturated = True
                    break

        # Top-k counter: count unique-per-batch, then add to global counter.
        # Periodic prune to keep memory bounded.
        uniq, counts = np.unique(arr, return_counts=True)
        for v, c in zip(uniq.tolist(), counts.tolist()):
            v_int = int(v)
            self.value_counter[v_int] = self.value_counter.get(v_int, 0) + int(c)
        if len(self.value_counter) > self.COUNTER_CAP:
            # Keep top COUNTER_TRIM by count.
            top_items = sorted(self.value_counter.items(), key=lambda x: -x[1])[:self.COUNTER_TRIM]
            self.value_counter = dict(top_items)

    def finalize_fid(self) -> dict[str, Any]:
        unique_str: int | str = (f">={self.UNIQUE_CAP}" if self.unique_saturated
                                 else len(self.unique_set))
        zero_frac = self.zero_count / self.n_vals if self.n_vals > 0 else 0.0
        neg_frac = self.neg_count / self.n_vals if self.n_vals > 0 else 0.0
        top3 = sorted(self.value_counter.items(), key=lambda x: -x[1])[:3]
        return {
            "n_vals": self.n_vals,
            "unique": unique_str,
            "zero_frac": zero_frac,
            "neg_frac": neg_frac,
            "min": self.min_v if self.min_v is not None else 0,
            "max": self.max_v if self.max_v is not None else 0,
            "top3_val_count": [(int(v), int(c)) for v, c in top3],
        }


class H2H3Accumulator:
    """H2 + H3: 4 个 domain 各 fid 的 unique / zero_frac / top3 / len distribution.

    Memory-safe streaming: per-fid state is bounded (unique_set cap, counter cap).
    Seq len is reservoir sampled (up to 500k per domain for percentile calc)."""

    RESERVOIR_CAP = 500_000

    def __init__(self) -> None:
        self.domain_states: dict[str, dict[str, Any]] = {}
        for domain_name, cfg in DOMAIN_CONFIG.items():
            self.domain_states[domain_name] = {
                "ts_fid": cfg["ts_fid"],
                "fid_states": {fid: _FidState() for fid in cfg["fids"]},
                "fid_all_none": {fid: True for fid in cfg["fids"]},
                "row_count": 0,
                "empty_row_count": 0,
                "seq_len_sample": [],
                "seq_len_n": 0,
            }

    def update(self, tbl) -> None:
        for domain_name, cfg in DOMAIN_CONFIG.items():
            state = self.domain_states[domain_name]
            prefix = cfg["prefix"]
            ts_fid = cfg["ts_fid"]
            ts_col_name = f"{prefix}_{ts_fid}"

            # Compute per-row seq lengths from ts_fid column (cheap).
            if ts_col_name in tbl.column_names:
                ts_col_py = tbl.column(ts_col_name).to_pylist()
                state["row_count"] += len(ts_col_py)
                for x in ts_col_py:
                    length = len(x) if x is not None else 0
                    if length == 0:
                        state["empty_row_count"] += 1
                    # Reservoir sample for percentile.
                    if len(state["seq_len_sample"]) < self.RESERVOIR_CAP:
                        state["seq_len_sample"].append(length)
                    state["seq_len_n"] += 1

            for fid in cfg["fids"]:
                col_name = f"{prefix}_{fid}"
                if col_name not in tbl.column_names:
                    continue
                col_py = tbl.column(col_name).to_pylist()
                concat = _concat_list_col(col_py)
                if concat is None:
                    continue
                state["fid_all_none"][fid] = False
                state["fid_states"][fid].update(concat)

    def finalize(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for domain_name, state in self.domain_states.items():
            fid_stats: dict[str, Any] = {}
            for fid, fs in state["fid_states"].items():
                if state["fid_all_none"][fid]:
                    fid_stats[f"fid_{fid}"] = {"n_vals": 0, "note": "all None/empty"}
                    continue
                fid_info = fs.finalize_fid()
                fid_info["is_ts_fid"] = (fid == state["ts_fid"])
                fid_stats[f"fid_{fid}"] = fid_info

            len_arr = np.asarray(state["seq_len_sample"], dtype=np.int64)
            row_count = state["row_count"]
            results[domain_name] = {
                "ts_fid": state["ts_fid"],
                "row_count": row_count,
                "seq_len_stats": _summary_stats(len_arr),
                "seq_len_percentiles": _percentiles(len_arr),
                "seq_empty_rows_frac": (state["empty_row_count"] / row_count
                                         if row_count > 0 else float("nan")),
                "fids": fid_stats,
            }
        return {"hypothesis": "H2+H3", "domains": results}


def check_h2_h3_seq_domain_structure(tbl) -> dict[str, Any]:
    """One-shot wrapper."""
    acc = H2H3Accumulator()
    acc.update(tbl)
    return acc.finalize()


class H4Accumulator:
    """H4: user_id / item_id 重复性 (streaming via global Counter).

    Memory: 1.9M rows → user/item Counter up to few MB each. OK."""

    def __init__(self) -> None:
        self.user_counts: dict[int, int] = {}
        self.item_counts: dict[int, int] = {}
        self.total_rows = 0

    def update(self, tbl) -> None:
        uid = tbl.column("user_id").to_numpy(zero_copy_only=False).astype(np.int64)
        iid = tbl.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
        self.total_rows += len(uid)

        u_uniq, u_c = np.unique(uid, return_counts=True)
        for v, c in zip(u_uniq.tolist(), u_c.tolist()):
            self.user_counts[int(v)] = self.user_counts.get(int(v), 0) + int(c)
        i_uniq, i_c = np.unique(iid, return_counts=True)
        for v, c in zip(i_uniq.tolist(), i_c.tolist()):
            self.item_counts[int(v)] = self.item_counts.get(int(v), 0) + int(c)

    def finalize(self) -> dict[str, Any]:
        u_counts_arr = np.asarray(list(self.user_counts.values()), dtype=np.int64)
        i_counts_arr = np.asarray(list(self.item_counts.values()), dtype=np.int64)

        def _count_dist(counts: np.ndarray) -> dict[str, int]:
            if len(counts) == 0:
                return {"=1": 0, "=2": 0, "3-5": 0, "6-10": 0, "11-50": 0, ">=51": 0}
            return {
                "=1": int((counts == 1).sum()),
                "=2": int((counts == 2).sum()),
                "3-5": int(((counts >= 3) & (counts <= 5)).sum()),
                "6-10": int(((counts >= 6) & (counts <= 10)).sum()),
                "11-50": int(((counts >= 11) & (counts <= 50)).sum()),
                ">=51": int((counts >= 51).sum()),
            }

        return {
            "hypothesis": "H4",
            "total_rows": self.total_rows,
            "unique_users": len(self.user_counts),
            "unique_items": len(self.item_counts),
            "user_impressions_per_user_mean": float(u_counts_arr.mean()) if len(u_counts_arr) > 0 else 0.0,
            "user_impressions_per_user_max": int(u_counts_arr.max()) if len(u_counts_arr) > 0 else 0,
            "user_count_dist": _count_dist(u_counts_arr),
            "item_impressions_per_item_mean": float(i_counts_arr.mean()) if len(i_counts_arr) > 0 else 0.0,
            "item_impressions_per_item_max": int(i_counts_arr.max()) if len(i_counts_arr) > 0 else 0,
            "item_count_dist": _count_dist(i_counts_arr),
        }


def check_h4_user_item_uniqueness(tbl) -> dict[str, Any]:
    """One-shot wrapper."""
    acc = H4Accumulator()
    acc.update(tbl)
    return acc.finalize()


class H5Accumulator:
    """H5: seq_ts - sample_ts 距离分布 (streaming).

    Per-domain state: bucket counters (precise) + reservoir sample of deltas
    for percentile calc (cap 500k per domain, 16MB × 4 = 64MB total).
    """

    RESERVOIR_CAP = 500_000
    DAY = 86400

    def __init__(self) -> None:
        self.domain_states: dict[str, dict[str, Any]] = {}
        for domain_name in DOMAIN_CONFIG.keys():
            self.domain_states[domain_name] = {
                "n_valid_deltas": 0,
                "sum_v": 0.0,
                "sum_sq_v": 0.0,
                "min_v": None,
                "max_v": None,
                "within_5min": 0,
                "within_1h": 0,
                "within_1d": 0,
                "within_7d": 0,
                "within_30d": 0,
                "over_30d": 0,
                "sample": [],
            }

    def update(self, tbl) -> None:
        sample_ts = tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
        for domain_name, cfg in DOMAIN_CONFIG.items():
            ts_col_name = f"{cfg['prefix']}_{cfg['ts_fid']}"
            if ts_col_name not in tbl.column_names:
                continue
            ts_col = tbl.column(ts_col_name).to_pylist()
            state = self.domain_states[domain_name]

            # Concat all deltas from this batch (all valid, positive-only).
            batch_deltas: list[np.ndarray] = []
            for i, seq_ts in enumerate(ts_col):
                if seq_ts is None or len(seq_ts) == 0:
                    continue
                seq_ts_arr = np.asarray(seq_ts, dtype=np.int64)
                valid = seq_ts_arr > 0
                if not valid.any():
                    continue
                delta = (sample_ts[i] - seq_ts_arr[valid]).astype(np.int64)
                delta = delta[delta >= 0]
                if len(delta) > 0:
                    batch_deltas.append(delta)
            if not batch_deltas:
                continue

            concat = np.concatenate(batch_deltas)
            state["n_valid_deltas"] += int(len(concat))

            # Bucket counters (precise).
            state["within_5min"] += int((concat <= 300).sum())
            state["within_1h"] += int((concat <= 3600).sum())
            state["within_1d"] += int((concat <= 1 * self.DAY).sum())
            state["within_7d"] += int((concat <= 7 * self.DAY).sum())
            state["within_30d"] += int((concat <= 30 * self.DAY).sum())
            state["over_30d"] += int((concat > 30 * self.DAY).sum())

            # Stats accumulators (exact).
            state["sum_v"] += float(concat.sum())
            state["sum_sq_v"] += float((concat.astype(np.float64) ** 2).sum())
            mn = int(concat.min()); mx = int(concat.max())
            state["min_v"] = mn if state["min_v"] is None else min(state["min_v"], mn)
            state["max_v"] = mx if state["max_v"] is None else max(state["max_v"], mx)

            # Reservoir sample up to cap (truncated-reservoir, simpler than true
            # reservoir algorithm; within-domain earlier-file bias, but 500k is
            # already larger than typical p99 calc needs).
            remain = self.RESERVOIR_CAP - len(state["sample"])
            if remain > 0:
                take = concat[:remain].tolist() if len(concat) > remain else concat.tolist()
                state["sample"].extend(take)

    def finalize(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for domain_name, state in self.domain_states.items():
            n = state["n_valid_deltas"]
            if n == 0:
                results[domain_name] = {"note": "no valid deltas"}
                continue
            sample_arr = np.asarray(state["sample"], dtype=np.int64)
            mean = state["sum_v"] / n
            var = state["sum_sq_v"] / n - mean * mean
            std = float(np.sqrt(max(var, 0.0)))
            results[domain_name] = {
                "n_valid_deltas": n,
                "delta_sec_stats": {
                    "n": n,
                    "min": float(state["min_v"]),
                    "max": float(state["max_v"]),
                    "mean": mean,
                    "std": std,
                },
                "delta_sec_percentiles": _percentiles(sample_arr),
                "within_5min_frac": state["within_5min"] / n,
                "within_1h_frac": state["within_1h"] / n,
                "within_1d_frac": state["within_1d"] / n,
                "within_7d_frac": state["within_7d"] / n,
                "within_30d_frac": state["within_30d"] / n,
                "over_30d_frac": state["over_30d"] / n,
            }
        return {"hypothesis": "H5", "domains": results}


def check_h5_seq_ts_distance(tbl) -> dict[str, Any]:
    """One-shot wrapper."""
    acc = H5Accumulator()
    acc.update(tbl)
    return acc.finalize()


# --------------------------------------------------------------------------
# Pretty printer
# --------------------------------------------------------------------------


def _print_h1(r: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("H1 · label_time - timestamp 分布（正样本 = label_type==2）")
    print("=" * 72)
    print(f"  label_type 分布: {r.get('label_type_dist')}")
    print(f"  pos_rate: {r.get('pos_rate'):.4f} (n_pos={r['n_pos']})")
    print(f"  label_time 为 0 的比例: {r.get('label_time_zero_frac'):.4f}")
    if r["n_pos"] == 0:
        return
    stats = r["delta_stats_sec"]
    pct = r["delta_percentiles_sec"]
    print(f"  delta (sec): min={stats['min']:.0f}, max={stats['max']:.0f}, mean={stats['mean']:.1f}, std={stats['std']:.1f}")
    print(f"  delta percentiles: p5={pct['p5']:.0f}, p25={pct['p25']:.0f}, p50={pct['p50']:.0f}, "
          f"p75={pct['p75']:.0f}, p95={pct['p95']:.0f}, p99={pct['p99']:.0f}")
    print(f"  正样本转化在 15min 内: {r['delta_within_15min_frac']*100:.1f}%")
    print(f"  正样本转化在 1h 内:   {r['delta_within_1h_frac']*100:.1f}%")
    print(f"  正样本转化在 24h 内:  {r['delta_within_24h_frac']*100:.1f}%")


def _print_h2h3(r: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("H2 + H3 · 4 个 domain 的语义结构诊断")
    print("=" * 72)
    for domain_name, d in r["domains"].items():
        print(f"\n  [{domain_name}]  ts_fid={d['ts_fid']}  rows={d['row_count']}")
        s = d["seq_len_stats"]
        p = d["seq_len_percentiles"]
        print(f"    seq_len: mean={s['mean']:.1f}, max={s['max']:.0f}, "
              f"p50={p['p50']:.0f}, p95={p['p95']:.0f}, p99={p['p99']:.0f}  "
              f"empty_rows={d['seq_empty_rows_frac']*100:.1f}%")
        print(f"    per-fid: fid    unique   zero%  neg%   top3 (val→count)")
        for fid_name, fid_info in d["fids"].items():
            if "unique" not in fid_info:
                continue
            top3 = fid_info["top3_val_count"]
            top3_str = " ".join([f"{v}→{c}" for v, c in top3])
            ts_tag = " *ts*" if fid_info.get("is_ts_fid") else ""
            print(f"      {fid_name:>8}  {fid_info['unique']:>7}  "
                  f"{fid_info['zero_frac']*100:>4.1f}%  {fid_info['neg_frac']*100:>4.1f}%  "
                  f"{top3_str}{ts_tag}")


def _print_h4(r: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("H4 · user_id / item_id 重复性")
    print("=" * 72)
    print(f"  total_rows: {r['total_rows']}")
    print(f"  unique_users: {r['unique_users']} ({r['unique_users']/r['total_rows']*100:.2f}% of rows)")
    print(f"  unique_items: {r['unique_items']} ({r['unique_items']/r['total_rows']*100:.2f}% of rows)")
    print(f"  user impressions: mean={r['user_impressions_per_user_mean']:.3f}, "
          f"max={r['user_impressions_per_user_max']}")
    print(f"    count dist: {r['user_count_dist']}")
    print(f"  item impressions: mean={r['item_impressions_per_item_mean']:.3f}, "
          f"max={r['item_impressions_per_item_max']}")
    print(f"    count dist: {r['item_count_dist']}")


def _print_h5(r: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("H5 · seq_ts - sample_ts 距离分布 (sample_ts > seq_ts only, 历史行为)")
    print("=" * 72)
    for domain_name, d in r["domains"].items():
        if "note" in d:
            print(f"  [{domain_name}] {d['note']}")
            continue
        s = d["delta_sec_stats"]
        p = d["delta_sec_percentiles"]
        print(f"\n  [{domain_name}]  n_deltas={d['n_valid_deltas']}")
        print(f"    sec: p25={p['p25']:.0f}, p50={p['p50']:.0f}, p75={p['p75']:.0f}, "
              f"p95={p['p95']:.0f}, p99={p['p99']:.0f}, max={s['max']:.0f}")
        print(f"    within 5min: {d['within_5min_frac']*100:>5.1f}%  |  "
              f"within 1h: {d['within_1h_frac']*100:>5.1f}%  |  "
              f"within 1d: {d['within_1d_frac']*100:>5.1f}%")
        print(f"    within 7d:   {d['within_7d_frac']*100:>5.1f}%  |  "
              f"within 30d: {d['within_30d_frac']*100:>5.1f}%  |  "
              f"over 30d:  {d['over_30d_frac']*100:>5.1f}%")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Platform EDA · 5 hypothesis checks (streaming, OOM-safe)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .parquet files")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="Cap total rows in final concat table (0 = no cap).")
    parser.add_argument("--max_files", type=int, default=100,
                        help="Cap number of parquet files scanned (0 = all). Files are sampled "
                             "evenly across the sorted file list for temporal coverage. "
                             "Default 100 files × 1000 rows/file ≈ 100k rows → ~5GB peak, "
                             "well within container limit (training uses ~17GB RAM).")
    parser.add_argument("--rows_per_file", type=int, default=1000,
                        help="Max rows to read from each sampled file (0 = full file). "
                             "Default 1000 × 100 files = 100k rows → statistically rich "
                             "for all 5 hypotheses (H5 gets ~280M delta points).")
    parser.add_argument("--output_dir", type=str, default="/tmp/eda",
                        help="Directory to write eda_report.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = _find_parquet_files(data_dir)
    print(f"[eda] Found {len(all_files)} parquet file(s) under {data_dir}")

    # Even sub-sampling across files for temporal coverage (files are sorted).
    if args.max_files > 0 and len(all_files) > args.max_files:
        step = len(all_files) // args.max_files
        files = [all_files[i * step] for i in range(args.max_files)]
        print(f"[eda] Sampling {len(files)} files evenly "
              f"(every {step}-th file) from {len(all_files)} total.")
    else:
        files = all_files

    # Streaming accumulators: per-RG update, never concat full table.
    import gc
    acc_h1 = H1Accumulator()
    acc_h2h3 = H2H3Accumulator()
    acc_h4 = H4Accumulator()
    acc_h5 = H5Accumulator()

    total_rows = 0
    files_scanned = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        rows_read_this_file = 0
        per_file_cap = args.rows_per_file if args.rows_per_file > 0 else 10**12
        for rg_idx in range(pf.num_row_groups):
            if rows_read_this_file >= per_file_cap:
                break
            tbl = pf.read_row_group(rg_idx)
            remain_file = per_file_cap - rows_read_this_file
            if tbl.num_rows > remain_file:
                tbl = tbl.slice(0, int(remain_file))
            if args.max_rows > 0:
                remain_total = args.max_rows - total_rows
                if remain_total <= 0:
                    break
                if tbl.num_rows > remain_total:
                    tbl = tbl.slice(0, remain_total)

            acc_h1.update(tbl)
            acc_h2h3.update(tbl)
            acc_h4.update(tbl)
            acc_h5.update(tbl)

            total_rows += tbl.num_rows
            rows_read_this_file += tbl.num_rows
            del tbl
        files_scanned += 1
        if files_scanned % 30 == 0 or files_scanned == len(files):
            print(f"[eda] Scanned {files_scanned}/{len(files)} files, "
                  f"{total_rows} rows so far")
        gc.collect()
        if args.max_rows > 0 and total_rows >= args.max_rows:
            break

    print(f"[eda] Streaming scan complete: {total_rows} rows from {files_scanned} files")

    print("[eda] Finalizing H1 · label_time delta...")
    r_h1 = acc_h1.finalize()
    print("[eda] Finalizing H2 + H3 · seq domain structure...")
    r_h2h3 = acc_h2h3.finalize()
    print("[eda] Finalizing H4 · user/item uniqueness...")
    r_h4 = acc_h4.finalize()
    print("[eda] Finalizing H5 · seq_ts distance...")
    r_h5 = acc_h5.finalize()

    # CRITICAL: build report + print SUMMARY + BLOB FIRST before any pretty
    # prints. Platform log UI truncates to top 1000 rows; pretty prints take
    # ~80 rows, if run before blob they push blob out of log window.
    report = {
        "meta": {
            "data_dir": str(data_dir),
            "num_files_total": len(all_files),
            "num_files_sampled": len(files),
            "rows_loaded": total_rows,
            "rows_per_file_cap": args.rows_per_file,
            "max_files_cap": args.max_files,
            "max_rows_cap": args.max_rows,
        },
        "H1": r_h1,
        "H2_H3": r_h2h3,
        "H4": r_h4,
        "H5": r_h5,
    }
    # (Disk write is useless on this platform — ckpt dir not downloadable —
    # but keep it as no-op best-effort.)
    out_path = out_dir / "eda_report.json"
    try:
        with out_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[eda] Report written to {out_path} (may not be downloadable)")
    except Exception as e:
        print(f"[eda] Warning: disk write failed: {e}")

    # Also write to TRAIN_LOG_PATH if set (some platforms expose log dir but not ckpt).
    log_path_env = os.environ.get("TRAIN_LOG_PATH")
    if log_path_env:
        log_dir = Path(log_path_env)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_copy = log_dir / "eda_report.json"
        try:
            with log_copy.open("w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[eda] Also copied to {log_copy} (TRAIN_LOG_PATH)")
        except Exception as e:
            print(f"[eda] Warning: could not write to TRAIN_LOG_PATH: {e}")

    # --------------------------------------------------------------------
    # Platform-safe summary + JSON dump to stdout
    # --------------------------------------------------------------------
    # Platform log may truncate long pretty prints (observed 2026-05-10).
    # Print a compact 20-line summary FIRST (small buffer, guaranteed flush),
    # THEN dump the full JSON to stdout as backup (1~2k lines, usually OK).
    print("\n" + "#" * 72)
    print("## EDA COMPACT SUMMARY · 20 key numbers (platform log safe)")
    print("#" * 72)
    print(f"# rows_loaded: {total_rows}, files_sampled: {len(files)} / {len(all_files)}")
    # H1: label_time delta
    if r_h1.get("n_pos", 0) > 0:
        p = r_h1["delta_percentiles_sec"]
        print(f"# H1 label_time_delta: pos_rate={r_h1['pos_rate']:.4f}, "
              f"p50={p['p50']:.0f}s, p95={p['p95']:.0f}s, "
              f"within_15min={r_h1['delta_within_15min_frac']*100:.1f}%, "
              f"within_24h={r_h1['delta_within_24h_frac']*100:.1f}%")
    # H2+H3: seq domain key numbers
    for dn in ["seq_a", "seq_b", "seq_c", "seq_d"]:
        d = r_h2h3["domains"].get(dn, {})
        s = d.get("seq_len_stats", {})
        p = d.get("seq_len_percentiles", {})
        # key signal: ts_fid + 1 (usually the "primary id" fid); show its unique+zero
        # instead probe the LARGEST-unique fid (the "item-like" fid)
        max_u = 0
        max_u_fid = None
        for fn, fi in d.get("fids", {}).items():
            if fi.get("is_ts_fid"):
                continue
            u = fi.get("unique", 0)
            # Saturated unique (str like '>=200000') treated as large sentinel.
            u_num = 10**18 if isinstance(u, str) else u
            if u_num > max_u:
                max_u = u_num
                max_u_fid = fn
        max_u_display = d.get("fids", {}).get(max_u_fid, {}).get("unique", 0) if max_u_fid else 0
        zfrac = d.get("fids", {}).get(max_u_fid, {}).get("zero_frac", -1) if max_u_fid else -1
        print(f"# H2+H3 {dn}: seq_len p50={p.get('p50', 0):.0f} p95={p.get('p95', 0):.0f}, "
              f"empty={d.get('seq_empty_rows_frac', 0)*100:.1f}%, "
              f"top_unique_fid={max_u_fid} unique={max_u_display} zero%={zfrac*100:.1f}")
    # H4
    print(f"# H4: total_rows={r_h4['total_rows']}, unique_users={r_h4['unique_users']} "
          f"({r_h4['unique_users']/r_h4['total_rows']*100:.2f}%), "
          f"user_mean_imp={r_h4['user_impressions_per_user_mean']:.2f}, "
          f"user_max_imp={r_h4['user_impressions_per_user_max']}, "
          f"unique_items={r_h4['unique_items']} ({r_h4['unique_items']/r_h4['total_rows']*100:.2f}%), "
          f"item_max_imp={r_h4['item_impressions_per_item_max']}")
    # H5 key: within_5min / 1h / 1d for each domain (the "recency" fingerprint)
    for dn in ["seq_a", "seq_b", "seq_c", "seq_d"]:
        d = r_h5["domains"].get(dn, {})
        if "note" in d:
            print(f"# H5 {dn}: {d['note']}")
            continue
        print(f"# H5 {dn}: within_5min={d.get('within_5min_frac', 0)*100:.2f}%, "
              f"within_1h={d.get('within_1h_frac', 0)*100:.2f}%, "
              f"within_1d={d.get('within_1d_frac', 0)*100:.2f}%, "
              f"within_7d={d.get('within_7d_frac', 0)*100:.2f}%, "
              f"within_30d={d.get('within_30d_frac', 0)*100:.2f}%, "
              f"over_30d={d.get('over_30d_frac', 0)*100:.2f}%")
    print("#" * 72)
    sys.stdout.flush()

    # --------------------------------------------------------------------
    # Base64(gzip(JSON)) blob · platform log truncates at top 1000 rows.
    # Pretty print and raw JSON were too long (1300+ rows), lost the SUMMARY
    # and JSON. Use compressed blob: demo 13.7KB raw → 3.7KB gzip → 4948 base64
    # chars ≈ 42 rows at 120 chars/line. Platform 100k data report will be
    # slightly larger but still ≤100 rows total.
    # --------------------------------------------------------------------
    import base64
    import gzip
    raw = json.dumps(report, default=str).encode("utf-8")
    gz = gzip.compress(raw)
    b64 = base64.b64encode(gz).decode("ascii")
    chunk_size = 120
    print("\n" + "#" * 72)
    print("## EDA JSON BLOB · base64(gzip(JSON))")
    print(f"## raw_bytes={len(raw)} gz_bytes={len(gz)} b64_chars={len(b64)}")
    print("## Decode locally: base64 -d | gunzip")
    print("#" * 72)
    print("<<<EDA_BLOB_START>>>")
    for i in range(0, len(b64), chunk_size):
        print(b64[i:i + chunk_size])
    print("<<<EDA_BLOB_END>>>")
    sys.stdout.flush()

    # Pretty prints LAST (nice-to-have, OK if truncated by platform log limit).
    _print_h1(r_h1)
    _print_h2h3(r_h2h3)
    _print_h4(r_h4)
    _print_h5(r_h5)


if __name__ == "__main__":
    main()
