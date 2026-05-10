"""Platform-ready EDA: 验证 5 个关键假设 (H1~H5)

Usage:
    python scripts/eda_platform.py --data_dir ./data/demo --max_rows 0
    python scripts/eda_platform.py --data_dir /path/to/platform/train --max_rows 200000

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

Design Principles:
    - stdlib + pyarrow + numpy only (no pandas, no sklearn, no torch)
    - streaming row-group read, max_rows cap 控制内存
    - 不改训练 code, 不影响 baseline/
    - 本地 demo_1000 smoke <10s; 平台 1.9M 数据 <5min 即可完成
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


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


def check_h1_label_time_delta(tbl) -> dict[str, Any]:
    """H1: label_time - timestamp 分布 (正样本 label_type==2 only)."""
    lt = tbl.column("label_time").to_numpy(zero_copy_only=False).astype(np.int64)
    ts = tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
    lab = tbl.column("label_type").to_numpy(zero_copy_only=False).astype(np.int64)

    pos_mask = lab == 2
    n_pos = int(pos_mask.sum())
    if n_pos == 0:
        return {"hypothesis": "H1", "note": "no positive samples", "n_pos": 0}

    delta_pos = (lt[pos_mask] - ts[pos_mask]).astype(np.int64)
    lt_null_frac = float((lt == 0).mean())

    return {
        "hypothesis": "H1",
        "label_type_dist": dict(zip(*[arr.tolist() for arr in np.unique(lab, return_counts=True)])),
        "n_pos": n_pos,
        "pos_rate": float(pos_mask.mean()),
        "label_time_zero_frac": lt_null_frac,
        "delta_stats_sec": _summary_stats(delta_pos),
        "delta_percentiles_sec": _percentiles(delta_pos),
        "delta_within_15min_frac": float((delta_pos <= 900).mean()),
        "delta_within_1h_frac": float((delta_pos <= 3600).mean()),
        "delta_within_24h_frac": float((delta_pos <= 86400).mean()),
    }


def check_h2_h3_seq_domain_structure(tbl) -> dict[str, Any]:
    """H2 + H3: 4 个 domain 各 fid 的 unique / zero_frac / top3 / len distribution."""
    results: dict[str, Any] = {}
    for domain_name, cfg in DOMAIN_CONFIG.items():
        prefix = cfg["prefix"]
        fid_stats: dict[str, Any] = {}
        per_row_lens: list[int] = []
        ts_fid = cfg["ts_fid"]
        for fid in cfg["fids"]:
            col_name = f"{prefix}_{fid}"
            if col_name not in tbl.column_names:
                continue
            col_py = tbl.column(col_name).to_pylist()
            if fid == ts_fid:
                per_row_lens = [len(x) if x is not None else 0 for x in col_py]

            concat = _concat_list_col(col_py)
            if concat is None:
                fid_stats[f"fid_{fid}"] = {"n_vals": 0, "note": "all None/empty"}
                continue
            uniq = len(np.unique(concat))
            zero_frac = float((concat == 0).mean())
            neg_frac = float((concat < 0).mean())
            top3 = _top_k_values(concat, k=3)
            fid_stats[f"fid_{fid}"] = {
                "n_vals": int(len(concat)),
                "unique": uniq,
                "zero_frac": zero_frac,
                "neg_frac": neg_frac,
                "min": int(concat.min()),
                "max": int(concat.max()),
                "top3_val_count": top3,
                "is_ts_fid": (fid == ts_fid),
            }

        len_arr = np.asarray(per_row_lens, dtype=np.int64)
        results[domain_name] = {
            "ts_fid": ts_fid,
            "row_count": int(len(per_row_lens)),
            "seq_len_stats": _summary_stats(len_arr),
            "seq_len_percentiles": _percentiles(len_arr),
            "seq_empty_rows_frac": float((len_arr == 0).mean()) if len(len_arr) > 0 else float("nan"),
            "fids": fid_stats,
        }
    return {"hypothesis": "H2+H3", "domains": results}


def check_h4_user_item_uniqueness(tbl) -> dict[str, Any]:
    """H4: user_id / item_id 重复性 (demo 100% unique, 真实可能不同)."""
    uid = tbl.column("user_id").to_numpy(zero_copy_only=False).astype(np.int64)
    iid = tbl.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)

    u_uniq, u_counts = np.unique(uid, return_counts=True)
    i_uniq, i_counts = np.unique(iid, return_counts=True)

    def _count_dist(counts: np.ndarray) -> dict[str, int]:
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
        "total_rows": int(len(uid)),
        "unique_users": int(len(u_uniq)),
        "unique_items": int(len(i_uniq)),
        "user_impressions_per_user_mean": float(u_counts.mean()),
        "user_impressions_per_user_max": int(u_counts.max()),
        "user_count_dist": _count_dist(u_counts),
        "item_impressions_per_item_mean": float(i_counts.mean()),
        "item_impressions_per_item_max": int(i_counts.max()),
        "item_count_dist": _count_dist(i_counts),
    }


def check_h5_seq_ts_distance(tbl) -> dict[str, Any]:
    """H5: seq_ts - sample_ts 每个 domain 的距离分布 (sec)."""
    sample_ts = tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)

    results: dict[str, Any] = {}
    for domain_name, cfg in DOMAIN_CONFIG.items():
        ts_col_name = f"{cfg['prefix']}_{cfg['ts_fid']}"
        if ts_col_name not in tbl.column_names:
            continue
        ts_col = tbl.column(ts_col_name).to_pylist()

        all_deltas: list[np.ndarray] = []
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
                all_deltas.append(delta)

        if not all_deltas:
            results[domain_name] = {"note": "no valid deltas"}
            continue

        concat = np.concatenate(all_deltas)
        results[domain_name] = {
            "n_valid_deltas": int(len(concat)),
            "delta_sec_stats": _summary_stats(concat),
            "delta_sec_percentiles": _percentiles(concat),
            "within_5min_frac": float((concat <= 300).mean()),
            "within_1h_frac": float((concat <= 3600).mean()),
            "within_1d_frac": float((concat <= 86400).mean()),
            "within_7d_frac": float((concat <= 7 * 86400).mean()),
            "within_30d_frac": float((concat <= 30 * 86400).mean()),
            "over_30d_frac": float((concat > 30 * 86400).mean()),
        }
    return {"hypothesis": "H5", "domains": results}


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
    parser = argparse.ArgumentParser(description="Platform EDA · 5 hypothesis checks")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing .parquet files")
    parser.add_argument("--max_rows", type=int, default=0,
                        help="Cap total rows (0 = no cap). Useful for quick platform runs.")
    parser.add_argument("--output_dir", type=str, default="/tmp/eda",
                        help="Directory to write eda_report.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _find_parquet_files(data_dir)
    print(f"[eda] Found {len(files)} parquet file(s) under {data_dir}")

    # streaming concat
    tables = []
    total_rows = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for rg_idx in range(pf.num_row_groups):
            if args.max_rows > 0 and total_rows >= args.max_rows:
                break
            tbl = pf.read_row_group(rg_idx)
            if args.max_rows > 0:
                remain = args.max_rows - total_rows
                if tbl.num_rows > remain:
                    tbl = tbl.slice(0, remain)
            tables.append(tbl)
            total_rows += tbl.num_rows
        if args.max_rows > 0 and total_rows >= args.max_rows:
            break

    import pyarrow as pa
    big_tbl = pa.concat_tables(tables)
    print(f"[eda] Loaded {big_tbl.num_rows} rows, {big_tbl.num_columns} columns")

    print("[eda] Running H1 · label_time delta...")
    r_h1 = check_h1_label_time_delta(big_tbl)
    print("[eda] Running H2 + H3 · seq domain structure...")
    r_h2h3 = check_h2_h3_seq_domain_structure(big_tbl)
    print("[eda] Running H4 · user/item uniqueness...")
    r_h4 = check_h4_user_item_uniqueness(big_tbl)
    print("[eda] Running H5 · seq_ts distance...")
    r_h5 = check_h5_seq_ts_distance(big_tbl)

    _print_h1(r_h1)
    _print_h2h3(r_h2h3)
    _print_h4(r_h4)
    _print_h5(r_h5)

    report = {
        "meta": {
            "data_dir": str(data_dir),
            "num_files": len(files),
            "total_rows": big_tbl.num_rows,
            "max_rows_cap": args.max_rows,
        },
        "H1": r_h1,
        "H2_H3": r_h2h3,
        "H4": r_h4,
        "H5": r_h5,
    }
    out_path = out_dir / "eda_report.json"
    with out_path.open("w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[eda] Report written to {out_path}")


if __name__ == "__main__":
    main()
