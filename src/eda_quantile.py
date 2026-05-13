"""Independent EDA scanner for TAAC 2026 pcvr parquet data.

Runs outside training loop -- pure pyarrow + numpy, no torch. Designed to:
- Finish in ~5-10 min on platform data (1.9M rows, 1044 RGs)
- Produce the same 4 EDA questions as the in-training EDA collector, but
  fully vectorized (no Python per-row loops, no dict counters on hot path)
- Write `eda_summary.json` alongside the parquet dir

EDA Questions:
  Q1  user_id / item_id frequency distribution
  Q2  per-fid (user_int + item_int) null rate × CVR gap
  Q4  per-domain sequence true length quantiles
  Q5  per-domain time_diff quantiles (timestamp_ref - seq_ts)

Performance design:
- Scan parquet row group by row group (streaming, bounded memory)
- For Q1: accumulate uid/iid via numpy.unique per RG, merge final counts
- For Q2: per-fid null mask = (arr <= 0 | isnull) computed as numpy vectorized
- For Q5: arrow ListArray -> flat numpy values + offsets, single concat, one
          sort per reservoir merge (no per-row Python loop)

Usage::

  python scripts/eda_quantile.py \\
    --parquet_dir ./data/demo_rg100 \\
    --schema_path ./data/demo_rg100/schema.json \\
    --out ./eda_summary.json \\
    [--max_row_groups 0]  # 0 = all RGs; set small for local smoke

Platform usage (self-contained on training container):

  python eda_quantile.py \\
    --parquet_dir /data_ams/industrial_training_data \\
    --schema_path /data_ams/industrial_training_data/schema.json \\
    --out $TRAIN_LOG_PATH/eda_summary.json

"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)
LOG = logging.getLogger('eda')


# ───── Helpers ─────

def _reservoir_merge(
    buf: List[np.ndarray],
    new: np.ndarray,
    max_size: int,
    rng: np.random.Generator,
) -> None:
    """Mutate buf in place: append new data, subsample if > max_size.

    We keep this O(N) in aggregate by accumulating all new arrays in `buf`
    and only subsampling at finalize. That yields **approximate** reservoir
    semantics (not strict Algorithm R) but is adequate for quantile est
    and 10-100x cheaper than per-item reservoir decisions.
    """
    buf.append(new)


def _reservoir_finalize(
    buf: List[np.ndarray],
    max_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not buf:
        return np.empty(0, dtype=np.int64)
    arr = np.concatenate(buf)
    if len(arr) > max_size:
        idx = rng.choice(len(arr), size=max_size, replace=False)
        arr = arr[idx]
    return arr


def _quantiles(arr: np.ndarray) -> Dict[str, Any]:
    if arr.size == 0:
        return {'n': 0}
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    return {
        'n': int(arr.size),
        'mean': float(arr.mean()),
        'min': int(arr.min()),
        'max': int(arr.max()),
        'quantiles': {f'p{q}': float(np.percentile(arr, q)) for q in qs},
    }


def _pa_list_to_numpy(col: pa.ChunkedArray) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate a ChunkedArray of list<int64> into (values, offsets).

    Offsets are re-computed relative to the concatenated values buffer.
    """
    values_parts: List[np.ndarray] = []
    offsets_parts: List[np.ndarray] = []
    cum = 0
    for chunk in col.chunks:
        vals = chunk.values.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        offs = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        if values_parts:
            offs = offs + cum
            offs = offs[1:]
        values_parts.append(vals)
        offsets_parts.append(offs)
        cum += len(vals)
    values = np.concatenate(values_parts) if values_parts else np.empty(
        0, dtype=np.int64)
    offsets = np.concatenate(offsets_parts) if offsets_parts else np.array(
        [0], dtype=np.int64)
    return values, offsets


def _scalar_to_numpy(col: pa.ChunkedArray) -> np.ndarray:
    """Safe scalar int cast (nulls become 0, following dataset.py convention)."""
    arr = col.fill_null(0).to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=np.int64)


def _list_any_scalar_null(col: pa.ChunkedArray) -> np.ndarray:
    """For a list<int64> column, return per-row bool of 'effectively null'.

    A row is deemed null iff its list is either empty, all entries <= 0, or
    Arrow-null at the row level. Matches dataset.py collapse rule.
    """
    masks: List[np.ndarray] = []
    for chunk in col.chunks:
        vals = chunk.values.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        offs = chunk.offsets.to_numpy(zero_copy_only=False).astype(np.int64,
                                                                   copy=False)
        n = len(offs) - 1
        out = np.ones(n, dtype=bool)
        # Vectorized positive-entry count per row via bincount + gt0 mask
        gt0 = vals > 0
        if len(vals) > 0 and n > 0:
            row_id = np.repeat(np.arange(n, dtype=np.int64),
                               offs[1:] - offs[:-1])
            if row_id.size > 0:
                pos_per_row = np.bincount(row_id[gt0], minlength=n) if gt0.any() \
                    else np.zeros(n, dtype=np.int64)
                out = pos_per_row == 0
        # also mark arrow-null rows as null
        if chunk.null_count > 0:
            valid = np.asarray(chunk.is_valid())
            out |= ~valid
        masks.append(out)
    return np.concatenate(masks) if masks else np.empty(0, dtype=bool)


# ───── Main ─────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet_dir', type=str, required=True)
    ap.add_argument('--schema_path', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--max_row_groups', type=int, default=0,
                    help='Cap RGs scanned for smoke; 0 = all')
    ap.add_argument('--reservoir_size', type=int, default=200000,
                    help='Per-domain reservoir size for Q4/Q5')
    ap.add_argument('--id_counter_cap', type=int, default=0,
                    help='Cap distinct uids/iids tracked; 0 = no cap')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load schema
    with open(args.schema_path) as f:
        schema = json.load(f)
    user_int_plan: List[Tuple[int, int, int]] = [  # (fid, vs, dim)
        tuple(e) for e in schema['user_int']
    ]
    item_int_plan: List[Tuple[int, int, int]] = [
        tuple(e) for e in schema['item_int']
    ]
    seq_cfg: Dict[str, Dict[str, Any]] = schema['seq']

    # Enumerate parquet files
    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir,
                                                  '*.parquet')))
    if not parquet_files:
        raise SystemExit(f'No *.parquet under {args.parquet_dir}')

    LOG.info('Parquet files: %d · schema: %d user_int / %d item_int / %d seq',
             len(parquet_files), len(user_int_plan), len(item_int_plan),
             len(seq_cfg))

    # Accumulators
    total_rows = 0
    total_pos = 0
    # Q1
    uid_counter: Dict[int, int] = {}
    iid_counter: Dict[int, int] = {}
    # Q2: per-fid [n_null, n_total, pos_null, pos_notnull]
    user_int_acc = {fid: [0, 0, 0, 0] for fid, _, _ in user_int_plan}
    item_int_acc = {fid: [0, 0, 0, 0] for fid, _, _ in item_int_plan}
    # Q4/Q5: reservoirs
    seq_len_buf: Dict[str, List[np.ndarray]] = {d: [] for d in seq_cfg}
    time_diff_buf: Dict[str, List[np.ndarray]] = {d: [] for d in seq_cfg}
    # seq domain prefix / ts_fid
    dom_prefix = {d: c['prefix'] for d, c in seq_cfg.items()}
    dom_ts_fid = {d: c.get('ts_fid') for d, c in seq_cfg.items()}

    rg_seen = 0
    t_start = time.time()

    for fpath in parquet_files:
        pf = pq.ParquetFile(fpath)
        for rg_idx in range(pf.num_row_groups):
            if args.max_row_groups and rg_seen >= args.max_row_groups:
                break
            rg = pf.read_row_group(rg_idx)
            n = rg.num_rows
            total_rows += n

            # Labels: label_type == 2 → positive (mirrors dataset.py)
            label_type = _scalar_to_numpy(rg.column('label_type'))
            labels = (label_type == 2).astype(np.int64)
            total_pos += int(labels.sum())

            # Q1 user_id / item_id counter
            uids = _scalar_to_numpy(rg.column('user_id'))
            iids = _scalar_to_numpy(rg.column('item_id'))
            u_uniq, u_counts = np.unique(uids, return_counts=True)
            i_uniq, i_counts = np.unique(iids, return_counts=True)
            for u, c in zip(u_uniq.tolist(), u_counts.tolist()):
                if args.id_counter_cap and len(uid_counter) \
                        >= args.id_counter_cap and u not in uid_counter:
                    continue
                uid_counter[u] = uid_counter.get(u, 0) + c
            for u, c in zip(i_uniq.tolist(), i_counts.tolist()):
                if args.id_counter_cap and len(iid_counter) \
                        >= args.id_counter_cap and u not in iid_counter:
                    continue
                iid_counter[u] = iid_counter.get(u, 0) + c

            # Q2 · user_int per-fid missing
            for fid, vs, dim in user_int_plan:
                col_name = f'user_int_feats_{fid}'
                if col_name not in rg.schema.names:
                    continue
                col = rg.column(col_name)
                if pa.types.is_list(col.type):
                    is_null = _list_any_scalar_null(col)
                else:
                    arr = _scalar_to_numpy(col)
                    is_null = arr <= 0
                n_null = int(is_null.sum())
                acc = user_int_acc[fid]
                acc[0] += n_null
                acc[1] += n
                acc[2] += int(labels[is_null].sum())
                acc[3] += int(labels[~is_null].sum())
            for fid, vs, dim in item_int_plan:
                col_name = f'item_int_feats_{fid}'
                if col_name not in rg.schema.names:
                    continue
                col = rg.column(col_name)
                if pa.types.is_list(col.type):
                    is_null = _list_any_scalar_null(col)
                else:
                    arr = _scalar_to_numpy(col)
                    is_null = arr <= 0
                n_null = int(is_null.sum())
                acc = item_int_acc[fid]
                acc[0] += n_null
                acc[1] += n
                acc[2] += int(labels[is_null].sum())
                acc[3] += int(labels[~is_null].sum())

            # Q4 / Q5 · per-domain seq length + time_diff
            ref_ts = _scalar_to_numpy(rg.column('timestamp'))
            for dom, cfg in seq_cfg.items():
                ts_fid = dom_ts_fid.get(dom)
                if ts_fid is None:
                    continue
                ts_col_name = f'{dom_prefix[dom]}_{ts_fid}'
                if ts_col_name not in rg.schema.names:
                    continue
                ts_col = rg.column(ts_col_name)
                values, offsets = _pa_list_to_numpy(ts_col)
                # Per-row valid count = bincount of row_id for positive entries
                gt0 = values > 0
                row_lens = offsets[1:] - offsets[:-1]
                # Vectorized true length per row
                if values.size > 0:
                    row_id = np.repeat(np.arange(n, dtype=np.int64), row_lens)
                    if gt0.any():
                        true_len = np.bincount(row_id[gt0], minlength=n)
                    else:
                        true_len = np.zeros(n, dtype=np.int64)
                else:
                    true_len = np.zeros(n, dtype=np.int64)
                _reservoir_merge(seq_len_buf[dom], true_len,
                                 args.reservoir_size, rng)

                # Time diff: broadcast ref_ts over rows' valid entries
                if values.size > 0 and gt0.any():
                    valid_values = values[gt0]
                    valid_row = row_id[gt0]
                    diffs = ref_ts[valid_row] - valid_values
                    diffs = diffs[diffs >= 0]
                    if diffs.size > 0:
                        _reservoir_merge(time_diff_buf[dom], diffs,
                                         args.reservoir_size, rng)

            rg_seen += 1
            if rg_seen % 50 == 0:
                elapsed = time.time() - t_start
                rate = rg_seen / max(elapsed, 1e-6)
                LOG.info('RG %d · rows=%d · pos=%d · %.1f RG/s',
                         rg_seen, total_rows, total_pos, rate)

        if args.max_row_groups and rg_seen >= args.max_row_groups:
            break

    elapsed = time.time() - t_start
    LOG.info('Scan complete · %d RG · %d rows · %d pos · %.1fs',
             rg_seen, total_rows, total_pos, elapsed)

    # ───── Finalize ─────

    def _freq_stats(counter: Dict[int, int]) -> Dict[str, Any]:
        if not counter:
            return {'distinct': 0}
        freqs = np.array(list(counter.values()), dtype=np.int64)
        return {
            'distinct': int(len(freqs)),
            'freq_1': float((freqs == 1).sum() / len(freqs)),
            'freq_2_10': float(((freqs >= 2) & (freqs <= 10)).sum()
                               / len(freqs)),
            'freq_gt10': float((freqs > 10).sum() / len(freqs)),
            'max_freq': int(freqs.max()),
            'mean_freq': float(freqs.mean()),
            'total_observations': int(freqs.sum()),
        }

    def _missing_summary(
        acc: Dict[int, List[int]],
        plan: List[Tuple[int, int, int]],
        prefix: str,
    ) -> List[Dict[str, Any]]:
        out = []
        for fid, vs, dim in plan:
            if fid not in acc:
                continue
            n_null, n_total, pos_null, pos_notnull = acc[fid]
            if n_total == 0:
                continue
            n_notnull = n_total - n_null
            cvr_null = (pos_null / n_null) if n_null > 0 else 0.0
            cvr_notnull = (pos_notnull / n_notnull) if n_notnull > 0 else 0.0
            out.append({
                'col': f'{prefix}_{fid}',
                'fid': fid,
                'vocab_size': vs,
                'dim': dim,
                'null_rate': float(n_null / n_total),
                'cvr_null': float(cvr_null),
                'cvr_notnull': float(cvr_notnull),
                'cvr_gap': float(cvr_null - cvr_notnull),
                'n_null': int(n_null),
                'n_total': int(n_total),
            })
        out.sort(key=lambda x: -abs(x['cvr_gap']))
        return out

    summary = {
        'version': 1,
        'source': {
            'parquet_dir': args.parquet_dir,
            'num_row_groups_scanned': rg_seen,
            'total_rows': total_rows,
            'elapsed_sec': round(elapsed, 2),
            'max_row_groups': args.max_row_groups,
            'reservoir_size': args.reservoir_size,
        },
        'global': {
            'total_samples_observed': total_rows,
            'total_pos': total_pos,
            'global_cvr': (total_pos / total_rows) if total_rows else 0.0,
        },
        'q1_user_id': _freq_stats(uid_counter),
        'q1_item_id': _freq_stats(iid_counter),
        'q2_user_int_missing_top': _missing_summary(user_int_acc,
                                                     user_int_plan,
                                                     'user_int_feats'),
        'q2_item_int_missing_top': _missing_summary(item_int_acc,
                                                     item_int_plan,
                                                     'item_int_feats'),
        'q4_seq_length': {
            d: _quantiles(_reservoir_finalize(seq_len_buf[d],
                                              args.reservoir_size, rng))
            for d in seq_cfg
        },
        'q5_time_diff': {
            d: _quantiles(_reservoir_finalize(time_diff_buf[d],
                                              args.reservoir_size, rng))
            for d in seq_cfg
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    LOG.info('Wrote %s', args.out)


if __name__ == '__main__':
    main()
