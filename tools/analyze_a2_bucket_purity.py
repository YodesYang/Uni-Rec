"""tools/analyze_a2_bucket_purity.py · 5/17 EXP-067 v2 · 13:00 redesign.

A2 v2 · Hash bucket purity analysis · the **mechanistically correct** way
to measure seq encoder pathway signal in 100% user cold-start scenario.

Background (5/17 13:00 critique of EXP-067 v1):
- v1 measured "raw item_id train↔test overlap" → mechanistically meaningless
  because model uses seq_hash_vocab=500000 modulo, not raw item id
- v1 always predicts ~0% raw overlap (per pigeonhole on 278M raw items)
  even though mode performance is unaffected (hash buckets near 100%)
- v2 measures **bucket-level signal/noise** decomposition:
    pure_signal: test item raw == some train item raw (model truly reuses train signal)
    noise:       test item raw not in train, but hashes to a bucket train uses
                 (model's bucket embedding learns A's signal but tests on B → noise)
    dead:        test item raw not in train, and hash bucket also not in train
                 (zero signal from this token; effectively padding)

Inputs:
- train EDA blob: eda_data_distribution.json from src/eda_data_distribution.py
                  (a2_train_seq_item_freq + a2_train_seq_item_distinct_per_fid)
- test EDA blob: base64+gzip JSON emitted by src/infer.py::_emit_test_eda_blob
                 (seq_item_id_top_k + seq_item_id_distinct_per_fid)
                 Decode via tools/decode_eda_blob.py first.

Output: per (domain, fid) row with:
- raw_overlap_rate = |train_raw ∩ test_raw| / |test_raw|
- bucket_overlap_rate = |train_buckets ∩ test_buckets| / |test_buckets|
- pure_signal_rate = |test_raw ∩ train_raw| / |test_raw|  (= raw_overlap)
- noise_rate = |test_raw - train_raw, but bucket(test_raw) ∈ bucket(train_raw)|
              / |test_raw|
- dead_rate = 1 - pure_signal_rate - noise_rate
- weighted variants (by frequency) where available

Decision rule (mechanistic):
- pure_signal_rate ≥ 0.5 → seq encoder pathway has strong train→test signal
  transfer · current architecture sound
- pure_signal_rate ∈ [0.2, 0.5) → mixed · partial transfer + significant noise
- pure_signal_rate < 0.2 + noise_rate > 0.5 → bucket collision dominant ·
  considered re-architecting (larger hash_vocab? different encoder?)
- pure_signal_rate < 0.1 → seq encoder essentially noise · path B (UE split)
  is the right strategic direction since it doesn't depend on seq item ids

Run:
    python tools/analyze_a2_bucket_purity.py \\
        --train_blob path/to/eda_data_distribution.json \\
        --test_blob path/to/decoded_test_blob.json \\
        --hash_vocab 500000

Output: stdout summary + optional --output_json path
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _hash_to_bucket(item_id: int, hash_vocab: int) -> int:
    """Replicate src/model.py line 3154-3156 hash logic.

    Original:
        nonzero = vals != 0
        vals = vals.clone()
        vals[nonzero] = (vals[nonzero] % self.seq_hash_vocab).clamp(min=1)
    """
    if item_id == 0:
        return 0
    bucket = item_id % hash_vocab
    return max(bucket, 1)


def _items_to_bucket_set(
    items_with_freq: List[List[int]], hash_vocab: int
) -> Set[int]:
    """[[item_id, freq], ...] → set of buckets."""
    return {_hash_to_bucket(int(it[0]), hash_vocab) for it in items_with_freq}


def _items_to_freq_map(
    items_with_freq: List[List[int]],
) -> Dict[int, int]:
    """[[item_id, freq], ...] → {item_id: freq}."""
    return {int(it[0]): int(it[1]) for it in items_with_freq}


def analyze_one_fid(
    domain_fid_key: str,
    train_items: List[List[int]],
    test_items: List[List[int]],
    hash_vocab: int,
    train_distinct: Optional[int] = None,
    test_distinct: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute purity metrics for one (domain, fid) pair."""
    train_freq = _items_to_freq_map(train_items)
    test_freq = _items_to_freq_map(test_items)

    train_raw_set: Set[int] = set(train_freq.keys())
    test_raw_set: Set[int] = set(test_freq.keys())

    train_buckets: Set[int] = {
        _hash_to_bucket(it, hash_vocab) for it in train_raw_set
    }
    test_buckets: Set[int] = {
        _hash_to_bucket(it, hash_vocab) for it in test_raw_set
    }

    # Distinct count metrics
    raw_overlap = train_raw_set & test_raw_set
    bucket_overlap = train_buckets & test_buckets

    # Per-test-item classification
    pure_signal_count = 0  # test raw ∈ train raw
    noise_count = 0  # test raw ∉ train raw, but hash bucket ∈ train buckets
    dead_count = 0  # test raw ∉ train raw, hash bucket ∉ train buckets
    pure_signal_freq = 0
    noise_freq = 0
    dead_freq = 0
    total_test_freq = sum(test_freq.values())

    for test_item, freq in test_freq.items():
        if test_item in train_raw_set:
            pure_signal_count += 1
            pure_signal_freq += freq
        else:
            test_bucket = _hash_to_bucket(test_item, hash_vocab)
            if test_bucket in train_buckets:
                noise_count += 1
                noise_freq += freq
            else:
                dead_count += 1
                dead_freq += freq

    n_test = len(test_raw_set)

    return {
        "domain_fid": domain_fid_key,
        "train_distinct_in_blob": len(train_raw_set),
        "test_distinct_in_blob": len(test_raw_set),
        "train_distinct_full": train_distinct,  # may exceed blob top-K
        "test_distinct_full": test_distinct,
        "train_buckets_count": len(train_buckets),
        "test_buckets_count": len(test_buckets),
        # Distinct-level overlap (no frequency weight)
        "raw_overlap_count": len(raw_overlap),
        "raw_overlap_rate_distinct": (
            len(raw_overlap) / n_test if n_test else 0.0),
        "bucket_overlap_count": len(bucket_overlap),
        "bucket_overlap_rate_distinct": (
            len(bucket_overlap) / len(test_buckets)
            if test_buckets else 0.0),
        # Per-test-item classification (distinct-level)
        "pure_signal_count": pure_signal_count,
        "noise_count": noise_count,
        "dead_count": dead_count,
        "pure_signal_rate_distinct": (
            pure_signal_count / n_test if n_test else 0.0),
        "noise_rate_distinct": noise_count / n_test if n_test else 0.0,
        "dead_rate_distinct": dead_count / n_test if n_test else 0.0,
        # Frequency-weighted (more meaningful for model behavior)
        "total_test_freq": int(total_test_freq),
        "pure_signal_freq": int(pure_signal_freq),
        "noise_freq": int(noise_freq),
        "dead_freq": int(dead_freq),
        "pure_signal_rate_weighted": (
            pure_signal_freq / total_test_freq
            if total_test_freq else 0.0),
        "noise_rate_weighted": (
            noise_freq / total_test_freq if total_test_freq else 0.0),
        "dead_rate_weighted": (
            dead_freq / total_test_freq if total_test_freq else 0.0),
    }


def render_summary(result: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(
        f"A2 bucket purity · {result['domain_fid']}"
    )
    lines.append("-" * 72)
    lines.append(
        f"train distinct (in blob top-K) = {result['train_distinct_in_blob']}"
        f" / full = {result['train_distinct_full']}"
    )
    lines.append(
        f"test  distinct (in blob top-K) = {result['test_distinct_in_blob']}"
        f" / full = {result['test_distinct_full']}"
    )
    lines.append(
        f"hash buckets: train {result['train_buckets_count']} · "
        f"test {result['test_buckets_count']} · "
        f"overlap {result['bucket_overlap_count']}"
    )
    lines.append("")
    lines.append("Distinct-level breakdown (per unique test item id):")
    lines.append(
        f"  pure_signal (test raw ∈ train raw)         = "
        f"{result['pure_signal_rate_distinct']:.4f} "
        f"({result['pure_signal_count']})"
    )
    lines.append(
        f"  noise       (test raw ∉ train, bucket hit) = "
        f"{result['noise_rate_distinct']:.4f} "
        f"({result['noise_count']})"
    )
    lines.append(
        f"  dead        (test raw ∉ train, bucket miss)= "
        f"{result['dead_rate_distinct']:.4f} "
        f"({result['dead_count']})"
    )
    lines.append("")
    lines.append("Frequency-weighted breakdown (model-behavior weighted):")
    lines.append(
        f"  pure_signal_rate_weighted = "
        f"{result['pure_signal_rate_weighted']:.4f} "
        f"({result['pure_signal_freq']}/{result['total_test_freq']} freq)"
    )
    lines.append(
        f"  noise_rate_weighted       = "
        f"{result['noise_rate_weighted']:.4f} "
        f"({result['noise_freq']}/{result['total_test_freq']} freq)"
    )
    lines.append(
        f"  dead_rate_weighted        = "
        f"{result['dead_rate_weighted']:.4f} "
        f"({result['dead_freq']}/{result['total_test_freq']} freq)"
    )
    return "\n".join(lines)


def render_decision(per_fid: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("DECISION RULE · Mechanistic verdict per (domain, fid)")
    lines.append("=" * 72)
    for r in per_fid:
        psw = r["pure_signal_rate_weighted"]
        nrw = r["noise_rate_weighted"]
        if psw >= 0.5:
            verdict = "STRONG ✅ · seq encoder has strong train→test signal"
        elif psw >= 0.2:
            verdict = "MIXED 🟡 · partial transfer + meaningful noise"
        elif psw < 0.1 and nrw > 0.5:
            verdict = "BUCKET-COLLISION-DOMINANT 🔴 · need larger hash_vocab or new encoder"
        elif psw < 0.1:
            verdict = "DEAD 🔴 · seq encoder essentially noise · path B (UE) is right strategic direction"
        else:
            verdict = "WEAK 🟠 · low signal · possible re-architecture worth exploring"
        lines.append(
            f"  {r['domain_fid']:25s} pure={psw:.3f} noise={nrw:.3f} "
            f"dead={r['dead_rate_weighted']:.3f}  → {verdict}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2 v2 · hash bucket purity analysis (5/17 EXP-067)"
    )
    parser.add_argument(
        "--train_blob", type=str, required=True,
        help="Path to eda_data_distribution.json (train-side EDA output)",
    )
    parser.add_argument(
        "--test_blob", type=str, required=True,
        help="Path to decoded test EDA blob JSON (from "
             "tools/decode_eda_blob.py applied to evaluation log)",
    )
    parser.add_argument(
        "--hash_vocab", type=int, default=500_000,
        help="seq_hash_vocab from training config (default 500000 · M22 / "
             "M49 / M55 baseline value)",
    )
    parser.add_argument(
        "--output_json", type=str, default="",
        help="If set, write structured results to this path",
    )
    args = parser.parse_args()

    train_path = Path(args.train_blob).expanduser().resolve()
    test_path = Path(args.test_blob).expanduser().resolve()

    if not train_path.exists():
        print(f"[error] train_blob not found: {train_path}", file=sys.stderr)
        sys.exit(1)
    if not test_path.exists():
        print(f"[error] test_blob not found: {test_path}", file=sys.stderr)
        sys.exit(1)

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_freq = train_data.get("a2_train_seq_item_freq", {})
    train_distinct = train_data.get("a2_train_seq_item_distinct_per_fid", {})

    test_freq = test_data.get("seq_item_id_top_k", {})
    test_distinct = test_data.get("seq_item_id_distinct_per_fid", {})

    if not train_freq:
        print(
            "[error] train_blob has no 'a2_train_seq_item_freq' field; "
            "did you run src/eda_data_distribution.py?",
            file=sys.stderr,
        )
        sys.exit(2)
    if not test_freq:
        print(
            "[error] test_blob has no 'seq_item_id_top_k' field; did you "
            "run an evaluation with the latest src/infer.py?",
            file=sys.stderr,
        )
        sys.exit(2)

    common_keys = sorted(set(train_freq.keys()) & set(test_freq.keys()))
    if not common_keys:
        print(
            f"[error] No common (domain, fid) keys between train and test "
            f"blobs.\n  train: {list(train_freq.keys())}\n  "
            f"test: {list(test_freq.keys())}",
            file=sys.stderr,
        )
        sys.exit(3)

    print(
        f"# A2 v2 bucket purity analysis · "
        f"hash_vocab={args.hash_vocab} · "
        f"common_keys={common_keys}"
    )

    per_fid_results: List[Dict[str, Any]] = []
    for key in common_keys:
        result = analyze_one_fid(
            key,
            train_freq[key],
            test_freq[key],
            args.hash_vocab,
            train_distinct=train_distinct.get(key),
            test_distinct=test_distinct.get(key),
        )
        per_fid_results.append(result)
        print(render_summary(result))

    print(render_decision(per_fid_results))

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_dict = {
            "meta": {
                "train_blob": str(train_path),
                "test_blob": str(test_path),
                "hash_vocab": args.hash_vocab,
                "common_keys": common_keys,
            },
            "per_fid": per_fid_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, ensure_ascii=False, indent=2)
        print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
