"""scripts/smoke_test_eda_a2.py · 5/17 EXP-067 · Smoke test for A2 changes
in src/infer.py::_TestEdaState (seq_item_id_counter).

Tests:
1. _TestEdaState.from_dataset constructs seq_item_fid_plan correctly
2. update() collects seq item id frequencies into seq_item_id_counter
3. finalize() emits seq_item_id_top_k + seq_item_id_distinct_per_fid +
   seq_item_fid_plan + user_id_set_capped fields
4. Bit-identical to baseline when seq_item_fid_plan is empty (no high-vocab
   fids · e.g. tiny demo schema with all vocab < 1M)

Run: python scripts/smoke_test_eda_a2.py
Expected: 4/4 PASS
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.infer import _TestEdaState  # noqa: E402


def _make_dataset_mock(
    seq_domains=("seq_a", "seq_b", "seq_c", "seq_d"),
    sideinfo_fids=None,
    seq_vocab_sizes=None,
    seq_maxlen=None,
    user_int_entries=None,
    item_int_entries=None,
):
    """Mock dataset with the read-only attrs _TestEdaState needs."""
    if sideinfo_fids is None:
        sideinfo_fids = {
            "seq_a": [38, 40, 41, 42, 43, 44, 45, 46],
            "seq_b": [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 88],
            "seq_c": [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 47],
            "seq_d": [17, 18, 19, 20, 21, 22, 23, 24, 25],
        }
    if seq_vocab_sizes is None:
        seq_vocab_sizes = {
            "seq_a": {38: 1_201_293, 40: 19, 41: 12, 42: 1018, 43: 3450,
                      44: 15147, 45: 9213, 46: 18},
            "seq_b": {68: 28, 69: 143_233_600, 70: 734, 71: 2691, 72: 10490,
                      73: 7239, 74: 638_924, 75: 30, 76: 164_032, 77: 168,
                      78: 4320, 79: 11781, 88: 288_959},
            "seq_c": {28: 75, 29: 8_227_608, 30: 858, 31: 7086, 32: 7, 33: 4,
                      34: 1_986_300, 35: 3004, 36: 1_511_293, 37: 9975,
                      47: 278_677_640},
            "seq_d": {17: 5, 18: 975, 19: 3420, 20: 11298, 21: 5106,
                      22: 510_191, 23: 674_034, 24: 603, 25: 15},
        }
    if seq_maxlen is None:
        seq_maxlen = {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}
    if user_int_entries is None:
        user_int_entries = [(1, 0, 1), (3, 1, 1)]
    if item_int_entries is None:
        item_int_entries = [(5, 0, 1), (6, 1, 1)]

    ds = MagicMock()
    ds.seq_domains = list(seq_domains)
    ds.sideinfo_fids = sideinfo_fids
    ds.seq_vocab_sizes = seq_vocab_sizes
    ds._seq_maxlen = seq_maxlen
    ds.user_int_schema = MagicMock()
    ds.user_int_schema.entries = user_int_entries
    ds.item_int_schema = MagicMock()
    ds.item_int_schema.entries = item_int_entries
    return ds


def test_1_from_dataset_builds_plan() -> bool:
    """from_dataset auto-discovers high-vocab fids per domain."""
    ds = _make_dataset_mock()
    state = _TestEdaState.from_dataset(ds, seq_item_id_min_vocab=1_000_000)

    plan = state.seq_item_fid_plan
    assert "seq_a" in plan, "seq_a should have fid 38 (vocab 1.2M)"
    assert "seq_b" in plan, "seq_b should have fid 69 (vocab 143M)"
    assert "seq_c" in plan, "seq_c should have fids 29/34/36/47"
    assert "seq_d" not in plan, "seq_d has no fid > 1M"

    seq_a_fids = [fid for _slot, fid in plan["seq_a"]]
    assert seq_a_fids == [38], f"seq_a expected [38], got {seq_a_fids}"

    seq_c_fids = sorted(fid for _slot, fid in plan["seq_c"])
    assert seq_c_fids == [29, 34, 36, 47], (
        f"seq_c expected [29,34,36,47], got {seq_c_fids}")

    # slot index = position in sideinfo_fids (e.g. seq_c fid 47 is at slot 10)
    seq_c_slot_47 = [s for s, f in plan["seq_c"] if f == 47][0]
    assert seq_c_slot_47 == 10, (
        f"seq_c fid 47 expected slot 10, got {seq_c_slot_47}")

    print(
        "  test_1: from_dataset plan = "
        f"{ {d: len(lst) for d, lst in plan.items()} } OK")
    return True


def test_2_update_collects_seq_item_freq() -> bool:
    """update() accumulates seq_item_id_counter from batch tensor."""
    ds = _make_dataset_mock()
    state = _TestEdaState.from_dataset(ds)

    B = 4
    n_feats_seq_a = len(ds.sideinfo_fids["seq_a"])  # 8
    n_feats_seq_c = len(ds.sideinfo_fids["seq_c"])  # 11
    max_len_a = 256
    max_len_c = 512

    # Build a synthetic batch
    batch = {
        "user_int_feats": torch.zeros(B, 2, dtype=torch.int64),
        "item_int_feats": torch.zeros(B, 2, dtype=torch.int64),
        "timestamp": torch.tensor(
            [1772725000, 1772725100, 1772725200, 1772725300],
            dtype=torch.int64),
        "user_id": ["u1", "u2", "u3", "u1"],  # u1 repeats
        "seq_a": torch.zeros(B, n_feats_seq_a, max_len_a, dtype=torch.int64),
        "seq_a_len": torch.tensor([3, 2, 1, 0], dtype=torch.int64),
        "seq_b": torch.zeros(B, len(ds.sideinfo_fids["seq_b"]), 256,
                             dtype=torch.int64),
        "seq_b_len": torch.tensor([2, 1, 0, 0], dtype=torch.int64),
        "seq_c": torch.zeros(B, n_feats_seq_c, max_len_c, dtype=torch.int64),
        "seq_c_len": torch.tensor([5, 4, 3, 2], dtype=torch.int64),
        "seq_d": torch.zeros(B, len(ds.sideinfo_fids["seq_d"]), 512,
                             dtype=torch.int64),
        "seq_d_len": torch.tensor([2, 1, 1, 0], dtype=torch.int64),
    }

    # seq_a fid 38 is at slot 0 (sideinfo_fids[seq_a][0] = 38)
    # Place item ids 100, 200, 300, ... in slot 0
    batch["seq_a"][0, 0, 0] = 100
    batch["seq_a"][0, 0, 1] = 200
    batch["seq_a"][1, 0, 0] = 100  # 100 appears twice across rows
    batch["seq_a"][2, 0, 0] = 300

    # seq_c fid 47 is at slot 10
    batch["seq_c"][0, 10, 0] = 999_999
    batch["seq_c"][0, 10, 1] = 999_999  # same value twice
    batch["seq_c"][1, 10, 0] = 12345

    state.update(batch)

    assert state.total_samples == B
    # A2: seq_a fid 38 should have items {100: 2, 200: 1, 300: 1}
    counter_a38 = state.seq_item_id_counter[("seq_a", 38)]
    assert counter_a38[100] == 2, f"100 expected 2, got {counter_a38[100]}"
    assert counter_a38[200] == 1
    assert counter_a38[300] == 1

    # A2: seq_c fid 47 should have items {999999: 2, 12345: 1}
    counter_c47 = state.seq_item_id_counter[("seq_c", 47)]
    assert counter_c47[999_999] == 2
    assert counter_c47[12345] == 1

    # F3: user_id_counter
    assert state.user_id_counter["u1"] == 2
    assert state.user_id_counter["u2"] == 1
    assert state.user_id_counter["u3"] == 1

    print(
        f"  test_2: seq_a fid 38 = {dict(counter_a38)}, "
        f"seq_c fid 47 = {dict(counter_c47)}, "
        f"user_ids = {dict(state.user_id_counter)} OK")
    return True


def test_3_finalize_emits_a2_f3_fields() -> bool:
    """finalize() includes seq_item_id_top_k + user_id_set_capped fields."""
    ds = _make_dataset_mock()
    state = _TestEdaState.from_dataset(ds)

    # Manually populate a small bit of data
    state.total_samples = 2
    state.user_id_counter["u_test"] = 1
    state.user_id_counter["u_other"] = 2
    state.seq_item_id_counter[("seq_c", 47)][111] = 5
    state.seq_item_id_counter[("seq_c", 47)][222] = 3

    out = state.finalize()

    # New A2 fields
    assert "seq_item_id_top_k" in out
    assert "seq_item_id_distinct_per_fid" in out
    assert "seq_item_fid_plan" in out
    assert "seq_c_fid_47" in out["seq_item_id_top_k"]
    top = out["seq_item_id_top_k"]["seq_c_fid_47"]
    assert top[0] == [111, 5], f"top item expected [111, 5], got {top[0]}"
    assert top[1] == [222, 3]
    assert out["seq_item_id_distinct_per_fid"]["seq_c_fid_47"] == 2

    # New F3 fields
    assert "user_id_set_capped" in out
    uids = out["user_id_set_capped"]
    assert "u_test" in uids and "u_other" in uids

    # Old fields still present (backward compat)
    assert "user_id_distinct" in out
    assert "user_id_freq_hist_top20" in out
    assert "seq_length" in out

    print(
        f"  test_3: finalize keys = {list(out.keys())[:6]}... + "
        f"{len(out)} total fields OK")
    return True


def test_4_no_high_vocab_no_overhead() -> bool:
    """When all seq fids have vocab < min_vocab, plan is empty + no panic."""
    ds = _make_dataset_mock(
        seq_vocab_sizes={
            "seq_a": {38: 1000, 40: 19},
            "seq_b": {68: 28, 69: 100},
            "seq_c": {28: 75, 29: 100},
            "seq_d": {17: 5, 18: 100},
        },
        sideinfo_fids={
            "seq_a": [38, 40], "seq_b": [68, 69],
            "seq_c": [28, 29], "seq_d": [17, 18],
        },
    )
    state = _TestEdaState.from_dataset(ds, seq_item_id_min_vocab=1_000_000)

    # No high-vocab fid → plan is empty
    assert state.seq_item_fid_plan == {}, (
        f"expected empty plan, got {state.seq_item_fid_plan}")

    # update() should not crash even with no plan
    batch = {
        "user_int_feats": torch.zeros(2, 2, dtype=torch.int64),
        "item_int_feats": torch.zeros(2, 2, dtype=torch.int64),
        "timestamp": torch.tensor([1, 2], dtype=torch.int64),
        "user_id": ["a", "b"],
        "seq_a": torch.zeros(2, 2, 4, dtype=torch.int64),
        "seq_a_len": torch.tensor([1, 1], dtype=torch.int64),
        "seq_b": torch.zeros(2, 2, 4, dtype=torch.int64),
        "seq_b_len": torch.tensor([1, 1], dtype=torch.int64),
        "seq_c": torch.zeros(2, 2, 4, dtype=torch.int64),
        "seq_c_len": torch.tensor([1, 1], dtype=torch.int64),
        "seq_d": torch.zeros(2, 2, 4, dtype=torch.int64),
        "seq_d_len": torch.tensor([1, 1], dtype=torch.int64),
    }
    state.update(batch)
    out = state.finalize()
    assert out["seq_item_id_top_k"] == {}, (
        f"expected empty seq_item_id_top_k, got {out['seq_item_id_top_k']}")
    print(
        "  test_4: no high-vocab → plan empty · no overhead, no crash OK")
    return True


if __name__ == "__main__":
    tests = [
        ("test_1_from_dataset_builds_plan", test_1_from_dataset_builds_plan),
        ("test_2_update_collects_seq_item_freq",
         test_2_update_collects_seq_item_freq),
        ("test_3_finalize_emits_a2_f3_fields",
         test_3_finalize_emits_a2_f3_fields),
        ("test_4_no_high_vocab_no_overhead", test_4_no_high_vocab_no_overhead),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok, None))
        except (AssertionError, ValueError, TypeError, AttributeError,
                ImportError) as e:
            results.append((name, False, repr(e)))

    print()
    print("=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"A2 smoke: {n_pass}/{n_total} PASS")
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        suffix = "" if ok else f" :: {err}"
        print(f"  {status}: {name}{suffix}")
    sys.exit(0 if n_pass == n_total else 1)
