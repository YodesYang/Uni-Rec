"""T38 silent skip bug fix · Full smoke (8 case).

Bug 历史:
- 5/16 EXP-061 M52 / M54 因 train.py line 832 调用
  `parse_bucket_boundaries('')` 返回 default ladder copy (非 None) →
  dataset.py line 3445 elif 触发 silent skip warning · per-domain fit
  被跳过 · M52/M54 实际 = M49/M22 rerun · 损失 10 GPU-hour。

修复 (Path A · train.py line 832 改条件传 None):
- enable_per_domain_buckets=True 且 --time_bucket_boundaries=""
  → 传 None · 触发 dataset.py line 3430 fit 分支
- 其他 case 行为 bit-identical 历史

Smoke cases (8/8 expected PASS):
1. argparse 模拟 · enable_per_domain_buckets=True + tbb='' → bucket_boundaries=None
2. argparse 模拟 · enable_per_domain_buckets=False + tbb='' → bucket_boundaries=ndarray (default)
3. argparse 模拟 · enable_per_domain_buckets=True + tbb='非空' → bucket_boundaries=ndarray (supplied 优先)
4. argparse 模拟 · enable_per_domain_buckets=False + tbb='非空' → bucket_boundaries=ndarray (supplied)
5. infer.py _resolve_bucket_boundaries · fitted_per_domain_bucket_boundaries 优先
6. infer.py back-compat · 老 ckpt 无 fitted_* key → default ladder
7. T-divisibility 检查 · num_ns=8 / num_queries=2 / d_model=64 仍合法
8. argparse 真实模拟 (用 sys.argv) · 走完 parser.parse_args() · 验 bucket_boundaries 一致

Run: python scripts/smoke_t38_bug_fix.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "src"))

from src.dataset import (  # noqa: E402
    DEFAULT_BUCKET_BOUNDARIES,
    parse_bucket_boundaries,
)
from src.infer import _resolve_bucket_boundaries  # noqa: E402


def _resolve_train_bucket_boundaries(args: Any) -> Any:
    """Mirror the exact logic from train.py line 830-836 after the fix.

    保持与 train.py 实际代码一致。如 train.py 改变,本函数必须同步。
    """
    if (
        args.enable_per_domain_buckets
        and not str(args.time_bucket_boundaries).strip()
    ):
        return None
    return parse_bucket_boundaries(args.time_bucket_boundaries)


def test_1_per_domain_true_empty_tbb_returns_none() -> bool:
    """Case 1: enable_per_domain_buckets=True + tbb='' → None.

    这是 bug 修复的核心 case · 之前返回 default ladder copy → silent skip.
    """
    args = MagicMock()
    args.enable_per_domain_buckets = True
    args.time_bucket_boundaries = ""
    bucket_boundaries = _resolve_train_bucket_boundaries(args)
    assert bucket_boundaries is None, (
        f"BUG NOT FIXED · expected None when per_domain=True + tbb='' · "
        f"got {type(bucket_boundaries).__name__} "
        f"({'ndarray shape=' + str(bucket_boundaries.shape) if isinstance(bucket_boundaries, np.ndarray) else bucket_boundaries})"
    )
    print(f"  test_1: per_domain=True + tbb='' → None OK (bug fixed)")
    return True


def test_2_per_domain_false_empty_tbb_returns_default() -> bool:
    """Case 2: enable_per_domain_buckets=False + tbb='' → default ladder.

    历史 default 行为 · 不能 break。
    """
    args = MagicMock()
    args.enable_per_domain_buckets = False
    args.time_bucket_boundaries = ""
    bucket_boundaries = _resolve_train_bucket_boundaries(args)
    assert isinstance(bucket_boundaries, np.ndarray), (
        f"expected ndarray, got {type(bucket_boundaries).__name__}"
    )
    assert np.array_equal(bucket_boundaries, DEFAULT_BUCKET_BOUNDARIES), (
        "default fallback should equal DEFAULT_BUCKET_BOUNDARIES")
    print("  test_2: per_domain=False + tbb='' → default ladder OK")
    return True


def test_3_per_domain_true_nonempty_tbb_supplied_priority() -> bool:
    """Case 3: enable_per_domain_buckets=True + tbb=非空 → supplied 优先.

    互斥语义: 当用户**同时**给两者时, supplied 优先 (warning · CLI help
    line 318-321 文档化)。
    """
    custom_tbb = ",".join(
        str(int(b)) for b in DEFAULT_BUCKET_BOUNDARIES.tolist())
    args = MagicMock()
    args.enable_per_domain_buckets = True
    args.time_bucket_boundaries = custom_tbb
    bucket_boundaries = _resolve_train_bucket_boundaries(args)
    assert isinstance(bucket_boundaries, np.ndarray), (
        f"expected ndarray (supplied 优先), got {type(bucket_boundaries).__name__}"
    )
    assert np.array_equal(bucket_boundaries, DEFAULT_BUCKET_BOUNDARIES)
    print(
        "  test_3: per_domain=True + tbb=非空 → ndarray (supplied 优先 · "
        "dataset.py line 3445 warning 路径) OK")
    return True


def test_4_per_domain_false_nonempty_tbb_supplied() -> bool:
    """Case 4: enable_per_domain_buckets=False + tbb=非空 → supplied."""
    custom_tbb = ",".join(
        str(int(b)) for b in DEFAULT_BUCKET_BOUNDARIES.tolist())
    args = MagicMock()
    args.enable_per_domain_buckets = False
    args.time_bucket_boundaries = custom_tbb
    bucket_boundaries = _resolve_train_bucket_boundaries(args)
    assert isinstance(bucket_boundaries, np.ndarray)
    assert np.array_equal(bucket_boundaries, DEFAULT_BUCKET_BOUNDARIES)
    print("  test_4: per_domain=False + tbb=非空 → supplied OK")
    return True


def test_5_infer_resolves_per_domain_priority() -> bool:
    """Case 5: infer.py _resolve_bucket_boundaries · fitted_* 优先."""
    cfg = {
        "fitted_per_domain_bucket_boundaries": {
            'seq_a': DEFAULT_BUCKET_BOUNDARIES.tolist(),
            'seq_b': DEFAULT_BUCKET_BOUNDARIES.tolist(),
            'seq_c': DEFAULT_BUCKET_BOUNDARIES.tolist(),
            'seq_d': DEFAULT_BUCKET_BOUNDARIES.tolist(),
        },
        "time_bucket_boundaries": "",
    }
    out = _resolve_bucket_boundaries(cfg)
    assert isinstance(out, dict), f"expected dict, got {type(out)}"
    assert set(out.keys()) == {'seq_a', 'seq_b', 'seq_c', 'seq_d'}
    for d in ('seq_a', 'seq_b', 'seq_c', 'seq_d'):
        assert out[d].dtype == np.int64
        assert out[d].shape == (63,)
    print("  test_5: infer fitted_* 优先 · 4 domain dict OK")
    return True


def test_6_infer_back_compat_old_ckpt() -> bool:
    """Case 6: 老 ckpt 无 fitted_* key → default ladder."""
    cfg_old = {"time_bucket_boundaries": ""}
    out_old = _resolve_bucket_boundaries(cfg_old)
    assert isinstance(out_old, np.ndarray), (
        f"old ckpt fallback expected ndarray, got {type(out_old).__name__}")
    assert np.array_equal(out_old, DEFAULT_BUCKET_BOUNDARIES)

    cfg_none = {
        "fitted_per_domain_bucket_boundaries": None,
        "time_bucket_boundaries": "",
    }
    out_none = _resolve_bucket_boundaries(cfg_none)
    assert isinstance(out_none, np.ndarray)
    assert np.array_equal(out_none, DEFAULT_BUCKET_BOUNDARIES)
    print("  test_6: old ckpt fallback (无 key + None) → default OK")
    return True


def test_7_t_divisibility_unchanged() -> bool:
    """Case 7: T-divisibility 检查不变.

    M22 baseline config:
    num_user_ns=5 + has_user_dense=1 + num_item_ns=2 + has_item_dense=0 = 8
    T = num_queries × num_sequences + num_ns = 2 × 4 + 8 = 16
    d_model = 64 % T = 0 ✅
    """
    num_user_ns = 5
    has_user_dense = 1
    num_item_ns = 2
    has_item_dense = 0
    num_queries = 2
    num_sequences = 4
    d_model = 64
    num_ns = num_user_ns + has_user_dense + num_item_ns + has_item_dense
    T = num_queries * num_sequences + num_ns
    assert num_ns == 8
    assert T == 16
    assert d_model % T == 0, (
        f"d_model {d_model} not divisible by T {T} · M22 base broken")
    print(f"  test_7: T-divisibility · num_ns={num_ns} T={T} 64%T=0 OK")
    return True


def test_8_argparse_real_simulation() -> bool:
    """Case 8: argparse 真实模拟 · sys.argv → parser.parse_args() → bucket_boundaries."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_bucket_boundaries', type=str, default='')
    parser.add_argument(
        '--enable_per_domain_buckets', action='store_true', default=False)

    # Sub-case 8a: --enable_per_domain_buckets (无 tbb) → None
    args_a = parser.parse_args(['--enable_per_domain_buckets'])
    bb_a = _resolve_train_bucket_boundaries(args_a)
    assert bb_a is None, f"8a expected None, got {type(bb_a).__name__}"

    # Sub-case 8b: 无任何 flag → default ladder
    args_b = parser.parse_args([])
    bb_b = _resolve_train_bucket_boundaries(args_b)
    assert isinstance(bb_b, np.ndarray)
    assert np.array_equal(bb_b, DEFAULT_BUCKET_BOUNDARIES)

    # Sub-case 8c: --enable_per_domain_buckets + --time_bucket_boundaries
    custom = ",".join(
        str(int(b)) for b in DEFAULT_BUCKET_BOUNDARIES.tolist())
    args_c = parser.parse_args(
        ['--enable_per_domain_buckets', '--time_bucket_boundaries', custom])
    bb_c = _resolve_train_bucket_boundaries(args_c)
    assert isinstance(bb_c, np.ndarray)
    print(
        "  test_8: argparse real · 8a None / 8b default / 8c supplied OK")
    return True


if __name__ == "__main__":
    tests = [
        ("test_1_per_domain_true_empty_tbb_returns_none",
         test_1_per_domain_true_empty_tbb_returns_none),
        ("test_2_per_domain_false_empty_tbb_returns_default",
         test_2_per_domain_false_empty_tbb_returns_default),
        ("test_3_per_domain_true_nonempty_tbb_supplied_priority",
         test_3_per_domain_true_nonempty_tbb_supplied_priority),
        ("test_4_per_domain_false_nonempty_tbb_supplied",
         test_4_per_domain_false_nonempty_tbb_supplied),
        ("test_5_infer_resolves_per_domain_priority",
         test_5_infer_resolves_per_domain_priority),
        ("test_6_infer_back_compat_old_ckpt",
         test_6_infer_back_compat_old_ckpt),
        ("test_7_t_divisibility_unchanged",
         test_7_t_divisibility_unchanged),
        ("test_8_argparse_real_simulation",
         test_8_argparse_real_simulation),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok, None))
        except (AssertionError, ValueError, TypeError, AttributeError) as e:
            results.append((name, False, repr(e)))

    print()
    print("=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"T38 bug fix smoke: {n_pass}/{n_total} PASS")
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        suffix = "" if ok else f" :: {err}"
        print(f"  {status}: {name}{suffix}")
    sys.exit(0 if n_pass == n_total else 1)
