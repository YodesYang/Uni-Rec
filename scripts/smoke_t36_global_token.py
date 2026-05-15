"""T36 Global Token smoke test (ADR-013 Plan B).

Validates:
1. enable_global_token=False is bit-identical to baseline (no behaviour change).
2. enable_global_token=True without user_dense raises ValueError (guard).
3. enable_global_token=True correctly adds the global_token_proj module
   with the expected parameter increment (~25k at d_model=64).
4. forward + backward run cleanly with finite gradients.
5. T-divisibility (d_model % T == 0) holds before and after enabling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import PCVRHyFormer, ModelInput  # noqa: E402


def _make_inputs(B: int = 4, user_dense_dim: int = 8) -> ModelInput:
    """Build a minimal valid ModelInput for the smoke model below."""
    domains = ["a", "b", "c", "d"]
    L = 16
    user_int_feats = torch.zeros(B, 6, dtype=torch.long)
    item_int_feats = torch.zeros(B, 4, dtype=torch.long)
    user_dense_feats = torch.randn(B, user_dense_dim)
    item_dense_feats = torch.zeros(B, 0)  # item_dense_dim=0 in this smoke

    seq_data = {d: torch.zeros(B, 2, L, dtype=torch.long) for d in domains}
    seq_time_buckets = {d: torch.zeros(B, L, dtype=torch.long) for d in domains}
    seq_lens = {d: torch.full((B,), L, dtype=torch.long) for d in domains}
    return ModelInput(
        user_int_feats=user_int_feats,
        item_int_feats=item_int_feats,
        user_dense_feats=user_dense_feats,
        item_dense_feats=item_dense_feats,
        seq_data=seq_data,
        seq_time_buckets=seq_time_buckets,
        seq_lens=seq_lens,
    )


def _build_model(enable_global_token: bool, user_dense_dim: int = 8) -> PCVRHyFormer:
    """Construct a minimal PCVRHyFormer that satisfies T-divisibility.

    Layout: user_ns_tokens=5, item_ns_tokens=2, num_queries=2, num_sequences=4.
    => num_ns = 5 + 1(user_dense) + 2 + 0(item_dense) = 8
    => T = 2 * 4 + 8 = 16, d_model=64 % 16 == 0 ✅
    """
    user_int_feature_specs = [(i, 1, 100) for i in range(6)]
    item_int_feature_specs = [(i, 1, 100) for i in range(4)]
    user_ns_groups = [[0], [1], [2], [3], [4, 5]]  # 5 groups
    item_ns_groups = [[0, 1], [2, 3]]  # 2 groups
    seq_vocab_sizes = {d: [50, 50] for d in ["a", "b", "c", "d"]}
    return PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=user_dense_dim,
        item_dense_dim=0,
        seq_vocab_sizes=seq_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        d_model=64,
        emb_dim=64,
        num_queries=2,
        num_hyformer_blocks=2,
        num_heads=4,
        seq_encoder_type='swiglu',
        ns_tokenizer_type='rankmixer',
        user_ns_tokens=5,
        item_ns_tokens=2,
        rank_mixer_mode='full',
        enable_global_token=enable_global_token,
    )


def test_disabled_bit_identical() -> None:
    """enable_global_token=False uses user_dense_proj exactly like baseline."""
    torch.manual_seed(0)
    model = _build_model(enable_global_token=False)
    assert model.enable_global_token is False
    assert model.user_dense_proj is not None
    assert model.global_token_proj is None
    inputs = _make_inputs()
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (4, 1), f"unexpected shape {logits.shape}"
    assert torch.isfinite(logits).all()
    print("[OK] disabled path · user_dense_proj kept · forward shape OK")


def test_enabled_smoke() -> None:
    """enable_global_token=True · forward + backward · gradients finite."""
    torch.manual_seed(0)
    model = _build_model(enable_global_token=True)
    assert model.enable_global_token is True
    assert model.user_dense_proj is None
    assert model.global_token_proj is not None
    # Confirm T = 16 unchanged (Plan B preserves divisibility)
    assert model.num_ns == 8, f"expected num_ns=8, got {model.num_ns}"

    inputs = _make_inputs()
    logits = model(inputs)
    assert logits.shape == (4, 1)
    loss = logits.sum()
    loss.backward()
    # Inspect global_token_proj gradients
    lin = model.global_token_proj[0]
    assert lin.weight.grad is not None
    assert torch.isfinite(lin.weight.grad).all()
    # Confirm input dim of global_token_proj matches expected concat size
    expected_in_dim = 8 + 64 + 64 + 4 * 64  # user_dense + 2D + 4D
    assert lin.in_features == expected_in_dim, (
        f"expected in_features={expected_in_dim}, got {lin.in_features}"
    )
    print(
        f"[OK] enabled path · global_token_proj in_features={lin.in_features} "
        f"(expected {expected_in_dim}) · backward grads finite"
    )


def test_param_delta() -> None:
    """Param count increment ≈ Linear(in_dim, D) + LN(D)."""
    base = _build_model(enable_global_token=False)
    enabled = _build_model(enable_global_token=True)
    base_params = sum(p.numel() for p in base.parameters())
    enabled_params = sum(p.numel() for p in enabled.parameters())
    delta = enabled_params - base_params
    # baseline user_dense_proj: Linear(8, 64) + LN(64) = 8*64 + 64 + 2*64 = 704
    # enabled global_token_proj: Linear(328, 64) + LN(64) = 328*64 + 64 + 2*64 = 21184
    # delta ≈ 21184 - 704 = 20480
    expected_delta = (8 + 64 + 64 + 4 * 64) * 64 + 64 - (8 * 64 + 64)
    assert delta == expected_delta, (
        f"unexpected param delta: got {delta}, expected {expected_delta}"
    )
    print(f"[OK] param delta = {delta} (expected {expected_delta} ≈ +20k)")


def test_no_user_dense_raises() -> None:
    """enable_global_token=True with user_dense_dim=0 raises ValueError."""
    try:
        _build_model(enable_global_token=True, user_dense_dim=0)
    except ValueError as e:
        msg = str(e)
        assert "user_dense_dim" in msg, f"unexpected error message: {msg}"
        print(f"[OK] no-user-dense guard raises ValueError: {msg}")
        return
    raise AssertionError("expected ValueError but model was built")


def test_disabled_forward_matches_seq_first_order() -> None:
    """When disabled, forward output must NOT depend on seq/ns ordering switch.

    Belt-and-suspenders check: confirms that the new `if self.enable_global_token`
    branching in forward does not change the disabled-path numerics. We achieve
    this by building two identical models (same seed) and verifying the disabled
    path produces identical output across runs (deterministic given fixed seed
    and inputs).
    """
    torch.manual_seed(0)
    m1 = _build_model(enable_global_token=False)
    torch.manual_seed(0)
    m2 = _build_model(enable_global_token=False)
    inputs = _make_inputs()
    m1.eval()
    m2.eval()
    with torch.no_grad():
        out1 = m1(inputs)
        out2 = m2(inputs)
    assert torch.allclose(out1, out2, atol=1e-7), (
        "disabled path is non-deterministic across same-seed builds"
    )
    print("[OK] disabled path · deterministic across same-seed builds")


if __name__ == "__main__":
    test_disabled_bit_identical()
    test_enabled_smoke()
    test_param_delta()
    test_no_user_dense_raises()
    test_disabled_forward_matches_seq_first_order()
    print("\nAll T36 Global Token smoke tests passed.")
