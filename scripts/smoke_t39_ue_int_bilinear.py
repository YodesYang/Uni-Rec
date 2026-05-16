"""T39 / M55 smoke test · UE × user_int Bilinear gating.

Tests:
1. enable_ue_int_bilinear=False · bit-identical (baseline)
2. enable_ue_int_bilinear=True without bypass → ValueError
3. enable_ue_int_bilinear=True + enable_ue_split=True · forward + predict OK
4. zero-init: gating ≈ 1 → output ≈ baseline
5. Param count: bilinear adds Linear(D → bypass_dense_dim) + 1 scalar alpha

Lesson learned (5/16 T38 NameError): smoke test must cover platform-startup
paths AND forward/predict paths · not just unit-level helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model import PCVRHyFormer, ModelInput  # noqa: E402


def _build_sota_model(
    enable_ue_split: bool = True,
    enable_dense_bypass: bool = False,
    enable_ue_int_bilinear: bool = False,
    ue_int_bilinear_alpha_init: float = 0.5,
    user_dense_dim: int = 988,
    ue_offset: int = 0,
    ue_dim: int = 256,
) -> PCVRHyFormer:
    """Mirror M22 config + 988-dim user_dense (production-like)."""
    user_int_feature_specs = [(1000, i, 1) for i in range(46)]
    item_int_feature_specs = [(1000, i, 1) for i in range(14)]
    user_ns_groups = [list(range(0, 10)), list(range(10, 20)),
                      list(range(20, 30)), list(range(30, 40)),
                      list(range(40, 46))]
    item_ns_groups = [list(range(0, 7)), list(range(7, 14))]
    seq_vocab_sizes = {d: [500, 500] for d in ["a", "b", "c", "d"]}
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
        item_conditioned_query=True,
        enable_din_interest=True,
        enable_dense_bypass=enable_dense_bypass,
        enable_ue_split=enable_ue_split,
        ue_offset=ue_offset,
        ue_dim=ue_dim,
        enable_ue_int_bilinear=enable_ue_int_bilinear,
        ue_int_bilinear_alpha_init=ue_int_bilinear_alpha_init,
    )


def _make_inputs(B: int = 4, user_dense_dim: int = 988) -> ModelInput:
    domains = ["a", "b", "c", "d"]
    L_map = {"a": 256, "b": 256, "c": 512, "d": 512}
    user_int_feats = torch.randint(0, 1000, (B, 46), dtype=torch.long)
    item_int_feats = torch.randint(0, 1000, (B, 14), dtype=torch.long)
    user_dense_feats = torch.randn(B, user_dense_dim) * 0.1
    item_dense_feats = torch.zeros(B, 0)
    seq_data = {d: torch.randint(0, 500, (B, 2, L_map[d]), dtype=torch.long) for d in domains}
    seq_time_buckets = {d: torch.zeros(B, L_map[d], dtype=torch.long) for d in domains}
    seq_lens = {d: torch.full((B,), L_map[d] // 2, dtype=torch.long) for d in domains}
    return ModelInput(
        user_int_feats=user_int_feats,
        item_int_feats=item_int_feats,
        user_dense_feats=user_dense_feats,
        item_dense_feats=item_dense_feats,
        seq_data=seq_data,
        seq_time_buckets=seq_time_buckets,
        seq_lens=seq_lens,
    )


def test_1_disabled_baseline() -> bool:
    """enable_ue_int_bilinear=False · 不应有 bilinear-related modules."""
    torch.manual_seed(0)
    m = _build_sota_model(
        enable_ue_split=True, enable_ue_int_bilinear=False)
    assert m.enable_ue_int_bilinear is False
    assert not hasattr(m, 'ue_int_bilinear_proj') or m.ue_int_bilinear_proj is None  # noqa
    inputs = _make_inputs()
    with torch.no_grad():
        logits = m(inputs)
    assert logits.shape == (4, 1)
    assert torch.isfinite(logits).all()
    print(f"  test_1_disabled_baseline: forward OK · logits[0]={logits[0].item():.4f}")
    return True


def test_2_rejects_no_bypass() -> bool:
    """bilinear=True · ue_split=False · dense_bypass=False → ValueError."""
    raised = False
    try:
        _build_sota_model(
            enable_ue_split=False,
            enable_dense_bypass=False,
            enable_ue_int_bilinear=True,
        )
    except ValueError as e:
        raised = True
        assert "has_dense_bypass=True" in str(e)
    assert raised, "expected ValueError"
    print("  test_2_rejects_no_bypass: ValueError raised OK")
    return True


def test_3_forward_with_bilinear() -> bool:
    """enable_ue_int_bilinear=True + ue_split=True · forward + predict 都 OK."""
    torch.manual_seed(0)
    m = _build_sota_model(
        enable_ue_split=True, enable_ue_int_bilinear=True)
    assert m.enable_ue_int_bilinear is True
    assert isinstance(m.ue_int_bilinear_proj, torch.nn.Linear)
    assert m.ue_int_bilinear_alpha.item() == 0.5
    # zero-init verify
    assert torch.all(m.ue_int_bilinear_proj.weight == 0)
    assert torch.all(m.ue_int_bilinear_proj.bias == 0)

    inputs = _make_inputs()
    m.eval()
    with torch.no_grad():
        logits_fwd = m(inputs)
        logits_pred = m.predict(inputs)
        if isinstance(logits_pred, tuple):
            logits_pred = logits_pred[0]
    assert logits_fwd.shape == (4, 1)
    assert logits_pred.shape == (4, 1)
    assert torch.isfinite(logits_fwd).all()
    assert torch.isfinite(logits_pred).all()
    print(f"  test_3_forward_with_bilinear: forward + predict OK · "
          f"fwd[0]={logits_fwd[0].item():.4f} pred[0]={logits_pred[0].item():.4f}")
    return True


def test_4_zero_init_matches_baseline() -> bool:
    """zero-init · gating = 1 + 0.5 * tanh(0) = 1 · forward 应等同 baseline."""
    torch.manual_seed(42)
    m_baseline = _build_sota_model(
        enable_ue_split=True, enable_ue_int_bilinear=False)
    inputs = _make_inputs()
    m_baseline.eval()
    with torch.no_grad():
        logits_base = m_baseline(inputs)

    # Build bilinear model · copy baseline params (filter out missing keys)
    torch.manual_seed(42)
    m_bilinear = _build_sota_model(
        enable_ue_split=True, enable_ue_int_bilinear=True)
    sd_base = m_baseline.state_dict()
    sd_bilinear = m_bilinear.state_dict()
    for k in sd_base:
        if k in sd_bilinear:
            sd_bilinear[k] = sd_base[k]
    # bilinear-specific params already zero-init
    m_bilinear.load_state_dict(sd_bilinear)
    m_bilinear.eval()
    with torch.no_grad():
        logits_bi = m_bilinear(inputs)

    diff = (logits_base - logits_bi).abs().max().item()
    assert diff < 1e-5, f"zero-init should match baseline · diff={diff}"
    print(f"  test_4_zero_init_matches_baseline: max abs diff = {diff:.2e} OK")
    return True


def test_5_param_count() -> bool:
    """Param count delta = bilinear_proj (Linear(64 → bypass_dim)) +
    alpha (1 scalar)."""
    user_dense_dim = 988
    ue_dim = 256
    bypass_dim = user_dense_dim - ue_dim  # = 732

    m_base = _build_sota_model(
        enable_ue_split=True, enable_ue_int_bilinear=False)
    m_bi = _build_sota_model(
        enable_ue_split=True, enable_ue_int_bilinear=True)
    n_base = sum(p.numel() for p in m_base.parameters())
    n_bi = sum(p.numel() for p in m_bi.parameters())
    delta = n_bi - n_base
    expected_delta = (64 * bypass_dim) + bypass_dim + 1  # Linear weight + bias + alpha
    assert delta == expected_delta, (
        f"unexpected param delta · {delta} vs expected {expected_delta}")
    print(f"  test_5_param_count: Δparams = {delta} (= 64*{bypass_dim} + "
          f"{bypass_dim} + 1) OK")
    return True


if __name__ == "__main__":
    tests = [
        ("test_1_disabled_baseline", test_1_disabled_baseline),
        ("test_2_rejects_no_bypass", test_2_rejects_no_bypass),
        ("test_3_forward_with_bilinear", test_3_forward_with_bilinear),
        ("test_4_zero_init_matches_baseline", test_4_zero_init_matches_baseline),
        ("test_5_param_count", test_5_param_count),
    ]
    results = []
    for name, fn in tests:
        try:
            ok = fn()
            results.append((name, ok, None))
        except Exception as e:
            import traceback
            results.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print()
    print("=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    print(f"T39/M55 smoke: {n_pass}/{n_total} PASS")
    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        suffix = "" if ok else f" :: {err}"
        print(f"  {status}: {name}{suffix}")
    sys.exit(0 if n_pass == n_total else 1)
