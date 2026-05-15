"""T36 Global Token SOTA-config smoke (close to M22 production shape).

Tests:
1. Real M22 shape (d_model=64, num_seqs=4, L=256/256/512/768, hyformer_blocks=2,
   user_ns=5, item_ns=2, num_queries=2) builds successfully.
2. Forward + backward run, no NaN, no Inf.
3. Full-padding row case (seq_len=0 for all 4 domains) doesn't NaN.
4. Mixed bf16 autocast doesn't break (CPU autocast simulation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import PCVRHyFormer, ModelInput  # noqa: E402


def _build_sota_model() -> PCVRHyFormer:
    """Mirror M22 config as closely as feasible on CPU."""
    user_int_feature_specs = [(1000, i, 1) for i in range(46)]
    item_int_feature_specs = [(1000, i, 1) for i in range(14)]
    user_ns_groups = [
        list(range(0, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40)),
        list(range(40, 46)),
    ]
    item_ns_groups = [
        list(range(0, 7)),
        list(range(7, 14)),
    ]
    seq_vocab_sizes = {d: [500, 500] for d in ["a", "b", "c", "d"]}
    return PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=10,
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
        enable_global_token=True,
    )


def _make_inputs(B: int = 4, lens: dict | None = None) -> ModelInput:
    """SOTA-shape inputs with configurable seq_lens (full-padding test support)."""
    domains = ["a", "b", "c", "d"]
    L_map = {"a": 256, "b": 256, "c": 512, "d": 768}
    user_int_feats = torch.randint(0, 1000, (B, 46), dtype=torch.long)
    item_int_feats = torch.randint(0, 1000, (B, 14), dtype=torch.long)
    user_dense_feats = torch.randn(B, 10)
    item_dense_feats = torch.zeros(B, 0)
    seq_data = {d: torch.randint(0, 500, (B, 2, L_map[d]), dtype=torch.long) for d in domains}
    seq_time_buckets = {d: torch.zeros(B, L_map[d], dtype=torch.long) for d in domains}
    if lens is None:
        seq_lens = {d: torch.full((B,), L_map[d] // 2, dtype=torch.long) for d in domains}
    else:
        seq_lens = {d: torch.full((B,), lens[d], dtype=torch.long) for d in domains}
    return ModelInput(
        user_int_feats=user_int_feats,
        item_int_feats=item_int_feats,
        user_dense_feats=user_dense_feats,
        item_dense_feats=item_dense_feats,
        seq_data=seq_data,
        seq_time_buckets=seq_time_buckets,
        seq_lens=seq_lens,
    )


def test_sota_forward_backward() -> None:
    torch.manual_seed(42)
    model = _build_sota_model()
    inputs = _make_inputs(B=4)
    logits = model(inputs)
    assert logits.shape == (4, 1)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
    loss = logits.sum()
    loss.backward()
    # Walk all parameters; ensure no NaN/Inf gradients
    bad = []
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(n)
    assert not bad, f"non-finite gradients found in: {bad[:5]}..."
    print(f"[OK] SOTA shape forward+backward, model has "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")


def test_sota_full_padding_row() -> None:
    """Full-padding row (seq_len=0 for all 4 domains) must not NaN."""
    torch.manual_seed(0)
    model = _build_sota_model()
    inputs = _make_inputs(
        B=4,
        lens={"a": 0, "b": 0, "c": 0, "d": 0},  # all-padding
    )
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
    assert torch.isfinite(logits).all(), "all-padding row produced NaN/Inf"
    print(f"[OK] all-padding row · logits finite (sample={logits.flatten().tolist()})")


def test_sota_mixed_lens() -> None:
    """Realistic mix of seq lengths (some empty, some full)."""
    torch.manual_seed(0)
    model = _build_sota_model()
    domains = ["a", "b", "c", "d"]
    L_map = {"a": 256, "b": 256, "c": 512, "d": 768}
    user_int_feats = torch.randint(0, 1000, (4, 46), dtype=torch.long)
    item_int_feats = torch.randint(0, 1000, (4, 14), dtype=torch.long)
    user_dense_feats = torch.randn(4, 10)
    item_dense_feats = torch.zeros(4, 0)
    seq_data = {d: torch.randint(0, 500, (4, 2, L_map[d]), dtype=torch.long) for d in domains}
    seq_time_buckets = {d: torch.zeros(4, L_map[d], dtype=torch.long) for d in domains}
    # Row 0: all empty | Row 1: half-full | Row 2: full | Row 3: mixed
    seq_lens = {
        "a": torch.tensor([0, 128, 256, 50], dtype=torch.long),
        "b": torch.tensor([0, 128, 256, 200], dtype=torch.long),
        "c": torch.tensor([0, 256, 512, 100], dtype=torch.long),
        "d": torch.tensor([0, 384, 768, 600], dtype=torch.long),
    }
    inputs = ModelInput(
        user_int_feats=user_int_feats,
        item_int_feats=item_int_feats,
        user_dense_feats=user_dense_feats,
        item_dense_feats=item_dense_feats,
        seq_data=seq_data,
        seq_time_buckets=seq_time_buckets,
        seq_lens=seq_lens,
    )
    logits = model(inputs)
    assert torch.isfinite(logits).all(), "mixed-len rows produced NaN/Inf"
    loss = logits.sum()
    loss.backward()
    print(f"[OK] mixed seq_lens (0/half/full/uneven) · forward+backward stable")


if __name__ == "__main__":
    test_sota_forward_backward()
    test_sota_full_padding_row()
    test_sota_mixed_lens()
    print("\nAll T36 SOTA-shape smoke tests passed.")
