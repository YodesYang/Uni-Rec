"""Evaluation entry point for PCVRHyFormer.

This mirrors the training-side model and dataset construction from
``train.py`` and is required for checkpoints trained with synthetic dense
features, because platform default inference code may not know how to rebuild
those in-memory features.
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import PCVRParquetDataset, FeatureSchema, NUM_TIME_BUCKETS, parse_bucket_boundaries, parse_int_csv
from model import ModelInput, PCVRHyFormer


def _build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for _, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(raw: Any) -> Dict[str, int]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    out: Dict[str, int] = {}
    for pair in str(raw).split(","):
        if not pair.strip():
            continue
        key, value = pair.split(":")
        out[key.strip()] = int(value.strip())
    return out


def _find_model_dir(model_output_path: str) -> str:
    root = Path(model_output_path)
    if (
        (root / "model.pt").exists()
        and (root / "train_config.json").exists()
        and (root / "schema.json").exists()
    ):
        return str(root)

    candidates = []
    for model_path in glob.glob(str(root / "**" / "model.pt"), recursive=True):
        model_dir = Path(model_path).parent
        if (model_dir / "train_config.json").exists() and (model_dir / "schema.json").exists():
            candidates.append(model_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No model.pt + train_config.json + schema.json found under {root}")

    candidates.sort(key=lambda p: (".best_model" not in p.name, str(p)))
    return str(candidates[0])


def _resolve_ns_groups_path(model_dir: str, cfg: Dict[str, Any]) -> str:
    raw = cfg.get("ns_groups_json") or ""
    if not raw:
        return ""
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    candidates.append(os.path.join(model_dir, raw))
    candidates.append(os.path.join(model_dir, os.path.basename(raw)))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _resolve_history_cvr_path(model_dir: str, cfg: Dict[str, Any]) -> str:
    raw = cfg.get("history_cvr_cache_path") or ""
    if not raw:
        return ""
    candidates = []
    if os.path.isabs(raw):
        candidates.append(raw)
    candidates.append(os.path.join(model_dir, raw))
    candidates.append(os.path.join(model_dir, os.path.basename(raw)))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _parse_disable_seq_fids(raw: Any) -> Optional[List[int]]:
    """Parse ``disable_seq_fids`` from ``train_config.json``.

    Accepts a comma-separated string (what ``train.py`` writes) or a
    list (future-compat). Returns ``None`` when no fids are requested
    so ``PCVRParquetDataset`` takes its default "no-op" branch and we
    stay bit-identical to pre-T23 infer behaviour.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        parts = [s.strip() for s in raw.split(",") if s.strip()]
        if not parts:
            return None
        return [int(p) for p in parts]
    if isinstance(raw, (list, tuple)):
        out = [int(x) for x in raw]
        return out or None
    raise ValueError(
        f"Unexpected type for disable_seq_fids: {type(raw).__name__}")


def _load_ns_groups(
    path: str,
    user_schema: FeatureSchema,
    item_schema: FeatureSchema,
) -> Tuple[List[List[int]], List[List[int]]]:
    if not path:
        return (
            [[i] for i in range(len(user_schema.entries))],
            [[i] for i in range(len(item_schema.entries))],
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    user_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(user_schema.entries)
    }
    item_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(item_schema.entries)
    }
    user_groups = [
        [user_fid_to_idx[f] for f in fids]
        for fids in cfg["user_ns_groups"].values()
    ]
    item_groups = [
        [item_fid_to_idx[f] for f in fids]
        for fids in cfg["item_ns_groups"].values()
    ]
    return user_groups, item_groups


def _make_model_input(batch: Dict[str, Any], device: torch.device) -> ModelInput:
    seq_domains = batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = batch[domain].to(device, non_blocking=True)
        seq_lens[domain] = batch[f"{domain}_len"].to(device, non_blocking=True)
        time_key = f"{domain}_time_bucket"
        if time_key in batch:
            seq_time_buckets[domain] = batch[time_key].to(device, non_blocking=True)
        else:
            bsz = batch[domain].shape[0]
            seq_len = batch[domain].shape[2]
            seq_time_buckets[domain] = torch.zeros(
                bsz, seq_len, dtype=torch.long, device=device)

    return ModelInput(
        user_int_feats=batch["user_int_feats"].to(device, non_blocking=True),
        item_int_feats=batch["item_int_feats"].to(device, non_blocking=True),
        user_dense_feats=batch["user_dense_feats"].to(device, non_blocking=True),
        item_dense_feats=batch["item_dense_feats"].to(device, non_blocking=True),
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def _load_state_dict(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        sd = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location=device)
    # Strip "_orig_mod." prefix if present. Such prefix appears when a ckpt
    # was saved from a torch.compile()-wrapped model (ADR-008 T27 code had
    # an early bug where EarlyStopping received the compiled wrapper and
    # torch.save'd its state_dict directly). infer.py itself never compiles,
    # so its model.state_dict() keys are always un-prefixed; renaming at
    # load time keeps ancient ckpts evaluate-able without retraining.
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()
        }
    return sd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model_output_path = os.environ["MODEL_OUTPUT_PATH"]
    eval_data_path = os.environ["EVAL_DATA_PATH"]
    eval_result_path = os.environ["EVAL_RESULT_PATH"]

    model_dir = _find_model_dir(model_output_path)
    cfg_path = os.path.join(model_dir, "train_config.json")
    schema_path = os.path.join(model_dir, "schema.json")
    model_path = os.path.join(model_dir, "model.pt")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    logging.info(f"Using model dir: {model_dir}")
    logging.info(f"Loaded train_config from {cfg_path}")
    logging.info(f"Using schema: {schema_path}")

    seq_max_lens = _parse_seq_max_lens(cfg.get("seq_max_lens"))
    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(os.environ.get("EVAL_NUM_WORKERS", cfg.get("num_workers", 8)))

    # ── Test-set EDA dump (default ON · 2026-05-14 user decision) ──────
    # Always collect test-set EDA stats (no labels) and emit base64+gzip
    # blob to stdout at end. Used to study train↔test covariate shift.
    # Cost: forces num_workers=0 for clean accumulation, ~+5-10% wall-time.
    # Override: set EDA_BLOB=0 in env to disable (e.g. for time-critical
    # evaluation where every minute matters; ddl day fallback).
    # Quota: 0 extra (piggybacks on regular evaluation submission).
    enable_test_eda = os.environ.get("EDA_BLOB", "1") != "0"
    if enable_test_eda and num_workers > 0:
        logging.warning(
            "EDA_BLOB=1 (default) forces num_workers=0 for inference; "
            "dataset._eda_state is only updated in the main process. "
            "Wall-time impact estimated +5~10%."
        )
        num_workers = 0

    dataset = PCVRParquetDataset(
        parquet_path=eval_data_path,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        clip_vocab=bool(cfg.get("clip_vocab", True)),
        is_training=False,
        enable_count_features=bool(cfg.get("enable_count_features", False)),
        enable_seq_stats_features=bool(cfg.get("enable_seq_stats_features", False)),
        enable_history_cvr_features=bool(cfg.get("enable_history_cvr_features", False)),
        history_cvr_cache_path=_resolve_history_cvr_path(model_dir, cfg),
        history_cvr_time_mode=str(cfg.get("history_cvr_time_mode", "timestamp_cutoff")),
        history_cvr_cutoff_sec=int(cfg.get("history_cvr_cutoff_sec", 86400)),
        history_cvr_available_lag_sec=int(cfg.get("history_cvr_available_lag_sec", 0)),
        history_cvr_prior_strength=float(cfg.get("history_cvr_prior_strength", 20.0)),
        enable_time_of_day_features=bool(cfg.get("enable_time_of_day_features", False)),
        enable_hour_only_features=bool(cfg.get("enable_hour_only_features", False)),
        disable_seq_fids=_parse_disable_seq_fids(cfg.get("disable_seq_fids", "")),
        bucket_boundaries=parse_bucket_boundaries(cfg.get("time_bucket_boundaries", "")),
        enable_is_missing=bool(cfg.get("enable_is_missing", False)),
        is_missing_user_int_fids=parse_int_csv(cfg.get("is_missing_user_int_fids", "")),
        is_missing_item_int_fids=parse_int_csv(cfg.get("is_missing_item_int_fids", "")),
        enable_eda_dump=enable_test_eda,
    )

    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )

    user_ns_groups, item_ns_groups = _load_ns_groups(
        _resolve_ns_groups_path(model_dir, cfg),
        dataset.user_int_schema,
        dataset.item_int_schema,
    )

    model_args = {
        "user_int_feature_specs": _build_feature_specs(
            dataset.user_int_schema, dataset.user_int_vocab_sizes),
        "item_int_feature_specs": _build_feature_specs(
            dataset.item_int_schema, dataset.item_int_vocab_sizes),
        "user_dense_dim": dataset.user_dense_schema.total_dim,
        "item_dense_dim": dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": int(cfg.get("d_model", 64)),
        "emb_dim": int(cfg.get("emb_dim", 64)),
        "num_queries": int(cfg.get("num_queries", 1)),
        "num_hyformer_blocks": int(cfg.get("num_hyformer_blocks", 2)),
        "num_heads": int(cfg.get("num_heads", 4)),
        "seq_encoder_type": cfg.get("seq_encoder_type", "transformer"),
        "hidden_mult": int(cfg.get("hidden_mult", 4)),
        "dropout_rate": float(cfg.get("dropout_rate", 0.01)),
        "seq_top_k": int(cfg.get("seq_top_k", 50)),
        "seq_causal": bool(cfg.get("seq_causal", False)),
        "action_num": int(cfg.get("action_num", 1)),
        "num_time_buckets": NUM_TIME_BUCKETS if cfg.get("use_time_buckets", True) else 0,
        "rank_mixer_mode": cfg.get("rank_mixer_mode", "full"),
        "rank_mixer_ffn_mode": cfg.get("rank_mixer_ffn_mode", "shared"),
        "use_rope": bool(cfg.get("use_rope", False)),
        "rope_base": float(cfg.get("rope_base", 10000.0)),
        "emb_skip_threshold": int(cfg.get("emb_skip_threshold", 0)),
        "seq_id_threshold": int(cfg.get("seq_id_threshold", 10000)),
        "ns_tokenizer_type": cfg.get("ns_tokenizer_type", "rankmixer"),
        "user_ns_tokens": int(cfg.get("user_ns_tokens", 0)),
        "item_ns_tokens": int(cfg.get("item_ns_tokens", 0)),
        "use_onetrans": bool(cfg.get("use_onetrans", False)),
        "onetrans_top_k": int(cfg.get("onetrans_top_k", 50)),
        "seq_hash_vocab": int(cfg.get("seq_hash_vocab", 0)),
        "item_conditioned_query": bool(cfg.get("item_conditioned_query", False)),
        "target_item_seq_injection": str(cfg.get("target_item_seq_injection", "off")),
        "target_inject_alpha_init": float(cfg.get("target_inject_alpha_init", 0.0)),
        "enable_dense_bypass": bool(cfg.get("enable_dense_bypass", False)),
        "enable_din_interest": bool(cfg.get("enable_din_interest", False)),
        "enable_din_integrated": bool(cfg.get("enable_din_integrated", False)),
        "din_integrated_alpha_init": float(cfg.get("din_integrated_alpha_init", 0.1)),
        "enable_nlir_gating": bool(cfg.get("enable_nlir_gating", False)),
        "enable_fafe": bool(cfg.get("enable_fafe", False)),
        "enable_dcn_cross": bool(cfg.get("enable_dcn_cross", False)),
        # T34: cfg stores LIST INDICES (already mapped from raw schema fids
        # by train.py before saving train_config.json). infer.py forwards
        # them as-is — no re-mapping needed because the model's
        # user_int_feature_specs ordering is identical between train and
        # infer (both derive from schema.json::user_int).
        "dcn_cross_user_fids": cfg.get("dcn_cross_user_fids"),
        "dcn_cross_item_fids": cfg.get("dcn_cross_item_fids"),
        "dcn_cross_layers": int(cfg.get("dcn_cross_layers", 2)),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Building PCVRHyFormer with cfg: {model_args}")
    model = PCVRHyFormer(**model_args).to(device)
    model.load_state_dict(_load_state_dict(model_path, device))
    model.eval()

    predictions: Dict[str, float] = {}
    seen_rows = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            model_input = _make_model_input(batch, device)
            logits, _ = model.predict(model_input)
            if logits.ndim == 2 and logits.shape[1] > 1:
                logits = logits[:, -1]
            else:
                logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs_list = probs.detach().cpu().tolist()
            for user_id, prob in zip(batch["user_id"], probs_list):
                predictions[str(user_id)] = float(prob)
            seen_rows += len(probs_list)
            if step % 100 == 0:
                logging.info(
                    f"Inference progress: step={step}, rows={seen_rows}, "
                    f"unique_user_ids={len(predictions)}")

    os.makedirs(eval_result_path, exist_ok=True)
    out_path = os.path.join(eval_result_path, "predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f)
    logging.info(
        f"Saved {len(predictions)} predictions from {seen_rows} rows to {out_path}")

    if enable_test_eda:
        _emit_test_eda_blob(dataset)


def _emit_test_eda_blob(dataset: PCVRParquetDataset) -> None:
    """Emit base64+gzip JSON blob of test-set EDA stats to stdout.

    Placed at the very END of stdout (after predictions are saved and after
    any other logging) so the platform Logs UI tail-1000-lines window
    contains the blob even if log volume is high. Sentinels make extraction
    by ``tools/decode_eda_blob.py`` reliable.

    No-ops if dataset has no EDA state (e.g. EDA_BLOB unset or empty
    inference). Wraps in try/except so any EDA failure NEVER fails the
    overall evaluation (predictions.json is the contract; EDA is bonus).
    """
    import base64
    import gzip
    import sys
    try:
        summary = dataset.finalize_eda()
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(f"[test-eda] finalize_eda failed: {exc!r}")
        return
    if summary is None:
        logging.warning("[test-eda] EDA_BLOB=1 but no EDA state collected")
        return
    try:
        payload = json.dumps(summary, sort_keys=True).encode("utf-8")
        gz = gzip.compress(payload, compresslevel=9)
        b64 = base64.b64encode(gz).decode("ascii")
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(f"[test-eda] blob encode failed: {exc!r}")
        return

    chunk_size = 120
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]

    sys.stdout.flush()
    print()
    print("<<<EDA_BLOB_START>>>")
    for ch in chunks:
        print(ch)
    print("<<<EDA_BLOB_END>>>")
    sys.stdout.flush()
    logging.info(
        f"[test-eda] blob emitted: {len(b64)} chars, "
        f"{len(chunks)} chunks, raw {len(payload)} bytes")


if __name__ == "__main__":
    main()
