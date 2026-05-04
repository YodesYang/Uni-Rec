"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--early_stop_metric', type=str, default='auc',
                        choices=['auc', 'logloss'],
                        help="Metric used for early stopping: 'auc' (maximize, default) "
                             "or 'logloss' (minimize). Using logloss may preserve "
                             "better-calibrated checkpoints that AUC would skip.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--limit_train_rgs', type=int, default=0,
                        help='Fast-proxy switch: cap train Row Groups after the '
                             'normal split, keeping the tail closest to the '
                             'validation cut (0 = disabled)')
    parser.add_argument('--limit_valid_rgs', type=int, default=0,
                        help='Fast-proxy switch: cap validation Row Groups after '
                             'the normal split, keeping the first N validation '
                             'Row Groups (0 = disabled)')
    parser.add_argument('--valid_split_strategy', type=str, default='filename',
                        choices=['filename', 'time'],
                        help='How to order Row Groups before taking the tail '
                             'as validation: '
                             "'filename' (default, behavior-preserving) keeps "
                             'the sorted(glob)+RG-index order assumed by the '
                             'baseline; '
                             "'time' sorts RGs by max(timestamp) ASC so the "
                             'validation tail is strictly later than train, '
                             'matching the TAAC 2026 test-set time gap.')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--max_train_steps', type=int, default=0,
                        help='Fast-proxy switch: stop after N optimizer steps, '
                             'run a final validation, and save best checkpoint '
                             'if improved (0 = disabled/full training)')
    parser.add_argument('--enable_row_time_cutoff', action='store_true', default=False,
                        help='T8.5 switch: after the Row Group split, filter rows '
                             'by the validation minimum timestamp so train keeps '
                             'timestamp < cutoff and valid keeps timestamp >= cutoff '
                             '(default off)')
    parser.add_argument('--enable_count_features', action='store_true', default=False,
                        help='P0-A switch: append user/item non-zero feature counts '
                             'to user dense features (default off)')
    parser.add_argument('--enable_seq_stats_features', action='store_true', default=False,
                        help='P0-B switch: append per-domain sequence length and '
                             'timestamp-range features to user dense features '
                             '(default off)')
    parser.add_argument('--enable_history_cvr_features', action='store_true', default=False,
                        help='P0-C switch: append leak-safe historical item CVR '
                             'features from a training-only sidecar (default off)')
    parser.add_argument('--history_cvr_cache_path', type=str, default=None,
                        help='Path for the generated P0-C history CVR sidecar '
                             '(defaults to <log_dir>/history_cvr_stats.npz)')
    parser.add_argument('--history_cvr_item_fids', type=str, default='scalar',
                        help="Item int fids for P0-C history CVR features. "
                             "'scalar' means all scalar item_int fids; or pass "
                             "a comma-separated list such as '5,7,12,16'.")
    parser.add_argument('--history_cvr_bin_sec', type=int, default=3600,
                        help='Time-bin width for cumulative P0-C history stats')
    parser.add_argument('--history_cvr_cutoff_sec', type=int, default=86400,
                        help='Only use history rows at least this many seconds '
                             'older than the current row')
    parser.add_argument('--history_cvr_time_mode', type=str,
                        default='timestamp_cutoff',
                        choices=['timestamp_cutoff', 'available'],
                        help='History-CVR time semantics: timestamp_cutoff bins '
                             'rows by event timestamp; available bins positives '
                             'by label_time and negatives by matured availability')
    parser.add_argument('--history_cvr_available_lag_sec', type=int, default=0,
                        help='Additional lookup lag in seconds when '
                             '--history_cvr_time_mode=available')
    parser.add_argument('--history_cvr_prior_strength', type=float, default=20.0,
                        help='Smoothing strength for historical CVR rates')
    parser.add_argument('--enable_mature_negative_weighting',
                        action='store_true', default=False,
                        help='Delayed-feedback switch: downweight negatives '
                             'whose conversion window has not fully elapsed '
                             'by the train observation end timestamp')
    parser.add_argument('--negative_maturity_sec', type=int, default=86400,
                        help='Conversion observation window used by available '
                             'history-CVR and mature-negative weighting')
    parser.add_argument('--immature_negative_weight', type=float, default=0.0,
                        help='Sample weight for immature negatives; 0 drops '
                             'them from the conversion loss')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--multi_task_loss', action='store_true', default=False,
                        help='P1 switch: train action_num=2 heads as '
                             'click(label_type>=1) + conversion(label_type==2); '
                             'inference/eval use the conversion head')
    parser.add_argument('--click_loss_weight', type=float, default=1.0,
                        help='Loss weight for the click head when '
                             '--multi_task_loss is enabled')
    parser.add_argument('--conversion_loss_weight', type=float, default=2.0,
                        help='Loss weight for the conversion head when '
                             '--multi_task_loss is enabled')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--dense_log1p_fids', type=str, default='62,63,64,65,66',
                        help='Comma-separated list of user_dense fids whose raw values '
                             'are large-magnitude counts/frequencies and should be '
                             'compressed with log1p before the dense projection layer. '
                             'EDA confirmed fids 62-66 store raw counts (10^3-10^5) '
                             'while other dense columns (fid 61/87) are pre-normalised '
                             'embeddings. Pass "" to disable all scaling.')
    parser.add_argument('--enable_time_of_day_features', action='store_true', default=False,
                        help='Append 5 synthetic dense features derived from the impression '
                             'timestamp: hour_sin, hour_cos (hour-of-day circular encoding), '
                             'dow_sin, dow_cos (day-of-week circular encoding), is_weekend. '
                             'Captures diurnal and weekly purchase-intent periodicity absent '
                             'from existing sequence features. (T12)')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    parser.add_argument('--use_onetrans', action='store_true', default=False,
                        help='Replace HyFormer query-decoding blocks with OneTrans '
                             'Mixed-Parameterization unified attention. When enabled, '
                             'seq_encoder_type is used for the per-domain lightweight '
                             'SwiGLU pre-encoder; the num_queries and rank_mixer_mode '
                             'flags are ignored.')
    parser.add_argument('--onetrans_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept per domain before '
                             'the unified OneTrans attention (default 50)')
    parser.add_argument('--seq_hash_vocab', type=int, default=0,
                        help='T13: when > 0, sequence features with vocab > '
                             'emb_skip_threshold use a hash embedding of this size '
                             'instead of a zero vector. Values are mapped via '
                             'value %% seq_hash_vocab before embedding lookup. '
                             '0 = disabled (default).')
    parser.add_argument('--item_conditioned_query', action='store_true', default=False,
                        help='Enable item-conditioned query generation (DIN-style). '
                             'Each sequence domain learns a gate that blends the standard '
                             'global-info query with a target-item-aware query branch, '
                             'letting the model focus user history on the target ad.')

    parser.add_argument('--temporal_weight_alpha', type=float, default=0.0,
                        help='Recency reweighting: give samples closer to the test period '
                             'higher training weight via weight=exp(alpha*norm_ts). '
                             'norm_ts is linearly scaled to [0,1] over the training split. '
                             '0 = disabled (default); 2~4 = moderate upweighting of '
                             'recent samples; useful for reducing train/test temporal gap.')
    parser.add_argument('--enable_hour_only_features', action='store_true', default=False,
                        help='Append hour-of-day sin/cos (2 dims) to user_dense. '
                             'Captures diurnal periodicity WITHOUT day-of-week, which '
                             'avoids overfitting to the weekday distribution of training '
                             'data (train is dominated by a single weekday ~63%%). '
                             'Use this instead of --enable_time_of_day_features when '
                             'DOW features hurt leaderboard generalization.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()

    for name in (
        'limit_train_rgs',
        'limit_valid_rgs',
        'max_train_steps',
        'history_cvr_cutoff_sec',
        'history_cvr_available_lag_sec',
        'negative_maturity_sec',
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be >= 0")
    if args.history_cvr_bin_sec <= 0:
        parser.error("--history_cvr_bin_sec must be > 0")
    if args.history_cvr_prior_strength < 0:
        parser.error("--history_cvr_prior_strength must be >= 0")
    if args.immature_negative_weight < 0:
        parser.error("--immature_negative_weight must be >= 0")

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    # TRAIN_LOG_PATH is project-internal (not part of the platform's
    # documented env var set); fall back to a writable temp dir so the
    # script does not crash when neither env nor --log_dir is provided.
    args.log_dir = (os.environ.get('TRAIN_LOG_PATH', args.log_dir)
                    or '/tmp/taac_train_log')
    args.tf_events_dir = (
        os.environ.get('TRAIN_TF_EVENTS_PATH')
        or os.path.join(args.log_dir, 'tb')
    )
    if args.enable_history_cvr_features and not args.history_cvr_cache_path:
        args.history_cvr_cache_path = os.path.join(
            args.log_dir, 'history_cvr_stats.npz')
    if args.multi_task_loss:
        args.action_num = 2

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
        valid_split_strategy=args.valid_split_strategy,
        limit_train_rgs=args.limit_train_rgs,
        limit_valid_rgs=args.limit_valid_rgs,
        enable_row_time_cutoff=args.enable_row_time_cutoff,
        enable_count_features=args.enable_count_features,
        enable_seq_stats_features=args.enable_seq_stats_features,
        enable_history_cvr_features=args.enable_history_cvr_features,
        history_cvr_cache_path=args.history_cvr_cache_path,
        history_cvr_item_fids=args.history_cvr_item_fids,
        history_cvr_bin_sec=args.history_cvr_bin_sec,
        history_cvr_cutoff_sec=args.history_cvr_cutoff_sec,
        history_cvr_time_mode=args.history_cvr_time_mode,
        history_cvr_available_lag_sec=args.history_cvr_available_lag_sec,
        history_cvr_prior_strength=args.history_cvr_prior_strength,
        enable_mature_negative_weighting=args.enable_mature_negative_weighting,
        negative_maturity_sec=args.negative_maturity_sec,
        immature_negative_weight=args.immature_negative_weight,
        dense_log1p_fids=frozenset(
            int(x) for x in args.dense_log1p_fids.split(',') if x.strip()
        ),
        enable_time_of_day_features=args.enable_time_of_day_features,
        temporal_weight_alpha=args.temporal_weight_alpha,
        enable_hour_only_features=args.enable_hour_only_features,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
        "use_onetrans": args.use_onetrans,
        "onetrans_top_k": args.onetrans_top_k,
        "seq_hash_vocab": args.seq_hash_vocab,
        "item_conditioned_query": args.item_conditioned_query,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    es_mode = 'min' if args.early_stop_metric == 'logloss' else 'max'
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
        mode=es_mode,
    )
    logging.info(f"EarlyStopping metric={args.early_stop_metric}, mode={es_mode}")

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        multi_task_loss=args.multi_task_loss,
        click_loss_weight=args.click_loss_weight,
        conversion_loss_weight=args.conversion_loss_weight,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        max_train_steps=args.max_train_steps,
        train_config=vars(args),
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
