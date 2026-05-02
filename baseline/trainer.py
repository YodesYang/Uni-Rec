"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import os
import glob
import shutil
import logging
import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = 'bce',
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        max_train_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model: nn.Module = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})")
            logging.info(f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})")
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params, lr=sparse_lr, weight_decay=sparse_weight_decay
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params, lr=lr, betas=(0.9, 0.98)
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=(0.9, 0.98)
            )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.max_train_steps: int = max_train_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.validation_history = []
        self.best_epoch: Optional[int] = None
        self.best_global_step: Optional[int] = None
        self.best_val_auc: Optional[float] = None
        self.best_val_logloss: Optional[float] = None
        self.best_checkpoint_dir: Optional[str] = None
        self.last_epoch: Optional[int] = None
        self.last_global_step: int = 0
        self.last_val_auc: Optional[float] = None
        self.last_val_logloss: Optional[float] = None

        logging.info(f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
                     f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
                     f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}")
        if self.max_train_steps < 0:
            raise ValueError("max_train_steps must be >= 0")
        if self.max_train_steps > 0:
            logging.info(
                f"Fast proxy enabled: max_train_steps={self.max_train_steps}")

    def _record_validation_result(
        self,
        epoch: int,
        total_step: int,
        val_auc: float,
        val_logloss: float,
        previous_best: Optional[float],
    ) -> None:
        """Track compact validation history for end-of-run summaries."""
        is_new_best = (
            self.early_stopping.best_score is not None
            and (previous_best is None
                 or self.early_stopping.best_score != previous_best)
        )
        row = {
            "epoch": int(epoch),
            "global_step": int(total_step),
            "val_auc": float(val_auc),
            "val_logloss": float(val_logloss),
            "is_best": bool(is_new_best),
        }
        self.validation_history.append(row)
        self.last_epoch = int(epoch)
        self.last_global_step = int(total_step)
        self.last_val_auc = float(val_auc)
        self.last_val_logloss = float(val_logloss)
        if is_new_best:
            self.best_epoch = int(epoch)
            self.best_global_step = int(total_step)
            self.best_val_auc = float(val_auc)
            self.best_val_logloss = float(val_logloss)
            self.best_checkpoint_dir = os.path.dirname(
                self.early_stopping.checkpoint_path)

    def _write_training_summary(self, stop_reason: str) -> None:
        """Emit a grep-friendly summary and persist it next to logs/checkpoints."""
        summary = {
            "stop_reason": stop_reason,
            "best_val_auc": self.best_val_auc,
            "best_val_logloss": self.best_val_logloss,
            "best_epoch": self.best_epoch,
            "best_global_step": self.best_global_step,
            "best_checkpoint_dir": self.best_checkpoint_dir,
            "last_val_auc": self.last_val_auc,
            "last_val_logloss": self.last_val_logloss,
            "last_epoch": self.last_epoch,
            "last_global_step": self.last_global_step,
            "early_stopping_counter": self.early_stopping.counter,
            "early_stopping_patience": self.early_stopping.patience,
            "validation_history": self.validation_history,
            "key_train_config": {},
        }
        if self.train_config:
            key_names = (
                "seq_encoder_type", "valid_split_strategy", "valid_ratio",
                "train_ratio", "limit_train_rgs", "limit_valid_rgs",
                "max_train_steps", "d_model", "num_queries",
                "dropout_rate", "loss_type", "batch_size", "num_workers",
                "seed",
            )
            summary["key_train_config"] = {
                k: self.train_config.get(k) for k in key_names
                if k in self.train_config
            }

        logging.info("[summary] ===== Training Summary =====")
        logging.info(f"[summary] stop_reason={stop_reason}")
        logging.info(
            f"[summary] best_val_AUC={self.best_val_auc} "
            f"best_val_logloss={self.best_val_logloss} "
            f"best_epoch={self.best_epoch} "
            f"best_global_step={self.best_global_step}")
        logging.info(
            f"[summary] last_val_AUC={self.last_val_auc} "
            f"last_val_logloss={self.last_val_logloss} "
            f"last_epoch={self.last_epoch} "
            f"last_global_step={self.last_global_step}")
        logging.info(f"[summary] best_checkpoint_dir={self.best_checkpoint_dir}")
        if self.validation_history:
            compact = ", ".join(
                f"E{r['epoch']}@{r['global_step']}:"
                f"{r['val_auc']:.6f}"
                f"{'*' if r['is_best'] else ''}"
                for r in self.validation_history
            )
            logging.info(f"[summary] val_auc_history={compact}")
        logging.info("[summary] ===== End Training Summary =====")

        out_paths = [os.path.join(self.save_dir, "training_summary.json")]
        if self.train_config and self.train_config.get("log_dir"):
            out_paths.append(os.path.join(
                self.train_config["log_dir"], "training_summary.json"))
        for path in out_paths:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)
                logging.info(f"[summary] wrote {path}")
            except Exception as e:
                logging.warning(f"[summary] failed to write {path}: {e}")

    def _build_step_dir_name(
        self,
        global_step: int,
        is_best: bool = False,
        val_auc: Optional[float] = None,
    ) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.auc=0.865000][.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        if is_best and val_auc is not None:
            # Platform checkpoint names must be <=300 chars and only use
            # alnum, "_", "-", "=", ".". Fixed precision is readable and
            # stable enough for selecting the best local-validation ckpt.
            parts.append(f"auc={val_auc:.6f}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json
            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump['ns_groups_json'] = os.path.basename(
                    self.ns_groups_path)
            with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        val_auc: Optional[float] = None,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            val_auc: optional validation AUC included in best checkpoint names.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(
            global_step, is_best=is_best, val_auc=val_auc)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        old_best = self.early_stopping.best_score
        is_likely_new_best = (
            old_best is None
            or val_auc > old_best + self.early_stopping.delta
        )
        if not is_likely_new_best:
            # No new best anticipated: leave disk untouched. The previous
            # best_model dir (with its model.pt + sidecars) remains valid.
            self.early_stopping(val_auc, self.model, {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            })
            return

        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(
                total_step, is_best=True, val_auc=val_auc),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        self._remove_old_best_dirs()

        self.early_stopping(val_auc, self.model, {
            "best_val_AUC": val_auc,
            "best_val_logloss": val_logloss,
        })

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if self.early_stopping.best_score != old_best and os.path.exists(
            self.early_stopping.checkpoint_path
        ):
            self._save_step_checkpoint(
                total_step, is_best=True, val_auc=val_auc,
                skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                              dynamic_ncols=True)
            loss_sum = 0.0
            epoch_steps = 0

            for step, batch in train_pbar:
                loss = self._train_step(batch)
                total_step += 1
                epoch_steps += 1
                loss_sum += loss
                validated_this_step = False

                if self.writer:
                    self.writer.add_scalar('Loss/train', loss, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if self.eval_every_n_steps > 0 and total_step % self.eval_every_n_steps == 0:
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

                    if self.writer:
                        self.writer.add_scalar('AUC/valid', val_auc, total_step)
                        self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                    previous_best = self.early_stopping.best_score
                    self._handle_validation_result(total_step, val_auc, val_logloss)
                    self._record_validation_result(
                        epoch, total_step, val_auc, val_logloss, previous_best)
                    validated_this_step = True

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        self._write_training_summary(
                            f"early_stop_step_{total_step}")
                        return

                if self.max_train_steps > 0 and total_step >= self.max_train_steps:
                    avg_loss = loss_sum / max(1, epoch_steps)
                    logging.info(
                        f"Reached max_train_steps={self.max_train_steps} at "
                        f"epoch {epoch}, step {step + 1}; "
                        f"Average Loss: {avg_loss}")

                    if not validated_this_step:
                        val_auc, val_logloss = self.evaluate(epoch=epoch)
                        self.model.train()
                        torch.cuda.empty_cache()

                        logging.info(
                            f"Max-step Validation | AUC: {val_auc}, "
                            f"LogLoss: {val_logloss}")

                        if self.writer:
                            self.writer.add_scalar('AUC/valid', val_auc, total_step)
                            self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

                        previous_best = self.early_stopping.best_score
                        self._handle_validation_result(
                            total_step, val_auc, val_logloss)
                        self._record_validation_result(
                            epoch, total_step, val_auc, val_logloss,
                            previous_best)

                    logging.info(
                        f"Stopping training after max_train_steps="
                        f"{self.max_train_steps}")
                    self._write_training_summary(
                        f"max_train_steps_{self.max_train_steps}")
                    return

            logging.info(
                f"Epoch {epoch}, Average Loss: "
                f"{loss_sum / max(1, epoch_steps)}")

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}")

            if self.writer:
                self.writer.add_scalar('AUC/valid', val_auc, total_step)
                self.writer.add_scalar('LogLoss/valid', val_logloss, total_step)

            previous_best = self.early_stopping.best_score
            self._handle_validation_result(total_step, val_auc, val_logloss)
            self._record_validation_result(
                epoch, total_step, val_auc, val_logloss, previous_best)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                self._write_training_summary(f"early_stop_epoch_{epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if epoch >= self.reinit_sparse_after_epoch and self.sparse_optimizer is not None:
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group['params']:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(self.reinit_cardinality_threshold)
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params, lr=self.sparse_lr, weight_decay=self.sparse_weight_decay
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                             f"restored optimizer state for {restored} low-cardinality params")
        else:
            self._write_training_summary("max_epochs_completed")

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch['_seq_domains']
        seq_data: Dict[str, torch.Tensor] = {}
        seq_lens: Dict[str, torch.Tensor] = {}
        seq_time_buckets: Dict[str, torch.Tensor] = {}
        for domain in seq_domains:
            seq_data[domain] = device_batch[domain]
            seq_lens[domain] = device_batch[f'{domain}_len']
            B = device_batch[domain].shape[0]
            L = device_batch[domain].shape[2]
            seq_time_buckets[domain] = device_batch.get(
                f'{domain}_time_bucket',
                torch.zeros(B, L, dtype=torch.long, device=self.device))
        return ModelInput(
            user_int_feats=device_batch['user_int_feats'],
            item_int_feats=device_batch['item_int_feats'],
            user_dense_feats=device_batch['user_dense_feats'],
            item_dense_feats=device_batch['item_dense_feats'],
            seq_data=seq_data,
            seq_lens=seq_lens,
            seq_time_buckets=seq_time_buckets,
        )

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        # foreach=False: avoids a PyTorch _foreach_norm CUDA kernel bug observed
        # with certain tensor shapes in this project.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, foreach=False)

        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

        return loss.item()

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        # Per-row diagnostic side-channel (only populated if dataset emits the
        # _diag_* fields; safe to be empty otherwise).
        diag_keys = ('_diag_user_int_nz', '_diag_item_int_nz',
                     '_diag_seq_total_len', '_diag_label_type_raw')
        all_diag: Dict[str, list] = {k: [] for k in diag_keys}

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                for k in diag_keys:
                    if k in batch:
                        all_diag[k].append(batch[k].numpy())

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
        else:
            logloss = float('inf')

        # Slice diagnostics (best-effort; never blocks the main metric).
        if any(len(v) > 0 for v in all_diag.values()):
            try:
                diag_concat = {k: np.concatenate(v) for k, v in all_diag.items() if len(v) > 0}
                if nan_mask.any():
                    keep = ~nan_mask
                    diag_concat = {k: arr[keep] for k, arr in diag_concat.items()}
                self._slice_diagnose(probs, labels_np, diag_concat)
            except Exception as e:
                logging.warning(f"[diag] slice diagnose failed: {e}")

        return auc, logloss

    def _slice_diagnose(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        diag: Dict[str, np.ndarray],
    ) -> None:
        """Log per-slice (n / pos_rate / AUC / pred_mean) tables for valid set.

        Slices:
        - ``user_int_nonzero`` (4-quantile): how many user_int features are
          non-padding. Detects how the model handles cold/sparse users.
        - ``item_int_nonzero`` (4-quantile): same on the item side.
        - ``seq_total_len`` (4-quantile): sum of seq lengths across all 4
          domains. Detects how the model handles low-activity users.
        - ``label_type`` (categorical {0,1,2}): per-class AUC and base rate.
          Sanity-checks that label_type==2 is the only positive class.

        AUC bucket-vs-overall delta > 0.02 typically indicates a slice the
        model is materially worse on, i.e. a feature-engineering target.
        """
        n_total = len(probs)
        if n_total == 0:
            return

        header = (f"  {'bucket':<28s} {'n':>10s} {'pos_rate':>10s} "
                  f"{'AUC':>8s} {'pred_mean':>10s}")

        def _row(name: str, mask: np.ndarray) -> str:
            n = int(mask.sum())
            if n == 0:
                return f"  {name:<28s} {n:>10d}"
            p, s = labels[mask], probs[mask]
            pos_rate = float(p.mean())
            pred_mean = float(s.mean())
            if len(np.unique(p)) < 2:
                auc_str = '     nan'
            else:
                auc_str = f"{roc_auc_score(p, s):>8.4f}"
            return (f"  {name:<28s} {n:>10d} {pos_rate:>10.4f} "
                    f"{auc_str} {pred_mean:>10.4f}")

        def _print_quantile(name: str, arr: np.ndarray, n_buckets: int = 4) -> None:
            qs = np.unique(np.quantile(arr, np.linspace(0, 1, n_buckets + 1)))
            if len(qs) < 2:
                logging.info(f"[diag] {name}: degenerate (single value {qs[0]}), skipped")
                return
            # searchsorted on the interior cut points; side='right' so that
            # the maximum value lands in the last bucket.
            bucket = np.searchsorted(qs[1:-1], arr, side='right')
            lines = [f"[diag] slice by {name} ({len(qs) - 1}-quantile):", header]
            for b in range(len(qs) - 1):
                mask = bucket == b
                label = f"[{int(qs[b])},{int(qs[b + 1])}]"
                lines.append(_row(label, mask))
            logging.info('\n'.join(lines))

        if '_diag_user_int_nz' in diag:
            _print_quantile('user_int_nonzero', diag['_diag_user_int_nz'])
        if '_diag_item_int_nz' in diag:
            _print_quantile('item_int_nonzero', diag['_diag_item_int_nz'])
        if '_diag_seq_total_len' in diag:
            _print_quantile('seq_total_len', diag['_diag_seq_total_len'])
        if '_diag_label_type_raw' in diag:
            arr = diag['_diag_label_type_raw']
            lines = ["[diag] slice by label_type (raw 0/1/2):", header]
            for lt in (0, 1, 2):
                lines.append(_row(f"label_type={lt}", arr == lt))
            logging.info('\n'.join(lines))

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label']

        model_input = self._make_model_input(device_batch)
        logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
