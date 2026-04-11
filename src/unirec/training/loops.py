from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from unirec.eval.offline_report import build_offline_report
from unirec.tasks.cvr import compute_cvr_loss
from unirec.utils.io import ensure_dir


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(sub_value, device) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = compute_cvr_loss(outputs, batch["labels"])
        loss.backward()
        optimizer.step()

        batch_size = int(batch["labels"].shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    labels: list[float] = []
    preds: list[float] = []
    sequence_lengths: list[int] = []

    for batch in dataloader:
        batch = move_to_device(batch, device)
        outputs = model(batch)
        labels.extend(batch["labels"].detach().cpu().tolist())
        preds.extend(outputs["probs"].detach().cpu().tolist())
        sequence_lengths.extend(batch["context"]["sequence_length_total"].detach().cpu().tolist())

    return build_offline_report(labels, preds, sequence_lengths, eval_config)


def save_checkpoint(model: torch.nn.Module, output_dir: str | Path, filename: str = "model.pt") -> str:
    output_dir = ensure_dir(output_dir)
    checkpoint_path = Path(output_dir) / filename
    torch.save(model.state_dict(), checkpoint_path)
    return str(checkpoint_path)
