from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unirec.datasets.parquet_dataset import build_dataloader, build_split_dataset, load_parquet_rows
from unirec.datasets.splits import time_based_split
from unirec.models.registry import build_model
from unirec.training.loops import evaluate_model
from unirec.utils.io import resolve_default_config
from unirec.utils.logging import get_logger


LOGGER = get_logger("eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TAAC2026 sequence CVR baseline")
    parser.add_argument("--config-root", default=str(ROOT / "configs"))
    parser.add_argument("--data-path", required=True, help="Local parquet file path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_default_config(args.config_root)
    config["data"]["path"] = args.data_path

    rows, metadata = load_parquet_rows(config)
    config = metadata["runtime_config"]
    splits = time_based_split(
        rows=rows,
        timestamp_column=config["data"]["timestamp_column"],
        train_ratio=float(config["data"]["split"]["train_ratio"]),
        valid_ratio=float(config["data"]["split"]["valid_ratio"]),
        test_ratio=float(config["data"]["split"]["test_ratio"]),
        order=str(config["data"]["split"].get("order", "ascending")),
    )
    dataset = build_split_dataset(rows, splits[args.split])
    dataloader = build_dataloader(dataset, config, shuffle=False)

    device = torch.device(config["train"]["device"])
    model = build_model(config["model"], config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    report = evaluate_model(model, dataloader, device, config["eval"])
    LOGGER.info("%s split report: %s", args.split, report)


if __name__ == "__main__":
    main()
