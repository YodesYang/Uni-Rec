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
from unirec.training.loops import evaluate_model, save_checkpoint, train_one_epoch
from unirec.utils.io import ensure_dir, resolve_default_config, save_json, save_yaml
from unirec.utils.logging import get_logger
from unirec.utils.seed import seed_everything


LOGGER = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TAAC2026 sequence CVR baseline")
    parser.add_argument("--config-root", default=str(ROOT / "configs"))
    parser.add_argument("--data-path", default=None, help="Local parquet file path")
    parser.add_argument("--output-dir", default=None, help="Directory for checkpoints and reports")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_default_config(args.config_root)

    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.output_dir:
        config["train"]["output_dir"] = args.output_dir
    if args.max_rows is not None:
        config["data"]["max_rows"] = args.max_rows

    if not config["data"].get("path"):
        raise ValueError("--data-path is required for the first run because no default parquet is bundled")

    seed_everything(int(config["train"]["seed"]))
    output_dir = ensure_dir(ROOT / config["train"]["output_dir"])

    rows, metadata = load_parquet_rows(config)
    config = metadata["runtime_config"]
    save_yaml(output_dir / "resolved_config.yaml", config)
    splits = time_based_split(
        rows=rows,
        timestamp_column=config["data"]["timestamp_column"],
        train_ratio=float(config["data"]["split"]["train_ratio"]),
        valid_ratio=float(config["data"]["split"]["valid_ratio"]),
        test_ratio=float(config["data"]["split"]["test_ratio"]),
        order=str(config["data"]["split"].get("order", "ascending")),
    )

    train_dataset = build_split_dataset(rows, splits["train"])
    valid_dataset = build_split_dataset(rows, splits["valid"])
    test_dataset = build_split_dataset(rows, splits["test"])

    train_loader = build_dataloader(train_dataset, config, shuffle=True)
    valid_loader = build_dataloader(valid_dataset, config, shuffle=False)
    test_loader = build_dataloader(test_dataset, config, shuffle=False)

    device = torch.device(config["train"]["device"])
    model = build_model(config["model"], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    LOGGER.info("Loaded %s rows. train=%s valid=%s test=%s", len(rows), len(train_dataset), len(valid_dataset), len(test_dataset))
    LOGGER.info("Model=%s params=%s", config["model"]["name"], model.count_parameters())

    best_valid_auc = float("-inf")
    best_state = None
    best_epoch = -1
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_report = evaluate_model(model, valid_loader, device, config["eval"])
        valid_auc = valid_report["overall"]["auc"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "valid_auc": float(valid_report["overall"]["auc"]),
                "valid_logloss": float(valid_report["overall"]["logloss"]),
            }
        )
        LOGGER.info(
            "epoch=%s train_loss=%.6f valid_auc=%.6f valid_logloss=%.6f",
            epoch,
            train_loss,
            valid_report["overall"]["auc"],
            valid_report["overall"]["logloss"],
        )

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(config["train"]["early_stop_patience"]):
                LOGGER.info("Early stop triggered at epoch=%s", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = save_checkpoint(model, output_dir)
    valid_report = evaluate_model(model, valid_loader, device, config["eval"])
    test_report = evaluate_model(model, test_loader, device, config["eval"])

    manifest = {
        "data_path": str(config["data"]["path"]),
        "row_count": len(rows),
        "schema_column_count": len(metadata["schema"]),
        "effective_feature_counts": {
            "id_features": len(config["features"].get("id_features", [])),
            "user_sparse": len(config["features"].get("user_sparse", [])),
            "user_list_sparse": len(config["features"].get("user_list_sparse", [])),
            "user_dense_only": len(config["features"].get("user_dense_only", [])),
            "aligned_dense_pairs": len(config["features"].get("aligned_dense_pairs", [])),
            "item_sparse": len(config["features"].get("item_sparse", [])),
            "item_list_sparse": len(config["features"].get("item_list_sparse", [])),
            "sequence_domains": {
                domain: len(domain_cfg.get("columns", []))
                for domain, domain_cfg in config["features"].get("sequence_domains", {}).items()
            },
        },
        "splits": {key: len(value) for key, value in splits.items()},
        "positive_label_values": config["data"]["positive_label_values"],
        "best_epoch": best_epoch,
        "best_valid_auc": best_valid_auc,
        "checkpoint_path": checkpoint_path,
        "model_name": config["model"]["name"],
        "parameter_count": model.count_parameters(),
        "history": history,
        "valid_report": valid_report,
        "test_report": test_report,
    }
    save_json(output_dir / "run_manifest.json", manifest)
    LOGGER.info("Finished training. checkpoint=%s", checkpoint_path)


if __name__ == "__main__":
    main()
