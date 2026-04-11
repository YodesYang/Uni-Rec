from __future__ import annotations

from typing import Any

import torch
from torch import nn

from unirec.features.encoders import DenseStatsEncoder, HashEmbedding, SequenceEmbeddingEncoder, WeightedSequenceEncoder
from unirec.features.multi_domain import aggregate_domain_vectors

from .base import BaseCVRModel


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class MLPCVRModel(BaseCVRModel):
    def __init__(self, model_config: dict[str, Any], feature_config: dict[str, Any]) -> None:
        super().__init__()
        features = feature_config["features"]
        embed_dim = int(model_config["embed_dim"])
        dense_proj_dim = int(model_config["dense_proj_dim"])
        bucket_size = int(model_config["hash_bucket_size"])

        self.static_sparse = nn.ModuleDict()
        for column in features.get("id_features", []) + features.get("user_sparse", []) + features.get("item_sparse", []):
            self.static_sparse[column] = HashEmbedding(bucket_size=bucket_size, embed_dim=embed_dim)

        self.list_sparse = nn.ModuleDict()
        for column in features.get("user_list_sparse", []) + features.get("item_list_sparse", []):
            self.list_sparse[column] = SequenceEmbeddingEncoder(bucket_size=bucket_size, embed_dim=embed_dim)

        self.aligned_dense_pairs = nn.ModuleDict()
        for pair in features.get("aligned_dense_pairs", []):
            self.aligned_dense_pairs[pair["name"]] = WeightedSequenceEncoder(
                bucket_size=bucket_size,
                embed_dim=embed_dim,
            )

        self.dense_only = nn.ModuleDict()
        for column in features.get("user_dense_only", []):
            self.dense_only[column] = DenseStatsEncoder(output_dim=dense_proj_dim)

        self.domain_sequences = nn.ModuleDict()
        for domain, domain_cfg in features.get("sequence_domains", {}).items():
            encoders = nn.ModuleDict()
            for column in domain_cfg.get("columns", []):
                encoders[column] = SequenceEmbeddingEncoder(bucket_size=bucket_size, embed_dim=embed_dim)
            self.domain_sequences[domain] = encoders

        feature_dim = (
            len(self.static_sparse) * embed_dim
            + len(self.list_sparse) * embed_dim
            + len(self.aligned_dense_pairs) * embed_dim
            + len(self.domain_sequences) * embed_dim
            + len(self.dense_only) * dense_proj_dim
        )

        layers: list[nn.Module] = []
        input_dim = feature_dim
        for hidden_dim in model_config.get("hidden_dims", []):
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(_activation(model_config.get("activation", "relu")))
            layers.append(nn.Dropout(float(model_config.get("dropout", 0.0))))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        vectors: list[torch.Tensor] = []

        static_sparse = batch["static_sparse"]
        for column, encoder in self.static_sparse.items():
            vectors.append(encoder(static_sparse[column]))

        list_sparse = batch["list_sparse"]
        for column, encoder in self.list_sparse.items():
            entry = list_sparse[column]
            vectors.append(encoder(entry["values"], entry["mask"]))

        aligned_dense_pairs = batch["aligned_dense_pairs"]
        for name, encoder in self.aligned_dense_pairs.items():
            entry = aligned_dense_pairs[name]
            vectors.append(encoder(entry["ids"], entry["weights"], entry["mask"]))

        dense_only = batch["dense_only"]
        for column, encoder in self.dense_only.items():
            entry = dense_only[column]
            vectors.append(encoder(entry["values"], entry["mask"], entry["lengths"]))

        domain_sequences = batch["domain_sequences"]
        for domain, encoders in self.domain_sequences.items():
            column_vectors = []
            for column, encoder in encoders.items():
                entry = domain_sequences[domain][column]
                column_vectors.append(encoder(entry["values"], entry["mask"]))
            vectors.append(aggregate_domain_vectors(column_vectors))

        features = torch.cat(vectors, dim=1)
        logits = self.mlp(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        return {"logits": logits, "probs": probs}
