"""Utility helpers for loading post-processing configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    """Structured configuration for the HPO normalization pipeline."""

    raw: Dict[str, Any]

    @property
    def input_pkl(self) -> Path:
        return Path(self.raw.get("input_pkl", "data/PhenoGemini_g2p_20250127.pkl"))

    @property
    def output_pkl(self) -> Path:
        return Path(self.raw.get("output_pkl", "data/PhenoGemini_g2p_20250127_mapped.pkl"))

    @property
    def hpo_path(self) -> Path:
        return Path(self.raw.get("hpo_resource", {}).get("path", "data/hp_terms.tsv"))

    @property
    def hpo_format(self) -> str:
        return self.raw.get("hpo_resource", {}).get("format", "tsv")

    @property
    def hpo_embedding_cache(self) -> Path:
        path = self.raw.get("hpo_resource", {}).get("cache_embeddings")
        return Path(path) if path else Path("data/hp_terms_embeddings.npy")

    @property
    def candidate_cfg(self) -> Dict[str, Any]:
        return self.raw.get("candidate_generation", {})

    @property
    def embedding_cfg(self) -> Dict[str, Any]:
        return self.raw.get("embedding_model", {})

    @property
    def fusion_cfg(self) -> Dict[str, Any]:
        return self.raw.get("fusion", {})

    @property
    def output_cfg(self) -> Dict[str, Any]:
        return self.raw.get("output_options", {})

    @property
    def reranker_cfg(self) -> Dict[str, Any]:
        return self.raw.get("reranker", {})

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.raw.get("checkpoint_dir", "data/postprocess_chunks"))

    @property
    def batching_cfg(self) -> Dict[str, Any]:
        return self.raw.get("batching", {})

    @property
    def index_cfg(self) -> Dict[str, Any]:
        return self.raw.get("embedding_index", {})

    @property
    def ontology_cfg(self) -> Dict[str, Any]:
        return self.raw.get("ontology", {})


def load_config(path: Path) -> PipelineConfig:
    """Read a YAML config file and return a structured wrapper."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(raw)
