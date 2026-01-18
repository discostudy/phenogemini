"""Lightweight Cross-Encoder reranker (optional).

This module provides a configurable reranker that scores (phrase, candidate)
pairs using a cross-encoder model from `sentence-transformers`. If disabled in
the config, a no-op reranker is used and has zero runtime overhead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency via sentence-transformers
    CrossEncoder = None  # type: ignore

from .candidate_generator import Candidate

LOGGER = logging.getLogger(__name__)


@dataclass
class RerankerConfig:
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cuda"
    batch_size: int = 64
    max_length: int = 128
    progress_bar: bool = False


class BaseReranker:
    def score(self, phrase: str, candidates: List[Candidate]) -> List[Candidate]:
        return candidates


class NoOpReranker(BaseReranker):
    pass


class CrossEncoderReranker(BaseReranker):
    def __init__(self, cfg: RerankerConfig) -> None:
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers CrossEncoder not available")
        LOGGER.info("Initialising cross-encoder %s on %s", cfg.model, cfg.device)
        self.model = CrossEncoder(cfg.model, device=cfg.device, max_length=cfg.max_length)
        self.batch_size = int(cfg.batch_size)
        self.show_progress = bool(cfg.progress_bar)

    def _pair_inputs(self, phrase: str, texts: Iterable[str]) -> List[list[str]]:
        return [[phrase, t] for t in texts]

    def score(self, phrase: str, candidates: List[Candidate]) -> List[Candidate]:
        if not candidates:
            return candidates
        inputs = self._pair_inputs(phrase, [c.source_text or c.label for c in candidates])
        try:
            scores = self.model.predict(
                inputs,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
            )
        except TypeError:
            scores = self.model.predict(inputs, batch_size=self.batch_size)

        scores_list = [float(s) for s in (scores.tolist() if hasattr(scores, "tolist") else scores)]
        for cand, s in zip(candidates, scores_list):
            cand.cross_score = s
        return candidates


def build_reranker(raw_cfg: dict) -> BaseReranker:
    enabled = bool(raw_cfg.get("enabled", False))
    if not enabled:
        LOGGER.info("Cross-encoder reranker disabled")
        return NoOpReranker()
    cfg = RerankerConfig(
        enabled=True,
        model=raw_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        device=raw_cfg.get("device", "cuda"),
        batch_size=int(raw_cfg.get("batch_size", 64)),
        max_length=int(raw_cfg.get("max_length", 128)),
        progress_bar=bool(raw_cfg.get("progress_bar", False)),
    )
    return CrossEncoderReranker(cfg)


