"""Fusion of candidate signals into final scores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .candidate_generator import Candidate


@dataclass
class FusionWeights:
    lexical: float = 0.5
    embedding: float = 0.4
    cross: float = 0.0
    prior: float = 0.1
    ontology: float = 0.0
    accept_threshold: float = 0.6
    review_threshold: float = 0.45


class CandidateRanker:
    """Combine multiple scoring channels into a final ranking."""

    def __init__(self, weights: FusionWeights, keep_top_k: int = 3, depth_map: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights
        self.keep_top_k = keep_top_k
        self.depth_map = depth_map or {}

    def rank(self, candidates: List[Candidate]) -> Dict[str, List[Dict]]:
        if not candidates:
            return {"accepted": [], "review": [], "rejected": []}

        scored = []
        for cand in candidates:
            ont_score = 0.0
            if self.depth_map and self.weights.ontology > 0.0:
                ont_score = float(self.depth_map.get(cand.hpo_id, 0.0))
            final_score = (
                cand.lexical_score * self.weights.lexical
                + cand.embedding_score * self.weights.embedding
                + cand.cross_score * self.weights.cross
                + cand.prior_score * self.weights.prior
                + ont_score * self.weights.ontology
            )
            entry = cand.as_dict()
            entry["ontology_score"] = ont_score
            entry["final_score"] = float(final_score)
            scored.append(entry)

        scored.sort(key=lambda x: x["final_score"], reverse=True)
        accepted: List[Dict] = []
        review: List[Dict] = []
        rejected: List[Dict] = []

        for rank, entry in enumerate(scored, start=1):
            entry["rank"] = rank
            score = entry["final_score"]
            if score >= self.weights.accept_threshold:
                accepted.append(entry)
            elif score >= self.weights.review_threshold:
                review.append(entry)
            else:
                rejected.append(entry)

        # truncate lists to keep_top_k for storage efficiency
        accepted = accepted[: self.keep_top_k]
        review = review[: self.keep_top_k]
        rejected = rejected[: self.keep_top_k]
        return {"accepted": accepted, "review": review, "rejected": rejected}
