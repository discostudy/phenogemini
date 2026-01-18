"""Lightweight lexical candidate retriever for HPO texts.

Uses a TF-IDF word n-gram model to generate top-k lexical candidates for a
given phrase against the HPO text inventory (labels + synonyms as provided by
CandidateGenerator). Designed for small vocab corpora (~20-50k entries).
"""
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize
    from scipy.sparse import csr_matrix
except Exception:  # pragma: no cover - optional dependency via requirements
    TfidfVectorizer = None  # type: ignore
    sk_normalize = None  # type: ignore
    csr_matrix = None  # type: ignore


class LexicalRetriever:
    def __init__(
        self,
        corpus_texts: List[str],
        analyzer: str = "word",
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = None,
    ) -> None:
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is required for LexicalRetriever")
        self.corpus = corpus_texts
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            lowercase=True,
            dtype=np.float32,
            max_features=max_features,
        )
        self.doc_mat: csr_matrix = self.vectorizer.fit_transform(self.corpus)
        # l2-normalise for cosine via dot product
        self.doc_mat = sk_normalize(self.doc_mat, norm="l2", copy=False)

    def _score_query(self, phrase: str, top_k: int) -> List[Tuple[int, float]]:
        if not phrase:
            return []
        q = self.vectorizer.transform([phrase])
        q = sk_normalize(q, norm="l2", copy=False)
        sims = (self.doc_mat @ q.T).toarray().ravel()  # (N,)
        if top_k <= 0 or top_k >= sims.shape[0]:
            idx = np.argsort(-sims)
        else:
            idx = np.argpartition(sims, -top_k)[-top_k:]
            idx = idx[np.argsort(sims[idx])[::-1]]
        return [(int(i), float(sims[i])) for i in idx]

    def score(self, phrase: str, top_k: int) -> List[Tuple[int, float]]:
        return self._score_query(phrase, top_k)

