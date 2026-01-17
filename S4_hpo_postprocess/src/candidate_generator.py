"""GPU-first candidate generator for HPO entity linking."""
from __future__ import annotations

import logging
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch might not be installed
    torch = None
    F = None

from .hpo_loader import HPOTerm, load_hpo_canonical_map
from .lexical_retriever import LexicalRetriever

LOGGER = logging.getLogger(__name__)


@dataclass
class Candidate:
    hpo_id: str
    label: str
    source_text: str
    lexical_score: float = 0.0
    embedding_score: float = 0.0
    cross_score: float = 0.0
    prior_score: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "hpo_id": self.hpo_id,
            "label": self.label,
            "source_text": self.source_text,
            "lexical_score": self.lexical_score,
            "embedding_score": self.embedding_score,
            "cross_score": self.cross_score,
            "prior_score": self.prior_score,
        }


class CandidateGenerator:
    """Generate GPU-based embedding candidates for phenotype phrases."""

    def __init__(
        self,
        terms: List[HPOTerm],
        embedding_top_k: int = 25,
        include_priors: bool = True,
        embed_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        encode_device: str = "cuda",
        similarity_device: Optional[str] = None,
        cache_path: Optional[Path] = None,
        multi_gpu_devices: Optional[List[str]] = None,
        label_only: bool = True,
        min_syn_len: int = 3,
        lexical_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.terms = terms
        self.embedding_top_k = embedding_top_k
        self.include_priors = include_priors
        self.cache_path = cache_path
        self.multi_gpu_devices = multi_gpu_devices or []
        self.encode_device = self._validate_device(encode_device)
        self.similarity_device = similarity_device or encode_device
        self.embed_model_name = embed_model
        # lexical retriever config (defaults)
        self.lex_top_k = 0
        self.lex: Optional[LexicalRetriever] = None

        self._text_lookup: List[str] = []
        self._hpo_lookup: List[str] = []
        self._label_lookup: Dict[str, str] = {}
        for term in terms:
            self._label_lookup[term.hpo_id] = term.label
        # deterministic index: primary labels only (or labels+synonyms filtered) sorted by (hpo_id, text)
        use_label_only = bool(label_only)
        min_len = int(min_syn_len)
        pairs: List[tuple[str, str]] = []
        for term in terms:
            if use_label_only:
                pairs.append((term.hpo_id, term.label))
            else:
                texts = [term.label] + [s for s in term.synonyms if s and len(s) >= min_len]
                for t in texts:
                    pairs.append((term.hpo_id, t))
        # dedup and sort
        seen = set()
        uniq: List[tuple[str, str]] = []
        for hid, txt in pairs:
            key = (hid, txt)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(key)
        uniq.sort(key=lambda x: (x[0], x[1].lower()))
        for hid, txt in uniq:
            if txt:
                self._hpo_lookup.append(hid)
                self._text_lookup.append(txt)

        if not self._text_lookup:
            raise ValueError("No HPO strings available for candidate generation. Check ontology resource.")

        LOGGER.info("Initialising sentence-transformer model %s on %s", embed_model, self.encode_device)
        self.model = SentenceTransformer(embed_model, device=self.encode_device)
        self.embedding_matrix = self._load_or_build_embeddings()
        self.embedding_matrix = self._normalise(self.embedding_matrix)
        self.hpo_matrix = None
        self._prepare_similarity_matrix()

        # Try to build lexical retriever using config from YAML if available
        lex_cfg = lexical_cfg or {}
        if bool(lex_cfg.get('enable', False)):
            try:
                analyzer = lex_cfg.get('analyzer', 'word')
                ngram = lex_cfg.get('ngram_range', [1, 2])
                if isinstance(ngram, list) and len(ngram) == 2:
                    ngram_tuple = (int(ngram[0]), int(ngram[1]))
                else:
                    ngram_tuple = (1, 2)
                max_feats = lex_cfg.get('max_features')
                self.lex_top_k = int(lex_cfg.get('top_k', 100))
                self.lex = LexicalRetriever(
                    self._text_lookup,
                    analyzer=analyzer,
                    ngram_range=ngram_tuple,
                    max_features=max_feats,
                )
                LOGGER.info(
                    "Lexical retriever enabled: analyzer=%s, ngram=%s, top_k=%d",
                    analyzer,
                    ngram_tuple,
                    self.lex_top_k,
                )
            except Exception as exc:
                LOGGER.warning("Lexical retriever disabled (%s)", exc)

    def _validate_device(self, device: str) -> str:
        dev = device or "cpu"
        if dev.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                LOGGER.warning("CUDA requested but unavailable. Falling back to CPU.")
                return "cpu"
        return dev

    def _cache_meta_path(self) -> Optional[Path]:
        if not self.cache_path:
            return None
        return self.cache_path.with_suffix(".meta.json")

    def _current_index_fingerprint(self) -> str:
        # small, deterministic fingerprint of the index content
        take = min(100, len(self._text_lookup))
        sample = "\n".join(self._text_lookup[:take]) + f"|N={len(self._text_lookup)}"
        return hashlib.sha1(sample.encode("utf-8")).hexdigest()

    def _load_or_build_embeddings(self) -> np.ndarray:
        if self.cache_path and self.cache_path.exists():
            LOGGER.info("Loading cached HPO embeddings from %s", self.cache_path)
            mat = np.load(self.cache_path)
            ok = True
            meta_ok = False
            if mat.shape[0] != len(self._text_lookup):
                LOGGER.warning("Cached embeddings rows (%d) != text index size (%d); rebuilding", mat.shape[0], len(self._text_lookup))
                ok = False
            meta_path = self._cache_meta_path()
            if ok and meta_path and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("model") != self.embed_model_name:
                        ok = False
                    if meta.get("index_size") != len(self._text_lookup):
                        ok = False
                    if meta.get("fingerprint") != self._current_index_fingerprint():
                        ok = False
                    meta_ok = True
                except Exception:
                    ok = False
            if ok:
                return mat
            else:
                if not meta_ok:
                    LOGGER.info("Cache metadata mismatch or missing; rebuilding HPO embeddings")

        LOGGER.info("Encoding %d HPO strings", len(self._text_lookup))
        matrix = self._encode_texts(self._text_lookup, batch_size=256, show_progress=True, normalise=False)
        matrix = self._normalise(matrix)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, matrix)
            meta_path = self._cache_meta_path()
            if meta_path:
                meta = {
                    "model": self.embed_model_name,
                    "index_size": len(self._text_lookup),
                    "fingerprint": self._current_index_fingerprint(),
                    "dim": int(matrix.shape[1]) if matrix.ndim == 2 else None,
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return matrix

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _prepare_similarity_matrix(self) -> None:
        if torch is None or F is None:
            LOGGER.warning("torch not available; similarity scoring will run on CPU")
            self.hpo_matrix = None
            return

        sim_device = self._validate_device(self.similarity_device)
        matrix = torch.tensor(self.embedding_matrix, dtype=torch.float32, device=sim_device)
        matrix = F.normalize(matrix, dim=1, eps=1e-12)
        self.hpo_matrix = matrix
        LOGGER.info("Loaded HPO embedding matrix on %s for similarity", sim_device)
        # build canonical map for ID normalization
        try:
            # assume JSON path used in pipeline
            from .config import load_config
            cfg = load_config(Path('configs/hpo_mapping.yaml'))
            canon = load_hpo_canonical_map(cfg.hpo_path, cfg.hpo_format)
            self._canon_map = canon
        except Exception:
            self._canon_map = {}

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool,
        normalise: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype="float32")

        if self.multi_gpu_devices:
            LOGGER.info(
                "Using multi-process encoding on devices: %s", ", ".join(self.multi_gpu_devices)
            )
            pool = self.model.start_multi_process_pool(target_devices=self.multi_gpu_devices)
            try:
                embeddings = self.model.encode_multi_process(texts, pool, batch_size=batch_size)
                embeddings = np.asarray(embeddings)
            finally:
                self.model.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

        if normalise:
            embeddings = self._normalise(embeddings)
        return embeddings

    def encode_phrases(
        self,
        phrases: List[str],
        batch_size: int = 256,
        show_progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        unique: List[str] = []
        seen = set()
        for phrase in phrases:
            if not phrase:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            unique.append(phrase)

        if not unique:
            return {}

        LOGGER.info(
            "Encoding %d unique phenotype strings (batch_size=%d, multi_gpu=%s)",
            len(unique), batch_size, bool(self.multi_gpu_devices),
        )
        embeddings = self._encode_texts(unique, batch_size=batch_size, show_progress=show_progress)
        return {phrase: emb for phrase, emb in zip(unique, embeddings)}

    def _similarity_scores(self, emb_vec: np.ndarray) -> List[Tuple[int, float]]:
        if emb_vec.ndim != 1:
            emb_vec = emb_vec.reshape(-1)

        if torch is not None and self.hpo_matrix is not None:
            vec = torch.from_numpy(emb_vec).to(self.hpo_matrix.device)
            vec = F.normalize(vec, dim=0, eps=1e-12)
            scores = torch.mv(self.hpo_matrix, vec)
            total = scores.numel()
            k = self.embedding_top_k if self.embedding_top_k > 0 else total
            k = min(k, total)
            values, indices = torch.topk(scores, k)
            return [(int(idx), float(val)) for idx, val in zip(indices.tolist(), values.tolist())]

        sims = np.dot(self.embedding_matrix, emb_vec)
        total = sims.shape[0]
        k = self.embedding_top_k if self.embedding_top_k > 0 else total
        k = min(k, total)
        idx = np.argpartition(sims, -k)[-k:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        return [(int(i), float(sims[i])) for i in idx]

    def generate(
        self,
        phrase: str,
        prior_ids: Optional[Iterable[str]] = None,
        phrase_embedding: Optional[np.ndarray] = None,
    ) -> List[Candidate]:
        phrase_clean = phrase.strip()
        if not phrase_clean:
            return []

        candidates: Dict[str, Candidate] = {}

        if phrase_embedding is not None:
            emb_vec = phrase_embedding
        else:
            emb = self._encode_texts([phrase_clean], batch_size=256, show_progress=False)
            emb_vec = emb[0]

        # dense embedding channel
        for idx, score in self._similarity_scores(emb_vec):
            hpo_id = self._hpo_lookup[idx]
            if hasattr(self, "_canon_map") and self._canon_map:
                hpo_id = self._canon_map.get(hpo_id, hpo_id)
            cand = candidates.get(hpo_id)
            if cand is None:
                candidates[hpo_id] = Candidate(
                    hpo_id=hpo_id,
                    label=self._label_lookup.get(hpo_id, ""),
                    source_text=self._text_lookup[idx],
                    embedding_score=score,
                )
            else:
                cand.embedding_score = max(cand.embedding_score, score)

        # lexical channel (optional)
        if self.lex and self.lex_top_k > 0:
            try:
                for idx, lscore in self.lex.score(phrase_clean, self.lex_top_k):
                    hpo_id = self._hpo_lookup[idx]
                    if hasattr(self, "_canon_map") and self._canon_map:
                        hpo_id = self._canon_map.get(hpo_id, hpo_id)
                    cand = candidates.get(hpo_id)
                    if cand is None:
                        candidates[hpo_id] = Candidate(
                            hpo_id=hpo_id,
                            label=self._label_lookup.get(hpo_id, ""),
                            source_text=self._text_lookup[idx],
                            lexical_score=float(lscore),
                        )
                    else:
                        cand.lexical_score = max(cand.lexical_score, float(lscore))
            except Exception:
                pass

        if self.include_priors and prior_ids:
            for hpo_id in set(prior_ids):
                if hasattr(self, "_canon_map") and self._canon_map:
                    hpo_id = self._canon_map.get(hpo_id, hpo_id)
                cand = candidates.get(hpo_id)
                if cand is None:
                    candidates[hpo_id] = Candidate(
                        hpo_id=hpo_id,
                        label=self._label_lookup.get(hpo_id, ""),
                        source_text="prior",
                        prior_score=1.0,
                    )
                else:
                    cand.prior_score = max(cand.prior_score, 1.0)

        return list(candidates.values())
