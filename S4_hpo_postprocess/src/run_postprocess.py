#!/usr/bin/env python3
"""Orchestrate HPO mapping post-processing."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import tempfile
import shlex
import re

import pandas as pd
from tqdm import tqdm

from .candidate_generator import CandidateGenerator
from .candidate_ranker import CandidateRanker, FusionWeights
from .reranker import build_reranker, BaseReranker
from .config import PipelineConfig, load_config
from .hpo_loader import HPOTerm, load_hpo_terms
from .phenotype_cleaner import PhenotypeCleaner

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process PhenoGemini phenotypes to HPO IDs")
    parser.add_argument("--config", default="configs/hpo_mapping.yaml", help="Path to YAML config")
    parser.add_argument("--input", help="Override input pickle path", nargs="?")
    parser.add_argument("--output", help="Override output pickle path", nargs="?")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing output")
    parser.add_argument("--limit", type=int, help="Limit number of rows for debugging")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def load_terms(cfg: PipelineConfig) -> List[HPOTerm]:
    LOGGER.info("Loading HPO resource from %s (%s)", cfg.hpo_path, cfg.hpo_format)
    return load_hpo_terms(cfg.hpo_path, cfg.hpo_format)

def _compute_hpo_depths_from_json(path: Path, root_id: str = "HP:0000118") -> Dict[str, float]:
    import json as _json
    from collections import deque

    def _convert_id(raw_id: str) -> Optional[str]:
        if raw_id.startswith("HP:"):
            return raw_id
        if raw_id.startswith("http://purl.obolibrary.org/obo/HP_"):
            tail = raw_id.rsplit("/", 1)[-1]
            return tail.replace("_", ":")
        return None

    payload = _json.loads(path.read_text(encoding="utf-8"))
    graphs = payload.get("graphs") or []
    if graphs:
        edges = []
        for g in graphs:
            edges.extend(g.get("edges", []) or [])
    else:
        edges = payload.get("edges", []) or []

    def _is_is_a(pred: str) -> bool:
        p = (pred or "").lower()
        return (
            p.endswith("is_a") or p.endswith("#is_a") or p.endswith("subclassof") or p.endswith("#subclassof") or ":is_a" in p or "subclass" in p
        )

    parent2children: Dict[str, set] = {}
    nodes = set()
    for e in edges:
        sub = _convert_id(str(e.get("sub"))) if e.get("sub") else None
        obj = _convert_id(str(e.get("obj"))) if e.get("obj") else None
        pred = str(e.get("pred", ""))
        if not sub or not obj:
            continue
        nodes.add(sub); nodes.add(obj)
        if not _is_is_a(pred):
            continue
        parent2children.setdefault(obj, set()).add(sub)

    if root_id not in nodes and "HP:0000001" in nodes:
        root_id = "HP:0000001"

    depth_int: Dict[str, int] = {}
    q: deque[str] = deque()
    depth_int[root_id] = 0
    q.append(root_id)
    while q:
        u = q.popleft()
        for v in parent2children.get(u, ()):  # children of u
            nd = depth_int[u] + 1
            if v in depth_int:
                if nd < depth_int[v]:
                    depth_int[v] = nd
                continue
            depth_int[v] = nd
            q.append(v)

    if not depth_int:
        return {}
    max_d = max(depth_int.values()) or 1
    depths: Dict[str, float] = {k: (v / max_d) for k, v in depth_int.items()}
    for n in nodes:
        depths.setdefault(n, 0.0)
    return depths


def build_generator(cfg: PipelineConfig, terms: List[HPOTerm]) -> CandidateGenerator:
    gen_cfg = cfg.candidate_cfg
    emb_cfg = cfg.embedding_cfg
    idx_cfg = cfg.index_cfg
    generator = CandidateGenerator(
        terms=terms,
        embedding_top_k=int(gen_cfg.get("embedding_top_k", 25)),
        include_priors=bool(gen_cfg.get("include_priors", True)),
        embed_model=emb_cfg.get("name", "cambridgeltl/SapBERT-from-PubMedBert-fulltext"),
        encode_device=emb_cfg.get("device", "cpu"),
        similarity_device=emb_cfg.get("similarity_device"),
        cache_path=cfg.hpo_embedding_cache,
        multi_gpu_devices=emb_cfg.get("multi_gpu_devices"),
        label_only=bool(idx_cfg.get("label_only", True)),
        min_syn_len=int(idx_cfg.get("min_syn_len", 3)),
        lexical_cfg=cfg.raw.get("lexical", {}),
    )
    return generator


def build_ranker(cfg: PipelineConfig) -> CandidateRanker:
    fusion_cfg = cfg.fusion_cfg
    weights = FusionWeights(
        lexical=float(fusion_cfg.get("lexical_weight", 0.45)),
        embedding=float(fusion_cfg.get("embedding_weight", 0.45)),
        cross=float(fusion_cfg.get("cross_weight", 0.0)),
        prior=float(fusion_cfg.get("prior_weight", 0.10)),
        ontology=float(cfg.ontology_cfg.get("depth_weight", 0.0)),
        accept_threshold=float(fusion_cfg.get("accept_threshold", 0.6)),
        review_threshold=float(fusion_cfg.get("review_threshold", 0.45)),
    )
    keep_top_k = int(cfg.output_cfg.get("keep_top_k", 3))
    depth_map: Dict[str, float] | None = None
    if bool(cfg.ontology_cfg.get("enable", False)) and cfg.hpo_format.lower() == "json":
        root_id = cfg.ontology_cfg.get("root_id", "HP:0000118")
        try:
            depth_map = _compute_hpo_depths_from_json(cfg.hpo_path, root_id=root_id)
            LOGGER.info("Loaded HPO depth map with %d nodes (root=%s)", len(depth_map), root_id)
        except Exception as exc:
            LOGGER.warning("Failed to compute HPO depths: %s", exc)
            depth_map = None
    return CandidateRanker(weights, keep_top_k=keep_top_k, depth_map=depth_map)


def map_phenotypes(
    phenos: List[str],
    cleaned: List[str],
    generator: CandidateGenerator,
    ranker: CandidateRanker,
    cfg: PipelineConfig,
    reranker: BaseReranker,
    phrase_embeddings: Dict[str, Any],
    candidate_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    include_candidates = bool(cfg.output_cfg.get("write_candidates", True))

    mapping_details: List[Dict[str, Any]] = []
    accepted_ids: List[str] = []
    confidence_scores: List[float] = []

    for original, clean_text in zip(phenos, cleaned):
        record: Dict[str, Any] = {
            "original": original,
            "cleaned": clean_text,
            "status": "unmapped",
            "best_id": None,
            "best_label": None,
            "confidence": 0.0,
        }
        if not clean_text:
            if include_candidates:
                record["candidates"] = {"accepted": [], "review": [], "rejected": []}
            mapping_details.append(record)
            continue

        cache_key = clean_text
        if cache_key in candidate_cache:
            ranked = copy.deepcopy(candidate_cache[cache_key])
        else:
            embedding = phrase_embeddings.get(clean_text)
            candidates = generator.generate(clean_text, phrase_embedding=embedding)
            candidates = reranker.score(clean_text, candidates)
            ranked_raw = ranker.rank(candidates)
            candidate_cache[cache_key] = ranked_raw
            ranked = copy.deepcopy(ranked_raw)

        best_entry: Optional[Dict[str, Any]] = None
        if ranked["accepted"]:
            best_entry = ranked["accepted"][0]
            record["status"] = "accepted"
        elif ranked["review"]:
            best_entry = ranked["review"][0]
            record["status"] = "review"
        elif ranked["rejected"]:
            best_entry = ranked["rejected"][0]
            record["status"] = "rejected"

        if best_entry:
            record["best_id"] = best_entry["hpo_id"]
            record["best_label"] = best_entry["label"]
            record["confidence"] = float(best_entry.get("final_score", 0.0))
            if record["status"] == "accepted":
                accepted_ids.append(best_entry["hpo_id"])
                confidence_scores.append(record["confidence"])

        if include_candidates:
            record["candidates"] = ranked

        mapping_details.append(record)

    unique_ids = []
    for hpo_id in accepted_ids:
        if hpo_id not in unique_ids:
            unique_ids.append(hpo_id)

    # Primary output with clearer name for side-by-side comparison
    result = {
        "sapbert_mapping_records": mapping_details,
        "hpo_sapbert": unique_ids,
        "hpo_sapbert_conf": confidence_scores,
    }
    return result


def _extract_hpo_ids(text: str, pattern: str = r"HP:\d{7}") -> List[str]:
    try:
        ids = re.findall(pattern, text)
        # preserve order + unique
        seen = set()
        out = []
        for h in ids:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out
    except Exception:
        return []


def _run_external_mapper_single(payload: str, cfg: Dict[str, Any]) -> List[str]:
    if not payload or not cfg.get("enable", False):
        return []
    cmd = cfg.get("cli") or cfg.get("cmd") or cfg.get("command")
    if not cmd:
        return []
    args = cfg.get("args", [])
    mode = (cfg.get("mode") or "stdin").lower()
    regex = cfg.get("regex", r"HP:\\d{7}")
    timeout = int(cfg.get("timeout", 60))

    def _exec_cmd(command_list: List[str], input_bytes: Optional[bytes]) -> str:
        try:
            res = subprocess.run(
                command_list,
                cwd=str(Path(__file__).resolve().parent.parent),
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            out = (res.stdout or b"").decode("utf-8", errors="ignore")
            if not out:
                # some tools print to stderr
                out = (res.stderr or b"").decode("utf-8", errors="ignore")
            return out
        except Exception:
            return ""

    if isinstance(cmd, str):
        cmd_list = shlex.split(cmd)
    else:
        cmd_list = list(cmd)
    if args:
        if isinstance(args, str):
            cmd_list += shlex.split(args)
        elif isinstance(args, list):
            cmd_list += [str(x) for x in args]

    if mode == "stdin":
        out = _exec_cmd(cmd_list, payload.encode("utf-8", errors="ignore"))
        return _extract_hpo_ids(out, regex)
    # file mode
    with tempfile.TemporaryDirectory() as td:
        inp = Path(td) / "input.txt"
        outp = Path(td) / "output.txt"
        inp.write_text(payload, encoding="utf-8")
        replaced = []
        used_out = False
        for tok in cmd_list:
            tok2 = tok.replace("{infile}", str(inp)).replace("{outfile}", str(outp))
            if "{outfile}" in tok:
                used_out = True
            replaced.append(tok2)
        out_text = _exec_cmd(replaced, None)
        if used_out and outp.exists():
            return _extract_hpo_ids(outp.read_text(encoding="utf-8", errors="ignore"), regex)
        # fallback: parse stdout/stderr of the process
        return _extract_hpo_ids(out_text or "", regex)


def _checkpoint_signature(cfg: PipelineConfig) -> Dict[str, Any]:
    sig: Dict[str, Any] = {
        "embedding_model": cfg.embedding_cfg,
        "candidate_generation": cfg.candidate_cfg,
        "fusion": cfg.fusion_cfg,
        "reranker": cfg.reranker_cfg,
        "ontology": cfg.ontology_cfg,
        "lexical": cfg.raw.get("lexical", {}),
        "index": cfg.index_cfg,
        "hpo": {
            "path": str(cfg.hpo_path),
            "format": cfg.hpo_format,
        },
    }
    try:
        if cfg.hpo_path.exists():
            stat = cfg.hpo_path.stat()
            sig["hpo"]["mtime"] = stat.st_mtime
            sig["hpo"]["size"] = stat.st_size
    except Exception:
        pass
    return sig


def _load_signature(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_signature(path: Path, sig: Dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(sig, indent=2), encoding="utf-8")
    except Exception:
        LOGGER.warning("Failed to write checkpoint signature: %s", path)


def process_dataframe(
    df: pd.DataFrame,
    cleaner: PhenotypeCleaner,
    generator: CandidateGenerator,
    ranker: CandidateRanker,
    reranker: BaseReranker,
    cfg: PipelineConfig,
    checkpoint_dir: Path,
    use_checkpoints: bool,
) -> pd.DataFrame:
    chunk_size = int(cfg.batching_cfg.get("chunk_size", 2000))
    show_progress = bool(cfg.batching_cfg.get("progress_bar", True))
    worker_count = int(cfg.batching_cfg.get("workers", 8))

    total_rows = len(df)
    LOGGER.info(
        "Processing %d rows (chunk_size=%d, workers=%d)",
        total_rows, chunk_size, worker_count
    )

    index_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
    cleaned_results: List[Optional[List[str]]] = [None] * total_rows
    mapping_results: List[Optional[List[Dict[str, Any]]]] = [None] * total_rows
    ids_results: List[Optional[List[str]]] = [None] * total_rows  # sapbert ids
    conf_results: List[Optional[List[float]]] = [None] * total_rows
    doc2hpo_results: List[Optional[List[str]]] = [None] * total_rows
    clinphen_results: List[Optional[List[str]]] = [None] * total_rows

    checkpoint_dir = checkpoint_dir
    processed_chunks: set[Tuple[int, int]] = set()
    signature_match = False
    current_sig = _checkpoint_signature(cfg)
    sig_path = checkpoint_dir / "signature.json"
    if use_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        force_recompute = os.environ.get("FORCE_RECOMPUTE", "").strip().lower() in ("1","true","yes","y")
        if force_recompute:
            LOGGER.info("FORCE_RECOMPUTE set; ignoring existing checkpoints and signature")
        else:
            old_sig = _load_signature(sig_path)
            if old_sig == current_sig and any(checkpoint_dir.glob("*.pkl")):
                signature_match = True
                LOGGER.info("Checkpoint signature matches current config; resuming from existing chunks")
                prefix = f"total{total_rows}_chunk_"
                for ckpt_file in sorted(checkpoint_dir.glob(f"{prefix}*.pkl")):
                    try:
                        chunk_df = pd.read_pickle(ckpt_file)
                    except Exception as exc:
                        LOGGER.warning("Failed to read checkpoint %s: %s", ckpt_file.name, exc)
                        continue
                    _apply_chunk_results(
                        chunk_df,
                        index_to_pos,
                        cleaned_results,
                        mapping_results,
                        ids_results,
                        conf_results,
                        doc2hpo_results,
                        clinphen_results,
                    )
                    range_part = ckpt_file.stem[len(prefix):]
                    try:
                        start_val, stop_val = map(int, range_part.split("_"))
                        processed_chunks.add((start_val, stop_val))
                        LOGGER.info("Loaded checkpoint %s with %d rows", ckpt_file.name, len(chunk_df))
                    except Exception:
                        LOGGER.warning("Could not parse chunk range from %s", ckpt_file.name)
            else:
                if old_sig is not None:
                    LOGGER.info("Checkpoint signature mismatch or empty; existing chunk files will be ignored for this run")
                else:
                    LOGGER.info("No checkpoint signature found; starting fresh")
    else:
        LOGGER.info("Checkpointing disabled for this run")

    iterator = range(0, total_rows, chunk_size)
    if show_progress:
        iterator = tqdm(iterator, desc="post-process", unit="chunk")

    global_cache: Dict[str, Dict[str, Any]] = {}

    def build_candidates(phrases: List[str], embeddings: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if not phrases:
            return {}
        LOGGER.info("Precomputing candidates for %d phrases (workers=%d)", len(phrases), worker_count)
        start_time = time.time()
        cache: Dict[str, Dict[str, Any]] = {}

        def worker(phrase: str) -> Tuple[str, Dict[str, Any]]:
            embedding = embeddings.get(phrase)
            cands = generator.generate(phrase, phrase_embedding=embedding)
            cands = reranker.score(phrase, cands)
            ranked = ranker.rank(cands)
            return phrase, ranked

        # Avoid excessive concurrent GPU calls if a cross-encoder reranker is active
        rerank_active = reranker.__class__.__name__ != "NoOpReranker"
        effective_workers = worker_count
        if rerank_active and worker_count > 8:
            effective_workers = 8
            LOGGER.info("Reranker enabled: capping candidate precompute workers to %d", effective_workers)

        if effective_workers <= 1:
            for phrase in phrases:
                key, ranked = worker(phrase)
                cache[key] = ranked
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {executor.submit(worker, phrase): phrase for phrase in phrases}
                for idx, future in enumerate(as_completed(futures), start=1):
                    key, ranked = future.result()
                    cache[key] = ranked
                    if show_progress and idx % 500 == 0:
                        elapsed = time.time() - start_time
                        LOGGER.info(
                            "    completed %d/%d candidate computations (%.1fs elapsed)",
                            idx, len(phrases), elapsed
                        )
        LOGGER.info("Finished candidate precomputation for chunk in %.2fs", time.time() - start_time)
        return cache

    prefix = f"total{total_rows}_chunk_"
    for start in iterator:
        stop = min(start + chunk_size, total_rows)
        chunk_range = (start, stop)
        chunk_file = checkpoint_dir / f"{prefix}{start}_{stop}.pkl"

        if use_checkpoints and chunk_range in processed_chunks and chunk_file.exists():
            LOGGER.info("Skipping rows %d-%d (checkpoint present)", start, stop)
            continue

        chunk = df.iloc[start:stop]
        LOGGER.info("Processing rows %d-%d (chunk size %d)", start, stop, len(chunk))
        chunk_t0 = time.time()

        cleaned_rows: List[Tuple[List[str], List[str]]] = []
        unique_phrases: List[str] = []
        seen_phrases = set()

        for _, row in chunk.iterrows():
            if isinstance(row.get("phenotypes_cleaned"), list) and row["phenotypes_cleaned"]:
                phenos = [str(x).strip() for x in row["phenotypes_cleaned"] if str(x).strip()]
                cleaned = phenos
            else:
                phenos = row.get("phenotype") or []
                if not isinstance(phenos, list):
                    phenos = []
                cleaned = cleaner.clean_list(phenos)
            cleaned_rows.append((phenos, cleaned))
            for phrase in cleaned:
                if phrase and phrase not in seen_phrases:
                    seen_phrases.add(phrase)
                    unique_phrases.append(phrase)

        LOGGER.info("Unique phrases in chunk: %d", len(unique_phrases))
        phrase_embeddings = generator.encode_phrases(
            unique_phrases,
            show_progress=bool(cfg.embedding_cfg.get("progress_bar", False))
        )
        LOGGER.info("Finished encoding phrases; generating candidates")

        candidate_cache: Dict[str, Dict[str, Any]] = {}
        new_phrases: List[str] = []
        for phrase in unique_phrases:
            if phrase in global_cache:
                candidate_cache[phrase] = global_cache[phrase]
            else:
                new_phrases.append(phrase)

        if new_phrases:
            new_cache = build_candidates(new_phrases, phrase_embeddings)
            global_cache.update(new_cache)
            candidate_cache.update(new_cache)
        else:
            LOGGER.info("All phrases already cached; skipping candidate generation")

        results_chunk = {
            "row_index": chunk.index.to_numpy(),
            "phenotypes_cleaned": [],
            "sapbert_mapping_records": [],
            "hpo_sapbert": [],
            "hpo_sapbert_conf": [],
            # backward-compat columns will be derived at the end
            # external mappers (optional)
            "hpo_doc2hpo": [],
            "hpo_clinphen": [],
        }

        total_chunk_rows = len(cleaned_rows)
        row_log_interval = max(1, total_chunk_rows // 10)
        LOGGER.info("Generating candidates for %d rows (log every %d rows)", total_chunk_rows, row_log_interval)

        ext_cfg = cfg.raw.get("external_mappers", {}) or {}
        use_doc2hpo = bool(ext_cfg.get("doc2hpo", {}).get("enable", False))
        use_clinphen = bool(ext_cfg.get("clinphen", {}).get("enable", False))

        # Per-phrase external mapping over phenotypes_cleaned
        doc2_cfg = ext_cfg.get("doc2hpo", {})
        cph_cfg = ext_cfg.get("clinphen", {})

        # Flatten unique phrases across this chunk for caching
        phrase_to_rows: Dict[str, List[int]] = {}
        for ridx0, (_, cleaned) in enumerate(cleaned_rows):
            for p in (cleaned or []):
                if p:
                    phrase_to_rows.setdefault(p, []).append(ridx0)

        doc2_phrase_map: Dict[str, List[str]] = {}
        clin_phrase_map: Dict[str, List[str]] = {}
        def _bulk_apply(texts: List[str], mapper_cfg: Dict[str, Any], name: str) -> Dict[str, List[str]]:
            out: Dict[str, List[str]] = {}
            if not mapper_cfg.get("enable", False) or not texts:
                return out
            max_workers = int(ext_cfg.get("workers", 8))
            mapper_cfg = dict(mapper_cfg)
            mapper_cfg["name"] = name
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_external_mapper_single, t, mapper_cfg): t for t in texts}
                for fut in tqdm(as_completed(futs), total=len(futs), disable=not show_progress, desc=f"{name} mapping"):
                    t = futs[fut]
                    try:
                        out[t] = fut.result() or []
                    except Exception:
                        out[t] = []
            return out

        uniq_phrases = list(phrase_to_rows.keys())
        if use_doc2hpo:
            doc2_phrase_map = _bulk_apply(uniq_phrases, doc2_cfg, "doc2hpo")
        if use_clinphen:
            clin_phrase_map = _bulk_apply(uniq_phrases, cph_cfg, "clinphen")

        # Aggregate per row: union of per-phrase HPO IDs; also record mapped/unmapped phrases
        doc2_outputs: List[List[str]] = [[] for _ in range(len(cleaned_rows))]
        doc2_mapped_texts: List[List[str]] = [[] for _ in range(len(cleaned_rows))]
        doc2_unmapped_texts: List[List[str]] = [[] for _ in range(len(cleaned_rows))]
        clin_outputs: List[List[str]] = [[] for _ in range(len(cleaned_rows))]
        clin_mapped_texts: List[List[str]] = [[] for _ in range(len(cleaned_rows))]
        clin_unmapped_texts: List[List[str]] = [[] for _ in range(len(cleaned_rows))]

        if use_doc2hpo or use_clinphen:
            for ridx0, (_, cleaned) in enumerate(cleaned_rows):
                ids_doc2: List[str] = []
                ids_clin: List[str] = []
                seen_doc2 = set()
                seen_clin = set()
                for p in (cleaned or []):
                    if use_doc2hpo:
                        h = doc2_phrase_map.get(p, [])
                        if h:
                            doc2_mapped_texts[ridx0].append(p)
                            for x in h:
                                if x not in seen_doc2:
                                    seen_doc2.add(x); ids_doc2.append(x)
                        else:
                            doc2_unmapped_texts[ridx0].append(p)
                    if use_clinphen:
                        h2 = clin_phrase_map.get(p, [])
                        if h2:
                            clin_mapped_texts[ridx0].append(p)
                            for x in h2:
                                if x not in seen_clin:
                                    seen_clin.add(x); ids_clin.append(x)
                        else:
                            clin_unmapped_texts[ridx0].append(p)
                doc2_outputs[ridx0] = ids_doc2 if use_doc2hpo else []
                clin_outputs[ridx0] = ids_clin if use_clinphen else []

        # per-row processing
        for ridx, (phenos, cleaned) in enumerate(cleaned_rows, start=1):
            mapped = map_phenotypes(phenos, cleaned, generator, ranker, cfg, reranker, phrase_embeddings, candidate_cache)
            results_chunk["phenotypes_cleaned"].append(cleaned)
            results_chunk["sapbert_mapping_records"].append(mapped["sapbert_mapping_records"])
            results_chunk["hpo_sapbert"].append(mapped["hpo_sapbert"])
            results_chunk["hpo_sapbert_conf"].append(mapped["hpo_sapbert_conf"])
            # external mappers (aggregated per-phrase)
            results_chunk["hpo_doc2hpo"].append(doc2_outputs[ridx-1] if use_doc2hpo else [])
            results_chunk["hpo_clinphen"].append(clin_outputs[ridx-1] if use_clinphen else [])
            # mapped/unmapped phrase tracking for external mappers
            if use_doc2hpo:
                results_chunk.setdefault("hpo_doc2hpo_mapped_texts", []).append(doc2_mapped_texts[ridx-1])
                results_chunk.setdefault("hpo_doc2hpo_unmapped_texts", []).append(doc2_unmapped_texts[ridx-1])
            if use_clinphen:
                results_chunk.setdefault("hpo_clinphen_mapped_texts", []).append(clin_mapped_texts[ridx-1])
                results_chunk.setdefault("hpo_clinphen_unmapped_texts", []).append(clin_unmapped_texts[ridx-1])

            if show_progress and ridx % row_log_interval == 0:
                LOGGER.info("  processed %d/%d rows in current chunk", ridx, total_chunk_rows)

            # additional periodic log: candidate cache size
            if show_progress and ridx % row_log_interval == 0:
                LOGGER.info("    cache size=%d", len(candidate_cache))

        chunk_df = pd.DataFrame(results_chunk)
        _apply_chunk_results(
            chunk_df,
            index_to_pos,
            cleaned_results,
            mapping_results,
            ids_results,
            conf_results,
            doc2hpo_results,
            clinphen_results,
        )

        if use_checkpoints:
            chunk_df.to_pickle(chunk_file)
            LOGGER.info("Saved checkpoint %s", chunk_file.name)

        LOGGER.info("Finished chunk %d-%d in %.2fs", start, stop, time.time() - chunk_t0)

    df["phenotypes_cleaned"] = [val if val is not None else [] for val in cleaned_results]
    df["sapbert_mapping_records"] = [val if val is not None else [] for val in mapping_results]
    df["hpo_sapbert"] = [val if val is not None else [] for val in ids_results]
    df["hpo_sapbert_conf"] = [val if val is not None else [] for val in conf_results]
    # external mappers columns from aggregated results
    df["hpo_doc2hpo"] = [val if val is not None else [] for val in doc2hpo_results]
    df["hpo_clinphen"] = [val if val is not None else [] for val in clinphen_results]

    # Derive convenience columns: which cleaned phrases mapped vs not mapped
    try:
        def _derive_lists(records: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
            mapped: List[str] = []
            unmapped: List[str] = []
            if not isinstance(records, list):
                return mapped, unmapped
            for rec in records:
                try:
                    st = (rec.get("status") or "").lower()
                    txt = rec.get("cleaned") or rec.get("original") or ""
                    if st == "accepted":
                        mapped.append(txt)
                    elif st == "unmapped":
                        unmapped.append(txt)
                except Exception:
                    continue
            return mapped, unmapped

        mapped_lists: List[List[str]] = []
        unmapped_lists: List[List[str]] = []
        for recs in df["sapbert_mapping_records"].tolist():
            m, u = _derive_lists(recs)
            mapped_lists.append(m)
            unmapped_lists.append(u)
        df["hpo_sapbert_mapped_texts"] = mapped_lists
        df["hpo_sapbert_unmapped_texts"] = unmapped_lists
    except Exception as e:
        LOGGER.warning("Failed to derive mapped/unmapped phrase lists: %s", e)
    LOGGER.info("Completed processing all chunks")
    if use_checkpoints:
        _save_signature(sig_path, current_sig)
    return df


def _apply_chunk_results(
    chunk_df: pd.DataFrame,
    index_to_pos: Dict[Any, int],
    cleaned_results: List[Optional[List[str]]],
    mapping_results: List[Optional[List[Dict[str, Any]]]],
    ids_results: List[Optional[List[str]]],
    conf_results: List[Optional[List[float]]],
    doc2hpo_results: List[Optional[List[str]]],
    clinphen_results: List[Optional[List[str]]],
) -> None:
    for row in chunk_df.itertuples():
        pos = index_to_pos.get(row.row_index)
        if pos is None:
            continue
        cleaned_results[pos] = row.phenotypes_cleaned
        mapping_val = getattr(row, 'sapbert_mapping_records', None)
        if mapping_val is None:
            mapping_val = getattr(row, 'phenotype_mapping_v2', [])
        mapping_results[pos] = mapping_val
        # Collect SapBERT outputs from the chunk
        ids_val = getattr(row, 'hpo_sapbert', None)
        if ids_val is None:
            ids_val = []
        conf_val = getattr(row, 'hpo_sapbert_conf', None)
        if conf_val is None:
            conf_val = []
        ids_results[pos] = ids_val
        conf_results[pos] = conf_val
        # external mappers from chunk (if present)
        try:
            doc2_val = getattr(row, 'hpo_doc2hpo', [])
        except Exception:
            doc2_val = []
        try:
            clin_val = getattr(row, 'hpo_clinphen', [])
        except Exception:
            clin_val = []
        doc2hpo_results[pos] = doc2_val
        clinphen_results[pos] = clin_val
        # optional mapped/unmapped phrase lists
        try:
            if "hpo_doc2hpo_mapped_texts" in chunk_df.columns:
                # pandas itertuples uses attribute name with underscores
                doc2_m = getattr(row, 'hpo_doc2hpo_mapped_texts')
                chunk_df.at[row.Index, 'hpo_doc2hpo_mapped_texts'] = doc2_m
        except Exception:
            pass

def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.input:
        cfg.raw["input_pkl"] = args.input
    if args.output:
        cfg.raw["output_pkl"] = args.output

    terms = load_terms(cfg)
    generator = build_generator(cfg, terms)
    ranker = build_ranker(cfg)
    reranker = build_reranker(cfg.reranker_cfg)
    cleaner = PhenotypeCleaner()

    checkpoint_dir = cfg.checkpoint_dir
    use_checkpoints = not args.dry_run
    if use_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading dataframe from %s", cfg.input_pkl)
    df = pd.read_pickle(cfg.input_pkl)
    if args.limit:
        df = df.head(args.limit)
        LOGGER.info("Limiting to first %d rows for debug", args.limit)

    try:
        df = process_dataframe(df, cleaner, generator, ranker, reranker, cfg, checkpoint_dir, use_checkpoints)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user. Exiting without writing output.")
        return

    if args.dry_run:
        LOGGER.info("Dry run enabled; not writing output.")
        return

    cfg.output_pkl.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing enhanced dataframe to %s", cfg.output_pkl)
    df.to_pickle(cfg.output_pkl)

    # also export json metadata for traceability
    meta_path = cfg.output_pkl.with_suffix(".mapping_meta.json")
    meta = {
        "config": cfg.raw,
        "rows": len(df),
        "hpo_terms": len(terms),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", meta_path)


if __name__ == "__main__":
    main()
