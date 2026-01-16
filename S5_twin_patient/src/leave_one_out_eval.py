#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PACKAGE_ROOT / "data" / "samples" / "atlas_embeddings_sample.pkl"



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate leave-one-out gene ranking for PhenoGemini embeddings"
    )
    ap.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Pickled dataset with patient_uid, entrez_gene_id, phenotypes",
    )
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model used to encode phenotype text",
    )
    ap.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Batch size for encoding phenotype texts",
    )
    ap.add_argument(
        "--mp-chunk-size",
        type=int,
        default=10000,
        help="When using multi-process encoding on multiple devices, split texts into chunks of this size and show progress",
    )
    ap.add_argument(
        "--encode-devices",
        default="",
        help="Comma-separated devices for multi-process encoding, e.g. cuda:0,cuda:1",
    )
    ap.add_argument(
        "--dedupe-phrases",
        action="store_true",
        help="Encode unique phenotype phrases only to reduce computation",
    )
    ap.add_argument(
        "--score-device",
        default="auto",
        help="Device for scoring: auto/cpu/cuda:0/...",
    )
    ap.add_argument("--sample-size", type=int, default=200, help="Number of random queries (0 = all)")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed for sampling")
    ap.add_argument("--topk", type=int, default=10, help="Backward-compatible top-k cutoff for hit-rate")
    ap.add_argument(
        "--ks",
        default="1,2,5,10,20,50,100",
        help="Comma-separated list of k values for hit-rate/nDCG (auto-includes --topk)",
    )
    ap.add_argument(
        "--pmid-column",
        default="pmid",
        help="Column containing PubMed IDs; used to drop same-article candidates",
    )
    ap.add_argument(
        "--exclude-unknown-genes",
        action="store_true",
        help="Exclude rows with unknown/invalid gene symbols (UNKNOWN/NONE/NA/N/A/ERROR/empty)",
    )
    ap.add_argument(
        "--gene-col",
        default="entrez_gene_id",
        help="Dataset column containing Entrez gene IDs (default: entrez_gene_id)",
    )
    ap.add_argument(
        "--gene-symbol-col",
        default="gene_symbol_norm",
        help="Optional column containing display gene symbols",
    )
    # Historical flag name kept for backwards compatibility; newer scripts use
    # --disable-pmid-mask to match the time-split evaluator.
    ap.add_argument(
        "--disable-pmid-filter",
        action="store_true",
        help="If set, do not exclude candidates sharing the query PMID",
    )
    ap.add_argument(
        "--disable-pmid-mask",
        action="store_true",
        help="Alias for --disable-pmid-filter",
    )
    ap.add_argument(
        "--out",
        default="6.evaluation/out/leave_one_out_metrics.json",
        help="Path to summary JSON output",
    )
    ap.add_argument(
        "--uids-file",
        default="",
        help="Optional file listing patient_uids (one per line) to evaluate; otherwise evaluate all",
    )
    ap.add_argument(
        "--case-out",
        default="",
        help="Optional TSV path to store per-query ranks and top predictions",
    )
    ap.add_argument(
        "--case-topk",
        type=int,
        default=5,
        help="How many top genes to record per query when writing --case-out",
    )
    return ap.parse_args()


def build_embeddings_from_dataset(data_path: Path,
                                  model_name: str,
                                  batch_size: int,
                                  pmid_col: str,
                                  gene_col: str,
                                  gene_symbol_col: str,
                                  exclude_unknown: bool,
                                  encode_devices: Optional[List[str]] = None,
                                  dedupe_phrases: bool = False,
                                  mp_chunk_size: int = 10000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    df = pd.read_pickle(data_path)
    cols = set(df.columns)
    for c in ("patient_uid", gene_col):
        if c not in cols:
            raise ValueError(f"Dataset missing column: {c}")
    if pmid_col not in cols:
        df[pmid_col] = ""

    df = df.copy()
    df[gene_col] = df[gene_col].astype(str).str.strip()
    valid_mask = df[gene_col] != ""
    if exclude_unknown:
        df = df[valid_mask]
    else:
        df = df[valid_mask]

    if df.empty:
        raise ValueError("No rows remain after filtering for valid Entrez gene IDs")

    # Collect phenotype phrases per row
    def iter_phrases(row) -> List[str]:
        if "phenotypes_cleaned" in df.columns and isinstance(row.phenotypes_cleaned, list):
            return [str(x).strip() for x in row.phenotypes_cleaned if str(x).strip() and str(x).strip().lower() != "error"]
        val = row.phenotype if "phenotype" in df.columns else None
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip() and str(x).strip().lower() != "error"]
        if isinstance(val, str) and val.strip():
            s = val.strip()
            return [s] if s.lower() != "error" else []
        return []

    rows = list(df.itertuples(index=False))
    patient_uids = df["patient_uid"].astype("string").fillna("").to_numpy()
    genes = df[gene_col].astype(str).to_numpy()
    if gene_symbol_col and gene_symbol_col in df.columns:
        gene_symbols = df[gene_symbol_col].astype("string").fillna("").to_numpy()
    else:
        gene_symbols = np.array(["" for _ in range(len(df))], dtype=object)
    pmids = df[pmid_col].astype("string").fillna("").str.strip().to_numpy()

    # Flatten phrases
    flat_idx: List[int] = []
    flat_texts: List[str] = []
    for i, r in enumerate(rows):
        phrases = iter_phrases(r)
        for p in phrases:
            flat_idx.append(i)
            flat_texts.append(p)

    if not flat_texts:
        raise ValueError("No phenotype text found to encode")

    # Encode
    model = SentenceTransformer(model_name)
    emb_dim = model.get_sentence_embedding_dimension()

    if dedupe_phrases:
        # Map to unique texts to reduce compute
        unique_map: Dict[str, int] = {}
        unique_texts: List[str] = []
        unique_ids: List[int] = []
        for t in flat_texts:
            uid = unique_map.get(t)
            if uid is None:
                uid = len(unique_texts)
                unique_map[t] = uid
                unique_texts.append(t)
            unique_ids.append(uid)
        texts_to_encode = unique_texts
    else:
        unique_ids = None
        texts_to_encode = flat_texts

    embs: np.ndarray
    devices = [d.strip() for d in (encode_devices or []) if d.strip()]
    if devices and len(devices) > 1:
        pool = model.start_multi_process_pool(devices)
        try:
            # Split into chunks to expose progress
            embs_list: List[np.ndarray] = []
            n = len(texts_to_encode)
            for start in tqdm(range(0, n, mp_chunk_size), desc="Batches", unit="chunk"):
                end = min(start + mp_chunk_size, n)
                part = model.encode_multi_process(texts_to_encode[start:end], pool)
                embs_list.append(np.asarray(part))
            embs = np.vstack(embs_list)
        finally:
            model.stop_multi_process_pool(pool)
    else:
        embs = model.encode(texts_to_encode, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    if dedupe_phrases:
        # Expand back to original order
        embs = embs[np.asarray(unique_ids, dtype=np.int64)]

    # Aggregate mean per row
    sums = np.zeros((len(rows), emb_dim), dtype=np.float32)
    cnts = np.zeros((len(rows),), dtype=np.int32)
    for i, vec in zip(flat_idx, embs):
        sums[i] += vec.astype(np.float32)
        cnts[i] += 1
    # For rows with no phrases, keep zeros; else divide
    mask = cnts > 0
    sums[mask] /= cnts[mask, None]

    return sums, genes, pmids, patient_uids, gene_symbols


def stack_matrix(vecs: Sequence[Sequence[float]]) -> np.ndarray:
    return np.vstack([np.asarray(v, dtype=np.float32) for v in vecs])


def build_gene_index(genes: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, int], np.ndarray]:
    mapping: Dict[str, list[int]] = {}
    for idx, gene in enumerate(genes):
        mapping.setdefault(gene, []).append(idx)
    gene_to_indices = {gene: np.asarray(indices, dtype=np.int32) for gene, indices in mapping.items()}
    # Assign compact ids
    gene_to_id: Dict[str, int] = {g: i for i, g in enumerate(gene_to_indices.keys())}
    idx_to_gid = np.empty(len(genes), dtype=np.int64)
    for g, idxs in gene_to_indices.items():
        idx_to_gid[idxs] = gene_to_id[g]
    return gene_to_indices, gene_to_id, idx_to_gid


def main() -> None:
    args = parse_args()
    devices = [d.strip() for d in args.encode_devices.split(',') if d.strip()]
    matrix, genes, pmids, patient_uids, gene_symbols = build_embeddings_from_dataset(
        Path(args.data),
        args.model,
        args.encode_batch_size,
        args.pmid_column,
        args.gene_col,
        args.gene_symbol_col,
        args.exclude_unknown_genes,
        devices,
        args.dedupe_phrases,
        args.mp_chunk_size,
    )
    pmid_available = True

    norms = np.linalg.norm(matrix, axis=1)
    mask = norms > 0
    if not np.all(mask):
        keep = np.count_nonzero(mask)
        print(f"[info] filtered {len(norms) - keep} zero-norm embeddings")
        genes = genes[mask]
        pmids = pmids[mask]
        matrix = matrix[mask]
        patient_uids = patient_uids[mask]
        gene_symbols = gene_symbols[mask]
        norms = norms[mask]

    gene_to_indices, gene_to_id, idx_to_gid = build_gene_index(genes)
    id_to_gene = [''] * len(gene_to_id)
    for gene, gid in gene_to_id.items():
        id_to_gene[gid] = gene

    symbol_lookup: Dict[str, str] = {}
    for gid, sym in zip(genes, gene_symbols):
        if gid and gid not in symbol_lookup and isinstance(sym, str) and sym:
            symbol_lookup[gid] = sym
    id_to_symbol = [''] * len(gene_to_id)
    for gene, gid in gene_to_id.items():
        id_to_symbol[gid] = symbol_lookup.get(gene, '')

    import torch
    if args.score_device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.score_device)
    mt = torch.from_numpy(matrix).to(device)
    nt = torch.from_numpy(norms).to(device)
    gid_t = torch.as_tensor(idx_to_gid, device=device, dtype=torch.long)

    total = len(matrix)
    uid_to_index = {str(uid): idx for idx, uid in enumerate(patient_uids)}
    eval_indices: List[int] = []
    requested_uids: List[int] = []
    missing_uids: List[int] = []
    if args.uids_file:
        for line in Path(args.uids_file).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            uid = line
            requested_uids.append(uid)
            idx = uid_to_index.get(uid)
            if idx is None:
                missing_uids.append(uid)
                continue
            eval_indices.append(idx)
    else:
        eval_indices = list(range(total))

    if not eval_indices:
        raise ValueError('No evaluation indices available (uids not found or empty dataset)')

    rng = np.random.default_rng(args.seed)
    if args.sample_size and args.sample_size > 0 and args.sample_size < len(eval_indices):
        eval_indices = rng.choice(eval_indices, size=args.sample_size, replace=False).tolist()

    ks = set()
    if args.topk and args.topk > 0:
        ks.add(args.topk)
    if args.ks:
        for token in args.ks.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                ks.add(int(token))
            except ValueError:
                raise ValueError(f"Invalid k value: {token}")
    ks = sorted(k for k in ks if k > 0)
    if not ks:
        raise ValueError("No valid k values provided")

    hit_counts = {k: 0 for k in ks}
    ndcg_totals = {k: 0.0 for k in ks}
    mrr = 0.0
    coverage = 0
    singletons = 0
    ranks: List[float] = []
    cases: Optional[List[Dict[str, object]]] = [] if args.case_out else None

    G = len(gene_to_indices)
    disable_pmid = args.disable_pmid_filter or args.disable_pmid_mask

    for idx in tqdm(eval_indices, desc="leave-one-out", unit="query"):
        true_gene = genes[idx]
        true_indices = gene_to_indices[true_gene]
        if len(true_indices) == 1:
            singletons += 1
        # Torch-based cosine scoring on device
        q = mt[idx]
        q_norm = torch.linalg.vector_norm(q)
        if q_norm <= 0:
            continue
        scores = (mt @ q) / (nt * q_norm + 1e-8)
        # mask out self
        scores[idx] = -float('inf')
        # mask out same PMID if required
        if not disable_pmid:
            query_pmid = pmids[idx]
            if query_pmid:
                same_mask = torch.from_numpy((pmids == query_pmid)).to(device)
                scores[same_mask] = -float('inf')

        # Per-gene max via scatter_reduce
        gene_max = torch.full((G,), -float('inf'), device=device, dtype=mt.dtype)
        gene_max.scatter_reduce_(0, gid_t, scores, reduce='amax', include_self=True)

        # rank by counting genes with higher score than true gene
        true_gid = gene_to_id[true_gene]
        true_score = gene_max[true_gid]
        if torch.isneginf(true_score):
            rank = float('inf')
        else:
            rank = int((gene_max > true_score).sum().item()) + 1
        coverage += 1
        ranks.append(rank)
        if np.isfinite(rank):
            inv_rank = 1.0 / rank
            mrr += inv_rank
            for k in ks:
                if rank <= k:
                    hit_counts[k] += 1
                    ndcg_totals[k] += 1.0 / np.log2(rank + 1)
        else:
            for k in ks:
                ndcg_totals[k] += 0.0

        if cases is not None:
            topk_genes: List[str] = []
            topk_scores: List[float] = []
            topk_gene_symbols: List[str] = []
            if args.case_out:
                topn = min(args.case_topk, G)
                if topn > 0:
                    top_vals, top_idx = torch.topk(gene_max, k=topn)
                    topk_genes = [id_to_gene[i] for i in top_idx.tolist()]
                    topk_scores = [float(v) for v in top_vals.tolist()]
                    topk_gene_symbols = [id_to_symbol[i] for i in top_idx.tolist()]
            cases.append({
                'patient_uid': str(patient_uids[idx]),
                'pmid': str(pmids[idx]),
                'true_gene': true_gene,
                'true_gene_symbol': symbol_lookup.get(true_gene, ''),
                'rank': int(rank) if np.isfinite(rank) else None,
                'correct_score': float(true_score) if torch.isfinite(true_score) else None,
                'pmid_masked': not disable_pmid,
                'top_genes': '|'.join(topk_genes) if topk_genes else '',
                'top_gene_symbols': '|'.join(topk_gene_symbols) if topk_gene_symbols else '',
                'top_scores': '|'.join(f'{s:.6f}' for s in topk_scores) if topk_scores else '',
            })

    evaluated = len(eval_indices)
    finite_ranks = [r for r in ranks if np.isfinite(r)]
    mean_rank = float(np.mean(finite_ranks)) if finite_ranks else None
    median_rank = float(np.median(finite_ranks)) if finite_ranks else None

    summary = {
        "records_total": int(total),
        "sampled": int(evaluated),
        "mrr": mrr / evaluated if evaluated else 0.0,
        "coverage": coverage / evaluated if evaluated else 0.0,
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "singletons_removed": int(singletons),
        "topk": args.topk,
        "k_list": ks,
        "sample_size": evaluated,
        "seed": args.seed,
        "pmid_column": args.pmid_column if pmid_available else None,
        "pmid_mask": not disable_pmid,
        "data": str(Path(args.data).resolve()),
        "model": args.model,
        "gene_column": args.gene_col,
        "gene_symbol_column": args.gene_symbol_col,
    }
    summary["uids_requested"] = len(requested_uids) if args.uids_file else None
    summary["uids_missing"] = len(missing_uids) if args.uids_file else 0

    for k in ks:
        summary[f"top{k}_hit_rate"] = hit_counts[k] / evaluated if evaluated else 0.0
        summary[f"ndcg@{k}"] = ndcg_totals[k] / evaluated if evaluated else 0.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.case_out:
        case_path = Path(args.case_out)
        case_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame(cases).to_csv(case_path, sep='\t', index=False)
        print(f"[info] wrote case breakdown -> {case_path}")


if __name__ == "__main__":
    main()
