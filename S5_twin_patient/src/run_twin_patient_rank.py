#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_EMBEDDINGS = Path(__file__).resolve().parents[1] / "data" / "samples" / "atlas_embeddings_sample.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PhenoGemini-style gene rankings from phenotype queries"
    )
    parser.add_argument("--queries", required=True, help="TSV with columns query_id and text")
    parser.add_argument("--out", required=True, help="Destination TSV (query_id, gene, score)")
    parser.add_argument(
        "--embeddings",
        default=str(DEFAULT_EMBEDDINGS),
        help="Pickle containing patient embeddings with columns [gene_column, embedding_column]",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="SentenceTransformer model name or path",
    )
    parser.add_argument(
        "--gene-column",
        default="entrez_gene_id",
        help="Column containing gene identifiers in the embeddings file",
    )
    parser.add_argument(
        "--embedding-column",
        default="patient_embeddings",
        help="Column containing list/array embeddings",
    )
    parser.add_argument("--topk", type=int, default=100, help="Rank depth to emit per query")
    parser.add_argument(
        "--phenotype-sep",
        default=";",
        help="Phenotype delimiter inside the query text (used to average embeddings)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip queries whose text is empty instead of encoding the empty string",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional cap on number of queries to process (0 means all)",
    )
    return parser.parse_args()


def load_embeddings(path: Path, gene_col: str, embedding_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_pickle(path)
    missing = [col for col in (gene_col, embedding_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Embeddings file {path} missing columns: {', '.join(missing)}")
    trimmed = df[[gene_col, embedding_col]].dropna()
    vectors = trimmed[embedding_col].apply(lambda vec: np.asarray(vec, dtype=np.float32))
    matrix = np.vstack(vectors.values)
    norms = np.linalg.norm(matrix, axis=1)
    genes = trimmed[gene_col].astype(str).str.strip().values
    return matrix, norms, genes


def encode_query(text: str,
                 model: SentenceTransformer,
                 sep: str,
                 skip_empty: bool) -> Optional[np.ndarray]:
    if not text or not text.strip():
        if skip_empty:
            return None
        return np.asarray(model.encode(text if text is not None else ""), dtype=np.float32)
    phenos = [chunk.strip() for chunk in text.split(sep) if chunk.strip()]
    if not phenos:
        phenos = [text.strip()]
    embeddings = model.encode(phenos)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings.mean(axis=0)


def rank_genes(query_vec: np.ndarray,
               emb_matrix: np.ndarray,
               emb_norms: np.ndarray,
               genes: np.ndarray,
               topk: int) -> List[tuple[str, float]]:
    q_norm = float(np.linalg.norm(query_vec))
    if q_norm == 0.0:
        return []
    sims = emb_matrix @ query_vec
    sims = sims / (emb_norms * q_norm + 1e-8)
    # shortlist for efficiency
    k_candidates = min(len(sims), max(topk * 10, 1000))
    shortlist = np.argpartition(-sims, k_candidates - 1)[:k_candidates]
    ordered = shortlist[np.argsort(-sims[shortlist])]
    seen = set()
    output: List[tuple[str, float]] = []
    for idx in ordered:
        gene = genes[idx]
        if gene in seen:
            continue
        seen.add(gene)
        output.append((gene, float(sims[idx])))
        if len(output) >= topk:
            break
    return output


def main() -> None:
    args = parse_args()
    emb_matrix, emb_norms, genes = load_embeddings(Path(args.embeddings), args.gene_column, args.embedding_column)
    print(f"[info] loaded embeddings: matrix={emb_matrix.shape}, genes={len(genes)}")

    model = SentenceTransformer(args.model)
    queries_df = pd.read_csv(args.queries, sep="\t")
    limit = args.max_queries if args.max_queries and args.max_queries > 0 else len(queries_df)

    rows: List[tuple[str, str, float]] = []
    for _, row in tqdm(list(queries_df.iterrows())[:limit], desc="twin-patient", unit="query"):
        qid = str(row["query_id"])
        text = str(row.get("text", ""))
        query_vec = encode_query(text, model, args.phenotype_sep, args.skip_empty)
        if query_vec is None:
            continue
        ranking = rank_genes(query_vec, emb_matrix, emb_norms, genes, args.topk)
        for gene, score in ranking:
            rows.append((qid, gene, score))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["query_id", "gene", "score"]).to_csv(out_path, sep="\t", index=False)
    print(f"[info] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
