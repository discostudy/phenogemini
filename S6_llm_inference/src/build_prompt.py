#!/usr/bin/env python3
"""Construct PhenoGemini-LLM prompts by retrieving twin patients from an atlas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS = PACKAGE_ROOT / "data" / "samples" / "atlas_sample.pkl"
GENE_TOKEN_PREFIX = "<|PhenoGemini-Special-Token-Entrez-ID-"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phenotypes", help="Comma-separated phenotype description")
    group.add_argument("--queries", type=Path, help="TSV with columns query_id,phenotypes")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS,
                        help=f"Atlas pickle containing patient embeddings (default: {DEFAULT_ATLAS})")
    parser.add_argument("--twin-count", type=int, default=20,
                        help="Number of twin patients per query")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                        help="SentenceTransformer backbone for query encoding")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional output TSV (query_id,prompt). If omitted, prints prompts")
    parser.add_argument("--verbose", action="store_true", help="Print twin metadata to stderr")
    return parser.parse_args()


def ensure_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def to_gene_token(entrez_id: str) -> str:
    entrez_id = (entrez_id or "").strip()
    if entrez_id.isdigit():
        return f"{GENE_TOKEN_PREFIX}{entrez_id}|>"
    return "<|PhenoGemini-Special-Token-UNKNOWN|>"


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def compute_query_embedding(terms: Iterable[str], model: SentenceTransformer) -> np.ndarray:
    cleaned = [t.strip() for t in terms if t.strip()]
    if not cleaned:
        raise ValueError("No valid phenotype terms provided")
    embeddings = model.encode(cleaned)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    vec = embeddings.mean(axis=0)
    return vec / max(np.linalg.norm(vec), 1e-12)


def build_prompt(phen_terms: Iterable[str], twin_rows: List[dict]) -> str:
    intro = f"For a patient who presents with {', '.join(phen_terms)}."
    bridge = "We can find some patients with similar phenotypes, such as:"
    closing = "Based on all the information above, and my own general knowledge, this patient is likely to have a mutation in"
    lines = [intro, bridge]
    for row in twin_rows:
        phen_text = row.get("phenotypes", "no documented phenotypes") or "no documented phenotypes"
        gene_token = row.get("gene_token") or row.get("gene_symbol") or "an unknown gene"
        lines.append(
            f"A patient presents with {phen_text}. This patient has a mutation in {gene_token}."
        )
    lines.append(closing)
    return "\n".join(lines)


def load_atlas(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    required = {"patient_embeddings", "phenotypes_cleaned", "entrez_gene_id", "gene_symbol_norm"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Atlas {path} missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["patient_embeddings"] = df["patient_embeddings"].apply(lambda v: np.asarray(v, dtype=np.float32))
    df["phenotypes_cleaned"] = df["phenotypes_cleaned"].apply(ensure_list)
    df["entrez_gene_id"] = df["entrez_gene_id"].astype(str).str.strip()
    df["gene_symbol_norm"] = df["gene_symbol_norm"].astype(str).str.strip()
    matrix = np.vstack(df["patient_embeddings"].values)
    df["_embedding_norm"] = np.linalg.norm(matrix, axis=1)
    df["_embedding"] = list(matrix)
    return df


def retrieve_twins(atlas: pd.DataFrame,
                   query_vec: np.ndarray,
                   twin_count: int) -> List[dict]:
    matrix = np.vstack(atlas["_embedding"].values)
    norms = atlas["_embedding_norm"].values
    scores = matrix @ query_vec / np.maximum(norms * np.linalg.norm(query_vec), 1e-12)
    twin_count = min(twin_count, len(scores))
    idx = np.argpartition(scores, -twin_count)[-twin_count:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    rows: List[dict] = []
    for order, i in enumerate(idx, start=1):
        row = atlas.iloc[i]
        phenos = row["phenotypes_cleaned"]
        rows.append(
            {
                "rank": order,
                "patient_uid": row.get("patient_uid", ""),
                "phenotypes": ", ".join(phenos) if phenos else "",
                "gene_symbol": row["gene_symbol_norm"],
                "gene_token": to_gene_token(row["entrez_gene_id"]),
                "similarity": float(scores[i]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    atlas = load_atlas(args.atlas)
    embed_matrix = np.vstack(atlas["_embedding"].values)
    embed_matrix = normalise_rows(embed_matrix)
    atlas["_embedding"] = list(embed_matrix)
    atlas["_embedding_norm"] = np.linalg.norm(embed_matrix, axis=1)

    model = SentenceTransformer(args.model)

    if args.queries:
        queries = pd.read_csv(args.queries, sep='\t')
        if 'query_id' not in queries.columns:
            raise SystemExit("Queries TSV must contain a query_id column")
        phen_col = 'phenotypes' if 'phenotypes' in queries.columns else 'text' if 'text' in queries.columns else None
        if phen_col is None:
            raise SystemExit("Queries TSV must include either a phenotypes or text column")
        outputs = []
        for row in queries.itertuples(index=False):
            raw = getattr(row, phen_col)
            terms = [p.strip() for p in str(raw).replace(',', ';').split(';') if p.strip()]
            query_vec = compute_query_embedding(terms, model)
            twins = retrieve_twins(atlas, query_vec, args.twin_count)
            prompt = build_prompt(terms, twins)
            outputs.append((row.query_id, prompt))
            if args.verbose:
                import sys
                print(f"[info] query {row.query_id}: {len(twins)} twins", file=sys.stderr)
                for twin in twins:
                    print(f"    #{twin['rank']:02d} gene={twin['gene_symbol']} sim={twin['similarity']:.3f}", file=sys.stderr)
        df = pd.DataFrame(outputs, columns=['query_id', 'prompt'])
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, sep='\t', index=False)
        else:
            for qid, prompt in outputs:
                print(f"# prompt for {qid}\n{prompt}\n")
    else:
        phenos = [p.strip() for p in args.phenotypes.split(',') if p.strip()]
        query_vec = compute_query_embedding(phenos, model)
        twins = retrieve_twins(atlas, query_vec, args.twin_count)
        prompt = build_prompt(phenos, twins)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(prompt, encoding='utf-8')
        else:
            print(prompt)


if __name__ == "__main__":
    main()
