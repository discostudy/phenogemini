#!/usr/bin/env python3
"""Compute Top-K accuracy for PhenoGemini-LLM outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="TSV with query_id,entrez_gene_id,probability")
    parser.add_argument("--gold", required=True, help="TSV with query_id,entrez_gene_id")
    parser.add_argument("--out", required=True, help="Output TSV for summary metrics")
    parser.add_argument("--ks", default="1,3,5,10,20,50,100", help="Comma-separated rank cutoffs")
    return parser.parse_args()


def load_predictions(path: Path) -> Dict[str, List[Tuple[str, float]]]:
    df = pd.read_csv(path, sep='\t')
    required = {'query_id', 'entrez_gene_id', 'probability'}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Prediction file {path} missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df['probability'] = pd.to_numeric(df['probability'], errors='coerce').fillna(0.0)
    grouped: Dict[str, List[Tuple[str, float]]] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(str(row.query_id), []).append((str(row.entrez_gene_id), float(row.probability)))
    for values in grouped.values():
        values.sort(key=lambda x: x[1], reverse=True)
    return grouped


def load_gold(path: Path) -> Dict[str, List[str]]:
    df = pd.read_csv(path, sep='\t')
    required = {'query_id', 'entrez_gene_id'}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Gold file {path} missing columns: {', '.join(sorted(missing))}")
    grouped: Dict[str, List[str]] = {}
    for row in df.itertuples(index=False):
        grouped.setdefault(str(row.query_id), []).append(str(row.entrez_gene_id))
    return grouped


def compute_metrics(preds: Dict[str, List[Tuple[str, float]]],
                    gold: Dict[str, List[str]],
                    ks: Iterable[int]) -> Dict[str, float]:
    ks = list(sorted(set(ks)))
    out: Dict[str, float] = {}
    recalls = {k: [] for k in ks}
    precisions = {k: [] for k in ks}
    for qid, rels in gold.items():
        rel_set = set(rels)
        candidates = preds.get(qid, [])
        for k in ks:
            top = candidates[:k]
            hits = sum(1 for doc_id, _ in top if doc_id in rel_set)
            recalls[k].append(hits / len(rel_set) if rel_set else 0.0)
            precisions[k].append(hits / k if k else 0.0)
    for k in ks:
        out[f'top{k}_recall'] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        out[f'top{k}_precision'] = float(np.mean(precisions[k])) if precisions[k] else 0.0
    return out


def main() -> None:
    args = parse_args()
    ks = [int(x) for x in args.ks.split(',') if x.strip()]
    preds = load_predictions(Path(args.pred))
    gold = load_gold(Path(args.gold))
    metrics = compute_metrics(preds, gold, ks)
    df = pd.DataFrame([metrics])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep='\t', index=False)
    print(df)


if __name__ == "__main__":
    main()
