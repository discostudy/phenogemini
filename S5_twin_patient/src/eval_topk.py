#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute top-k metrics for twin-patient runs")
    parser.add_argument("--qrels", required=True, help="TSV containing query_id and gold gene column")
    parser.add_argument("--runs", required=True, help="Run TSV or directory of TSV files")
    parser.add_argument("--out", required=True, help="Where to write the metric summary (CSV)")
    parser.add_argument("--ks", default="1,2,5,10,20,50,100", help="Comma-separated k values")
    parser.add_argument(
        "--qrels-id-column",
        default="entrez_gene_id",
        help="Column in qrels representing the relevant gene/document identifier",
    )
    parser.add_argument(
        "--run-id-column",
        default="gene",
        help="Column in the run file representing the candidate gene/document identifier",
    )
    parser.add_argument(
        "--score-column",
        default="score",
        help="Score column in the run file (used for sorting descending)",
    )
    return parser.parse_args()


def iter_run_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        for candidate in sorted(path.glob("*.tsv")):
            if candidate.is_file():
                yield candidate


def load_qrels(path: Path, id_column: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path, sep="\t")
    if "query_id" not in df.columns:
        raise ValueError(f"qrels file {path} must contain a query_id column")
    if id_column not in df.columns:
        raise ValueError(f"qrels file {path} missing column {id_column}")
    rels: Dict[str, Dict[str, float]] = {}
    for row in df.itertuples(index=False):
        qid = str(getattr(row, "query_id"))
        did = str(getattr(row, id_column))
        rels.setdefault(qid, {})[did] = 1.0
    return rels


def load_run(path: Path, q_column: str, d_column: str, s_column: str) -> Dict[str, List[Tuple[str, float]]]:
    df = pd.read_csv(path, sep="\t")
    for required in (q_column, d_column):
        if required not in df.columns:
            raise ValueError(f"Run file {path} missing column {required}")
    if s_column not in df.columns:
        if "rank" in df.columns:
            df[s_column] = -df["rank"].astype(float)
        else:
            raise ValueError(f"Run file {path} missing score column {s_column} and rank fallback")
    runs: Dict[str, List[Tuple[str, float]]] = {}
    for row in df.itertuples(index=False):
        qid = str(getattr(row, q_column))
        did = str(getattr(row, d_column))
        score = float(getattr(row, s_column))
        runs.setdefault(qid, []).append((did, score))
    for candidates in runs.values():
        candidates.sort(key=lambda x: x[1], reverse=True)
    return runs


def metrics_for_run(runs: Dict[str, List[Tuple[str, float]]],
                    qrels: Dict[str, Dict[str, float]],
                    ks: List[int]) -> Dict[str, float]:
    qids = list(qrels.keys())
    results: Dict[str, float] = {}
    recs = {k: [] for k in ks}
    precs = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    mrrs: List[float] = []
    coverage = 0

    for qid in qids:
        rels = qrels[qid]
        candidates = runs.get(qid, [])
        if candidates:
            coverage += 1
        ideal = sorted(rels.values(), reverse=True)
        total_rel = float(len(rels)) if rels else 0.0
        for k in ks:
            top = candidates[:k]
            rel_hits = sum(1.0 for doc_id, _ in top if rels.get(doc_id, 0.0) > 0.0)
            recs[k].append(rel_hits / total_rel if total_rel > 0 else 0.0)
            precs[k].append(rel_hits / k if k > 0 else 0.0)
            dcg = 0.0
            for rank, (doc_id, _) in enumerate(top, start=1):
                gain = rels.get(doc_id, 0.0)
                if gain > 0.0:
                    dcg += gain / np.log2(rank + 1)
            idcg = 0.0
            for rank, gain in enumerate(ideal[:k], start=1):
                idcg += gain / np.log2(rank + 1)
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)
        reciprocal = 0.0
        for rank, (doc_id, _) in enumerate(candidates, start=1):
            if rels.get(doc_id, 0.0) > 0.0:
                reciprocal = 1.0 / rank
                break
        mrrs.append(reciprocal)

    results["coverage"] = coverage / len(qids) if qids else 0.0
    results["MRR"] = float(np.mean(mrrs)) if mrrs else 0.0
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recs[k])) if recs[k] else 0.0
        results[f"Precision@{k}"] = float(np.mean(precs[k])) if precs[k] else 0.0
        results[f"nDCG@{k}"] = float(np.mean(ndcgs[k])) if ndcgs[k] else 0.0
    return results


def main() -> None:
    args = parse_args()
    ks = [int(value) for value in args.ks.split(",")]
    qrels = load_qrels(Path(args.qrels), args.qrels_id_column)

    rows: List[Dict[str, float]] = []
    for run_path in tqdm(list(iter_run_files(Path(args.runs))), desc="eval", unit="run"):
        runs = load_run(run_path, "query_id", args.run_id_column, args.score_column)
        metrics = metrics_for_run(runs, qrels, ks)
        record = {"run": run_path.name}
        record.update(metrics)
        rows.append(record)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[info] wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
