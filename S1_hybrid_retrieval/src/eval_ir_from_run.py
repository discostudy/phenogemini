#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_qrels(path: str) -> Dict[str, set]:
    rel = defaultdict(set)
    with open(path, 'r', encoding='utf-8') as f:
        header = None
        for i, ln in enumerate(f):
            parts = ln.rstrip('\n').split('\t')
            if i == 0 and parts and parts[0] == 'query_id':
                header = parts
                continue
            if not parts or len(parts) < 2:
                continue
            qid, did = parts[0], parts[1]
            rel[qid].add(did)
    return rel


def ensure_rank(df: pd.DataFrame) -> pd.DataFrame:
    if 'rank' in df.columns:
        return df
    df = df.sort_values(['query_id', 'score'], ascending=[True, False])
    df['rank'] = df.groupby('query_id').cumcount() + 1
    return df


def ndcg_at_k(rel_flags: List[int], k: int) -> float:
    # Binary gains; DCG@k and ideal DCG@k
    k = min(k, len(rel_flags))
    if k == 0:
        return 0.0
    dcg = 0.0
    for i in range(k):
        if rel_flags[i]:
            dcg += 1.0 / np.log2(i + 2)
    # ideal: top |rels| ones
    gains = sorted(rel_flags, reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(gains):
        if g:
            idcg += 1.0 / np.log2(i + 2)
    return (dcg / idcg) if idcg > 0 else 0.0


def average_precision(rel_flags: List[int]) -> float:
    # AP = mean of precisions at relevant ranks
    hits = 0
    prec_sum = 0.0
    for i, r in enumerate(rel_flags, start=1):
        if r:
            hits += 1
            prec_sum += hits / i
    return (prec_sum / hits) if hits > 0 else 0.0


def precision_at_k(rel_flags: List[int], k: int) -> float:
    k = min(k, len(rel_flags))
    if k == 0:
        return 0.0
    return float(sum(rel_flags[:k])) / k


def hits_by_k(rel_flags: List[int], ks: List[int]) -> Dict[int, int]:
    out = {}
    c = 0
    pos = 0
    for i, r in enumerate(rel_flags, start=1):
        if r:
            c += 1
        while pos < len(ks) and i == ks[pos]:
            out[ks[pos]] = c
            pos += 1
        if pos >= len(ks):
            break
    # Fill tail if ranked list shorter than max k
    for k in ks:
        out.setdefault(k, c)
    return out


def macro_pr_curve(per_query_rel_flags: Dict[str, List[int]], points: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    # Interpolated macro PR at recall grid [0,1] with given points
    grid = np.linspace(0.0, 1.0, points)
    precisions = np.zeros_like(grid)
    valid = 0
    for q, flags in per_query_rel_flags.items():
        m = sum(flags)
        if m == 0:
            continue
        valid += 1
        # Build stepwise recall->precision map
        prec_at_recall = []  # (recall, precision)
        hits = 0
        for i, r in enumerate(flags, start=1):
            if r:
                hits += 1
                prec = hits / i
                rec = hits / m
                prec_at_recall.append((rec, prec))
        # Interpolation: for each target recall, take max precision with recall>=t
        for idx, t in enumerate(grid):
            best = 0.0
            for rec, prec in prec_at_recall:
                if rec >= t and prec > best:
                    best = prec
            precisions[idx] += best
    if valid > 0:
        precisions /= valid
    return grid, precisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qrels', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--recall_k', type=int, default=5000)
    ap.add_argument('--pr_points', type=int, default=11)
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--out_pr', required=True)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    df = pd.read_csv(args.run, sep='\t', dtype={'query_id': str, 'doc_id': str, 'score': float})
    if 'query_id' not in df.columns or 'doc_id' not in df.columns:
        raise SystemExit('run must have columns: query_id, doc_id, [score|rank]')
    df = ensure_rank(df)
    df = df.sort_values(['query_id', 'rank'])

    # Prepare
    ks_grid = [args.recall_k]
    per_query_flags: Dict[str, List[int]] = {}
    n_rel_total = 0
    ndcgs = []
    aps = []
    p10s = []
    recall_hits = 0
    valid_q = 0

    for qid, sub in df.groupby('query_id'):
        rels = qrels.get(qid)
        if not rels:
            continue
        rels = set(rels)
        docs = sub['doc_id'].tolist()
        seen = set()
        ranked_docs = []
        for d in docs:
            if d not in seen:
                seen.add(d)
                ranked_docs.append(d)
        flags = [1 if d in rels else 0 for d in ranked_docs]
        per_query_flags[qid] = flags
        m = len(rels)
        if m == 0:
            continue
        valid_q += 1
        n_rel_total += m
        ndcgs.append(ndcg_at_k(flags, args.k))
        aps.append(average_precision(flags))
        p10s.append(precision_at_k(flags, 10))
        h = hits_by_k(flags, ks_grid)
        recall_hits += h[args.recall_k]

    mean_ndcg10 = float(np.mean(ndcgs)) if ndcgs else 0.0
    mean_map = float(np.mean(aps)) if aps else 0.0
    mean_p10 = float(np.mean(p10s)) if p10s else 0.0
    mean_r_at_k = (recall_hits / n_rel_total) if n_rel_total > 0 else 0.0

    # Macro PR curve and max F1
    r, p = macro_pr_curve(per_query_flags, points=args.pr_points)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    pr_df = pd.DataFrame({'recall': r, 'precision': p, 'F1': f1})
    os.makedirs(os.path.dirname(args.out_pr), exist_ok=True)
    pr_df.to_csv(args.out_pr, index=False)

    out = {
        'queries': valid_q,
        'mean_nDCG@10': mean_ndcg10,
        'mean_MAP': mean_map,
        'mean_P@10': mean_p10,
        f'mean_R@{args.recall_k}': mean_r_at_k,
        'max_F1': float(f1.max() if len(f1) else 0.0),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {args.out_json} and {args.out_pr}")


if __name__ == '__main__':
    main()
