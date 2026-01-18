#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import math
import os
import re
import multiprocessing as mp
from collections import Counter, defaultdict
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ek_phenopub.source.pubmed_repository import PubRepository

PM_PAT = re.compile(r'^PM:(\d+)$')
TOKEN_PAT = re.compile(r"[A-Za-z0-9_+-]+")


def _pm_to_int(doc_id: str) -> int:
    m = PM_PAT.match(doc_id)
    if not m:
        raise ValueError(f'Invalid doc_id (expect PM:<pmid>): {doc_id}')
    return int(m.group(1))


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    return TOKEN_PAT.findall(text)


def load_queries(queries_tsv: str) -> Dict[str, str]:
    df = pd.read_csv(queries_tsv, sep='\t')
    return dict(zip(df['query_id'].astype(str), df['query_text'].astype(str)))


def load_docs_list(docs_tsv: str) -> List[str]:
    df = pd.read_csv(docs_tsv, sep='\t')
    col = 'doc_id' if 'doc_id' in df.columns else df.columns[0]
    return [str(x) for x in df[col].tolist()]


def fetch_docs_text(repo: PubRepository, doc_ids: List[str]) -> Dict[str, str]:
    out = {}
    for d in tqdm(doc_ids, desc='fetch docs'):
        pmid = _pm_to_int(d)
        full = repo.get_pub_full(f'PM:{pmid}')
        if full and 'id' in full:
            rec = repo.get_pub_jtak_by_id(full['id'])
        else:
            rec = repo.get_pub_jtak_by_id(pmid)
        if not rec or not rec.get('text'):
            continue
        out[d] = rec['text']
    return out


def build_index(doc_texts_map: Dict[str, str]):
    # Returns: dids, doc_len, avgdl, postings (token->list of (doc_idx, tf)), idf
    dids = list(doc_texts_map.keys())
    N = len(dids)
    doc_len = np.zeros(N, dtype=np.int32)
    # Build postings and df
    postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    df: Dict[str, int] = defaultdict(int)

    for i, d in enumerate(tqdm(dids, desc='tokenize+index docs')):
        tokens = tokenize(doc_texts_map[d])
        doc_len[i] = len(tokens)
        if not tokens:
            continue
        tf = Counter(tokens)
        for t, f in tf.items():
            postings[t].append((i, f))
        for t in tf.keys():
            df[t] += 1

    avgdl = float(doc_len.mean()) if N > 0 else 0.0
    # Okapi BM25 idf
    idf = {}
    for t, dft in df.items():
        idf[t] = math.log((N - dft + 0.5) / (dft + 0.5) + 1.0)
    return dids, doc_len, avgdl, postings, idf


def bm25_score_query(q_tokens: List[str],
                     postings: Dict[str, List[Tuple[int, int]]],
                     idf: Dict[str, float],
                     doc_len: np.ndarray,
                     avgdl: float,
                     k1: float = 1.5,
                     b: float = 0.75) -> Dict[int, float]:
    scores: Dict[int, float] = defaultdict(float)
    if avgdl <= 0:
        return scores
    # Use unique query tokens (ignoring qf) for simplicity
    for t in set(q_tokens):
        if t not in idf:
            continue
        t_idf = idf[t]
        for di, tf in postings.get(t, []):
            denom = tf + k1 * (1.0 - b + b * (doc_len[di] / avgdl))
            s = t_idf * (tf * (k1 + 1.0) / denom)
            scores[di] += s
    return scores


POSTINGS = None
IDF = None
DOC_LEN = None
AVGDL = None
DIDS = None
K1 = 1.5
B = 0.75
TOPK = 0


def _init_worker(postings, idf, doc_len, avgdl, dids, k1, b, topk):
    global POSTINGS, IDF, DOC_LEN, AVGDL, DIDS, K1, B, TOPK
    POSTINGS = postings
    IDF = idf
    DOC_LEN = doc_len
    AVGDL = avgdl
    DIDS = dids
    K1 = k1
    B = b
    TOPK = topk


def _process_query_batch(batch: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    for qid, qtext in batch:
        q_tokens = tokenize(qtext)
        scores = bm25_score_query(q_tokens, POSTINGS, IDF, DOC_LEN, AVGDL, k1=K1, b=B)
        if not scores:
            # Keep query coverage aligned across models: even if no term matches,
            # emit TopK placeholder rows with score 0.0 so this query appears.
            topn = TOPK if TOPK and TOPK > 0 else len(DIDS)
            topn = min(topn, len(DIDS))
            for di in range(topn):
                rows.append((qid, DIDS[di], 0.0))
            continue
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if TOPK and TOPK > 0:
            items = items[:TOPK]
        for di, score in items:
            rows.append((qid, DIDS[di], float(score)))
    return rows


def _chunk_list(items: List[Tuple[str, str]], chunk_size: int) -> List[List[Tuple[str, str]]]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--docs', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', default='runs/bm25.tsv')
    ap.add_argument('--topk_out', type=int, default=1000)
    ap.add_argument('--k1', type=float, default=1.5)
    ap.add_argument('--b', type=float, default=0.75)
    ap.add_argument('--workers', type=int, default=1, help='Parallel worker processes for query scoring (default=1)')
    ap.add_argument('--query_chunk', type=int, default=100, help='Queries per worker batch')
    args = ap.parse_args()

    queries = load_queries(args.queries)
    dids_all = load_docs_list(args.docs)

    # Fetch doc texts
    repo = PubRepository(args.repo)
    doc_texts_map = fetch_docs_text(repo, dids_all)
    dids = [d for d in dids_all if d in doc_texts_map]

    # Build BM25 index
    dids, doc_len, avgdl, postings, idf = build_index(doc_texts_map)

    # Score per query
    q_items = [(qid, queries[qid]) for qid in sorted(queries.keys())]
    if args.workers > 1 and q_items:
        ctx = mp.get_context('fork')
        chunks = _chunk_list(q_items, args.query_chunk)
        total_chunks = len(chunks)
        with ctx.Pool(processes=args.workers, initializer=_init_worker,
                      initargs=(postings, idf, doc_len, avgdl, dids, args.k1, args.b, args.topk_out)) as pool:
            rows = []
            for idx, batch_rows in enumerate(pool.imap_unordered(_process_query_batch, chunks), 1):
                rows.extend(batch_rows)
                print(f"[bm25] processed chunks {idx}/{total_chunks}", flush=True)
    else:
        _init_worker(postings, idf, doc_len, avgdl, dids, args.k1, args.b, args.topk_out)
        rows = _process_query_batch(q_items)

    out_df = pd.DataFrame(rows, columns=['query_id', 'doc_id', 'score'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] wrote {args.out} with {len(out_df)} rows')

if __name__ == '__main__':
    main()
