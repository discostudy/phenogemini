#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
from multiprocessing import Pool
from typing import Dict, List

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

# global state for workers
GLOBAL_DOC_TOKENS = None
GLOBAL_QUERIES = None
GLOBAL_DIDS = None
GLOBAL_TOPK = None
GLOBAL_QDOC = None

def _init_worker(doc_tokens, queries, dids, topk_out, q_to_docs):
    global GLOBAL_DOC_TOKENS, GLOBAL_QUERIES, GLOBAL_DIDS, GLOBAL_TOPK, GLOBAL_QDOC
    GLOBAL_DOC_TOKENS = doc_tokens
    GLOBAL_QUERIES = queries
    GLOBAL_DIDS = dids
    GLOBAL_TOPK = topk_out
    GLOBAL_QDOC = q_to_docs


def _score_batch(batch):
    rows = []
    for qid in batch:
        rows.extend(_score_one(qid))
    return rows
def _score_one(qid: str):
    qtext = GLOBAL_QUERIES.get(qid)
    if not qtext:
        return []
    qtok = set(tokenize(qtext))
    if not qtok:
        return []
    did_list = GLOBAL_QDOC.get(qid)
    if not did_list:
        did_list = GLOBAL_DIDS

    rows = []
    scores = []
    did_map = []
    for d in did_list:
        tok = GLOBAL_DOC_TOKENS.get(d)
        if tok is None:
            continue
        scores.append(float(len(qtok & tok)))
        did_map.append(d)
    if not scores:
        # Align query coverage: emit placeholder TopK rows with 0.0 score so
        # every query appears with the same depth across models
        topn = GLOBAL_TOPK if GLOBAL_TOPK and GLOBAL_TOPK > 0 else len(GLOBAL_DIDS)
        topn = min(topn, len(GLOBAL_DIDS))
        for di in range(topn):
            rows.append((qid, GLOBAL_DIDS[di], 0.0))
        return rows
    scores = np.asarray(scores)
    order = np.argsort(-scores)
    if GLOBAL_TOPK and GLOBAL_TOPK > 0:
        order = order[:GLOBAL_TOPK]
    for idx in order:
        sc = float(scores[idx])
        rows.append((qid, did_map[idx], sc))
    return rows

def load_queries(path: str) -> Dict[str, str]:
    df = pd.read_csv(path, sep='\t')
    return dict(zip(df['query_id'].astype(str), df['query_text'].astype(str)))

def load_qdoc_pairs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    return df[['query_id', 'doc_id']].drop_duplicates().reset_index(drop=True)

def fetch_docs_text(repo: PubRepository, doc_ids: List[str]) -> Dict[str, str]:
    out = {}
    for d in tqdm(doc_ids, desc='fetch docs'):
        pmid = _pm_to_int(d)
        full = repo.get_pub_full(f'PM:{pmid}')
        rec = repo.get_pub_jtak_by_id(full['id']) if (full and 'id' in full) else repo.get_pub_jtak_by_id(pmid)
        if not rec or not rec.get('text'):
            continue
        out[d] = rec['text']
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queries', required=True)
    ap.add_argument('--qdoc_pairs', default=None)
    ap.add_argument('--docs', default=None)
    ap.add_argument('--topk_out', type=int, default=2000)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--query_chunk', type=int, default=200)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    qdoc_df = load_qdoc_pairs(args.qdoc_pairs) if args.qdoc_pairs else None
    if args.docs:
        df_docs = pd.read_csv(args.docs, sep='\t')
        col = 'doc_id' if 'doc_id' in df_docs.columns else df_docs.columns[0]
        dids = sorted(set(df_docs[col].astype(str).tolist()))
        qids = sorted(queries.keys())
    else:
        qids = sorted(set(qdoc_df['query_id'].astype(str)))
        dids = sorted(set(qdoc_df['doc_id'].astype(str)))

    repo = PubRepository(args.repo)
    doc_texts_map = fetch_docs_text(repo, dids)
    dids = [d for d in dids if d in doc_texts_map]

    print('[INFO] start tokenize docs', flush=True)
    doc_tokens = {d: set(tokenize(doc_texts_map[d])) for d in tqdm(dids, desc='tokenize docs')}

    q_to_docs = {}
    if qdoc_df is not None:
        for q, grp in qdoc_df.groupby('query_id'):
            q_to_docs[q] = grp['doc_id'].tolist()

    workers = max(1, args.workers)
    rows = []
    if workers > 1:
        chunk_size = max(1, args.query_chunk)
        chunks = [qids[i:i + chunk_size] for i in range(0, len(qids), chunk_size)]
        with Pool(processes=workers, initializer=_init_worker, initargs=(doc_tokens, queries, dids, args.topk_out, q_to_docs)) as pool:
            for chunk_rows in tqdm(pool.imap_unordered(_score_batch, chunks), total=len(chunks), desc='keywords rank', unit='chunk'):
                rows.extend(chunk_rows)
    else:
        _init_worker(doc_tokens, queries, dids, args.topk_out, q_to_docs)
        for q in tqdm(qids, desc='keywords rank'):
            rows.extend(_score_one(q))

    out_df = pd.DataFrame(rows, columns=['query_id', 'doc_id', 'score'])
    out_df.sort_values(['query_id', 'score'], ascending=[True, False], inplace=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] wrote {args.out} with {len(out_df)} rows', flush=True)

if __name__ == '__main__':
    main()
