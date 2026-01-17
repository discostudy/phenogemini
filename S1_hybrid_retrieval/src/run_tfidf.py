#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
from typing import Dict, List
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from ek_phenopub.source.pubmed_repository import PubRepository

PM_PAT = re.compile(r'^PM:(\d+)$')

def _pm_to_int(doc_id: str) -> int:
    m = PM_PAT.match(doc_id)
    if not m:
        raise ValueError(f'Invalid doc_id (expect PM:<pmid>): {doc_id}')
    return int(m.group(1))

def load_queries(queries_tsv: str) -> Dict[str, str]:
    df = pd.read_csv(queries_tsv, sep='\t')
    return dict(zip(df['query_id'].astype(str), df['query_text'].astype(str)))

def load_qdoc_pairs(qdoc_pairs_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(qdoc_pairs_tsv, sep='\t')
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
    ap.add_argument('--min_df', type=int, default=2)
    ap.add_argument('--max_df', type=float, default=0.95)
    ap.add_argument('--ngram_max', type=int, default=2)
    ap.add_argument('--query_chunk', type=int, default=200, help='Number of queries per scoring chunk (default=200)')
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

    print('[INFO] fit TF-IDF on docs', flush=True)
    vectorizer = TfidfVectorizer(ngram_range=(1, args.ngram_max), sublinear_tf=True, min_df=args.min_df, max_df=args.max_df)
    X = vectorizer.fit_transform([doc_texts_map[d] for d in dids])

    q_texts = [queries[q] for q in qids]
    print('[INFO] transform queries', flush=True)
    Q = vectorizer.transform(q_texts)

    rows = []
    topk = args.topk_out if args.topk_out and args.topk_out > 0 else None
    total_queries = len(qids)
    chunk_size = args.query_chunk if args.query_chunk > 0 else total_queries
    num_chunks = math.ceil(total_queries / chunk_size) if chunk_size else 1

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(total_queries, start + chunk_size)
        qblock_ids = qids[start:end]
        if not qblock_ids:
            continue
        Q_block = Q[start:end]
        print(f"[tfidf] compute scores chunk {chunk_idx+1}/{num_chunks} (queries {start+1}-{end})", flush=True)
        score_block = Q_block @ X.T
        for offset, qid in enumerate(qblock_ids):
            row: sparse.csr_matrix = score_block.getrow(offset)
            if row.nnz == 0:
                # Keep query coverage aligned: emit TopK placeholder rows with 0.0
                # scores so that every query appears with the same depth.
                topn = topk if topk is not None else len(dids)
                topn = min(topn, len(dids))
                for di in range(topn):
                    rows.append((qid, dids[di], 0.0))
                continue
            data = row.data
            indices = row.indices
            if topk is not None and row.nnz > topk:
                top_idx = np.argpartition(data, -topk)[-topk:]
                top_scores = data[top_idx]
                top_indices = indices[top_idx]
                order = np.argsort(-top_scores)
                for pos in order:
                    rows.append((qid, dids[top_indices[pos]], float(top_scores[pos])))
            else:
                order = np.argsort(-data)
                for pos in order:
                    rows.append((qid, dids[indices[pos]], float(data[pos])))
        print(f"[tfidf] processed {end}/{total_queries} queries", flush=True)

    out_df = pd.DataFrame(rows, columns=['query_id', 'doc_id', 'score'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] wrote {args.out} with {len(out_df)} rows', flush=True)

if __name__ == '__main__':
    main()
