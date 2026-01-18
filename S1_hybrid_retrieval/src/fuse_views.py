#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import math
import multiprocessing as mp
from typing import Iterable

import pandas as pd

GLOBAL_DF = None


def _init_worker(df: pd.DataFrame) -> None:
    global GLOBAL_DF
    GLOBAL_DF = df


def _aggregate_chunk(base_ids: Iterable[str]) -> pd.DataFrame:
    chunk = GLOBAL_DF[GLOBAL_DF['query_base'].isin(base_ids)]
    if chunk.empty:
        return pd.DataFrame(columns=['query_id', 'doc_id', 'score'])
    out = (
        chunk.groupby(['query_base', 'doc_id'], as_index=False)['rrf_score']
        .sum()
        .rename(columns={'query_base': 'query_id', 'rrf_score': 'score'})
    )
    return out


def main():
    ap = argparse.ArgumentParser(description='Fuse view-level run (OMIM#gene/def/case) into base OMIM by RRF.')
    ap.add_argument('--run', required=True, help='Input run TSV with columns: query_id, doc_id, score')
    ap.add_argument('--out', required=True, help='Output fused run TSV')
    ap.add_argument('--k', type=int, default=60, help='RRF constant')
    ap.add_argument('--topk_in', type=int, default=1000, help='Use only top-N of each view when fusing')
    ap.add_argument('--workers', type=int, default=1, help='Parallel workers for aggregation (default=1)')
    ap.add_argument('--chunk_size', type=int, default=0, help='Base-query chunk size per worker (default balanced)')
    args = ap.parse_args()

    df = pd.read_csv(args.run, sep='\t', dtype={'query_id': str, 'doc_id': str})
    if df.empty:
        pd.DataFrame(columns=['query_id', 'doc_id', 'score']).to_csv(args.out, sep='\t', index=False)
        print(f'[OK] wrote {args.out} rows=0')
        return

    df = df.sort_values(['query_id', 'score'], ascending=[True, False], kind='mergesort')
    df['rank'] = df.groupby('query_id').cumcount() + 1
    df = df[df['rank'] <= args.topk_in].copy()
    df['query_base'] = df['query_id'].astype(str).str.split('#', n=1).str[0]
    df['rrf_score'] = 1.0 / (args.k + df['rank'])

    workers = max(1, args.workers)
    if workers == 1:
        out = (
            df.groupby(['query_base', 'doc_id'], as_index=False)['rrf_score']
              .sum()
              .rename(columns={'query_base': 'query_id', 'rrf_score': 'score'})
              .sort_values(['query_id', 'score'], ascending=[True, False])
        )
    else:
        base_ids = df['query_base'].unique().tolist()
        if not base_ids:
            out = pd.DataFrame(columns=['query_id', 'doc_id', 'score'])
        else:
            chunk_size = args.chunk_size if args.chunk_size > 0 else math.ceil(len(base_ids) / workers)
            chunks = [base_ids[i:i + chunk_size] for i in range(0, len(base_ids), chunk_size)]
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(df,)) as pool:
                parts = pool.map(_aggregate_chunk, chunks)
            if parts:
                out = pd.concat(parts, ignore_index=True)
                out = (
                    out.groupby(['query_id', 'doc_id'], as_index=False)['score']
                       .sum()
                       .sort_values(['query_id', 'score'], ascending=[True, False])
                )
            else:
                out = pd.DataFrame(columns=['query_id', 'doc_id', 'score'])

    out.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] wrote {args.out} rows={len(out)}')


if __name__ == '__main__':
    main()
