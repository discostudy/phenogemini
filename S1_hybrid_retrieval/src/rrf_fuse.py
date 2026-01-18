#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
from collections import defaultdict


def load_run(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df = df.sort_values(['query_id', 'score'], ascending=[True, False])
    df['rank'] = df.groupby('query_id').cumcount() + 1
    return df[['query_id', 'doc_id', 'rank']]


def main():
    ap = argparse.ArgumentParser(description='RRF fuse multiple runs into one run.')
    ap.add_argument('--runs', nargs='+', required=True, help='List of input run TSVs')
    ap.add_argument('--out', required=True, help='Output fused run TSV')
    ap.add_argument('--k', type=int, default=60, help='RRF constant')
    ap.add_argument('--topk_in', type=int, default=1000, help='Only fuse top-N of each run per query')
    args = ap.parse_args()

    cands = []
    for p in args.runs:
        if not os.path.isfile(p):
            continue
        cands.append(load_run(p))

    if not cands:
        raise SystemExit('No valid runs to fuse.')

    scores = defaultdict(float)
    for df in cands:
        for q, d, r in df.itertuples(index=False):
            if r <= args.topk_in:
                scores[(q, d)] += 1.0 / (args.k + r)

    out = pd.DataFrame([
        {'query_id': q, 'doc_id': d, 'score': s}
        for (q, d), s in scores.items()
    ])
    out = out.sort_values(['query_id', 'score'], ascending=[True, False])
    out.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] wrote {args.out} rows={len(out)}')


if __name__ == '__main__':
    main()

