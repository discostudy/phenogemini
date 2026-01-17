#!/usr/bin/env python3
"""Apply a fine-tuned PubMedBERT reranker to (omim_id, pmid) pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ek_phenopub.source.pubmed_repository import PubRepository


def load_queries(paths: List[Path]) -> Dict[str, str]:
    texts: Dict[str, List[str]] = {}
    for path in paths:
        df = pd.read_csv(path, sep='\t')
        if 'query_id' not in df.columns or 'query_text' not in df.columns:
            raise ValueError(f'Query file {path} missing query_id/query_text columns')
        df['omim_id'] = df['query_id'].astype(str).str.split('#', n=1, expand=True)[0]
        for omim, group in df.groupby('omim_id')['query_text']:
            arr = [t.strip() for t in group.dropna().tolist() if t and isinstance(t, str)]
            if not arr:
                continue
            bucket = texts.setdefault(omim, [])
            for t in arr:
                if t not in bucket:
                    bucket.append(t)
    return {k: '\n'.join(v) for k, v in texts.items()}


def fetch_jtak(pmids: Iterable[int], repo_path: Path) -> Dict[int, str]:
    repo = PubRepository(str(repo_path))
    cache: Dict[int, str] = {}
    for pmid in tqdm(pmids, desc='fetch jtak', unit='pmid'):
        full = repo.get_pub_full(f'PM:{pmid}')
        if not full:
            continue
        doc_id = full['id']
        jtak = repo.get_pub_jtak_by_id(doc_id)
        if jtak and jtak.get('text'):
            cache[pmid] = jtak['text']
    return cache


def batch_indices(n: int, batch_size: int) -> Iterable[range]:
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield range(start, end)


def main() -> None:
    ap = argparse.ArgumentParser(description='Apply PubMedBERT reranker to (omim_id, pmid) pairs')
    ap.add_argument('--pairs', required=True, help='TSV with omim_id, pmid columns')
    ap.add_argument('--queries', required=True, help='Comma-separated list of query TSV files (query_id, query_text)')
    ap.add_argument('--out', required=True, help='Output TSV with scores')
    ap.add_argument('--model-dir', required=True, help='Directory containing fine-tuned model/tokenizer subfolders')
    ap.add_argument('--repo', required=True, help='Path to PubMed SQLite repository used by ek_phenopub')
    ap.add_argument('--max-length', type=int, default=384)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--shards', type=int, default=1, help='Total number of shards to split the input pairs')
    ap.add_argument('--shard-idx', type=int, default=0, help='Index of shard to process (0-based)')
    args = ap.parse_args()

    pairs_df = pd.read_csv(args.pairs, sep='\t')
    pairs_df['pmid'] = pairs_df['pmid'].astype(int)
    pairs_df['omim_id'] = pairs_df['omim_id'].astype(str)

    if args.shards > 1:
        total = len(pairs_df)
        shard_size = (total + args.shards - 1) // args.shards
        start = shard_size * args.shard_idx
        end = min(start + shard_size, total)
        pairs_df = pairs_df.iloc[start:end].reset_index(drop=True)
        print(f'[INFO] shard {args.shard_idx + 1}/{args.shards}: rows {start}-{end} of {total}')

    query_paths = [Path(p.strip()) for p in args.queries.split(',') if p.strip()]
    if not query_paths:
        raise ValueError('At least one query file must be provided via --queries')
    queries = load_queries(query_paths)
    has_query = pairs_df['omim_id'].isin(queries.keys())
    missing_query = pairs_df.loc[~has_query, 'omim_id'].unique()
    if len(missing_query) > 0:
        print(f'[WARN] missing query text for {len(missing_query)} OMIM entries; assigning prob=0')

    pmids = pairs_df['pmid'].unique().tolist()
    jtak_map = fetch_jtak(pmids, Path(args.repo))
    has_doc = pairs_df['pmid'].isin(jtak_map.keys())
    missing_doc = pairs_df.loc[~has_doc, 'pmid'].unique()
    if len(missing_doc) > 0:
        print(f'[WARN] missing JTAC text for {len(missing_doc)} PMIDs; assigning prob=0')

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / 'tokenizer', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir / 'model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    valid_mask = has_query & has_doc
    probs = np.zeros(len(pairs_df), dtype=np.float32)

    if valid_mask.any():
        valid_indices = np.nonzero(valid_mask.to_numpy())[0]
        for start in tqdm(range(0, len(valid_indices), args.batch_size), desc='scoring', unit='batch'):
            end = min(start + args.batch_size, len(valid_indices))
            batch_idx = valid_indices[start:end]
            q_batch = pairs_df['omim_id'].iloc[batch_idx].map(queries).tolist()
            d_batch = [jtak_map[pairs_df['pmid'].iloc[i]] for i in batch_idx]
            enc = tokenizer(
                q_batch,
                d_batch,
                truncation=True,
                padding='max_length',
                max_length=args.max_length,
                return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
                batch_probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs[batch_idx] = batch_probs

    out_df = pairs_df.copy()
    out_df['rerank_prob'] = probs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.shards > 1:
        shard_name = out_path.stem + f'.shard{args.shard_idx:03d}'
        out_path = out_path.with_name(shard_name + out_path.suffix)
    out_df.to_csv(out_path, sep='\t', index=False)
    print(f'[OK] wrote {out_path} ({len(out_df)} rows)')


if __name__ == '__main__':
    main()
