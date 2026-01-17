#!/usr/bin/env python3
import argparse
import math
import os
import sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


def _add_vendor_path():
    # Make S1_hybrid_retrieval/vendor importable
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..', '..'))
    vendor = os.path.join(repo_root, 'S1_hybrid_retrieval', 'vendor')
    if os.path.isdir(vendor) and vendor not in sys.path:
        sys.path.append(vendor)


_add_vendor_path()
try:
    from ek_phenopub.source.pubmed_repository import PubRepository
except Exception as e:
    print('[ERR] cannot import PubRepository from S1_hybrid_retrieval/vendor. Ensure path is correct.', file=sys.stderr)
    raise


def ts():
    import datetime
    return datetime.datetime.now().strftime('[%F %T]')


def parse_args():
    ap = argparse.ArgumentParser(description='Build candidate pool per OMIM (RRF run) and compute rank-only features (s1).')
    ap.add_argument('--repo', required=True)
    ap.add_argument('--diseases', required=True, help='TSV with omim_id column (first col)')
    ap.add_argument('--run', required=True, help='RRF fused run TSV (query_id/doc_id/score or omim_id/pmid/score)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--p0', type=float, default=0.10)
    ap.add_argument('--lmin', type=int, default=10)
    ap.add_argument('--lmax', type=int, default=2000)
    ap.add_argument('--topn', type=int, default=5000)
    ap.add_argument('--alpha_rank', type=float, default=1.0)
    return ap.parse_args()


def load_disease_set(path):
    S = set()
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline()
        if 'omim_id' not in header:
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            omim = line.split('\t', 1)[0].strip()
            if omim:
                S.add(omim)
    return S


def pmid_from_doc_id(doc_id):
    if not doc_id:
        return None
    s = str(doc_id)
    if s.startswith('PM:'):
        s = s[3:]
    return int(s) if s.isdigit() else None


def stream_rrf_and_pool(run_path, disease_set, p0, lmin, lmax, topn):
    # For each OMIM we will collect up to cap = min(lmax, max(lmin, ceil(p0*topn)))
    cap_per_topic = {}
    pool = defaultdict(list)  # omim -> list of (rank, pmid)
    done_topics = set()
    rank_counters = Counter()

    def cap_for(omim):
        if omim not in cap_per_topic:
            cap_per_topic[omim] = min(lmax, max(lmin, math.ceil(p0 * topn)))
        return cap_per_topic[omim]

    with open(run_path, 'r', encoding='utf-8') as f:
        first = f.readline()
        has_header = ('query_id' in first or 'omim' in first)
        if not has_header:
            f.seek(0)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a.startswith('OMIM:'):
                omim = a
                pmid = pmid_from_doc_id(b)
            else:
                continue
            if omim not in disease_set or pmid is None:
                continue
            if omim in done_topics:
                continue
            rank_counters[omim] += 1
            r = rank_counters[omim]
            pool[omim].append((r, pmid))
            if len(pool[omim]) >= cap_for(omim):
                done_topics.add(omim)
        # end for lines
    return pool


# Regex-based features removed: rank-only pipeline


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    diseases = load_disease_set(args.diseases)
    print(ts(), f'Loaded diseases: {len(diseases)}')

    print(ts(), f'Pool from run: {args.run} (p0={args.p0}, Lmin={args.lmin}, Lmax={args.lmax})')
    pool = stream_rrf_and_pool(args.run, diseases, args.p0, args.lmin, args.lmax, args.topn)
    topics = len(pool)
    pool_size = sum(len(v) for v in pool.values())
    print(ts(), f'Pool topics={topics}, pooled pairs={pool_size}')

    # Write pool_S0.tsv and mapping omim->pmid list
    pool_path = os.path.join(args.out_dir, 'pool_S0.tsv')
    with open(pool_path, 'w', encoding='utf-8') as w:
        w.write('omim_id\tpmid\trank\trank_pct\trank_decay\n')
        for omim, items in pool.items():
            for r, pmid in items:
                rank_pct = max(0.0, 1.0 - (r/args.topn))
                rank_decay = (-math.log(1.0 + r)) / (-math.log(1.0 + args.topn))
                w.write(f'{omim}\t{pmid}\t{r}\t{rank_pct:.6f}\t{rank_decay:.6f}\n')
    print(ts(), f'[OK] wrote pool -> {pool_path}')

    # Aggregate unique pmids
    unique_pmids = sorted({pmid for items in pool.values() for _, pmid in items})
    print(ts(), f'Unique pmids in pool: {len(unique_pmids)}')

    # Rank-only features per pmid
    feat_path = os.path.join(args.out_dir, 'features_S1.tsv')
    with open(feat_path, 'w', encoding='utf-8') as w:
        w.write('pmid\trank_pct_max\trank_decay_max\ts1\n')
        # compute maximal rank features across OMIMs for a pmid (best position)
        rank_pct_max = defaultdict(float)
        rank_decay_max = defaultdict(float)
        # Pre-scan pool file to get pmid-wise max rank features
        with open(pool_path, 'r', encoding='utf-8') as f:
            _ = f.readline()
            for line in f:
                _omim, pmid_s, _r, rp_s, rd_s = line.rstrip('\n').split('\t')
                pmid = int(pmid_s)
                rp = float(rp_s)
                rd = float(rd_s)
                if rp > rank_pct_max[pmid]:
                    rank_pct_max[pmid] = rp
                if rd > rank_decay_max[pmid]:
                    rank_decay_max[pmid] = rd
        # Write with rank-only s1
        for pmid in unique_pmids:
            rp = rank_pct_max.get(pmid, 0.0)
            rd = rank_decay_max.get(pmid, 0.0)
            s1 = args.alpha_rank * rp
            w.write(f'{pmid}\t{rp:.6f}\t{rd:.6f}\t{s1:.6f}\n')
    print(ts(), f'[OK] wrote features -> {feat_path}')

    # Also write mapping pairs for later fairness accounting
    pairs_path = os.path.join(args.out_dir, 'pool_pairs.tsv')
    with open(pairs_path, 'w', encoding='utf-8') as w:
        w.write('omim_id\tpmid\n')
        for omim, items in pool.items():
            for _, pmid in items:
                w.write(f'{omim}\t{pmid}\n')
    print(ts(), f'[OK] wrote pairs -> {pairs_path}')


if __name__ == '__main__':
    main()
