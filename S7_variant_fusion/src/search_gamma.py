#!/usr/bin/env python3
"""Grid-search the γ parameter for combining variant scores with LLM logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from fusion_utils import (
    TOP_K_CUTOFFS,
    compute_midrank,
    evaluate_strategy,
    load_baseline,
    load_llm_logits,
    load_truth,
    load_uid_list,
    merge_candidate_table,
    summarise,
)

DEFAULT_DATA = Path(__file__).resolve().parents[1] / 'data' / 'samples'
DEFAULT_BASELINE = DEFAULT_DATA / 'sample_variant_scores.tsv'
DEFAULT_LLM = DEFAULT_DATA / 'sample_llm_logits.tsv'
DEFAULT_TRUTH = DEFAULT_DATA / 'sample_ground_truth.tsv'
DEFAULT_UIDS = DEFAULT_DATA / 'sample_dev_uids.tsv'
DEFAULT_OUTPUT = Path('runs/dev')
DEFAULT_GAMMA_GRID = [round(x, 3) for x in np.linspace(0.0, 0.3, 16)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                        help='Baseline variant scores (TSV)')
    parser.add_argument('--llm', type=Path, default=DEFAULT_LLM,
                        help='LLM logits (TSV)')
    parser.add_argument('--truth', type=Path, default=DEFAULT_TRUTH,
                        help='Ground truth genes (TSV)')
    parser.add_argument('--uids', type=Path, default=DEFAULT_UIDS,
                        help='UID list (txt/csv/tsv) for γ search (dev split)')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT,
                        help='Output directory (default: runs/dev)')
    parser.add_argument('--gamma-grid', type=str, default=None,
                        help='Comma-separated γ values (default: 0.0→0.3 step 0.02)')
    parser.add_argument('--save-ranks', action='store_true',
                        help='Write per-patient midrank TSVs for baseline/LLM/best fusion')
    return parser.parse_args()


def parse_gamma_grid(spec: str) -> List[float]:
    values: List[float] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError as exc:
            raise SystemExit(f'Unable to parse γ value: "{chunk}"') from exc
    if not values:
        raise SystemExit('Gamma grid is empty; please provide valid numbers')
    return values


def evaluate_fusion(uids: Iterable[str],
                    baseline_df: pd.DataFrame,
                    llm_df: pd.DataFrame,
                    truth_map: Dict[str, str],
                    gamma: float) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for uid in uids:
        target = truth_map.get(uid)
        if not target:
            continue
        merged = merge_candidate_table(uid, baseline_df, llm_df)
        if merged.empty or merged['baseline_score'].dropna().empty:
            continue
        min_logit = merged['logit'].min() if 'logit' in merged.columns and not merged['logit'].dropna().empty else -10.0
        merged['logit'] = merged['logit'].fillna(min_logit - 1.0)
        merged['fused_score'] = merged['baseline_score'] + gamma * merged['logit']
        midrank = compute_midrank(merged, 'fused_score', target)
        if midrank is None:
            continue
        records.append({
            'patient_uid': uid,
            'gene_symbol': target,
            'mid_rank': midrank,
            'gamma': gamma,
        })
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = load_baseline(args.baseline)
    llm_df = load_llm_logits(args.llm)
    truth_map = load_truth(args.truth)
    dev_uids = load_uid_list(args.uids)

    gamma_values = parse_gamma_grid(args.gamma_grid) if args.gamma_grid else DEFAULT_GAMMA_GRID

    baseline_ranks = evaluate_strategy(dev_uids, baseline_df, None, truth_map, 'baseline_score')
    llm_ranks = evaluate_strategy(dev_uids, baseline_df, llm_df, truth_map, 'logit', fallback_logit=-10.0)

    baseline_metrics = summarise(baseline_ranks)
    baseline_metrics['strategy'] = 'baseline'
    llm_metrics = summarise(llm_ranks)
    llm_metrics['strategy'] = 'llm_only'

    metrics_rows = [baseline_metrics, llm_metrics]
    best_gamma = None
    best_top10 = -np.inf
    best_ranks = pd.DataFrame()

    for gamma in gamma_values:
        fused_ranks = evaluate_fusion(dev_uids, baseline_df, llm_df, truth_map, gamma)
        fused_metrics = summarise(fused_ranks)
        fused_metrics['gamma'] = gamma
        fused_metrics['strategy'] = 'fusion'
        metrics_rows.append(fused_metrics)
        top10 = fused_metrics.get('top10', float('nan'))
        if not np.isnan(top10) and top10 > best_top10:
            best_top10 = top10
            best_gamma = gamma
            best_ranks = fused_ranks.copy()

    summary_df = pd.DataFrame(metrics_rows)
    summary_path = args.out_dir / 'gamma_search_summary.tsv'
    summary_df.to_csv(summary_path, sep='\t', index=False)

    best_info = {'best_gamma': best_gamma, 'best_top10': best_top10}
    (args.out_dir / 'best_gamma.json').write_text(json.dumps(best_info, indent=2))

    if args.save_ranks:
        if not baseline_ranks.empty:
            baseline_ranks.to_csv(args.out_dir / 'baseline_ranks.tsv', sep='\t', index=False)
        if not llm_ranks.empty:
            llm_ranks.to_csv(args.out_dir / 'llm_ranks.tsv', sep='\t', index=False)
        if not best_ranks.empty:
            best_ranks.to_csv(args.out_dir / 'fusion_ranks.tsv', sep='\t', index=False)

    print(f'[info] γ search summary written to {summary_path}')
    print(f'[info] best γ = {best_gamma} (Top10={best_top10:.3f})')


if __name__ == '__main__':
    main()
