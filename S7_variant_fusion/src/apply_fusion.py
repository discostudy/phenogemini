#!/usr/bin/env python3
"""Apply a fixed γ to combine variant scores with LLM logits and report metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from fusion_utils import (
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
DEFAULT_UIDS = DEFAULT_DATA / 'sample_test_uids.tsv'
DEFAULT_GAMMA_JSON = Path('runs/dev/best_gamma.json')
DEFAULT_OUTPUT = Path('runs/test')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE,
                        help='Baseline variant scores (TSV)')
    parser.add_argument('--llm', type=Path, default=DEFAULT_LLM,
                        help='LLM logits (TSV)')
    parser.add_argument('--truth', type=Path, default=DEFAULT_TRUTH,
                        help='Ground truth genes (TSV)')
    parser.add_argument('--uids', type=Path, default=DEFAULT_UIDS,
                        help='UID list (txt/csv/tsv) to evaluate (test split)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Fixed γ value. If omitted, read from --gamma-json')
    parser.add_argument('--gamma-json', type=Path, default=DEFAULT_GAMMA_JSON,
                        help='JSON file containing {"best_gamma": ...}')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT,
                        help='Directory to store outputs (default: runs/test)')
    return parser.parse_args()


def resolve_gamma(args: argparse.Namespace) -> float:
    if args.gamma is not None:
        return args.gamma
    if not args.gamma_json.exists():
        raise SystemExit(f'Gamma JSON not found: {args.gamma_json}')
    data = json.loads(args.gamma_json.read_text())
    if 'best_gamma' not in data or data['best_gamma'] is None:
        raise SystemExit(f'best_gamma missing in {args.gamma_json}')
    return float(data['best_gamma'])


def evaluate_fusion(uids: Iterable[str],
                    baseline_df: pd.DataFrame,
                    llm_df: pd.DataFrame,
                    truth_map: Dict[str, str],
                    gamma: float) -> pd.DataFrame:
    records = []
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

    gamma = resolve_gamma(args)

    baseline_df = load_baseline(args.baseline)
    llm_df = load_llm_logits(args.llm)
    truth_map = load_truth(args.truth)
    test_uids = load_uid_list(args.uids)

    baseline_ranks = evaluate_strategy(test_uids, baseline_df, None, truth_map, 'baseline_score')
    llm_ranks = evaluate_strategy(test_uids, baseline_df, llm_df, truth_map, 'logit', fallback_logit=-10.0)
    fusion_ranks = evaluate_fusion(test_uids, baseline_df, llm_df, truth_map, gamma)

    baseline_metrics = summarise(baseline_ranks)
    baseline_metrics['strategy'] = 'baseline'
    llm_metrics = summarise(llm_ranks)
    llm_metrics['strategy'] = 'llm_only'
    fusion_metrics = summarise(fusion_ranks)
    fusion_metrics['strategy'] = f'fusion_gamma_{gamma:.3f}'

    metrics_df = pd.DataFrame([baseline_metrics, llm_metrics, fusion_metrics])
    metrics_path = args.out_dir / 'fusion_metrics_summary.tsv'
    metrics_df.to_csv(metrics_path, sep='\t', index=False)

    if not fusion_ranks.empty:
        fusion_ranks.to_csv(args.out_dir / 'fusion_ranks.tsv', sep='\t', index=False)

    print(f'[info] fusion metrics written to {metrics_path}')


if __name__ == '__main__':
    main()
