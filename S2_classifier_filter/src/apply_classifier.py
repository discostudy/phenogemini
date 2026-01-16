#!/usr/bin/env python3
"""Apply the histogram-gradient-boosting classifier to a document pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helpers copied from the training script so that inference is standalone.
# ---------------------------------------------------------------------------

def load_and_merge(feat_path: Path, rerank_path: Path) -> pd.DataFrame:
    feat = pd.read_csv(feat_path, sep='\t')
    rerank = pd.read_csv(rerank_path, sep='\t')
    if 'prob' in rerank.columns:
        rerank = rerank[['omim_id', 'pmid', 'prob']].rename(columns={'prob': 'rerank_prob'})
    elif 'rerank_prob' in rerank.columns:
        rerank = rerank[['omim_id', 'pmid', 'rerank_prob']].copy()
    else:
        raise ValueError('rerank file missing prob/rerank_prob column')
    merged = feat.merge(rerank, on=['omim_id', 'pmid'], how='left')
    return merged


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [
        c for c in df.columns
        if c.startswith('gpt_')
        or c in {
            'gpt_label_x', 'gpt_label_y', 'yes_flag', 'unknown_flag',
            'logistic_prob_latest', 'logistic_prob_latest_rule'
        }
    ]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.drop_duplicates(subset=['omim_id', 'pmid'])
    if 'jtak_label' in df.columns:
        df = df[df['jtak_label'].isin(['yes', 'no'])].copy()
        df['label'] = (df['jtak_label'] == 'yes').astype(int)
    if 's1' not in df.columns:
        s1_cols = [c for c in ['s1', 's1_x', 's1_y'] if c in df.columns]
        if s1_cols:
            df['s1'] = df[s1_cols].max(axis=1)
        else:
            df['s1'] = 0.0
    df = df.drop(columns=['s1_x', 's1_y'], errors='ignore')
    if 'rerank_prob' not in df.columns:
        df['rerank_prob'] = 0.0
    else:
        df['rerank_prob'] = pd.to_numeric(df['rerank_prob'], errors='coerce').fillna(0.0)
    if 'jtak_text_len' not in df.columns:
        df['jtak_text_len'] = 0.0
    df['jtak_text_len'] = pd.to_numeric(df['jtak_text_len'], errors='coerce').fillna(0.0)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'jtak_text_len_log' not in df.columns:
        df['jtak_text_len_log'] = np.log1p(df['jtak_text_len'])
    if 'rerank_logit' not in df.columns:
        eps = 1e-5
        p = np.clip(df['rerank_prob'].astype(float), eps, 1 - eps)
        df['rerank_logit'] = np.log(p / (1 - p))
    if 'rule_pass' not in df.columns:
        df['rule_pass'] = (df.get('rule_hits', 0) > 0).astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    blacklist = {
        'omim_id', 'pmid', 'label', 'jtak_label', 'weight'
    }
    cols = [c for c in df.columns if c not in blacklist and df[c].dtype != object]
    return cols

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Apply the trained hybrid classifier to pooled features')
    ap.add_argument('--features', required=True, help='TSV with pooled retrieval features (omim_id, pmid, ...)')
    ap.add_argument('--rerank', required=True, help='TSV with reranker scores (omim_id, pmid, rerank_prob)')
    ap.add_argument('--model', required=True, help='model.joblib produced by train_hybrid_classifier.py')
    ap.add_argument('--summary', required=True, help='summary.json produced by train_hybrid_classifier.py')
    ap.add_argument('--out', required=True, help='Output TSV with probabilities and binary decisions')
    ap.add_argument('--threshold', type=float, default=None, help='Optional override for decision threshold')
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    merged = load_and_merge(Path(args.features), Path(args.rerank))
    merged = clean_features(merged)
    merged = add_derived(merged)

    model_bundle: Dict[str, object] = joblib.load(args.model)
    feature_cols: List[str] = list(model_bundle['feature_cols'])
    estimator = model_bundle['model']
    calibrator = model_bundle['calibrator']

    X = merged[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    raw_scores = estimator.predict_proba(X)[:, 1]
    probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    with Path(args.summary).open('r', encoding='utf-8') as f:
        summary = json.load(f)
    threshold = args.threshold if args.threshold is not None else summary.get('threshold')
    if threshold is None:
        raise SystemExit('Threshold not provided (neither via --threshold nor summary.json)')

    out_df = merged[['omim_id', 'pmid']].copy()
    out_df['score_raw'] = raw_scores
    out_df['prob'] = probs
    out_df['decision'] = (probs >= threshold).astype(int)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, sep='\t', index=False)
    print(f'[OK] wrote {args.out} ({len(out_df)} rows) with threshold={threshold:.4f}')


if __name__ == '__main__':
    main()
