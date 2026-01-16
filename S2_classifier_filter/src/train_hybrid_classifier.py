#!/usr/bin/env python3
"""Train hybrid rerank classifier combining multi-view scores + BERT reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
import joblib



def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Train hybrid classifier without GPT features')
    ap.add_argument('--dev-feat', required=True, help='TSV with development features (omim_id, pmid, s1, ...)')
    ap.add_argument('--test-feat', required=True, help='TSV with test features')
    ap.add_argument('--dev-rerank', required=True, help='TSV with PubMedBERT scores for dev (omim_id, pmid, rerank_prob)')
    ap.add_argument('--test-rerank', required=True, help='TSV with PubMedBERT scores for test')
    ap.add_argument('--out-dir', required=True, help='Output directory for model + diagnostics')
    ap.add_argument('--precision-target', type=float, default=0.9)
    ap.add_argument('--min-preds', type=int, default=20)
    ap.add_argument('--pos-weight', type=float, default=3.0)
    ap.add_argument('--learning-rate', type=float, default=0.08)
    ap.add_argument('--max-depth', type=int, default=3)
    ap.add_argument('--max-iter', type=int, default=400)
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ks', default='10,20,50,100,200,500,1000')
    ap.add_argument('--cv', action='store_true', help='Use GroupKFold OOF calibration')
    ap.add_argument('--cv-splits', type=int, default=5)
    return ap.parse_args()


def load_and_merge(feat_path: Path, rerank_path: Path) -> pd.DataFrame:
    feat = pd.read_csv(feat_path, sep='\t')
    rerank = pd.read_csv(rerank_path, sep='\t')
    if 'prob' in rerank.columns:
        rerank = rerank[['omim_id', 'pmid', 'prob']].rename(columns={'prob': 'rerank_prob'})
    elif 'rerank_prob' in rerank.columns:
        rerank = rerank[['omim_id', 'pmid', 'rerank_prob']].copy()
    else:
        raise ValueError('Rerank file missing prob/rerank_prob column')
    merged = feat.merge(rerank, on=['omim_id', 'pmid'], how='left')
    return merged


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop_cols = [c for c in df.columns if c.startswith('gpt_') or c in {'gpt_label_x', 'gpt_label_y', 'yes_flag', 'unknown_flag', 'logistic_prob_latest', 'logistic_prob_latest_rule'}]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.drop_duplicates(subset=['omim_id', 'pmid'])
    df = df[df['jtak_label'].isin(['yes', 'no'])].copy()
    df['label'] = (df['jtak_label'] == 'yes').astype(int)
    if 's1' not in df.columns:
        s1_cols = [c for c in ['s1', 's1_x', 's1_y'] if c in df.columns]
        if s1_cols:
            df['s1'] = df[s1_cols].max(axis=1)
        else:
            df['s1'] = 0.0
    if 's1_x' in df.columns:
        df = df.drop(columns=['s1_x'], errors='ignore')
    if 's1_y' in df.columns:
        df = df.drop(columns=['s1_y'], errors='ignore')
    if 'rerank_prob' not in df.columns:
        df['rerank_prob'] = 0.0
    else:
        df['rerank_prob'] = pd.to_numeric(df['rerank_prob'], errors='coerce').fillna(0.0)

    if 'jtak_text_len' not in df.columns:
        df['jtak_text_len'] = 0.0
    else:
        df['jtak_text_len'] = pd.to_numeric(df['jtak_text_len'], errors='coerce').fillna(0.0)

    if 'rule_pass' not in df.columns:
        df['rule_pass'] = 0.0
    else:
        df['rule_pass'] = pd.to_numeric(df['rule_pass'], errors='coerce').fillna(0.0)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['jtak_text_len_log'] = np.log1p(df['jtak_text_len'])
    df['rerank_logit'] = np.log(df['rerank_prob'].clip(1e-6, 1-1e-6) / (1 - df['rerank_prob'].clip(1e-6, 1-1e-6)))
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    base_cols = [
        's1', 'rank_pct_max', 'rank_decay_max', 'jtak_text_len', 'jtak_text_len_log',
        'rerank_prob', 'rerank_logit', 'rule_pass'
    ]
    view_cols: List[str] = []
    for prefix in ['rrf', 'bm25', 'mpnet', 'tfidf', 'keywords']:
        col = f'{prefix}_score'
        if col in df.columns:
            view_cols.append(col)
    return base_cols + view_cols


def compute_sample_weights(labels: np.ndarray, base: np.ndarray, pos_weight: float) -> np.ndarray:
    weights = base.copy()
    weights[labels == 1] *= pos_weight
    return weights


def precision_target_threshold(scores: np.ndarray, labels: np.ndarray, target: float, min_preds: int) -> Dict[str, float]:
    precision, recall, thresh = precision_recall_curve(labels, scores)
    thresholds = np.r_[thresh, [1.0]]
    tp = precision * recall * labels.sum()
    fp = tp / np.clip(precision, 1e-12, None) - tp
    preds = tp + fp

    mask = (preds >= min_preds) & (precision >= target)
    if mask.any():
        idx = np.argmax(recall[mask])
        idx = np.flatnonzero(mask)[idx]
        achieved = True
    else:
        idx = np.argmax(precision)
        achieved = False

    return {
        'threshold': float(thresholds[idx]),
        'precision': float(precision[idx]),
        'recall': float(recall[idx]),
        'tp': float(tp[idx]),
        'fp': float(fp[idx]),
        'predicted_positives': float(preds[idx]),
        'achieved': achieved,
        'precision_curve': precision.tolist(),
        'recall_curve': recall.tolist(),
        'thresholds_curve': thresholds.tolist(),
    }


def metrics_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'predicted_positives': tp + fp,
    }


def recall_at_k(scores: np.ndarray, labels: np.ndarray, ks: List[int]) -> pd.DataFrame:
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    total_pos = labels.sum()
    rows = []
    for k in ks:
        if k <= 0:
            continue
        taken = sorted_labels[: min(k, len(sorted_labels))]
        recall = taken.sum() / total_pos if total_pos else 0.0
        rows.append({'k': k, 'recall_at_k': recall})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    dev_df = load_and_merge(Path(args.dev_feat), Path(args.dev_rerank))
    test_df = load_and_merge(Path(args.test_feat), Path(args.test_rerank))

    dev_df = clean_features(dev_df)
    test_df = clean_features(test_df)

    dev_df = add_derived(dev_df)
    test_df = add_derived(test_df)

    feature_cols = feature_columns(dev_df)

    X_dev = dev_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_dev = dev_df['label'].to_numpy(dtype=np.int32)
    X_test = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_test = test_df['label'].to_numpy(dtype=np.int32)

    base_weight = dev_df.get('weight', pd.Series(1.0, index=dev_df.index)).fillna(1.0).to_numpy(dtype=np.float32)
    sample_w = compute_sample_weights(y_dev, base_weight, args.pos_weight)

    def make_estimator() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            max_iter=args.max_iter,
            l2_regularization=args.l2,
            random_state=args.seed,
            min_samples_leaf=10,
        )

    groups = dev_df['omim_id'].astype(str).to_numpy()

    if args.cv:
        n_splits = min(args.cv_splits, len(np.unique(groups)))
        if n_splits < 2:
            raise SystemExit('Not enough unique topics for CV')
        gkf = GroupKFold(n_splits=n_splits)
        oof_raw = np.zeros_like(y_dev, dtype=float)
        for train_idx, val_idx in gkf.split(X_dev, y_dev, groups):
            est = make_estimator()
            est.fit(X_dev[train_idx], y_dev[train_idx], sample_weight=sample_w[train_idx])
            oof_raw[val_idx] = est.predict_proba(X_dev[val_idx])[:, 1]
        hgb = make_estimator()
        hgb.fit(X_dev, y_dev, sample_weight=sample_w)
        dev_scores_raw = oof_raw
        test_scores_raw = hgb.predict_proba(X_test)[:, 1]
    else:
        hgb = make_estimator()
        hgb.fit(X_dev, y_dev, sample_weight=sample_w)
        dev_scores_raw = hgb.predict_proba(X_dev)[:, 1]
        test_scores_raw = hgb.predict_proba(X_test)[:, 1]

    # Platt calibration using logistic regression on dev scores
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(dev_scores_raw.reshape(-1, 1), y_dev, sample_weight=sample_w)
    dev_scores = logreg.predict_proba(dev_scores_raw.reshape(-1, 1))[:, 1]
    test_scores = logreg.predict_proba(test_scores_raw.reshape(-1, 1))[:, 1]

    threshold_info = precision_target_threshold(dev_scores, y_dev, args.precision_target, args.min_preds)
    thr = threshold_info['threshold']

    dev_metrics = metrics_at(dev_scores, y_dev, thr)
    test_metrics = metrics_at(test_scores, y_test, thr)

    ks = [int(k) for k in args.ks.split(',') if k.strip()]
    at_k = recall_at_k(test_scores, y_test, ks)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'omim_id': dev_df['omim_id'],
        'pmid': dev_df['pmid'],
        'label': y_dev,
        'score': dev_scores
    }).to_csv(out_dir / 'dev_scores.tsv', sep='\t', index=False)

    pd.DataFrame({
        'omim_id': test_df['omim_id'],
        'pmid': test_df['pmid'],
        'label': y_test,
        'score': test_scores
    }).to_csv(out_dir / 'test_scores.tsv', sep='\t', index=False)

    at_k.to_csv(out_dir / 'recall_at_k_test.tsv', sep='\t', index=False)

    pr_dev_prec, pr_dev_rec, pr_dev_thr = precision_recall_curve(y_dev, dev_scores)
    pr_test_prec, pr_test_rec, pr_test_thr = precision_recall_curve(y_test, test_scores)
    pd.DataFrame({'precision': pr_dev_prec, 'recall': pr_dev_rec, 'threshold': np.r_[pr_dev_thr, [1.0]]}).to_csv(out_dir / 'pr_curve_dev.tsv', sep='\t', index=False)
    pd.DataFrame({'precision': pr_test_prec, 'recall': pr_test_rec, 'threshold': np.r_[pr_test_thr, [1.0]]}).to_csv(out_dir / 'pr_curve_test.tsv', sep='\t', index=False)

    summary = {
        'features': feature_cols,
        'precision_target': args.precision_target,
        'threshold': thr,
        'achieved_target': threshold_info['achieved'],
        'dev_metrics_at_threshold': dev_metrics,
        'test_metrics_at_threshold': test_metrics,
        'dev_average_precision': float(average_precision_score(y_dev, dev_scores)),
        'test_average_precision': float(average_precision_score(y_test, test_scores)),
        'dev_roc_auc': float(roc_auc_score(y_dev, dev_scores)),
        'test_roc_auc': float(roc_auc_score(y_test, test_scores)),
        'cv_used': bool(args.cv),
        'cv_splits': int(args.cv_splits) if args.cv else None,
        'model_params': {
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'max_iter': args.max_iter,
            'l2': args.l2,
            'pos_weight': args.pos_weight,
        },
    }

    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    joblib.dump({
        'feature_cols': feature_cols,
        'model': hgb,
        'calibrator': logreg,
        'threshold': thr,
        'precision_target': args.precision_target,
        'pos_weight': args.pos_weight,
        'l2': args.l2,
        'cv_used': bool(args.cv),
        'cv_splits': args.cv_splits if args.cv else None,
    }, out_dir / 'model.joblib')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    plt.plot(pr_dev_rec, pr_dev_prec, label='Dev', linewidth=1.6)
    plt.plot(pr_test_rec, pr_test_prec, label='Test', linewidth=1.6)
    plt.axhline(args.precision_target, color='grey', linestyle='--', linewidth=1, label=f'Precision target {args.precision_target:.2f}')
    idx = np.argmin(np.abs(np.r_[pr_dev_thr, [1.0]] - thr))
    plt.scatter([pr_dev_rec[idx]], [pr_dev_prec[idx]], c='red', s=40, label=f'Threshold={thr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(frameon=False, fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / 'pr_curve.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    main()
