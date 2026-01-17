from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

TOP_K_CUTOFFS: Tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)


def load_uid_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"UID file not found: {path}")
    if path.suffix in {".csv", ".tsv"}:
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        column = df.columns[0]
        return df[column].astype(str).tolist()
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_baseline(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    required = {'patient_uid', 'gene_symbol', 'baseline_score'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline file missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df['patient_uid'] = df['patient_uid'].astype(str)
    df['gene_symbol'] = df['gene_symbol'].astype(str).str.upper()
    df['baseline_score'] = pd.to_numeric(df['baseline_score'], errors='coerce')
    df.dropna(subset=['baseline_score'], inplace=True)
    return df


def load_llm_logits(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    required = {'patient_uid', 'gene_symbol', 'logit'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"LLM file missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df['patient_uid'] = df['patient_uid'].astype(str)
    df['gene_symbol'] = df['gene_symbol'].astype(str).str.upper()
    df['logit'] = pd.to_numeric(df['logit'], errors='coerce')
    df.dropna(subset=['logit'], inplace=True)
    return df


def load_truth(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep='\t')
    required = {'patient_uid', 'true_gene_symbol'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Truth file missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df.dropna(subset=['true_gene_symbol'], inplace=True)
    df['patient_uid'] = df['patient_uid'].astype(str)
    df['true_gene_symbol'] = df['true_gene_symbol'].astype(str).str.upper()
    return dict(zip(df['patient_uid'], df['true_gene_symbol']))


def merge_candidate_table(uid: str,
                          baseline_df: pd.DataFrame,
                          llm_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    base = baseline_df[baseline_df['patient_uid'] == uid]
    llm = llm_df[llm_df['patient_uid'] == uid] if llm_df is not None else pd.DataFrame(columns=['patient_uid', 'gene_symbol', 'logit'])
    merged = pd.merge(base, llm[['patient_uid', 'gene_symbol', 'logit']],
                      on=['patient_uid', 'gene_symbol'], how='outer')
    merged = merged.dropna(subset=['gene_symbol']).copy()
    merged['patient_uid'] = uid
    return merged


def compute_midrank(candidates: pd.DataFrame, score_column: str, target_gene: str) -> Optional[float]:
    if score_column not in candidates.columns:
        return None
    subset = candidates[['gene_symbol', score_column]].dropna()
    if subset.empty:
        return None
    subset = subset.groupby('gene_symbol', as_index=False)[score_column].max()
    subset.sort_values(score_column, ascending=False, inplace=True, kind='mergesort')
    subset['mid_rank'] = subset[score_column].rank(method='average', ascending=False)
    row = subset[subset['gene_symbol'] == target_gene]
    if row.empty:
        return None
    return float(row['mid_rank'].iloc[0])


def topk_metrics(mid_ranks: Iterable[float]) -> Dict[str, float]:
    values = list(mid_ranks)
    if not values:
        metrics = {f'top{k}': float('nan') for k in TOP_K_CUTOFFS}
        metrics['other'] = float('nan')
        return metrics
    arr = np.asarray(values, dtype=float)
    metrics = {f'top{k}': float(np.mean(arr <= k)) for k in TOP_K_CUTOFFS}
    metrics['other'] = float(np.mean(arr > 100))
    metrics['evaluated'] = float(len(arr))
    return metrics


def evaluate_strategy(uids: Iterable[str],
                      baseline_df: pd.DataFrame,
                      llm_df: Optional[pd.DataFrame],
                      truth: Dict[str, str],
                      score_column: str,
                      fallback_logit: Optional[float] = None) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for uid in uids:
        target = truth.get(uid)
        if not target:
            continue
        merged = merge_candidate_table(uid, baseline_df, llm_df)
        if score_column not in merged.columns:
            continue
        if fallback_logit is not None and 'logit' in merged.columns:
            min_logit = merged['logit'].min() if not merged['logit'].dropna().empty else fallback_logit
            merged['logit'] = merged['logit'].fillna(min_logit - 1.0)
        midrank = compute_midrank(merged, score_column, target)
        if midrank is None:
            continue
        records.append({
            'patient_uid': uid,
            'gene_symbol': target,
            'mid_rank': midrank,
            'score_column': score_column,
        })
    return pd.DataFrame(records)


def summarise(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        metrics = {f'top{k}': float('nan') for k in TOP_K_CUTOFFS}
        metrics['other'] = float('nan')
        metrics['evaluated'] = 0.0
        return metrics
    metrics = topk_metrics(df['mid_rank'].to_numpy())
    metrics['evaluated'] = float(len(df))
    return metrics
