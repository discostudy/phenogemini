# High-confidence Case Report Classifier (Minimal Release)

This folder contains the core scripts used in Section 1.2 (quota-balanced
filter + calibrated classifier).  Large artefacts (full feature tables,
running logs, production checkpoints) are not included; instead, concise
samples and instructions are provided so reviewers can inspect or reproduce the
pipeline with their own data.

## Contents

```
release/S2_classifier_filter/
├── README.md
├── requirements.txt
├── src/
│   ├── build_pool_and_features.py    # assemble retrieval features
│   ├── train_hybrid_classifier.py    # train HistGradientBoosting classifier
│   └── apply_classifier.py           # score pools with a trained model
├── rerank/
│   └── apply_pubmedbert_reranker.py  # obtain PubMedBERT cross-encoder logits
├── data_samples/
│   ├── features_sample.tsv           # truncated feature rows
│   ├── rerank_scores_sample.tsv      # truncated reranker scores
│   └── gpt_labels_sample.tsv         # truncated GPT labels
└── docs/
    ├── metrics_overview.md           # dev/test precision & recall
    └── summary_example.json          # example classifier summary (weights/threshold)
```

## 1. Install dependencies

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# allow local ek_phenopub helper to be discovered
export PYTHONPATH=$(pwd)/vendor:${PYTHONPATH:-}
```

## 2. Prepare inputs

The classifier consumes three main resources per split (dev/test):

1. **Retrieval features**: TSV with columns such as `omim_id`, `pmid`, fused
   scores (`s1`, `s2`, ...), rank-based features, etc.  Generate with
   `src/build_pool_and_features.py` by merging the hybrid retrieval outputs,
   quota allocations, and other signals.
2. **Cross-encoder scores**: `rerank/apply_pubmedbert_reranker.py` scores each
   (omim, pmid) pair using PubMedBERT; produces a TSV with `rerank_prob`.
3. **Expert/GPT labels**: binary labels (`yes`/`no`) used for supervision.

The truncated samples under `data_samples/` illustrate the expected schema but
are *not* sufficient for meaningful training or evaluation.

## 3. Train the classifier

```
python src/train_hybrid_classifier.py \
  --dev-feat path/to/dev_features.tsv \
  --test-feat path/to/test_features.tsv \
  --dev-rerank path/to/dev_rerank_scores.tsv \
  --test-rerank path/to/test_rerank_scores.tsv \
  --out-dir outputs/pubmedbert_hgb \
  --precision-target 0.88
```

This script fits a Histogram-based Gradient Boosting model that combines the
retrieval scores and reranker logits.  It writes:

- `summary.json` with feature list, standardisation statistics, threshold, and
  evaluation metrics.
- `model.joblib` (scikit-learn estimator).
- DEV/TEST diagnostics (PR curves, confusion matrices, recall@k tables).

The default arguments reproduce the configuration described in the manuscript;
feel free to adjust the learning rate, depth, or class weights if exploring
variants.

## 4. Apply the classifier

```
python src/apply_classifier.py \
  --features path/to/pool_features.tsv \
  --rerank path/to/pool_rerank_scores.tsv \
  --model outputs/pubmedbert_hgb/model.joblib \
  --summary outputs/pubmedbert_hgb/summary.json \
  --out filtered_pairs.tsv \
  --threshold 0.6691
```

The script merges the pooled features with reranker scores, infers probabilities
using the saved model/calibrator, and emits binary decisions based on the
threshold (either supplied explicitly or taken from `summary.json`).

## 5. Metrics

`docs/metrics_overview.md` summarises the key dev/test results reported in the
paper (Precision ≥ 0.88 at the calibrated threshold).  The full confusion matrix
and PR curves are saved by the training script inside `out-dir`.

## 6. Notes

- The classifier expects that retrieval outputs already went through the quota
  balancing stage described in the manuscript.  Re-running `build_pool_and_features.py`
  requires the hybrid retrieval runs and quota plan.
- Cross-encoder scoring utilises a PubMedBERT checkpoint; the helper script
  `rerank/apply_pubmedbert_reranker.py` expects a TSV of `(omim_id, pmid)`
  pairs, the corresponding query files, and a local PubMed SQLite repository.
- No proprietary data (HGMD/GPT annotations) are distributed here.  Reviewers
  should generate analogous resources from their own licensed copies.
