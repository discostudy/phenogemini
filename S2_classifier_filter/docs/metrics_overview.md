# Classifier Metrics Overview

The calibrated classifier (Histogram-based Gradient Boosting + PubMedBERT
logit) achieves the following scores at the operating threshold 0.6691:

| Split | Precision | Recall | F1   | AP    | ROC-AUC |
|-------|-----------|--------|------|-------|---------|
| Dev (N=3,236)  | 0.880 | 0.845 | 0.862 | 0.934 | 0.959 |
| Test (N=1,375) | 0.842 | 0.842 | 0.842 | 0.925 | 0.949 |

These numbers correspond to the manually curated gold-standard described in the
manuscript.  Reproducing them requires the full feature tables and label sets,
which are not included in this release.
