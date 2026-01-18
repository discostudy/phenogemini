# Extraction Quality Summary

Manual evaluation on 75 patients (held-out subset) reported the following:

- **Gene extraction**: Precision = 1.000, Recall = 1.000
- **Phenotype extraction**:
  - Exact match: Precision = 0.391, Recall = 0.716
  - Allowing 1-hop HPO relaxation: Precision = 0.524, Recall = 0.870

The lower precision for phenotypes reflects the system's tendency to capture
more granular descriptions than the reference annotations (18.24 vs 10.29
phenotypes per patient on average).

Reproducing these numbers requires the manually curated evaluation set, which
includes copyrighted content and therefore is not part of this release.
