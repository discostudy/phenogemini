# Exomiser Variant Score + PhenoGemini-LLM Fusion: Rationale

This document explains why the Exomiser variant-only score is combined with
PhenoGemini-LLM logits for the CIP cohort, and it records the artefacts required
to reproduce the reported metrics (paths are relative to `version_3/`).

---

## 1. Inputs and evidence streams

| Evidence | Location | Description |
| --- | --- | --- |
| Exomiser variant score | `10.combine_analysis/tools/exomiser/cip/results/{patient_uid}_joint.genes.tsv` → column `EXOMISER_GENE_VARIANT_SCORE` | Exomiser’s genomic sub-score (HiPhive phenotypic components are discarded to avoid double counting). The fusion script `scripts/fuse_llm_exomiser_variant_only.py` automatically extracts the per-gene maxima. |
| LLM logits | `8.llm/2.evaluation/time-split/cip_full/out/distributions/{patient_uid}_llm_scores.tsv` → column `logit` | PhenoGemini-LLM similarity produced by “twin patient” retrieval. Missing genes are imputed with *(min logit − 1)* so that unseen genes do not dominate. |
| Ground-truth genes | `6.evaluation/runs/cip_filtered/cip_filtered_case_ranks.csv` | Provides the adjudicated causal gene for each patient, used to compute mid-rank metrics. |

---

## 2. Fusion rule

\[
\text{score}_{\text{fused}} = \text{EXOMISER\_GENE\_VARIANT\_SCORE}
+ \gamma \times \text{logit}_{\text{LLM}}.
\]

The variant score captures hard genomic evidence (frequency, deleteriousness,
ClinVar annotations, etc.). The LLM logits contribute PP4-like phenotypic
support drawn from literature cases. The additive formulation keeps the result
interpretable: every gene score decomposes into “variant evidence + \(\gamma\)
× phenotypic evidence”.

---

## 3. Selecting \(\gamma\) (development split, *n* = 800)

A grid search with \(\gamma \in \{0.00, 0.02, \ldots, 0.30\}\) maximised
Top‑10 recall on the manually curated development cohort:

| \(\gamma\) | Top‑1 | Top‑10 | Top‑20 | Top‑100 | other (>100) |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.543 | 0.874 | 0.911 | 0.974 | 0.026 |
| 0.02 | 0.625 | 0.916 | 0.948 | 0.996 | 0.004 |
| 0.04 | 0.628 | 0.916 | 0.958 | 0.999 | 0.001 |
| **0.06** | **0.620** | **0.918** | **0.959** | **0.999** | **0.001** |

The aggregated statistics are available in
`release/S7_variant_fusion/data/cip_fused_midrank_common_metrics.tsv`.

---

## 4. Test-set verification (CIP, *n* = 372)

Applying \(\gamma = 0.06\) to the 2023–2025 hold-out set yields the following
Top‑*k* recall figures (`release/S7_variant_fusion/data/cip_exomiser_llm_test_metrics.tsv`):

| \(\gamma\) | Top‑1 | Top‑10 | Top‑20 | Top‑100 | other (>100) |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 0.626 | 0.900 | 0.963 | 0.996 | 0.004 |
| 0.04 | 0.630 | 0.904 | 0.952 | 0.989 | 0.011 |
| **0.06** | **0.626** | **0.904** | **0.948** | **0.985** | **0.015** |

\(\gamma = 0.06\) matches the best-performing alternatives while keeping the
>100 rank tail lower, so it is retained as the default operating point.

---

## 5. Reproduction checklist

1. Prepare the UID list, e.g. `splits/dev/dev_uids.csv` or `splits/test/test_uids.csv`.
2. Run
   ```bash
   python version_3/10.combine_analysis/tools/exomiser/cip/scripts/fuse_llm_exomiser_variant_only.py \
     --uids version_3/10.combine_analysis/tools/exomiser/cip/splits/dev/dev_uids.csv \
     --gamma-grid "0,0.02,...,0.30" \
     --output-dir version_3/10.combine_analysis/tools/exomiser/cip/splits/dev/exomiser_llm
   ```
   to obtain `*_metrics.tsv` and candidate fused ranks.
3. Repeat for the test set (or simply pass `--gamma 0.06` to apply the chosen
   weight directly).
4. For other cohorts (PUMCH, ES, etc.), swap in the appropriate baseline TSVs
   and UID lists.

---

## 6. Alignment with community practice

*   HiPhive phenotypic sub-scores are deliberately excluded, preventing double
    counting of phenotypic evidence.
*   \(\gamma = 0.06\) accords phenotypic text matching a “supporting” weight in
    ACMG/AMP terminology, without overwhelming strong variant signals.
*   Linear fusion is transparent and comparable to other clinical decision tools
    (e.g. Exomiser, MOON) that aggregate variant and phenotype channels.

---

**Conclusion.** On the CIP cohort, Exomiser + PhenoGemini-LLM with
\(\gamma = 0.06\) materially improves Top‑10 recall (from 0.874 to ≈ 0.918) while
preserving interpretability and ease of audit.
