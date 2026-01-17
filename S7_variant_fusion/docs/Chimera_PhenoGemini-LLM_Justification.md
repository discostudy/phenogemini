# Variant Baseline + PhenoGemini-LLM Fusion: Rationale

This note summarises why the Chimera variant baseline is linearly combined with
PhenoGemini-LLM logits for the CIP cohort, and it records the exact artefacts
needed to reproduce the reported metrics (paths are relative to `version_3/`).

---

## 1. Complementary sources of evidence

| Evidence | Location | Description |
| --- | --- | --- |
| Variant baseline | `10.combine_analysis/tools/exomiser/cip/time-split/{dev,test}/variant_llm/variant_llm_*_baseline.csv` | Produced by `fuse_llm_variant_pipeline.py`; aggregates variant-centric features (impact, population frequency, in-silico predictors, ACMG score). See `Chimera_Scoring_Methods.md` for full feature definitions. |
| LLM logits | `8.llm/2.evaluation/time-split/cip_full/out/distributions/{patient_uid}_llm_scores.tsv` | `logit` values returned by PhenoGemini-LLM after “twin-patient” retrieval, measuring free-text phenotypic concordance. |

Clinical interpretation routinely combines genetic plausibility with phenotypic
concordance. The fusion layer mirrors this principle: the Chimera score captures
hard genomic evidence, whereas the LLM logits reflect the closeness of the
observed phenotype to previously reported cases.

---

## 2. Fusion rule

\[
\text{score}_{\text{fused}} = \text{score}_{\text{variant}} + \gamma \times \text{logit}_{\text{LLM}}.
\]

*   **Interpretability** – both quantities are linear; the fused score remains
    an additive decomposition of genomic evidence and phenotypic support.
*   **Independence** – genomic annotation and case-based textual matching are
    largely orthogonal; addition avoids pathological failure modes associated
    with multiplicative fusion.
*   **Tunability** – a single parameter \(\gamma\) controls how strongly
the phenotypic channel influences the final ranking.

---

## 3. Choosing \(\gamma\) (CIP)

*   Development cohort: `time-split/dev/dev_uids.csv` (publications ≤ 2022,
    *n* = 428).
*   Grid: \(\gamma \in \{0.00, 0.02, \ldots, 0.30\}\); objective: Top‑10 recall.
*   Results: `time-split/dev/variant_llm/variant_llm_dev_ts_metrics.tsv`.

\[
\begin{array}{c|ccccc}
\gamma & \text{Top-1} & \text{Top-10} & \text{Top-20} & \text{Top-100} & \text{other} (>100) \\
\hline
0.00 & 0.617 & 0.874 & 0.911 & 0.974 & 0.0140 \\
0.12 & \mathbf{0.682} & \mathbf{0.951} & \mathbf{0.971} & \mathbf{0.996} & \mathbf{0.0023}
\end{array}
\]

The retained release also includes
`release/S7_variant_fusion/data/cip_variant_fused_common_metrics.tsv`, which
summarises Top-*k* performance when \(\gamma=0.12\) is applied to all
800 cases in aggregate.

---

## 4. Implementation details

*   Fusion script: `10.combine_analysis/tools/exomiser/cip/scripts/fuse_llm_variant_pipeline.py`.
*   Missing logits are imputed with *(minimum logit – 1)* to down‑weight genes
    never retrieved by the LLM channel.
*   \(\gamma = 0.12\) approximates a “supporting” PP4 contribution in the
    ACMG/AMP framework (one LLM logit unit, typically in [−5, 5], corresponds to
    0.12 units of variant evidence).
*   Outputs include `variant_llm_dev_ts_best_gamma0.12.csv`,
    `variant_llm_test_ts_best_gamma0.12.csv`, and their associated
    `*_metrics.tsv` tables.

---

## 5. Discussion

*   Chimera scores encode deterministic genomic evidence; PhenoGemini-LLM logits
    provide case-based textual corroboration.
*   Linear fusion is easy to audit and consistent with clinical practice that
    weighs genetic and phenotypic evidence separately.
*   On the CIP cohort, \(\gamma=0.12\) delivers large gains in Top‑*k* recall,
    underpinning the improvements reported in the manuscript’s main text.
