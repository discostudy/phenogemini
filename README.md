# PhenoGemini Release Packages

This directory organises the artefacts referenced in the manuscript into
modular supplements. Each subdirectory aligns with a section of the Methods/Results and
contains only the minimal code or data needed for review. Large datasets are
omitted only when privacy or licensing prevents redistribution.

| ID | Package | Manuscript coverage | Notes |
| -- | ------- | ------------------- | ----- |
| S1 | `S1_hybrid_retrieval` | Section 1.1 (hybrid sparse+dense search) | Python modules, vendor helper, sample queries/qrels. |
| S2 | `S2_classifier_filter` | Section 1.2 (quota-balanced classifier) | Feature builders, classifier scripts, reranker CLI. |
| S3 | `S3_fulltext_extraction` | Section 1.3 (GPT-4.1 extraction) | Prompt configs, extractor CLI, validation utilities. |
| S4 | `S4_hpo_postprocess` | Section 1.4 (HPO normalisation) | Entity linking pipeline with sample configs. |
| S5 | `S5_twin_patient` | Section 2 (Twin Patient retrieval) | Time-split/LOO queries & metrics, evaluation scripts. |
| S6 | `S6_llm_inference` | Section 3 (PhenoGemini-LLM inference) | Prompt builder, inference driver, Top-K metrics. |
| S7 | `S7_variant_fusion` | Section 3–4 (Variant fusion with γ) | Fusion scripts and samples; see S9 for full inputs. |
| S8 | `S8_gold_standards` | Appendix / Quality audits | Manual gold datasets for extraction, HPO mapping, and classifier labels. |
| S9 | `S9_eval_sets` | Sections 2–4 evaluation cohorts | Full time-split/LOO/query sets, real-world cohorts, and fusion baselines. |

## Usage

Each package includes its own README with installation instructions and example
commands. Smoke tests rely on the bundled sample data; reproducing manuscript
numbers requires substituting full datasets (see package-specific notes).

## Contact

For questions about individual modules or accessing additional data, please
reach out to the corresponding author or the PhenoGemini data curation team.
