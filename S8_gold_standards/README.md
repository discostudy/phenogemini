# Gold Standard Annotations

This directory packages the manual gold standards referenced throughout the manuscript. All files are fully de-identified and may be shared publicly.

## Contents

| File | Description | Origin |
| ---- | ----------- | ------ |
| `llm_extraction_gold.tsv` | 75 manually curated literature cases with gold gene and HPO annotations used to validate the GPT-4.1 extraction pipeline (Section 1.3). Columns: `patient_uid`, `pmid`, `doi`, `gene_symbol`, `entrez_gene_id`, `gold_hpo_ids` (semicolon-separated HPO IDs). | version_3/4.extraction/eval_phenoapt/out |
| `pg_el_phrase_gold.tsv` | Phrase-to-HPO gold pairs (columns: `annotation_id`, `phrase`, `gold_hpo`, `confidence`) used to benchmark PG-EL against ClinPhen/Doc2HPO (Section 1.4). | version_3/5.post_procession/data/eval |
| `classifier_labels.tsv` | GPT-verified case-report labels for the retrieval classifier (columns: `pmid`, `omim_id`, `label`, `reviewer_label`, `status`, `rrf_score`). `label` is `yes`/`no`; `reviewer_label` stores the manual GPT review ("yes"/"no"/"i dont know"); `status` indicates scraping status, and `rrf_score` is the retrieval score used for sampling. | version_3/3.filter/gold_standard |

All fields are either public identifiers (PMID, DOI) or structured annotations. No verbatim clinical text or personal identifiers are included.

## Usage

- Use `llm_extraction_gold.tsv` as the reference set when recomputing precision/recall in `S3_fulltext_extraction` or `S5_twin_patient` evaluations.
- The phrase gold set can be fed into `release/S4_hpo_postprocess/src/make_eval_gold_from_details.py` or any custom evaluator to reproduce PG-EL metrics.
- `classifier_labels.tsv` serves as the supervision signal for `release/S2_classifier_filter/src/train_hybrid_classifier.py` (map `label == "yes"` to positive class).

If additional gold standards are generated, append them here and update the table above with provenance and schema details.
