# Evaluation Cohorts

This directory provides the full, de-identified datasets used to evaluate PhenoGemini and PhenoGemini-LLM. All patient identifiers have been anonymised and only structured phenotype/gene labels are included.

## Directory structure

```
S9_eval_sets/
├── time_split/
│   ├── queries_time.tsv       # 13,356 phenotype queries (query_id, text)
│   ├── labels_time.tsv        # Gold genes for the same queries (query_id, entrez_gene_id, gene_symbol_norm)
│   └── meta.json              # Dataset description and schema
├── loo/
│   ├── queries_loo.tsv        # 34,705 phenotype queries for leave-one-out analysis
│   ├── labels_loo.tsv         # Gold genes for the LOO set
│   └── meta.json
├── real_world_cip.tsv         # 800 CIP patients (patient_uid, gene_symbol, entrez_gene_id, phenotypes, cohort)
├── real_world_pumch.tsv       # 578 PUMCH patients (same schema as above)
├── real_world_diagnostic_lab.tsv  # 426 diagnostic lab cases (same schema)
└── variant_fusion/
    ├── dev_uids.csv           # Dev split patient UIDs for fusion tuning
    ├── test_uids.csv          # Test split patient UIDs for held-out evaluation
    ├── exomiser_midrank.tsv   # Exomiser variant mid-ranks (all 938 CIP patients)
    ├── exomiser_phenotype_midrank.tsv # Exomiser phenotype-only mid-ranks
    ├── variant_baseline_dev.csv       # Chimera variant baseline mid-ranks (dev)
    ├── variant_baseline_test.csv      # Chimera variant baseline mid-ranks (test)
    ├── exomiser_baseline_dev.csv      # Exomiser baseline mid-ranks (dev subset)
    ├── exomiser_baseline_test.csv     # Exomiser baseline mid-ranks (test subset)
    └── llm_logits.tsv         # PhenoGemini-LLM logits for all CIP patients (patient_uid, entrez_gene_id, gene_symbol, logit)
```

## Notes

- Phenotype text fields describe high-level clinical features (often semicolon-separated). They contain no verbatim clinical notes or protected health information.
- Gene identifiers follow Entrez nomenclature. Empty gene symbols indicate cases without confirmed diagnoses.
- `llm_logits.tsv` aggregates the per-patient logit files produced by the time-split inference pipeline (Section 3). Values are raw logits (pre-softmax).
- Mid-rank tables originate from `version_3/10.combine_analysis/tools/exomiser`; file paths have been removed for portability.

If additional cohorts are released, append them here and document any cohort-specific caveats.
