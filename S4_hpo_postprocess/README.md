# HPO Entity Linking (Minimal Release)

This directory contains the minimal artefacts needed to reproduce Section 1.4
of the manuscript—the normalization of free-text phenotypes into HPO IDs using
the PhenoGemini Entity Linking (PG-EL) strategy.  Large datasets (full SapBERT
indexes, production pickles, ClinPhen/Doc2HPO outputs) are intentionally
omitted; follow the instructions below to supply your own resources.

## Contents

```
release/S4_hpo_postprocess/
├── README.md
├── requirements.txt
├── configs/
│   └── hpo_mapping.example.yaml   # template configuration
├── data_samples/
│   ├── hpo_sample.tsv             # toy HPO resource
│   └── phenogemini_sample.pkl     # tiny input dataframe (one row)
└── src/
    ├── candidate_generator.py
    ├── candidate_ranker.py
    ├── config.py
    ├── hpo_loader.py
    ├── lexical_retriever.py
    ├── phenotype_cleaner.py
    ├── reranker.py
    ├── run_postprocess.py
    └── __init__.py
```

## 1. Install dependencies

The pipeline relies on PyTorch and SentenceTransformers; CPU-only setups are
supported for small-scale tests.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Prepare configuration & data

1. Copy and edit the example config:
   ```bash
   cp configs/hpo_mapping.example.yaml configs/hpo_mapping.yaml
   ```
2. Update the following fields:
   - `input_pkl`: pickled pandas DataFrame containing columns such as
     `omim_id`, `phenotypes_cleaned` (list of strings per record).  The sample
     pickle in `data_samples/` shows the expected schema.
   - `hpo_resource.path`: path to an HPO TSV/JSON/OBO file.  For real runs use
     the official `hp.json` (Monarch Initiative) or an up-to-date TSV.
   - `embedding_model.name`: (optional) choose any SentenceTransformer model;
     GPU usage is controlled via the `device` field.
   - `reranker.enabled`: set to `true` if you wish to use a cross-encoder
     reranker (requires additional GPU resources).

Optional: enable external mappers (Doc2HPO, ClinPhen) by adding the
`external_mappers` section from the original configuration; corresponding
executables are **not** included in this release.

## 3. Run the normalization pipeline

```
python src/run_postprocess.py --config configs/hpo_mapping.yaml
```

This command loads the input pickle, generates HPO candidates using SapBERT,
optionally applies lexical/cross-encoder rerankers, and writes an enhanced
DataFrame to the path specified in `output_pkl`.  Chunk-level checkpoints are
stored under `checkpoint_dir` and allow interrupted runs to resume.

### Input schema

The DataFrame is expected to contain at least:

- `phenotypes_cleaned`: list of phenotype phrases (strings) per row
- `omim_id`: identifier used for grouping topics (optional but recommended)

If `phenotypes_cleaned` is missing, the script falls back to a raw `phenotype`
array and cleans it with `PhenotypeCleaner`.

### Output columns

Key columns added by the pipeline:

- `sapbert_mapping_records`: detailed mapping info per phrase
- `hpo_sapbert`: accepted HPO IDs (top candidates)
- `hpo_sapbert_conf`: confidence scores
- `hpo_sapbert_mapped_texts` / `_unmapped_texts`: phrases classified as mapped or
  unmapped

When external mappers are enabled, additional columns (`hpo_doc2hpo`,
`hpo_clinphen`, etc.) are included.

## 4. Smoke test

The sample resources allow a quick dry-run:

```
python src/run_postprocess.py \
  --config configs/hpo_mapping.example.yaml \
  --dry-run --limit 1
```

This will build a small SapBERT index on the toy HPO TSV and process a single
row without writing outputs.

## 5. Notes for full reproduction

- Use the full HPO JSON (`hp.json`) and the latest SapBERT checkpoint
  (`cambridgeltl/SapBERT-from-PubMedBert-fulltext`) for best performance.
- Set `embedding_model.device` to `cuda` and provide `multi_gpu_devices` if you
  have multiple GPUs.
- External tools (Doc2HPO, ClinPhen) are optional; integrate them by specifying
  the CLI paths and enabling the corresponding flags in the config.
- Large embedding caches (`hp_terms_embeddings.npy`) and intermediate run files
  are not distributed but can be regenerated automatically by the script.

## 6. Reported metrics

The manuscript reports Average Precision = 0.903 (exact matches) on the manual
validation set (N=190), improving to 0.968 with a 1-hop HPO relaxation.  These
metrics are not reproduced here due to licensing constraints on the evaluation
set.
