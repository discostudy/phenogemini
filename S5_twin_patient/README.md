# Twin Patient Retrieval & Evaluation (Release)

This package provides the minimal assets needed to reproduce the "Twin Patient" stage of PhenoGemini, covering the time-split and leave-one-out (LOO) evaluations discussed in the manuscript Section 2. It is scoped for reviewer verification: heavy resources (full atlas embeddings, GPU clusters) must be supplied separately, but a lightweight sample is included for smoke testing.

## Contents

- `src/` – standalone scripts for ranking, evaluation and LOO analysis.
- `data/timesplit/` – full query/qrels metadata for the 13,356-case temporal hold-out cohort.
- `data/loo/` – query/qrels metadata for the 34,705-case LOO benchmark.
- `data/samples/` – small illustrative subsets (200 atlas embeddings plus run snippets) for quick sanity checks.
- `metrics/` – pre-computed Top-K tables used in the manuscript (time-split & LOO).
- `requirements.txt` – Python dependencies for the scripts.

## Quick start (smoke test)

1. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run the ranker on the first 50 time-split queries using the sample atlas subset:

   ```bash
   python src/run_twin_patient_rank.py \
     --queries data/timesplit/queries_time.tsv \
     --out runs/time_sample.tsv \
     --max-queries 50 \
     --topk 20
   ```

3. Evaluate the sample run against the provided qrels:

   ```bash
   python src/eval_topk.py \
     --qrels data/timesplit/labels_time.tsv \
     --runs runs/time_sample.tsv \
     --out runs/time_sample_metrics.csv
   ```

The resulting CSV reports recall/precision/nDCG at the requested cut-offs. Expect lower scores than the manuscript because only 200 embeddings are used.

## Reproducing manuscript results

To recover the reported numbers:

1. Replace `data/samples/atlas_embeddings_sample.pkl` with the full atlas embedding file (`PhenoGemini_g2p_20251014_conversion_all_gene_normalized_with_valid_gene_confirmed_phenotypes_filtered_with_patient_embeddings.pkl`, 4.2 GB). Either symlink it into `data/full/` and pass `--embeddings`, or override the default path on the command line.
2. Run `src/run_twin_patient_rank.py` for each phenotype source (e.g. `phenotypes_cleaned`, `clinphen_text`, etc.) to recreate the runs under `version_3/6.evaluation/runs/time/…`.
3. Evaluate each run with `src/eval_topk.py` and aggregate into tables matching `metrics/time_metrics_summary.tsv` and `metrics/time_split_topk_table.csv`.
4. For leave-one-out experiments, invoke `src/leave_one_out_eval.py --out <summary.json> --case-out <per_query.tsv>` with the full atlas file and optional patient UID lists. The script mirrors the production configuration: it masks same-publication candidates and supports GPU multi-processing through `--encode-devices`.

The precomputed TSV/CSV files in `metrics/` are the exact artifacts cited in the manuscript (Figure 3D–3G, Table S10–S12). They can be used to double-check regenerated results.

## Data notes

- `data/timesplit/labels_time.tsv` and `data/loo/labels_loo.tsv` treat each (query, gene) pair as a positive relevance judgement (`rel=1`). Evaluation scripts assume hits when the candidate gene matches the Entrez ID.
- The supplied sample embeddings retain only a handful of columns (`patient_uid`, `entrez_gene_id`, `gene_symbol_norm`, `phenotypes_cleaned`, `patient_embeddings`, `pmid`, `publication_year`, `journal`, `phenotype_text`) to minimise footprint.
- Timeseries metadata (`time_split_meta.json`, UID lists) are provided so reviewers can verify the temporal splits or define their own subsets.

## Alignment with manuscript

| Manuscript element | Resource in this release |
| --- | --- |
| Time-split benchmark (N = 13,356) | `data/timesplit/*.tsv`, `metrics/time_metrics_summary.tsv` |
| Leave-one-out benchmark (N = 34,705) | `data/loo/*.tsv`, `metrics/loo_metrics_summary.tsv` |
| Volume ablation / robustness tables | `metrics/time_split_topk_table.csv`, `metrics/loo_topk_table.csv` |
| Reproduction scripts | `src/run_twin_patient_rank.py`, `src/eval_topk.py`, `src/leave_one_out_eval.py` |

For additional subgroup analyses (atypicality deciles, noise injection, etc.) please refer to the original repository under `version_3/6.evaluation/subgroup_analysis/` – these require larger intermediate files and are excluded here to keep the release lightweight.
