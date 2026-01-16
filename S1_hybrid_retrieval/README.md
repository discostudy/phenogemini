# Hybrid Retrieval (Minimal Release)

This folder exposes the code that powers Section 1.1 of the manuscript.  It is
trimmed to the minimum artefacts needed for reviewers to inspect or reproduce
the sparse+dense retrieval experiments; large caches, indexes, and proprietary
datasets are deliberately omitted.

## Contents

```
release/S1_hybrid_retrieval/
├── README.md
├── config.env.example       # template for environment variables
├── requirements.txt         # Python dependencies
├── src/                     # core Python modules (BM25, SBERT, fusion, evaluation)
├── scripts/                 # empty (usage instructions below rely on python CLI)
├── vendor/ek_phenopub/      # lightweight PubMed repository helper
├── data_samples/            # tiny TSV snippets for smoke tests (not for metrics)
└── docs/metrics_overview.md # summary table of the reported metrics
```

## 1. Install dependencies

Use Python 3.9+ (the experiments were run with Python 3.10).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: install [Anserini](https://github.com/castorini/anserini) if you plan
to use the Lucene-based BM25 implementation; the default Python implementation
shipped here is sufficient for reproducing the reported numbers.

## 2. Prepare inputs

Copy the template and edit the paths for your environment:

```bash
cp config.env.example config.env
```

The following resources are required:

* **PubMed SQLite repository (`REPO`)** — created with the `ek_phenopub`
toolkit.  The helper class is provided under `vendor/`, but the actual database
must be generated separately (contains PubMed metadata + abstracts).
* **Queries / qrels** — the full TSVs used in the paper are derived from HGMD ×
OMIM and cannot be redistributed; use your own copies or regenerate them with
`src/build_multi_view_queries.py` and `src/build_hgmd_qrels.py`.  The
`data_samples/` directory only contains a handful of truncated rows for smoke
testing.
* *(Optional)* Documents pools (`data/dev/docs.tsv` etc.) if you want to skip
retrieval and only evaluate.  Otherwise the scripts below will fetch texts via
`PubRepository` on the fly.

## 3. Running retrieval and fusion

Activate the environment, load your `config.env`, and execute the modules
manually.  The examples below assume you are in `release/S1_hybrid_retrieval/`.

### 3.1 BM25 (lexical view)
```bash
source config.env
python src/run_bm25.py \
  --queries $QUERIES_SRC \
  --docs data/dev/docs.tsv \
  --repo $REPO \
  --out runs/dev/bm25.tsv \
  --workers 8 --query_chunk 50 --topk_out 5000
```

### 3.2 TF-IDF / keyword views
```bash
python src/run_tfidf.py \
  --queries $QUERIES_SRC \
  --docs data/dev/docs.tsv \
  --repo $REPO \
  --out runs/dev/tfidf.tsv \
  --topk_out 5000

python src/run_keywords.py \
  --queries $QUERIES_SRC \
  --docs data/dev/docs.tsv \
  --repo $REPO \
  --out runs/dev/keywords.tsv \
  --topk_out 5000 --workers 8 --chunk 50
```

### 3.3 SBERT (dense view)
```bash
python src/run_sbert.py \
  --queries $QUERIES_SRC \
  --docs data/dev/docs.tsv \
  --repo $REPO \
  --models sentence-transformers/all-mpnet-base-v2 \
  --devices cuda:0 \
  --out_dir runs/dev \
  --topk_out 5000
```

### 3.4 Fusion via Reciprocal Rank Fusion (RRF)
Assuming the individual run files are located under `runs/dev/`:
```bash
python src/fuse_views.py \
  --runs runs/dev/bm25.tsv,runs/dev/tfidf.tsv,runs/dev/keywords.tsv,runs/dev/all-mpnet-base-v2.tsv \
  --out runs/dev/hybrid_rrf.tsv \
  --scheme rrf --rrf-k 60
```

### 3.5 Evaluation (e.g., Recall@K)
```bash
python src/eval_ir_from_run.py \
  --run runs/dev/hybrid_rrf.tsv \
  --qrels $QRELS_SRC \
  --cutoffs 10,100,1000,5000 \
  --out analysis/dev_hybrid_metrics.tsv
```

Repeat the same recipe for the test split by changing the paths.  The
`docs/metrics_overview.md` file lists the aggregated metrics reported in the
manuscript (Dev: Recall@5000=45.0%, Test: Recall@5000=57.4%).

## 4. Smoke test with tiny samples

Set `QUERIES_SRC` and `QRELS_SRC` in `config.env` to the files under
`data_samples/` and run any of the commands above.  The resulting run files are
not meaningful for evaluation but allow verifying that the tooling is wired
correctly.

## 5. Notes

* All paths in the examples use the `dev` split; create analogous directories for
`test` or custom partitions.
* Large artefacts (Lucene indexes, vector caches, run files for all 25k topics)
  are intentionally absent.  Generation commands are documented above.
* If you wish to explore additional transformer checkpoints, pass a comma-
  separated list to `--models` (e.g., `sentence-transformers/all-mpnet-base-v2,sentence-transformers/msmarco-distilbert-base-tas-b`).
* The vendor helper relies purely on SQLite; no external API credentials are
  required.

