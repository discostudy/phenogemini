# LLM-driven Literature Extraction (Minimal Release)

This directory packages the essential scripts used in Section 1.3 for
extracting patient-level phenotype/genotype information from full-text
articles.  It is intentionally lightweight—no PDFs, GPT outputs, or large
checkpoints are included.  Instead, the focus is on the orchestration code,
configuration, and a few sample files so that reviewers can inspect the schema
and run smoke tests with their own inputs.

## Contents

```
release/S3_fulltext_extraction/
├── README.md
├── requirements.txt
├── config.env.example         # template for environment variables
├── src/
│   ├── config.py
│   ├── gpt41_responses_pdf_extractor.py
│   ├── hallucination_checker.py
│   ├── validator.py
│   └── process_pmids.py
├── data_samples/
│   ├── pmids_sample.csv       # toy list of PMIDs
│   └── extraction_sample.json # illustrative output structure
└── docs/
    ├── prompt.md              # schema-constrained prompt (see manuscript)
    └── quality_metrics.md     # summary of evaluation results
```

## 1. Install dependencies

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# expose helper package for PubMed metadata
export PYTHONPATH=$(pwd)/src:${PYTHONPATH:-}
```

This release targets Python 3.10+ and the official `openai` SDK ≥ 1.30.

## 2. Configure environment

Edit the template and provide your own credentials/paths:

```
cp config.env.example config.env
```

Key variables:

- `OPENAI_API_KEY` – required for GPT-4.1 Responses API access.
- `PDF_DIRECTORY` – location of source PDFs named `<pmid>.pdf`.
- `EXTRACTION_OUTPUT_DIR` – where JSON outputs are written.
- `MARKDOWN_TSV_PATH` – optional TSV used by the hallucination checker
  (can be left unset if unavailable).

Load the configuration before running the scripts:

```
source config.env
```

## 3. Extract structured records from a PMID list

Prepare a CSV with a `pmid` column (see `data_samples/pmids_sample.csv` for the
expected format) and run:

```
python src/process_pmids.py \
  --csv path/to/pmids.csv \
  --out-dir extraction_outputs \
  --checkpoint checkpoint.json \
  --pdf-dir /path/to/pdf_directory
```

The script iterates over PMIDs, validates the corresponding PDF, sends it to
GPT-4.1 with the schema-constrained prompt (`docs/prompt.md`), validates the
returned JSON, applies hallucination checks, and saves one JSON per PMID.  A
checkpoint file tracks progress so interrupted runs can be resumed.

### Important

- You must supply the PDFs yourself (for copyright reasons they are not
  distributed here).
- The OpenAI Responses API must support file uploads for your account.  Adjust
  the prompt/model in `config.env` as necessary.
- The hallucination checker (`hallucination_checker.py`) is optional; if the
  `MARKDOWN_TSV_PATH` does not exist the script will warn and proceed without
  applying the overlap heuristics.

## 4. Inspecting outputs

Each extraction result follows the schema illustrated in
`data_samples/extraction_sample.json`:

```json
{
  "document_info": {"pmid": "...", "total_patients_found": 1},
  "patients": [
    {
      "patient_id": "P1",
      "demographics": {"sex": "female", "age": 7},
      "phenotype": {"description": "...", "hpo_terms": [...]},
      "genotype": {"variants": [{"gene": "ABC1", "hgvs_cdna": "c.123A>G", ...}]}
    }
  ]
}
```

Downstream steps (e.g., entity linking to HPO, aggregation) are discussed in the
main manuscript.

## 5. Quality metrics

`docs/quality_metrics.md` summarises the manual evaluation reported in the
document (gene extraction precision=1.0; phenotype recall=0.716 exact, 0.870
within one-hop HPO relaxation).  Reproducing these numbers requires the
copyright-cleared evaluation set, which is not bundled here.

## 6. Smoke test

Without PDFs or API calls you can exercise the pipeline components using the
sample files:

```
python - <<'PY'
from pathlib import Path
import json
from src.validator import validate_extraction_result
example = json.loads(Path('data_samples/extraction_sample.json').read_text())
print(validate_extraction_result(example))
PY
```

This simply validates the sample schema.  To exercise the full extraction flow
you must provide real PDFs and API credentials as described above.

