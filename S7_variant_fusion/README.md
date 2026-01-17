# Variant Fusion with PhenoGemini-LLM (Release)

This package demonstrates how baseline variant rankings (Exomiser / Chimera) are
combined with PhenoGemini-LLM logits through a single γ parameter. It mirrors the
procedure used to produce Figure 5E–F and the time-split dev/test tables in the
manuscript.

## Contents

- `src/fusion_utils.py` – helper routines for loading TSV files, computing
  mid-ranks, and aggregating Top-K metrics.
- `src/search_gamma.py` – performs γ grid-search on a development split. Writes a
  metrics table and the best γ value (`best_gamma.json`).
- `src/apply_fusion.py` – applies the selected γ to a test split and reports
  baseline/LLM/fusion metrics.
- `data/cip_fused_midrank_common_metrics.tsv` – aggregated CIP dev split metrics showing the γ=0.06 operating point used in the manuscript.
- `data/cip_exomiser_llm_test_metrics.tsv` – CIP time-split test metrics for γ ∈ {0.00,…,0.12}, including the γ=0.06 row cited in the manuscript.
- `data/cip_variant_fused_common_metrics.tsv` – aggregated CIP variant-baseline fusion metrics at γ=0.12.
- `data/samples/` – toy TSV files (6 patients) illustrating the expected schema:
  variant baseline scores, LLM logits, dev/test UID lists, and ground truth genes.
- `docs/` – copies of the scoring justifications that define the underlying
  variant mid-rank logic (see below).

## Quick smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# γ search on the sample dev split
python src/search_gamma.py --save-ranks

# Apply the selected γ to the sample test split
python src/apply_fusion.py --gamma-json runs/dev/best_gamma.json
```

Outputs are written to `runs/dev/` and `runs/test/`, respectively.

## Reproducing manuscript results

1. Replace the sample TSVs with the full datasets:
   - Baseline variant mid-ranks from Exomiser or Chimera pipelines.
   - PhenoGemini-LLM logits for the same patient cohort.
   - Ground truth gene annotations and the dev/test UID lists (time-based split).
2. Run `src/search_gamma.py` on the dev split to determine the best γ. Use the
   same grid as the manuscript (0.00–0.30 step 0.02) or pass a custom list via
   `--gamma-grid`.
3. Apply the selected γ to the test split with `src/apply_fusion.py`. The
   resulting `fusion_metrics_summary.tsv` provides the Top-K recall/precision used
   in the manuscript tables.
4. For real-world cohorts (CIP, PUMCH, etc.), repeat the same sequence with the
   cohort-specific UID lists and baseline runs.

After running the full pipeline, cross-check the reported Top-K recalls with
`data/cip_fused_midrank_common_metrics.tsv` (CIP dev, γ=0.06),
`data/cip_exomiser_llm_test_metrics.tsv` (CIP test grid, highlighting γ=0.06), and
`data/cip_variant_fused_common_metrics.tsv` (CIP dev variant baseline, γ=0.12);
those tables are the aggregated numbers quoted in the manuscript.

## Scoring references

The computation of variant mid-ranks follows the methodology documented in:

- `docs/Chimera_Scoring_Justification.md`
- `docs/Chimera_PhenoGemini-LLM_Justification.md`
- `docs/Exomiser_PhenoGemini-LLM_Justificiation.md`

These documents describe how baseline variant scores are generated and why the
mid-rank aggregation is appropriate, providing the necessary provenance for
reviewers.

## File formats

All TSV inputs share the following conventions:

- **Variant baseline**: `patient_uid`, `gene_symbol`, `baseline_score`.
- **LLM logits**: `patient_uid`, `gene_symbol`, `logit` (pre-softmax score).
- **Ground truth**: `patient_uid`, `true_gene_symbol`.
- **UID lists**: newline-separated patient identifiers.

Gene symbols are compared case-insensitively after uppercasing. Missing logits
are imputed with *(min logit − 1)* during fusion so that unseen genes do not
unfairly dominate.

## Requirements

See `requirements.txt` for the minimal Python dependencies (`numpy`, `pandas`).
No GPU is needed for the example; full-scale runs rely on precomputed TSV files
from the upstream pipelines.
