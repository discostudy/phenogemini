# PhenoGemini-LLM Phenotype Inference (Release)

This bundle contains the minimal scripts and sample artefacts required to
reproduce the phenotype-only inference path of PhenoGemini-LLM described in the
manuscript (Figure 5C/F). It focuses on the stages from prompt construction to
gene-level probability outputs; fusion with variant information (γ search) is
handled separately.

## Layout

- `src/build_prompt.py` – retrieves top-K "Twin Patients" from an atlas and
  formats prompts for PhenoGemini-LLM.
- `src/run_llm_inference.py` – converts prompts into gene probability
  distributions. Supports real checkpoints or a deterministic `--mock` mode for
  smoke testing.
- `src/compute_topk_metrics.py` – evaluates Top-K recall/precision given ground
  truth gene labels.
- `data/samples/` – lightweight examples (atlas slice, query set, gene
  vocabulary, prompts, logits, metrics) that mirror the TSV/JSON structure used
  in production.
- `requirements.txt` – Python dependencies for the end-to-end scripts.

## Quick smoke test

1. Create a virtual environment and install requirements:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Build prompts for the bundled sample cohort (5 queries) using the sample
   atlas subset (200 individuals):

   ```bash
   python src/build_prompt.py \
     --queries data/samples/sample_queries.tsv \
     --twin-count 5 \
     --out data/samples/sample_prompts.tsv
   ```

3. Run PhenoGemini-LLM inference in mock mode to obtain gene probabilities:

   ```bash
   python src/run_llm_inference.py \
     --prompts data/samples/sample_prompts.tsv \
     --gene-vocab data/samples/sample_gene_vocab.tsv \
     --out data/samples/sample_logits.tsv \
     --mock --seed 42 --max-genes 25
   ```

4. Evaluate Top-K metrics against the supplied gold labels:

   ```bash
   python src/compute_topk_metrics.py \
     --pred data/samples/sample_logits.tsv \
     --gold data/samples/sample_gold.tsv \
     --out data/samples/sample_metrics.tsv
   ```

   The resulting TSV reports recall/precision at K = 1,3,5,10,20,50,100. (With
   mock logits the scores are deterministic but not clinically meaningful.)

## Reproducing manuscript results

To recover the numbers cited in the manuscript:

1. Replace `data/samples/atlas_sample.pkl` with the full PhenoGemini Atlas pickle
   that includes `patient_embeddings`, `phenotypes_cleaned`, and
   normalised gene annotations (4.2 GB). The same file is used in
   `release/S5_twin_patient`.
2. Regenerate prompts per evaluation split (time-split dev/test, leave-one-out,
   real-world cohorts) using `src/build_prompt.py` and the corresponding query
   TSVs. Save each prompt set under `out/prompts/…` as done in the main repo.
3. Run `src/run_llm_inference.py` with the fine-tuned Qwen3-MoE checkpoint used
   in the study. The model directory should contain merged weights (base + LoRA)
   compatible with `AutoModelForCausalLM`. Supply the comprehensive gene
   vocabulary covering all 4,739 gene tokens.
4. Point `src/compute_topk_metrics.py` at the resulting logits TSVs and the
   curated gold labels for each split (e.g., the time-split dev/test `gold`
   tables, real-world cohort annotations). Aggregate summaries feed into Figure
   5C/F and Supplementary Tables S19–S22.

The sample artefacts included here (`sample_prompts.tsv`, `sample_logits.tsv`,
`sample_metrics.tsv`) mirror the exact schema of the production outputs, letting
reviewers validate parsing without the heavy checkpoints.

## Notes on real inference

- `src/run_llm_inference.py` expects the prompt column to already include the
  HTML-like special gene tokens (`<|PhenoGemini-Special-Token-Entrez-ID-XXX|>`)
  appended during atlas preparation. When `--chat-template` is specified, the
  prompt column must contain a JSON array of chat messages matching the target
  tokenizer template.
- Multi-token gene identifiers are handled by taking the first token ID. Genes
  whose tokens are absent from the tokenizer vocabulary are skipped (probability
  set to zero).
- The script uses `torch_dtype=torch.bfloat16` and `device_map="auto"` by
  default. Adjust `--device` or pass additional keyword arguments through the
  environment if your checkpoint requires manual placement.
- γ-search and fusion with variant pipelines are not covered here; refer to the
  forthcoming release package for those components.

## Mapping to manuscript

| Manuscript element | Resource in this release |
| --- | --- |
| PhenoGemini-LLM prompt construction | `src/build_prompt.py`, `data/samples/*` |
| LLM inference → gene probabilities | `src/run_llm_inference.py`, `data/samples/sample_logits.tsv` |
| Top-K accuracy (Figure 5C/F, Tables S19–S22) | `src/compute_topk_metrics.py` |

This release is intentionally lightweight: large models, full atlases, and
patient-level gold labels must be supplied by the reviewer and are referenced in
context in the README.
