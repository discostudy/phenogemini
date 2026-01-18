# Rationale for the CIP Variant Scoring Heuristic

This note documents the genetic and methodological rationale behind the self-designed variant scoring function used for the CIP cohort (`version_3/10.combine_analysis/tools/exomiser/cip`). Every feature referenced below can be traced to the annotated Excel files (`vcf/cip_no_ai_filter/*.xlsx`) and supporting resources listed in [Chimera_Scoring_Methods.md](Chimera_Scoring_Methods.md); column names mentioned here correspond exactly to the Excel headers.

## 1. Variant-Level Features

### 1.1 Functional impact (`impact` column)
- **Genetic rationale**: ACMG/AMP PVS1 and PS1 emphasise that loss-of-function or disruptive missense variants are more likely pathogenic than synonymous or non-coding events.
- **Implementation**: map `impact` to weights HIGH=1.0, MODERATE=0.7, LOW=0.3, MODIFIER/other=0.1 (as in `fuse_llm_variant_pipeline.py`, lines 81–90). This mirrors the SnpEff consequence hierarchy and preserves the severity ordering that clinicians already use.

### 1.2 Population frequency (`gnomad_af_popmax` column)
- **Genetic rationale**: ACMG BA1/PM2 define allele-frequency thresholds that distinguish benign polymorphisms from pathogenic alleles.
- **Implementation**: a stepwise weight—≥1×10⁻² → 0.0, [1×10⁻³,1×10⁻²) → 0.5, [1×10⁻⁴,1×10⁻³) → 0.8, ≤1×10⁻⁴ or missing → 1.0 (script lines 92–110). The breakpoints follow BA1 (1%) and PM2 (0.1%) guidance; intermediate bins cushion measurement noise while still down-weighting likely polymorphisms.

### 1.3 Multi-model predictions (`revel_score`, `clinpred_score`, …)
- **Genetic rationale**: ACMG PP3/BP4 recommend exploiting concordant in-silico evidence but treating it as supporting evidence.
- **Implementation**: normalise each available predictor to [0,1] (BayesDel scores are mapped from [-1,1]) and take the arithmetic mean; default to 0.5 when all are missing (script lines 112–123). Equal weighting avoids overemphasising any single algorithm.

### 1.4 ACMG assertions (`acmg_score` column)
- **Genetic rationale**: manual ACMG/AMP classifications integrate evidence beyond sequence annotations and should directly modulate ranking.
- **Implementation**: convert text labels to additive offsets: pathogenic=1.0, likely pathogenic=0.7, VUS=0.4, likely benign=0.1, benign=0.0, default=0.3 (script lines 125–138). Using addition rather than multiplication ensures curated evidence can boost variants even when quantitative signals are weak.

### 1.5 Composite formula
- **Formula**: `variant_score = impact_weight × frequency_weight × (0.5 + 0.5 × in_silico_mean) + acmg_weight` (script lines 140–143).
- **Interpretation**: the multiplicative part requires simultaneous support from impact, rarity, and in-silico consensus; ACMG evidence then adjusts the score additively. A score ≤0 indicates benign/poorly supported variants, which are discarded.

## 2. Gene-Level Aggregation (script lines 145–206)
- **Motivation**: Mendelian disorders are gene-level diagnoses; two variants in the same gene must be evaluated jointly. The logic mirrors ACMG guidance on biallelic hits.
- **Rules**: group by `GENE_SYMBOL`, then
  1. Homozygous (`1/1`, `1|1`) → score = max(hom) × 1.2, inheritance = AR.
  2. ≥2 heterozygous (`0/1`, `1/0`, etc.) → take top two het scores, sum, ×0.8, inheritance = AR.
  3. Single heterozygous → max het score, inheritance = AD.
  4. Fallback (other GT) → max score ×0.5.
  5. Chromosome X/Y/MT → always take max score and label XL/MT.

These heuristics encode the expectations for autosomal recessive, autosomal dominant, and sex-linked inheritance modes without a dedicated pedigree.

## 3. Fusion with PhenoGemini-LLM
- **Evidence complementarity**: the variant score captures genomic plausibility; PhenoGemini-LLM logits (`time-split/cip_full/out/distributions/{uid}_llm_scores.tsv`) quantify phenotype concordance from literature-scale retrieval. Combining them parallels the clinical process of weighing variant pathogenicity against phenotype matching.
- **Fusion rule**: `variant_score + γ × logit`. A grid search on the CIP development split (`time-split/dev`) showed γ=0.12 maximises Top-10 recall (from 0.616 → 0.951). The same γ applied to the CIP test split yields Top-10 = 0.962.
- **Outputs**: fusion results and metrics are stored under `tools/exomiser/cip/time-split/{dev,test}/variant_llm/` and aggregated alongside Exomiser-based methods in `dev_fusion_summary.tsv`、`test_fusion_summary.tsv`.

## 4. Summary
- 每个参数选择都来自遗传学原则（功能影响、罕见频率、in-silico 共识、ACMG 支持、遗传模式）。
- 打分公式直接对应脚本中的实现，任何读者都可通过上述路径验证列名与数值。
- 与 PhenoGemini-LLM logits 的线性融合进一步提升了 CIP 的 Top-K 召回率，为方法部分提供了充分的实证依据。
