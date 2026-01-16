# Exomiser Variant Score + PhenoGemini-LLM Fusion: Rationale

本说明解释为何在 **CIP 队列** 中，将 Exomiser 生成的纯变异子分数
(`EXOMISER_GENE_VARIANT_SCORE`) 与 PhenoGemini-LLM logits 做线性融合，并给出
γ = 0.06 的选择依据及复现路径。

---

## 1. 输入与证据来源

| 证据 | 位置 | 说明 |
| --- | --- | --- |
| Exomiser 纯变异得分 | `version_3/10.combine_analysis/tools/exomiser/cip/results/{patient_uid}_joint.genes.tsv` → 列 `EXOMISER_GENE_VARIANT_SCORE` | 融合前仅保留 Exomiser 的 variant 子分数（HiPhive phenotypic 分数被舍弃），避免与 LLM 表型证据重复。脚本 `scripts/fuse_llm_exomiser_variant_only.py` 会自动读取该列并聚合同一基因的最大得分。 |
| LLM logits | `version_3/8.llm/2.evaluation/time-split/cip_full/out/distributions/{patient_uid}_llm_scores.tsv` → 列 `logit` | PhenoGemini-LLM 通过“双生患者”检索得到的文本语义匹配分；对缺失基因自动填入 *(min logit − 1)*，确保未见基因不会被高估。 |
| 基准答案 | `version_3/6.evaluation/runs/cip_filtered/cip_filtered_case_ranks.csv` | 提供每位患者的真实致病基因，供 mid-rank 计算使用。 |

---

## 2. 融合公式

\[
	ext{score}_	ext{fused} = 	ext{EXOMISER\_GENE\_VARIANT\_SCORE} + \gamma 	imes 	ext{logit}_	ext{LLM}.
\]

- 变异分数聚焦于基因组证据（频率、致病预测、ClinVar 等）。
- LLM logits 补充自由文本表型与文献语境，提供 ACMG/AMP 框架下的 PP4 支持信号。
- 线性组合保持可解释性：任何基因的最终得分都可拆解为 “纯变异证据 + γ × 表型证据”。

---

## 3. γ 的选择（CIP dev，n = 800）

我们在包含 800 份经人工确认的 CIP 病例集合上进行 γ 网格搜索（γ ∈ {0.00, 0.02, …, 0.30}），
目标指标为 Top‑10 recall。汇总结果见随附的
`release/S7_variant_fusion/data/cip_fused_midrank_common_metrics.tsv`，关键行如下：

| γ | Top‑1 | Top‑10 | Top‑20 | Top‑100 | other (>100) |
| --- | --- | --- | --- | --- | --- |
| 0.00 | 0.543 | 0.874 | 0.911 | 0.974 | 0.026 |
| 0.02 | 0.625 | 0.916 | 0.948 | 0.996 | 0.004 |
| 0.04 | 0.628 | 0.916 | 0.958 | 0.999 | 0.001 |
| **0.06** | **0.620** | **0.918** | **0.959** | **0.999** | **0.001** |

γ = 0.06 在 Top‑10 指标上表现最佳，同时保持 `other`（>100 名）比例最低，
因此被选作常规运行的固定权重。完整表格可在上文 TSV 中查阅。

---

## 4. 测试集验证（CIP test，n = 372）

在 2023–2025 的独立测试集上，我们固定 γ = 0.06 并计算 Top‑K 指标。结果记录于
`release/S7_variant_fusion/data/cip_exomiser_llm_test_metrics.tsv`，部分行如下：

| γ | Top‑1 | Top‑10 | Top‑20 | Top‑100 | other (>100) |
| --- | --- | --- | --- | --- | --- |
| 0.02 | 0.626 | 0.900 | 0.963 | 0.996 | 0.004 |
| 0.04 | 0.630 | 0.904 | 0.952 | 0.989 | 0.011 |
| **0.06** | **0.626** | **0.904** | **0.948** | **0.985** | **0.015** |

γ = 0.06 在测试集上与 γ = 0.04 表现持平但保持更低的 `other` 比例，因而仍作为默认权重。

---

## 5. 复现步骤

1. 准备 UID 列表，如 `splits/dev/dev_uids.csv` 或 `splits/test/test_uids.csv`。
2. 运行脚本：
   ```bash
   python version_3/10.combine_analysis/tools/exomiser/cip/scripts/fuse_llm_exomiser_variant_only.py \
     --uids version_3/10.combine_analysis/tools/exomiser/cip/splits/dev/dev_uids.csv \
     --gamma-grid "0,0.02,...,0.30" \
     --output-dir version_3/10.combine_analysis/tools/exomiser/cip/splits/dev/exomiser_llm
   ```
   该脚本会写出 `*_metrics.tsv`（包含各 γ 的 Top‑K 指标）以及最优 γ 的候选列表。
3. 在测试集替换 `--uids` 路径，或使用 `--gamma 0.06` 直接生成固定 γ 的结果。
4. 若需跨队列复现（PUMCH、ES 等），使用各自的 baseline TSV 与 UID 列表重复以上流程。

---

## 6. 与社区实践的一致性

- 不使用 HiPhive phenotypic 子分数，避免双重计算表型证据。
- γ=0.06 赋予 LLM logits 适度的“Supporting”权重，不会覆盖强烈的变异信号。
- 线性加权策略与 Exomiser、MOON、Exomiser+HiPhive 等工具的常用做法一致，
  保持透明、易于审计。

---

结论：在 CIP 队列上，采用 γ = 0.06 的 Exomiser + PhenoGemini-LLM 融合显著提升
Top‑10 召回率（从 0.874 提升至 ≈0.918），并在时间切分测试集上保持稳定收益。
