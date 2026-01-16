# Variant Baseline + PhenoGemini-LLM Fusion: Rationale

本说明文档阐述为什么要在 CIP 队列中将自研的变异基线打分（Chimera scoring）与 PhenoGemini-LLM logits 做线性融合，并给出可以直接复现的文件路径和指标。所有路径均相对于 `version_3/`。

---

## 1. 两类证据的互补性

| 证据 | 路径 | 描述 |
| --- | --- | --- |
| Variant baseline | `10.combine_analysis/tools/exomiser/cip/time-split/{dev,test}/variant_llm/variant_llm_*_baseline.csv` | 由 `fuse_llm_variant_pipeline.py` 计算，整合 `impact`/`gnomad_af_popmax`/in-silico/`acmg_score` 等列，代表纯粹的基因组证据（详见 `Chimera_Scoring_Methods.md`）。|
| LLM logits | `8.llm/2.evaluation/time-split/cip_full/out/distributions/{patient_uid}_llm_scores.tsv` | `logit` 列来自 PhenoGemini-LLM 的“孪生病例”检索，反映自由文本表型与基因的语义匹配度。|

临床判读往往需要同时考虑“基因是否具备潜在致病性”（variant 视角）以及“患者表型是否符合该基因已知的疾病谱”（phenotype 视角），因此我们让模型也遵循这一思路。

---

## 2. 融合规则

我们采用透明的线性模型：

\[
\text{score}_\text{fused} = \text{score}_\text{variant} + \gamma \times \text{logit}_\text{LLM}.
\]

理由：
1. **解释性**：两个分数在代码实现中都是线性指标；加法可拆解为 “原有变异证据 + 适量的表型支持”。
2. **独立性**：变异注释与文献检索基本独立，线性求和不会出现乘法那样“一方为 0 另一方完全失效”的问题。
3. **可调性**：单一 γ 决定表型证据对基线的影响程度；γ=0 时退化为纯 variant baseline。

---

## 3. γ 的选择（CIP）

- 开发集：`time-split/dev/dev_uids.csv`（2018–2022，n=428）。
- 网格：γ ∈ {0.00, 0.02, …, 0.30}；评估指标：Top‑10 recall。
- 结果记录在 `time-split/dev/variant_llm/variant_llm_dev_ts_metrics.tsv`。
- 最佳 γ = 0.12，指标如下：
  - Baseline (γ=0)：Top‑10 = 0.617；Top‑1 = 0.203。
  - Fusion (γ=0.12)：Top‑10 = 0.951；Top‑1 = 0.682；`other (>100)` 从 1.40% 降为 0.23%。

在测试集（`time-split/test/test_uids.csv`，2023+，n=372）上使用相同 γ 得到 Top‑10 = 0.962（baseline 为 0.503）。汇总文件位于 `time-split/dev/dev_fusion_summary.tsv` 与 `time-split/test/test_fusion_summary.tsv`，可直接引用。
同时，本发布包包含 `release/S7_variant_fusion/data/cip_variant_fused_common_metrics.tsv`，
记录了 γ = 0.12 在 800 例 CIP 聚合数据上的 Top‑K 指标，便于快速对照手稿中的数字。

---

## 4. 参数/实现细节

- 融合脚本：`10.combine_analysis/tools/exomiser/cip/scripts/fuse_llm_variant_pipeline.py`。
- `logit` 取值缺失时使用当前文件中的最小值（相当于“无表型支持”）。
- γ = 0.12 约等于“LLM 每有 1 单位的 logit（范围约 -5~+5），对得分的影响相当于 0.12 个 variant 单位”，对应 ACMG 中“表型一致性属于 Supporting 证据”。
- 输出文件：
  - `variant_llm_dev_ts_best_gamma0.12.csv` / `variant_llm_test_ts_best_gamma0.12.csv`
  - 汇总指标 `variant_llm_dev_ts_metrics.tsv`、`variant_llm_test_ts_metrics.tsv`

---

## 5. 讨论

- 变异打分来自结构化事实（注释字段），LLM logits 来自自由文本语义匹配，两者互补。
- 线性融合与 ACMG/AMP “variant + phenotype” 的综合判读方式一致，可审计、可解释。
- CIP 结果显示显著提升 Top‑K 召回率，是在 Methods 中进行描述的有力依据。
