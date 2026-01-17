# Metrics Overview

| Split | Configuration             | Recall@5000 | nDCG@10 | MAP   |
|-------|---------------------------|-------------|---------|-------|
| Dev   | BM25 (lexical)            | 31.8%       | 0.412   | 0.227 |
| Dev   | all-mpnet-base-v2 (dense) | 40.5%       | 0.298   | 0.208 |
| Dev   | Hybrid (RRF)              | **45.0%**   | 0.351   | 0.236 |
| Test  | BM25 (lexical)            | 43.1%       | 0.387   | 0.219 |
| Test  | all-mpnet-base-v2 (dense) | 52.6%       | 0.332   | 0.214 |
| Test  | Hybrid (RRF)              | **57.4%**   | 0.359   | **0.239** |

The hybrid configuration fuses the lexical and dense views via Reciprocal Rank
Fusion using five semantic query variants per topic.  Document-level recall at
K=5000 reaches 100% on both splits.
