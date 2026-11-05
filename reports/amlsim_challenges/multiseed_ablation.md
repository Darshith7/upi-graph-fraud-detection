# AMLSim multi-seed ablation (mean +/- std over seeds 42, 43, 44, 45, 46)

Each row retrains XGBoost and GraphSAGE from scratch; thresholds and blend weights are re-tuned on validation each time.

| Features | Scenario | XGBoost AP | GraphSAGE AP | Blend AP | Blend - XGB | Alert rate (policy) | Precision | Recall | Static-cutoff alert rate | Benign hubs alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| windowed | Baseline (Standard AMLSim) | 0.7775 +/- 0.0005 | 0.8329 +/- 0.0073 | 0.8344 +/- 0.0067 | 0.0572 +/- 0.0068 | 5.1% +/- 0.0% | 95.1% +/- 0.5% | 58.2% +/- 0.5% | 7.4% +/- 0.4% | n/a |
| windowed | Challenge 1 (Disguised Suspicious) | 0.6935 +/- 0.0014 | 0.7222 +/- 0.0213 | 0.7547 +/- 0.0127 | 0.0617 +/- 0.0125 | 4.7% +/- 0.0% | 90.2% +/- 0.8% | 50.5% +/- 0.7% | 6.1% +/- 0.4% | n/a |
| windowed | Challenge 2 (Benign Lookalike Hubs) | 0.7403 +/- 0.0010 | 0.7821 +/- 0.0157 | 0.7868 +/- 0.0092 | 0.0462 +/- 0.0098 | 5.2% +/- 0.0% | 88.1% +/- 1.4% | 55.9% +/- 0.8% | 8.8% +/- 0.2% | 26.3% +/- 3.2% |
| windowed | Challenge 3 (Partial History 50%) | 0.7569 +/- 0.0008 | 0.8224 +/- 0.0082 | 0.8249 +/- 0.0078 | 0.0681 +/- 0.0078 | 4.9% +/- 0.1% | 97.3% +/- 0.5% | 57.7% +/- 0.6% | 6.0% +/- 0.2% | n/a |
| cumulative | Baseline (Standard AMLSim) | 0.7642 +/- 0.0016 | 0.7807 +/- 0.0173 | 0.7886 +/- 0.0118 | 0.0246 +/- 0.0120 | 5.2% +/- 0.0% | 91.5% +/- 1.1% | 56.9% +/- 0.9% | 16.1% +/- 3.7% | n/a |
| cumulative | Challenge 1 (Disguised Suspicious) | 0.4151 +/- 0.0108 | 0.3312 +/- 0.0356 | 0.4016 +/- 0.0098 | -0.0130 +/- 0.0102 | 5.0% +/- 0.1% | 50.8% +/- 2.4% | 30.5% +/- 0.9% | 13.4% +/- 3.6% | n/a |
| cumulative | Challenge 2 (Benign Lookalike Hubs) | 0.7461 +/- 0.0021 | 0.7627 +/- 0.0191 | 0.7713 +/- 0.0136 | 0.0252 +/- 0.0134 | 5.2% +/- 0.0% | 89.6% +/- 1.3% | 57.2% +/- 1.0% | 17.3% +/- 3.6% | 10.5% +/- 4.0% |
| cumulative | Challenge 3 (Partial History 50%) | 0.7640 +/- 0.0008 | 0.7853 +/- 0.0264 | 0.7943 +/- 0.0166 | 0.0304 +/- 0.0167 | 5.0% +/- 0.1% | 94.6% +/- 0.3% | 56.1% +/- 1.3% | 7.7% +/- 1.2% | n/a |

## Precision at realistic base rates (deployed policy, negatives re-weighted)

| Features | Scenario | Precision @ 1% prevalence | Precision @ 0.1% prevalence |
|---|---|---|---|
| windowed | Baseline (Standard AMLSim) | 68.6% +/- 2.4% | 17.8% +/- 1.7% |
| windowed | Challenge 1 (Disguised Suspicious) | 50.5% +/- 2.2% | 9.2% +/- 0.8% |
| windowed | Challenge 2 (Benign Lookalike Hubs) | 45.8% +/- 3.2% | 7.8% +/- 0.9% |
| windowed | Challenge 3 (Partial History 50%) | 80.0% +/- 2.9% | 28.6% +/- 3.6% |
| cumulative | Baseline (Standard AMLSim) | 54.8% +/- 3.4% | 10.8% +/- 1.3% |
| cumulative | Challenge 1 (Disguised Suspicious) | 10.3% +/- 0.9% | 1.1% +/- 0.1% |
| cumulative | Challenge 2 (Benign Lookalike Hubs) | 49.6% +/- 3.6% | 9.0% +/- 1.2% |
| cumulative | Challenge 3 (Partial History 50%) | 65.9% +/- 1.2% | 16.1% +/- 0.7% |