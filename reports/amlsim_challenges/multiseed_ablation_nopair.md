# AMLSim multi-seed ablation (mean +/- std over seeds 42, 43, 44, 45, 46)

Each row retrains XGBoost and GraphSAGE from scratch; thresholds and blend weights are re-tuned on validation each time.

| Features | Scenario | XGBoost AP | GraphSAGE AP | Blend AP | Blend - XGB | Alert rate (policy) | Precision | Recall | Static-cutoff alert rate | Benign hubs alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| windowed_nopair | Baseline (Standard AMLSim) | 0.7719 +/- 0.0010 | 0.8035 +/- 0.0376 | 0.8123 +/- 0.0236 | 0.0406 +/- 0.0241 | 5.1% +/- 0.0% | 93.4% +/- 1.2% | 57.4% +/- 1.0% | 8.5% +/- 1.8% | n/a |
| windowed_nopair | Challenge 1 (Disguised Suspicious) | 0.4047 +/- 0.0148 | 0.4346 +/- 0.1051 | 0.4768 +/- 0.0530 | 0.0726 +/- 0.0485 | 3.9% +/- 1.0% | 64.9% +/- 1.6% | 29.9% +/- 7.4% | 4.9% +/- 2.1% | n/a |
| windowed_nopair | Challenge 2 (Benign Lookalike Hubs) | 0.7618 +/- 0.0006 | 0.7940 +/- 0.0384 | 0.8028 +/- 0.0246 | 0.0409 +/- 0.0249 | 5.1% +/- 0.0% | 92.6% +/- 1.4% | 58.3% +/- 1.2% | 8.8% +/- 1.8% | 3.5% +/- 2.1% |
| windowed_nopair | Challenge 3 (Partial History 50%) | 0.7499 +/- 0.0007 | 0.7989 +/- 0.0384 | 0.8042 +/- 0.0322 | 0.0544 +/- 0.0325 | 4.9% +/- 0.2% | 96.2% +/- 0.9% | 56.7% +/- 2.1% | 6.5% +/- 1.1% | n/a |

## Precision at realistic base rates (deployed policy, negatives re-weighted)

| Features | Scenario | Precision @ 1% prevalence | Precision @ 0.1% prevalence |
|---|---|---|---|
| windowed_nopair | Baseline (Standard AMLSim) | 61.3% +/- 4.4% | 13.7% +/- 2.1% |
| windowed_nopair | Challenge 1 (Disguised Suspicious) | 17.1% +/- 0.9% | 2.0% +/- 0.1% |
| windowed_nopair | Challenge 2 (Benign Lookalike Hubs) | 58.9% +/- 4.7% | 12.6% +/- 2.1% |
| windowed_nopair | Challenge 3 (Partial History 50%) | 74.2% +/- 4.9% | 22.7% +/- 4.7% |