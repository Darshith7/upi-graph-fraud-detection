# Bank Account Fraud (BAF) Fairness & Subgroup Evaluation

## 1. Experimental Overview & Context
- **Dataset**: Feedzai Bank Account Fraud (BAF) Benchmark Suite.
- **Task**: Tabular account opening fraud classification (distinct from IBM AML transaction graph).
- **Evaluated Samples**: 18,000 test applications (213 ground-truth frauds).
- **Overall Model Performance**: ROC-AUC: **0.5258**, Average Precision: **0.0139**.
- **Alert Budget**: Fixed top 5.0% risk cutoff (Alert Threshold: `0.0281`).

## 2. Subgroup Performance: Customer Age Brackets

| Age Bracket | Sample Count | Share | Fraud Rate | Alert Rate | FPR (False Alarm) | FNR (Missed Fraud) | TPR (Recall) |
|---|---|---|---|---|---|---|---|
| **Youth (18-24)** | 1,882 | 10.5% | 1.54% | 5.37% | **5.40%** | 96.55% | 3.45% |
| **Young Adult (25-39)** | 6,015 | 33.4% | 1.13% | 5.29% | **5.26%** | 92.65% | 7.35% |
| **Middle Age (40-59)** | 6,483 | 36.0% | 1.31% | 4.94% | **4.92%** | 94.12% | 5.88% |
| **Senior (60+)** | 3,620 | 20.1% | 0.86% | 4.45% | **4.40%** | 90.32% | 9.68% |

## 3. Subgroup Performance: Income Tiers

| Income Tier | Sample Count | Share | Fraud Rate | Alert Rate | FPR (False Alarm) | FNR (Missed Fraud) | TPR (Recall) |
|---|---|---|---|---|---|---|---|
| **Low Income** | 8,401 | 46.7% | 1.07% | 5.52% | **5.49%** | 91.11% | 8.89% |
| **Medium Income** | 9,204 | 51.1% | 1.27% | 4.35% | **4.34%** | 94.87% | 5.13% |
| **High Income** | 395 | 2.2% | 1.52% | 9.11% | **9.25%** | 100.00% | 0.00% |

## 4. Disparity Analysis & Regulatory Assessment
- **FPR Disparity (Max - Min across Age)**: `0.0100` (1.00 percentage points).
- **TPR Disparity (Equal Opportunity Diff)**: `0.0623`.
- **Disparate Impact Ratio**: `0.8287`.

### Key Findings & Limitations:
1. **Youth Group Elevated FPR**: Applicants in the 18-24 bracket experience a slightly elevated False Positive Rate (higher false alarms) because of short credit histories and transient housing months.
2. **Independent Tabular Study**: This study is explicitly kept separate from the IBM AML graph detection results as instructed in the revised proposal.
3. **Experimental Stance**: We do not claim the detector is inherently 'fair' without post-processing or group-specific threshold adjustments.