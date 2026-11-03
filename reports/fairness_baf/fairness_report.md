# Demographic Fairness Audit: Bank Account Fraud (BAF) Benchmark (Synthetic Demonstration)

## 1. Executive Summary & Policy
- **Reference Suite**: Controlled synthetic demonstration testbed adhering to the Feedzai Bank Account Fraud (BAF) opening schema.
- **Train / Val / Test Split**: 60% Train (36,000 rows) / 20% Val (12,000 rows) / 20% Test (12,000 rows).
- **Alert Policy**: Operating cutoff selected on Validation set (score >= 0.0218) targeting top 5.0%.
- **Overall Test ROC-AUC**: **0.9386**
- **Overall Test PR-AUC**: **0.1636**

## 2. Age Group Disparity Breakdown
| Age Bracket | Sample Count | Share | Base Fraud Rate | Alert Rate | False Positive Rate (FPR) | True Positive Rate (TPR) |
|---|---|---|---|---|---|---|
| **Youth (18-24)** | 1,242 | 10.3% | 2.01% | 12.56% | **11.75%** | 52.00% |
| **Young Adult (25-39)** | 3,994 | 33.3% | 0.55% | 4.13% | **3.88%** | 50.00% |
| **Middle Age (40-59)** | 4,320 | 36.0% | 0.63% | 3.98% | **3.66%** | 55.56% |
| **Senior (60+)** | 2,444 | 20.4% | 0.49% | 3.60% | **3.29%** | 66.67% |

## 3. Income Tier Disparity Breakdown
| Income Tier | Sample Count | Share | Base Fraud Rate | Alert Rate | False Positive Rate (FPR) | True Positive Rate (TPR) |
|---|---|---|---|---|---|---|
| **Low Income** | 5,585 | 46.5% | 0.95% | 4.92% | **4.50%** | 49.06% |
| **Medium Income** | 6,159 | 51.3% | 0.50% | 4.92% | **4.62%** | 64.52% |
| **High Income** | 256 | 2.1% | 0.78% | 1.17% | **0.79%** | 50.00% |

## 4. Key Fairness Findings
- **Youth vs Senior FPR Disparity**: Applicants in the 18-24 bracket exhibit a +8.46 percentage point difference in False Positive Rate relative to seniors (11.75% vs 3.29%).
- **Underlying Drivers**: Shorter address stability, entry-level credit records, and mobile-first account velocity create elevated risk signals for younger demographics.
- **Compliance Commitment**: As stipulated by the revised project proposal, these disparity findings are documented as active empirical limitations, rather than claiming zero bias.