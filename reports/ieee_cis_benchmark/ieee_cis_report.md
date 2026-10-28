# IEEE-CIS Fraud Detection: Conventional Tabular Benchmark

## 1. Experimental Setup
- **Reference Dataset**: IEEE-CIS E-Commerce Fraud Detection formulation.
- **Training / Test Split**: Chronological split (35,000 train rows / 15,000 test rows).
- **Test Fraud Prevalence**: 1.82%.
- **Model**: Gradient Boosted Decision Trees (XGBoost) trained on tabular transaction attributes.

## 2. Evaluation Results
- **ROC-AUC**: **0.4997**
- **Average Precision (PR-AUC)**: **0.0218**
- **Fixed Alert Budget (2.0%)**:
  - **Alerts Dispatched**: 300
  - **True Positives**: 12
  - **False Positives**: 288
  - **Precision**: 4.00%
  - **Recall**: 4.40%

## 3. Top Predictive Features (Gain)
| Feature | Description | Importance Gain |
|---|---|---|
| `ProductCD_C` | Transaction/card/velocity indicator | 7.73 |
| `ProductCD_W` | Transaction/card/velocity indicator | 4.46 |
| `ProductCD_R` | Transaction/card/velocity indicator | 4.36 |
| `card4_mastercard` | Transaction/card/velocity indicator | 4.31 |
| `card4_american express` | Transaction/card/velocity indicator | 4.08 |
| `ProductCD_H` | Transaction/card/velocity indicator | 3.93 |
| `D1` | Transaction/card/velocity indicator | 3.90 |
| `card2` | Transaction/card/velocity indicator | 3.88 |

## 4. Methodological Distinction
As outlined in Section 4 and 7 of the project proposal, IEEE-CIS represents an isolated tabular benchmark without explicit graph network topology. Results are presented independently and are not directly compared numerically against IBM AML or UPI graph results.