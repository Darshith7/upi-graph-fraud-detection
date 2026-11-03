# IEEE-CIS Fraud Detection: Conventional Tabular Benchmark (Synthetic Demonstration)

## 1. Experimental Setup
- **Reference Dataset**: Controlled synthetic demonstration testbed adhering to the IEEE-CIS E-Commerce Fraud Detection feature schema.
- **Train / Val / Test Split**: 60% Train (30,000 rows) / 20% Val (10,000 rows) / 20% Test (10,000 rows).
- **Validation-Selected Alert Cutoff**: score >= 0.3407 (targeted at 2.0% alert budget).
- **Test Fraud Prevalence**: 3.97%.
- **Model**: Gradient Boosted Decision Trees (XGBoost) trained on tabular transaction attributes.

## 2. Evaluation Results
- **ROC-AUC**: **0.9213**
- **Average Precision (PR-AUC)**: **0.4003**
- **Fixed Alert Budget (2.0%)**:
  - **Alerts Dispatched**: 217
  - **True Positives**: 108
  - **False Positives**: 109
  - **Precision**: 49.77%
  - **Recall**: 27.20%

## 3. Top Predictive Features (Gain)
| Feature | Description | Importance Gain |
|---|---|---|
| `ProductCD_C` | Transaction/card/velocity indicator | 130.12 |
| `card_credit` | Transaction/card/velocity indicator | 38.55 |
| `C1` | Transaction/card/velocity indicator | 23.04 |
| `TransactionAmt` | Transaction/card/velocity indicator | 19.87 |
| `C13` | Transaction/card/velocity indicator | 13.67 |
| `D1` | Transaction/card/velocity indicator | 9.55 |
| `dist1` | Transaction/card/velocity indicator | 9.24 |
| `ProductCD_R` | Transaction/card/velocity indicator | 3.21 |

## 4. Methodological Distinction
As outlined in Section 4 and 7 of the project proposal, IEEE-CIS represents an isolated tabular benchmark without explicit graph network topology. Results are presented independently and are not directly compared numerically against IBM AML or UPI graph results.