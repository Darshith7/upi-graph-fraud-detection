# Model Explainability Report: GNNExplainer & TreeSHAP

## 1. Objectives & Compliance
- **Dual Explanation Architecture**: TreeSHAP for tabular feature attribution and GNNExplainer for local graph topological edge importance.
- **Audit Trail Integrity**: Each explanation strictly logs prediction timestamp, graph history cutoff, and supporting transaction IDs.
- **Non-Prosecutorial Stance**: Explanations are provided as evidentiary aids for compliance analysts, not absolute proof of wrongdoing.

## 2. Exemplary Case Studies

### Case 1: Transaction #179904 (HIGH Risk - Score: 0.9828)
- **Transfer**: `acc_17800@paytm` ➔ `acc_08676@okhdfcbank`
- **Amount**: ₹14.03
- **Graph History Cutoff**: 1725804211 (UTC)
- **Recommended Action**: Freeze account and escalate to FIU-IND compliance

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `log1p_amount` | 2.71 | +6.336 | Pushes Risk HIGHER |
| `sender_log1p_seconds_since_out` | 12.0208 | -0.276 | Lowers Risk |
| `receiver_log1p_out_count` | 2.4849 | +0.231 | Pushes Risk HIGHER |
| `receiver_log1p_in_count` | 2.3979 | +0.225 | Pushes Risk HIGHER |
| `sender_log1p_out_count` | 1.0986 | -0.184 | Lowers Risk |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #171651 | `acc_11905@oksbi` | `acc_17800@paytm` | ₹301.14 | Inflow to Sender | 0.9621 |
| #159945 | `acc_17313@icici` | `acc_08676@okhdfcbank` | ₹170.83 | Rapid Inflow to Beneficiary | 0.3657 |
| #168494 | `acc_18273@icici` | `acc_08676@okhdfcbank` | ₹195.30 | Rapid Inflow to Beneficiary | 0.3648 |
| #177577 | `acc_17313@icici` | `acc_17800@paytm` | ₹170.83 | Inflow to Sender | 0.3457 |

### Case 2: Transaction #179931 (HIGH Risk - Score: 0.9865)
- **Transfer**: `acc_17995@oksbi` ➔ `acc_09872@okaxis`
- **Amount**: ₹3.01
- **Graph History Cutoff**: 1725804533 (UTC)
- **Recommended Action**: Freeze account and escalate to FIU-IND compliance

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `log1p_amount` | 1.3888 | +6.032 | Pushes Risk HIGHER |
| `receiver_log1p_in_count` | 3.6109 | +0.550 | Pushes Risk HIGHER |
| `sender_log1p_out_count` | 1.7918 | +0.233 | Pushes Risk HIGHER |
| `receiver_log1p_seconds_since_out` | 12.993 | -0.204 | Lowers Risk |
| `receiver_log1p_seconds_since_in` | 11.4209 | -0.118 | Lowers Risk |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #157244 | `acc_12589@oksbi` | `acc_17995@oksbi` | ₹292.17 | Inflow to Sender | 0.9646 |
| #151106 | `acc_13681@oksbi` | `acc_09872@okaxis` | ₹341.10 | Rapid Inflow to Beneficiary | 0.9635 |
| #151703 | `acc_02953@oksbi` | `acc_17995@oksbi` | ₹581.71 | Inflow to Sender | 0.9634 |
| #144452 | `acc_06660@okhdfcbank` | `acc_09872@okaxis` | ₹296.36 | Rapid Inflow to Beneficiary | 0.9626 |

### Case 3: Transaction #100984 (LOW Risk - Score: 0.0439)
- **Transfer**: `acc_03144@okhdfcbank` ➔ `acc_19184@okaxis`
- **Amount**: ₹487.98
- **Graph History Cutoff**: 1725239427 (UTC)
- **Recommended Action**: Permit standard clearing

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `receiver_log1p_in_count` | 0.0 | -0.621 | Lowers Risk |
| `sender_log1p_out_count` | 0.0 | -0.466 | Lowers Risk |
| `receiver_log1p_unique_senders` | 0.0 | +0.329 | Pushes Risk HIGHER |
| `log1p_amount` | 6.1923 | -0.248 | Lowers Risk |
| `sender_log1p_seconds_since_out` | 0.0 | +0.179 | Pushes Risk HIGHER |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
