# Model Explainability Report: GNNExplainer & TreeSHAP

## 1. Objectives & Compliance
- **Dual Explanation Architecture**: TreeSHAP for tabular feature attribution and GNNExplainer for local graph topological edge importance.
- **Audit Trail Integrity**: Each explanation strictly logs prediction timestamp, graph history cutoff, and supporting transaction IDs.
- **Non-Prosecutorial Stance**: Explanations are provided as evidentiary aids for compliance analysts, not absolute proof of wrongdoing.

## 2. Exemplary Case Studies

### Case 1: Transaction #179904 (HIGH Risk - Score: 0.9832)
- **Transfer**: `acc_17800@paytm` ➔ `acc_08676@okhdfcbank`
- **Amount**: ₹14.03
- **Graph History Cutoff**: 1725804211 (UTC)
- **Recommended Action**: Freeze account and escalate to FIU-IND compliance

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `log1p_amount` | 2.71 | +6.354 | Pushes Risk HIGHER |
| `sender_log1p_out_count` | 1.0986 | -0.293 | Lowers Risk |
| `receiver_log1p_out_count` | 2.4849 | +0.226 | Pushes Risk HIGHER |
| `sender_log1p_unique_receivers` | 1.0986 | +0.185 | Pushes Risk HIGHER |
| `sender_log1p_seconds_since_out` | 12.0208 | -0.182 | Lowers Risk |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #177577 | `acc_17313@icici` | `acc_17800@paytm` | ₹170.83 | Layering Outflow | 0.144 |
| #171490 | `acc_08676@okhdfcbank` | `acc_02426@okaxis` | ₹256.70 | Layering Outflow | 0.1435 |
| #154566 | `acc_08676@okhdfcbank` | `acc_03142@paytm` | ₹256.70 | Layering Outflow | 0.1349 |
| #166728 | `acc_08676@okhdfcbank` | `acc_14135@upi` | ₹256.70 | Layering Outflow | 0.1285 |

**Supporting Transaction IDs**: `177577, 171490, 154566, 166728`
---

### Case 2: Transaction #179931 (HIGH Risk - Score: 0.9863)
- **Transfer**: `acc_17995@oksbi` ➔ `acc_09872@okaxis`
- **Amount**: ₹3.01
- **Graph History Cutoff**: 1725804533 (UTC)
- **Recommended Action**: Freeze account and escalate to FIU-IND compliance

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `log1p_amount` | 1.3888 | +6.298 | Pushes Risk HIGHER |
| `receiver_log1p_in_count` | 3.6109 | +0.523 | Pushes Risk HIGHER |
| `receiver_log1p_seconds_since_out` | 12.993 | -0.346 | Lowers Risk |
| `sender_log1p_out_count` | 1.7918 | +0.136 | Pushes Risk HIGHER |
| `receiver_log1p_seconds_since_in` | 11.4209 | -0.052 | Lowers Risk |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #144452 | `acc_06660@okhdfcbank` | `acc_09872@okaxis` | ₹296.36 | Rapid Inflow to Mule | 0.9126 |
| #157244 | `acc_12589@oksbi` | `acc_17995@oksbi` | ₹292.17 | Layering Outflow | 0.9072 |
| #177639 | `acc_17995@oksbi` | `acc_15846@okhdfcbank` | ₹208.57 | Layering Outflow | 0.9068 |
| #161831 | `acc_07548@okhdfcbank` | `acc_17995@oksbi` | ₹129.96 | Layering Outflow | 0.9043 |

**Supporting Transaction IDs**: `144452, 157244, 177639, 161831`
---

### Case 3: Transaction #100984 (LOW Risk - Score: 0.0451)
- **Transfer**: `acc_03144@okhdfcbank` ➔ `acc_19184@okaxis`
- **Amount**: ₹487.98
- **Graph History Cutoff**: 1725239427 (UTC)
- **Recommended Action**: Permit standard clearing

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `receiver_log1p_in_count` | 0.0 | -0.730 | Lowers Risk |
| `sender_log1p_out_count` | 0.0 | -0.508 | Lowers Risk |
| `receiver_log1p_unique_senders` | 0.0 | +0.383 | Pushes Risk HIGHER |
| `log1p_amount` | 6.1923 | -0.296 | Lowers Risk |
| `receiver_log1p_seconds_since_out` | 0.0 | +0.179 | Pushes Risk HIGHER |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|

**Supporting Transaction IDs**: ``
---
