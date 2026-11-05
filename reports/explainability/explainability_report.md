# Model Explainability Report: GNNExplainer & TreeSHAP

## 1. Objectives & Compliance
- **Dual Explanation Architecture**: TreeSHAP for tabular feature attribution and GNNExplainer for local graph topological edge importance.
- **Audit Trail Integrity**: Each explanation strictly logs prediction timestamp, graph history cutoff, and supporting transaction IDs.
- **Non-Prosecutorial Stance**: Explanations are provided as evidentiary aids for compliance analysts, not absolute proof of wrongdoing.

## 2. Exemplary Case Studies

### Case 1: Transaction #211329 (HIGH Risk - Score: 0.9978)
- **Transfer**: `acc_19088@okaxis` ➔ `acc_19773@icici`
- **Amount**: ₹3.35
- **Graph History Cutoff**: 1726037582 (UTC)
- **Recommended Action**: Freeze account and escalate to FIU-IND compliance

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `log1p_amount` | 1.4702 | +4.908 | Pushes Risk HIGHER |
| `pair_first_seen_flag` | 0.0 | +2.280 | Pushes Risk HIGHER |
| `pair_log1p_repeat_cnt_72h` | 1.0986 | +0.255 | Pushes Risk HIGHER |
| `receiver_log1p_out_cnt_24h` | 2.0794 | +0.225 | Pushes Risk HIGHER |
| `sender_log1p_out_amount_24h` | 2.0931 | +0.189 | Pushes Risk HIGHER |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #206709 | `acc_11839@oksbi` | `acc_19773@icici` | ₹325.89 | Rapid Inflow to Beneficiary | 0.3351 |
| #211331 | `acc_19088@okaxis` | `acc_19773@icici` | ₹3.49 | Rapid Inflow to Beneficiary | 0.335 |
| #211328 | `acc_19088@okaxis` | `acc_19773@icici` | ₹3.62 | Rapid Inflow to Beneficiary | 0.335 |
| #209306 | `acc_19292@okaxis` | `acc_19088@okaxis` | ₹136.89 | Inflow to Sender | 0.3347 |

### Case 2: Transaction #197749 (HIGH Risk - Score: 0.9978)
- **Transfer**: `acc_18024@okhdfcbank` ➔ `acc_09987@icici`
- **Amount**: ₹3.70
- **Graph History Cutoff**: 1725925884 (UTC)
- **Recommended Action**: Freeze account and escalate to FIU-IND compliance

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `log1p_amount` | 1.5476 | +4.274 | Pushes Risk HIGHER |
| `pair_first_seen_flag` | 0.0 | +2.072 | Pushes Risk HIGHER |
| `sender_log1p_out_cnt_72h` | 2.3979 | +0.398 | Pushes Risk HIGHER |
| `sender_log1p_out_cnt_24h` | 2.0794 | +0.352 | Pushes Risk HIGHER |
| `receiver_log1p_out_cnt_24h` | 2.5649 | +0.314 | Pushes Risk HIGHER |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #196317 | `acc_09987@icici` | `acc_12961@oksbi` | ₹235.83 | Layering Outflow from Beneficiary | 0.3345 |
| #197336 | `acc_09987@icici` | `acc_08354@okaxis` | ₹235.83 | Layering Outflow from Beneficiary | 0.3345 |
| #194169 | `acc_09987@icici` | `acc_12443@upi` | ₹235.83 | Layering Outflow from Beneficiary | 0.3345 |
| #195238 | `acc_09987@icici` | `acc_09376@paytm` | ₹235.83 | Layering Outflow from Beneficiary | 0.3345 |

### Case 3: Transaction #188938 (LOW Risk - Score: 0.0521)
- **Transfer**: `acc_17085@icici` ➔ `acc_17143@oksbi`
- **Amount**: ₹508.67
- **Graph History Cutoff**: 1725865441 (UTC)
- **Recommended Action**: Permit standard clearing

#### TreeSHAP Top Attributions:
| Feature | Feature Value | SHAP Attribution | Direction |
|---|---|---|---|
| `receiver_log1p_in_cnt_72h` | 1.0986 | -0.497 | Lowers Risk |
| `receiver_log1p_unique_senders_72h` | 1.0986 | +0.245 | Pushes Risk HIGHER |
| `log1p_amount` | 6.2338 | -0.211 | Lowers Risk |
| `receiver_log1p_seconds_since_out` | 12.1021 | -0.203 | Lowers Risk |
| `receiver_log1p_in_cnt_24h` | 0.0 | -0.181 | Lowers Risk |

#### GNNExplainer Influential Subgraph Edges:
| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |
|---|---|---|---|---|---|
| #173437 | `acc_02433@icici` | `acc_17143@oksbi` | ₹203.57 | Rapid Inflow to Beneficiary | 0.3374 |
| #162975 | `acc_06710@okaxis` | `acc_17143@oksbi` | ₹239.08 | Rapid Inflow to Beneficiary | 0.3373 |
| #133060 | `acc_11908@paytm` | `acc_17143@oksbi` | ₹204.71 | Rapid Inflow to Beneficiary | 0.3372 |
| #187836 | `acc_17085@icici` | `acc_09159@icici` | ₹508.67 | Outflow from Sender | 0.3345 |
