# AI-Based Graph Fraud and Mule-Account Detection System for UPI-Like Transaction Networks

**Project Report | Revised Proposal Implementation**  
**Team**:
- **Anubola Sai Darshith**
- **Male Pranay Yashwin**
- **Krishna Gupta**

---

## 1. Executive Summary

Digital payment ecosystems (such as UPI) present unique anti-money laundering (AML) and fraud challenges. Fraudulent activity frequently spans multi-hop money mule networks characterized by rapid fan-in pooling, layering cycles, and coordinated pass-through disbursements. Conventional tabular fraud detection models evaluate isolated transaction rows and struggle to capture these structural dependencies without leaking future topology.

This project delivers a complete, production-ready AI-driven detection platform combining:
1. **Temporal Graph Neural Networks (Directional GraphSAGE)** and a rigorous tabular reference (**XGBoost**) evaluated under an identical temporal split and alert budget on the **IBM AML benchmark**.
2. **AMLSim Synthetic UPI Network with 3 Controlled Challenge Cases** testing disguised suspicious amounts/delays, benign lookalike merchant/payroll hubs, and incomplete earlier history under strict past-only feature calculation.
3. **Dual Model Explainability**:
   - **GNNExplainer**: Identifies influential local topological subgraphs and supporting transaction IDs strictly prior to cutoff.
   - **TreeSHAP**: Provides exact Shapley feature attributions for tabular predictors.
4. **Demographic Fairness & Subgroup Audit** on the **Feedzai Bank Account Fraud (BAF)** benchmark suite across age brackets and income tiers.
5. **Conventional Tabular Reference** on the **IEEE-CIS** fraud detection formulation.
6. **High-Throughput Real-Time Risk-Scoring API & Stream Replayer** achieving **median latency of 3.36 ms** and **p95 latency of 4.10 ms** (100% compliance with the sub-200ms operational target).
7. **Two AI Copilots**:
   - **Bank Analyst Copilot**: Generates structured, editable Suspicious Activity Reports (SAR) grounded in verifiable transaction IDs and graph topology.
   - **Consumer Safety Copilot**: Provides real-time, multilingual UPI payment safety advisories in 6 languages (English, Hindi, Telugu, Tamil, Marathi, Bengali) with explicit sparse-history alerts.
8. **Interactive Real-Time Web Dashboard** uniting live stream replay, real-time risk scoring, explainability inspection, SAR generation, and benchmark analytics.

---

## 2. System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Incoming UPI Transaction                 │
                    │        (Sender VPA, Receiver VPA, Amount, Time)        │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │             Past-Only Temporal Feature Store           │
                    │  (Calculates degree, in/out counts, velocity strictly   │
                    │      BEFORE mutating state; zero future data leakage)  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
        ┌────────────────────────────────┐            ┌────────────────────────────────┐
        │   Directional GraphSAGE (GNN)  │            │     Tabular XGBoost Baseline   │
        │ 1-hop neighbor convolutions &  │            │  Gradient boosted trees over   │
        │ directional flow embeddings    │            │  past-only engineered features │
        └────────────────┬───────────────┘            └────────────────┬───────────────┘
                         │                                             │
                         └──────────────────────┬──────────────────────┘
                                                │
                                                ▼
                    ┌────────────────────────────────────────────────────────┐
                    │            Ensemble Risk Engine & Reasoner             │
                    │ (Computes Risk Level: LOW / MED / HIGH, Alert Flag,    │
                    │  Insufficient-History Flag, Graph History Cutoff)      │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼ (Event committed into graph after scoring)
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
        ┌────────────────────────────────┐            ┌────────────────────────────────┐
        │       GNNExplainer Module      │            │       TreeSHAP Explainer       │
        │ Identifies top influential     │            │ Computes exact Shapley feature │
        │ prior transaction edges        │            │ attributions pushing risk      │
        └────────────────┬───────────────┘            └────────────────┬───────────────┘
                         │                                             │
                         └──────────────────────┬──────────────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
        ┌────────────────────────────────┐            ┌────────────────────────────────┐
        │     Bank Analyst Copilot       │            │     Consumer Safety Copilot    │
        │ Evidence-grounded SAR summary  │            │ Multilingual payment advisory  │
        │ with traceable transaction IDs │            │ with sparse history disclaimer │
        └────────────────────────────────┘            └────────────────────────────────┘
```

---

## 3. Experimental Results & Benchmarks

### A. IBM AML Main Benchmark (Held-out Test Split: 862,792 Transactions)
Both models were evaluated on the held-out test split under a fixed **0.5% alert budget** (4,313 alerts):

| Model | Test Average Precision (PR-AUC) | Alerts @ 0.5% Budget | True Positives | False Positives | Precision | Recall | FPs per 1,000 non-fraud |
|---|---|---|---|---|---|---|---|
| **XGBoost Baseline** | 0.1960 | 4,313 | 458 | 3,855 | 10.62% | 47.91% | 4.47 |
| **Directional GraphSAGE** | **0.2096** | 4,313 | **478** | **3,835** | **11.08%** | **50.00%** | **4.45** |

- **Key Takeaway**: Learned graph context improved recall by +2.09 percentage points (+20 true laundering cases detected) while decreasing false alarms under an identical review capacity.

---

### B. AMLSim Synthetic UPI Stream & 3 Controlled Challenge Scenarios
Evaluated over 120,558 transactions (Steps 1..90 Train, Steps 91..149 Test) using strict past-only feature calculation:

| Scenario | Evaluated Tx | Fraud Tx | XGBoost AP | GraphSAGE AP | XGB Recall @ 1% | GraphSAGE Recall @ 1% | Lookalike Hub FPs (XGB / GNN) |
|---|---|---|---|---|---|---|---|
| **Baseline (Standard AMLSim)** | 41,618 | 3,468 | 0.7985 | 0.4006 | 12.00% | 6.86% | 0 / 0 |
| **Challenge 1 (Disguised Suspicious)** | 42,210 | 4,060 | 0.5730 | 0.3211 | 9.31% | 5.67% | 0 / 0 |
| **Challenge 2 (Benign Lookalike Hubs)** | 42,453 | 3,468 | 0.7678 | 0.3984 | 12.23% | 6.98% | 0 / 0 |
| **Challenge 3 (Partial History 50%)** | 41,618 | 3,468 | 0.7772 | 0.3930 | 12.00% | 7.15% | 0 / 0 |

#### Insights on the Challenge Cases:
1. **Challenge 1 (Disguised Suspicious)**: Laundering amounts were split into micro-ticket sizes (₹115-₹485) and delayed by +3 to +13 steps. Both models experienced performance degradation, revealing that detection heuristics reliant purely on velocity are vulnerable to strategic obfuscation.
2. **Challenge 2 (Benign Lookalike Hubs)**: 2,500 legitimate transactions were injected across retail QR merchant hubs (high fan-in) and corporate payroll hubs (high fan-out). Both models achieved **zero false positive alerts** on these legitimate hubs by evaluating reciprocal neighbor relationships.
3. **Challenge 3 (Partial / Missing History)**: Suppressing 50% of earlier edges demonstrated the cold-start challenge. The system reliably raised the `insufficient_history` flag, preventing unwarranted reassurance.

---

### C. Bank Account Fraud (BAF) Demographic Fairness Audit
Evaluated on 18,000 test applications across demographic age brackets and income tiers:

| Subgroup | Sample Count | Share | Base Fraud Rate | Alert Rate | False Positive Rate (FPR) | True Positive Rate (TPR) |
|---|---|---|---|---|---|---|
| **Youth (18-24)** | 1,882 | 10.5% | 1.54% | 5.37% | **5.40%** | 3.45% |
| **Young Adult (25-39)** | 6,015 | 33.4% | 1.13% | 5.29% | **5.26%** | 7.35% |
| **Middle Age (40-59)** | 6,483 | 36.0% | 1.31% | 4.94% | **4.92%** | 5.88% |
| **Senior (60+)** | 3,620 | 20.1% | 0.86% | 4.45% | **4.40%** | 9.68% |

- **Fairness Finding**: Applicants in the 18-24 bracket exhibit a 1.00 percentage point higher False Positive Rate (5.40% vs 4.40% for seniors) due to shorter residential stability and credit history.
- **Methodological Stance**: As required by the proposal, these results are reported as experimental findings without claiming the system is bias-free.

---

### D. IEEE-CIS Conventional Tabular Fraud Benchmark
- **ROC-AUC**: **0.9238**
- **Average Precision (PR-AUC)**: **0.4682**
- **Precision @ 2% Alert Budget**: **35.33%**
- **Top Predictive Features**: `card1`, `TransactionAmt`, `C13`, `addr1`, `dist1`.

---

### E. Real-Time Scoring Latency Benchmark (1,000 Transactions Tested)
- **Median Latency (p50)**: **3.36 ms**
- **95th Percentile Latency (p95)**: **4.10 ms**
- **99th Percentile Latency (p99)**: **4.41 ms**
- **Maximum Latency**: **95.75 ms**
- **Throughput**: **280.6 transactions / second**
- **Sub-200ms Compliance**: **100.0%** (Operational Target: Met)

---

## 4. Repository Structure

```
├── README.md                           # Comprehensive documentation & report
├── requirements.txt                    # Project dependencies
├── data/
│   ├── raw/HI-Small_Trans.csv          # Raw IBM AML benchmark dataset
│   ├── processed/                      # SQLite prepared transactions and split manifest
│   ├── features/                       # Past-only feature arrays (.npy) and feature_schema.json
│   ├── temporal_graph/                 # Compressed CSR temporal graph index
│   ├── amlsim_raw/                     # Extracted AMLSim 20K synthetic network
│   ├── amlsim_challenges/              # Baseline + 3 Challenge datasets and manifest
│   └── baf/                            # Bank Account Fraud benchmark dataset
├── models/
│   ├── xgboost_baseline_v1/            # IBM AML XGBoost baseline (v1)
│   ├── xgboost_baseline_v2/            # IBM AML XGBoost baseline (v2)
│   ├── graphsage_v1/                   # IBM AML GraphSAGE checkpoint (v1)
│   ├── graphsage_v2/                   # IBM AML GraphSAGE checkpoint (v2)
│   └── amlsim_models/                  # Trained AMLSim UPI GraphSAGE and XGBoost models
├── reports/
│   ├── test_evaluation_v1/             # IBM AML test evaluation metrics & JSON comparison
│   ├── error_analysis_v1/              # IBM AML error review, case studies, HTML report
│   ├── amlsim_challenges/              # 3 Challenge cases metrics & summary report
│   ├── explainability/                 # GNNExplainer & TreeSHAP case studies & audit report
│   ├── fairness_baf/                   # BAF subgroup fairness metrics & report
│   ├── ieee_cis_benchmark/             # IEEE-CIS tabular benchmark report
│   └── api_benchmark/                  # Real-time scoring API latency benchmark report
├── src/
│   ├── prepare_data.py                 # SQLite database preparation from raw CSV
│   ├── prepare_graph.py                # Temporal graph index construction
│   ├── build_features.py               # Past-only temporal feature store builder
│   ├── train_baseline.py               # XGBoost training on IBM AML
│   ├── train_graphsage.py              # Directional temporal GraphSAGE training
│   ├── evaluate_models.py              # IBM AML test split evaluation
│   ├── analyze_errors.py               # Descriptive error analysis on IBM AML
│   ├── prepare_amlsim_challenges.py    # AMLSim UPI network & 3 challenge cases generator
│   ├── evaluate_amlsim_challenges.py   # Challenge cases evaluator & error tracker
│   ├── explainability.py               # GNNExplainer & TreeSHAP explanation module
│   ├── evaluate_fairness_baf.py        # Bank Account Fraud fairness evaluator
│   ├── evaluate_ieee_cis.py            # IEEE-CIS tabular benchmark experiment
│   ├── copilots.py                     # Bank Analyst SAR & Multilingual Consumer Copilots
│   ├── scoring_service.py              # Real-Time Scoring API & Web Application
│   └── benchmark_latency.py            # Automated API latency benchmark suite
└── tests/
    └── test_system_components.py       # End-to-end integration and verification suite
```

---

## 5. Quickstart & Execution Guide

### 1. Run Complete Test Suite
Verify that all feature stores, models, copilots, and benchmark reports are verified:
```bash
PYTHONPATH=src .venv/bin/python tests/test_system_components.py
```

### 2. Re-run AMLSim Challenge Scenarios
```bash
PYTHONPATH=src .venv/bin/python src/prepare_amlsim_challenges.py
PYTHONPATH=src .venv/bin/python src/evaluate_amlsim_challenges.py
```

### 3. Re-run Explainability Audit (GNNExplainer + TreeSHAP)
```bash
PYTHONPATH=src .venv/bin/python src/explainability.py
```

### 4. Re-run BAF Fairness Audit & IEEE-CIS Benchmark
```bash
PYTHONPATH=src .venv/bin/python src/evaluate_fairness_baf.py
PYTHONPATH=src .venv/bin/python src/evaluate_ieee_cis.py
```

### 5. Benchmark API Real-Time Scoring Latency
```bash
PYTHONPATH=src .venv/bin/python src/benchmark_latency.py --count 1000
```

### 6. Launch the Interactive Web Platform & REST API
```bash
PYTHONPATH=src .venv/bin/python src/scoring_service.py --port 8000
```
Open **`http://localhost:8000`** in any web browser to access:
- **⚡ Real-Time Live Scoring Feed**: Step-by-step or continuous streaming playback with real-time risk gauges.
- **🧪 3 Challenge Scenarios**: Interactive comparison table and breakdown of disguised, lookalike hub, and partial history tests.
- **🔍 GNN & SHAP Explainability**: Visual topological subgraphs and TreeSHAP waterfall charts.
- **💼 Bank Analyst Copilot**: One-click SAR generation with editable narrative and export.
- **🛡️ Consumer Safety Copilot**: Multilingual UPI ID safety search (English, Hindi, Telugu, Tamil, Marathi, Bengali).
- **📊 Model & Fairness Benchmarks**: Interactive IBM AML, BAF, and IEEE-CIS benchmark analytics.

