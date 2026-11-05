# AI-Based Graph Fraud and Mule-Account Detection System for UPI-Like Transaction Networks

**Project Report | Revised Proposal Implementation**  
**Team**:
- **Anubola Sai Darshith - 240958204**
- **Male Pranay Yashwin - 240958224**
- **Krishna Gupta - 240958222**

---

## 1. Executive Summary

Digital payment ecosystems (such as UPI) present unique anti-money laundering (AML) and fraud challenges. Fraudulent activity frequently spans multi-hop money mule networks characterized by rapid fan-in pooling, layering cycles, and coordinated pass-through disbursements. Conventional tabular fraud detection models evaluate isolated transaction rows and struggle to capture these structural dependencies without leaking future topology.

This project is a research prototype of an AI-driven detection platform (it is **not** validated on real UPI traffic and is not production-ready; see [Known Limitations](#6-known-limitations) and [MODEL_CARD.md](MODEL_CARD.md)). It combines:
1. **Temporal Graph Neural Networks (Directional GraphSAGE)** and a rigorous tabular reference (**XGBoost**), trained **causally** on a simulated UPI network and compared over five seeds with a feature-set ablation (sections A-B). *(A generic, non-UPI IBM AML comparison from the original proposal is kept in the supplementary section only.)*
2. **AMLSim Synthetic UPI Network with 3 Controlled Challenge Cases** (this is the model the scoring API serves) testing disguised suspicious amounts/delays, benign lookalike merchant/payroll hubs, and incomplete earlier history under strict past-only feature calculation. The accounts are simulated; the UPI-style handles are labels, not real UPI data.
3. **Dual Model Explainability**:
   - **GNNExplainer**: Identifies influential local topological subgraphs and supporting transaction IDs strictly prior to cutoff.
   - **TreeSHAP**: Provides exact Shapley feature attributions for tabular predictors.
4. **An independent held-out UPI simulation** (different generator, UPI limits, merchant-heavy traffic, five mule typologies the model never saw) used for zero-shot testing. The honest result is that the AMLSim-trained model **does not transfer** to it (section B).
5. **Real-Time Risk-Scoring API & Stream Replayer** with end-to-end median latency of **2.3 ms** and p95 **3.7 ms** (about 394 transactions/s, CPU, sequential requests; includes the SHAP explanation), optional API-key authentication and rate limiting.
6. **Two AI Copilots**:
   - **Bank Analyst Copilot**: Generates structured, editable Suspicious Activity Reports (SAR) grounded in verifiable transaction IDs and graph topology.
   - **Consumer Safety Copilot**: Provides real-time, multilingual UPI payment safety advisories in 6 languages (English, Hindi, Telugu, Tamil, Marathi, Bengali) with explicit sparse-history alerts. Merchant exemptions come from a verified-merchant registry (`data/verified_merchants.json`), never from the handle text.
7. **Interactive Real-Time Web Dashboard** uniting live stream replay, real-time risk scoring, explainability inspection, SAR generation, and benchmark analytics (smoke-tested in headless Chromium).

The project is **UPI-focused**. The IBM AML, BAF-style and IEEE-CIS-style material from the original proposal is generic financial-fraud work, is not used by the API, and is kept only in the clearly labelled supplementary section.

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
        │ 2-layer neighbor convolutions &│            │  Gradient boosted trees over   │
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

## 3. Experimental Results (UPI)

All data behind the results in sections A-C is **simulated** UPI-style data; there is no public UPI transaction dataset. See [docs/DATA.md](docs/DATA.md) for provenance and [MODEL_CARD.md](MODEL_CARD.md) for intended use and failure modes.

### A. Served model: AMLSim training data + 3 controlled stress tests
XGBoost uses 23 **windowed** past-only features (6h / 24h / 72h counts, unique counterparties, amounts, flow-through, repeat-pair, recency). GraphSAGE is trained **causally** (every training step sees only the graph of earlier steps; node features are a bounded running mean) and validated and tested on the identical evolving-graph code path used by the live service. The blend (0.25 x XGBoost + 0.75 x GraphSAGE) and the alert cutoff are tuned on validation only. Steps 1-75 train, 76-90 validation, 91-149 test.

#### A1. Deployed model (seed 42), rank-budget view

| Scenario | Evaluated Tx | Fraud Tx | XGBoost AP | GraphSAGE AP | Deployed Blend AP | XGB Recall @ 1% | GraphSAGE Recall @ 1% | Blend Recall @ 1% |
|---|---|---|---|---|---|---|---|---|
| **Baseline (Standard AMLSim)** | 41,618 | 3,468 | 0.7777 | 0.8400 | 0.8412 | 12.00% | 12.00% | 12.00% |
| **Challenge 1 (Disguised Suspicious)** | 41,618 | 3,468 | 0.6944 | 0.7447 | 0.7709 | 12.00% | 12.00% | 12.00% |
| **Challenge 2 (Benign Lookalike Hubs)** | 42,453 | 3,468 | 0.7414 | 0.7838 | 0.7897 | 12.23% | 12.20% | 12.23% |
| **Challenge 3 (Partial History 50%)** | 41,618 | 3,468 | 0.7578 | 0.8298 | 0.8323 | 12.00% | 12.00% | 12.00% |

(Recall at 1% is capped near 12% because 1% of transactions is far fewer alerts than the ~8% of transactions that are laundering in this simulation.)

#### A2. Five seeds and feature-set ablation (mean +/- std; every row retrains from scratch)

| Features | Scenario | XGBoost AP | GraphSAGE AP | Blend AP | Blend - XGB | Policy alert rate | Policy precision | Policy recall |
|---|---|---|---|---|---|---|---|---|
| windowed | Baseline (Standard AMLSim) | 0.7775 +/- 0.0005 | 0.8329 +/- 0.0073 | 0.8344 +/- 0.0067 | 0.0572 +/- 0.0068 | 5.1% +/- 0.0% | 95.1% +/- 0.5% | 58.2% +/- 0.5% |
| windowed | Challenge 1 (Disguised Suspicious) | 0.6935 +/- 0.0014 | 0.7222 +/- 0.0213 | 0.7547 +/- 0.0127 | 0.0617 +/- 0.0125 | 4.7% +/- 0.0% | 90.2% +/- 0.8% | 50.5% +/- 0.7% |
| windowed | Challenge 2 (Benign Lookalike Hubs) | 0.7403 +/- 0.0010 | 0.7821 +/- 0.0157 | 0.7868 +/- 0.0092 | 0.0462 +/- 0.0098 | 5.2% +/- 0.0% | 88.1% +/- 1.4% | 55.9% +/- 0.8% |
| windowed | Challenge 3 (Partial History 50%) | 0.7569 +/- 0.0008 | 0.8224 +/- 0.0082 | 0.8249 +/- 0.0078 | 0.0681 +/- 0.0078 | 4.9% +/- 0.1% | 97.3% +/- 0.5% | 57.7% +/- 0.6% |
| windowed_nopair | Baseline (Standard AMLSim) | 0.7719 +/- 0.0010 | 0.8035 +/- 0.0376 | 0.8123 +/- 0.0236 | 0.0406 +/- 0.0241 | 5.1% +/- 0.0% | 93.4% +/- 1.2% | 57.4% +/- 1.0% |
| windowed_nopair | Challenge 1 (Disguised Suspicious) | 0.4047 +/- 0.0148 | 0.4346 +/- 0.1051 | 0.4768 +/- 0.0530 | 0.0726 +/- 0.0485 | 3.9% +/- 1.0% | 64.9% +/- 1.6% | 29.9% +/- 7.4% |
| windowed_nopair | Challenge 2 (Benign Lookalike Hubs) | 0.7618 +/- 0.0006 | 0.7940 +/- 0.0384 | 0.8028 +/- 0.0246 | 0.0409 +/- 0.0249 | 5.1% +/- 0.0% | 92.6% +/- 1.4% | 58.3% +/- 1.2% |
| windowed_nopair | Challenge 3 (Partial History 50%) | 0.7499 +/- 0.0007 | 0.7989 +/- 0.0384 | 0.8042 +/- 0.0322 | 0.0544 +/- 0.0325 | 4.9% +/- 0.2% | 96.2% +/- 0.9% | 56.7% +/- 2.1% |
| cumulative | Baseline (Standard AMLSim) | 0.7642 +/- 0.0016 | 0.7807 +/- 0.0173 | 0.7886 +/- 0.0118 | 0.0246 +/- 0.0120 | 5.2% +/- 0.0% | 91.5% +/- 1.1% | 56.9% +/- 0.9% |
| cumulative | Challenge 1 (Disguised Suspicious) | 0.4151 +/- 0.0108 | 0.3312 +/- 0.0356 | 0.4016 +/- 0.0098 | -0.0130 +/- 0.0102 | 5.0% +/- 0.1% | 50.8% +/- 2.4% | 30.5% +/- 0.9% |
| cumulative | Challenge 2 (Benign Lookalike Hubs) | 0.7461 +/- 0.0021 | 0.7627 +/- 0.0191 | 0.7713 +/- 0.0136 | 0.0252 +/- 0.0134 | 5.2% +/- 0.0% | 89.6% +/- 1.3% | 57.2% +/- 1.0% |
| cumulative | Challenge 3 (Partial History 50%) | 0.7640 +/- 0.0008 | 0.7853 +/- 0.0264 | 0.7943 +/- 0.0166 | 0.0304 +/- 0.0167 | 5.0% +/- 0.1% | 94.6% +/- 0.3% | 56.1% +/- 1.3% |

* **Windowed features beat the legacy cumulative ones everywhere**, most of all on the disguised-transfers scenario (blend AP 0.402 -> 0.755). The cumulative counts keep growing, which made scores drift: with a fixed cutoff the alert rate was 16.1% on test against 7.4% with windowed features (the validation budget is 5%).
* **The graph component helps**: the blend beats XGBoost alone by +0.046 to +0.068 AP depending on the scenario (windowed features, mean over 5 seeds).
* **Trade-off, stated plainly:** the repeat-pair features are what lift Challenge 1 (blend AP 0.755 with them, 0.477 without), but they also make a legitimate recurring payment (same payee, daily) alert from about the 6th repeat, because AMLSim contains almost no benign recurring payments. `--feature-set windowed_nopair` removes that false-positive class and costs the Challenge 1 detection power.

#### A3. Deployed alert policy

An alert fires when `risk_score >= max(0.302, rolling cutoff)`: `0.302` is the validation score at a **5% alert budget** and the rolling cutoff is the score at the top 5% of the last 2,000 scores (past scores only).

| Scenario | Evaluated Tx | Alerts | Alert Rate | Precision | Recall | False Positives | Benign-hub alerts |
|---|---|---|---|---|---|---|---|
| **Baseline (Standard AMLSim)** | 41,618 | 2,133 | 5.13% | **95.31%** | **58.62%** | 100 | n/a |
| **Challenge 1 (Disguised Suspicious)** | 41,618 | 1,945 | 4.67% | **91.31%** | **51.21%** | 169 | n/a |
| **Challenge 2 (Benign Lookalike Hubs)** | 42,453 | 2,214 | 5.22% | **87.67%** | **55.97%** | 273 | 235 / 835 |
| **Challenge 3 (Partial History 50%)** | 41,618 | 2,066 | 4.96% | **97.82%** | **58.28%** | 45 | n/a |

Same model with only the fixed cutoff (no rolling budget):

| Scenario | Alert Rate | Precision | Recall |
|---|---|---|---|
| **Baseline (Standard AMLSim)** | 7.67% | 81.74% | 75.26% |
| **Challenge 1 (Disguised Suspicious)** | 6.45% | 82.64% | 63.99% |
| **Challenge 2 (Benign Lookalike Hubs)** | 9.01% | 68.22% | 75.26% |
| **Challenge 3 (Partial History 50%)** | 6.17% | 94.75% | 70.18% |

**Realistic base rates.** The simulation has ~8% laundering; real prevalence is far lower. Re-weighting the negatives (deployed policy, seed 42):

| Scenario | Precision @ 1% prevalence | Precision @ 0.1% prevalence | False alerts per true alert @ 0.1% |
|---|---|---|---|
| **Baseline (Standard AMLSim)** | 69.3% | 18.3% | 4.5 |
| **Challenge 1 (Disguised Suspicious)** | 53.9% | 10.4% | 8.6 |
| **Challenge 2 (Benign Lookalike Hubs)** | 44.7% | 7.4% | 12.5 |
| **Challenge 3 (Partial History 50%)** | 83.3% | 33.1% | 2.0 |

#### A4. Reading the stress tests

1. **Challenge 1 (disguised transfers):** blend AP 0.771 (seed 42); across seeds 0.7547 +/- 0.0127. Policy recall is only 50.5% +/- 0.7%.
2. **Challenge 2 (benign lookalike hubs):** 26.3% +/- 3.2% of benign hub events are still alerted (seed 42: 235 of 835). Behaviour features cannot fully separate a busy legitimate merchant from a collection mule.
3. **Challenge 3 (50% history withheld):** blend AP 0.8249 +/- 0.0078; the `insufficient_history` flag marks scores that rest on thin history.

---

### B. Independent held-out UPI simulation (zero-shot)

`src/generate_upi_simulation.py` is a separate generator (UPI per-payment limit, merchant-QR heavy traffic, payroll runs, ~1 payment per person per day, ~0.7% laundering) with **five mule typologies absent from the training data**: collection mule (8-80 senders), relay chain, structuring below limits, circular ring and fresh-account scam. The deployed model is applied with **nothing retrained or re-tuned**.

| Score | XGBoost | GraphSAGE | Blend | Random guess |
|---|---|---|---|---|
| Average precision | 0.0054 | 0.0045 | 0.0046 | 0.0084 |

* **Result: the model does not transfer.** AP is below random guessing, the fixed cutoff alerts on 56% of traffic, and the deployed policy catches 0.0% of laundering.
* **It is not a broken simulator:** XGBoost retrained on this simulation's own early period reaches AP 0.9949 on its later period. (That is also a warning: the simulation is easy once trained on, and amount alone has ROC-AUC 0.853.)
* **Why it fails:** the traffic mix differs. The median receiver handles 8 payments a day here against 1 in AMLSim (merchants dominate UPI), so merchant-like activity looks like laundering hubs. A deployment must (re)train on its own network's history.
* Version 1 of this simulator had a shortcut (amounts far above normal); it was fixed once, before the final evaluation, and not tuned against the model. Details in [docs/DATA.md](docs/DATA.md).

---

### C. Explanations and copilots

* **GNNExplainer fidelity (not demonstrated).** On 60 laundering transactions, removing the 2 edges the explainer ranks highest lowers the model's logit by 0.018 on average, against 0.011 for random edges (paired Wilcoxon p = 0.44). Individual prior edges barely move the score (mean absolute logit change 0.044), so the 'influential subgraph' is a lead for the investigator, not evidence; the SAR draft says so. TreeSHAP attributions are exact for XGBoost.
* **Consumer copilot** scores a typical payment to the recipient with the full model (dry run), instead of the hard-coded stand-in scores it used before. As a screening signal it separates future laundering receivers from benign ones with ROC-AUC 0.71 (150 vs 150 accounts, AMLSim): useful, not strong. Merchant exemptions come only from `data/verified_merchants.json`.
* **SAR draft** reads the blend weights from the checkpoint, fills in not-yet-streamed transactions by a read-only dry run, and returns 404 for unknown ids.

### D. Real-time scoring latency

Measured on the development machine (Windows, CPU only, one request at a time) by `src/benchmark_latency.py`. The GNN runs on the sender/receiver **2-hop in-neighbourhood only**, which is mathematically identical to the full-graph pass for a 2-layer mean-aggregation GraphSAGE (verified in a test, max logit difference 0), so the cost depends on local degree, not on the size of the graph.

| Measure | Median (p50) | p95 | p99 | Max |
|---|---|---|---|---|
| **End-to-end `score_transaction` (500 tx, incl. SHAP explanation)** | **2.34 ms** | **3.72 ms** | 4.60 ms | 24.10 ms |
| Staged (features + XGBoost + GraphSAGE + graph update, 1,000 tx) | 1.33 ms | 2.58 ms | 3.52 ms | 229.10 ms |

About **394 transactions/s** sequentially. (The earlier full-graph version took about 20 ms per transaction.) A hub with a very large in-degree makes its own transactions slower; this has not been load-tested beyond the roughly 120K-edge simulation.

### E. API security

Set `UPI_API_KEY` to require an `X-API-Key` header on every `/api/*` route (the dashboard asks for the key once). All `/api/*` routes are rate-limited per client (token bucket, default 600/min, burst 100; `UPI_RATE_LIMIT_PER_MIN`, `UPI_RATE_LIMIT_BURST`), request bodies are capped at 256 KB, and the server **refuses to bind a non-local interface without a key** unless `--allow-insecure` is passed. There are no user accounts, audit log or TLS: put it behind a reverse proxy for anything beyond local use. Covered by `test_api_key_and_rate_limit`.

---

## 3b. Supplementary: generic financial-fraud benchmarks (NOT UPI, optional)

These come from the original proposal. They are not used by the API or by any UPI result above and can be ignored for a UPI-focused reading. The BAF-style and IEEE-CIS-style sections run on **data generated by this repository**, not on the real datasets.

### S1. IBM AML Main Benchmark (Held-out Test Split: 862,792 Transactions)
Both models were evaluated on the held-out test split under a fixed **0.5% alert budget** (4,313 alerts):

| Model | Test Average Precision (PR-AUC) | Alerts @ 0.5% Budget | True Positives | False Positives | Precision | Recall | FPs per 1,000 non-fraud |
|---|---|---|---|---|---|---|---|
| **XGBoost Baseline** | 0.1960 | 4,313 | 458 | 3,855 | 10.62% | 47.91% | 4.47 |
| **Directional GraphSAGE** | **0.2096** | 4,313 | **478** | **3,835** | **11.08%** | **50.00%** | **4.45** |

- **Key Takeaway**: GraphSAGE detected 20 more laundering cases than XGBoost (recall +2.09 points) under an identical review capacity. This is a **single training run with no confidence interval on 956 positives, so the gap may be noise** and should not be read as proof that graph context helps on IBM AML.
- **Validation-to-test drop**: XGBoost scored AP 0.3234 / recall 58.4% on the validation split but AP 0.1960 / recall 47.9% on the test split, so performance degrades out of time (distribution shift between days).
- **Scope**: these IBM AML models are research baselines. The model served by the API is the AMLSim model in section A. The raw `HI-Small_Trans.csv` is not included in the repository, so section S1 cannot be reproduced without downloading it separately.

---

### S2. Fairness-Audit Protocol Check (BAF-style, **SYNTHETIC data - not the real Feedzai BAF**)
> **Read this first:** `evaluate_fairness_baf.py` *generates* its own dataset. The fraud score formula deliberately includes `+ 0.4 * (age < 25)`, so the youth disparity below is built into the generator. This section demonstrates that the subgroup-audit code runs; it is **not evidence about the model or about real BAF data**. To audit real data, download the Feedzai BAF dataset and point the script at it.

Evaluated on 12,000 synthetic holdout applications across age brackets and income tiers using validation-selected score threshold (threshold = 0.0218):

| Subgroup | Sample Count | Share | Base Fraud Rate | Alert Rate | False Positive Rate (FPR) | True Positive Rate (TPR) |
|---|---|---|---|---|---|---|
| **Youth (18-24)** | 1,242 | 10.3% | 2.01% | 12.56% | **11.75%** | 52.00% |
| **Young Adult (25-39)** | 3,994 | 33.3% | 0.55% | 4.13% | **3.88%** | 50.00% |
| **Middle Age (40-59)** | 4,320 | 36.0% | 0.63% | 3.98% | **3.66%** | 55.56% |
| **Senior (60+)** | 2,444 | 20.4% | 0.49% | 3.60% | **3.29%** | 66.67% |

- **What this shows**: the audit pipeline detects a subgroup disparity when one has been planted (11.75% vs 3.29% FPR). It does not show that any real system is biased toward youth.
- **Methodological Stance**: nothing here claims a model is bias-free; real fairness conclusions need real data.

---

### S3. Tabular-Benchmark Protocol Check (IEEE-CIS-style, **SYNTHETIC data - not the real IEEE-CIS**)
> **Read this first:** `evaluate_ieee_cis.py` generates its own data from a hand-written formula that was tuned to land near ROC-AUC 0.92 / PR-AUC 0.41 (see the comment in `generate_ieee_cis_dataset`). The numbers below therefore reflect the generator, **not model quality, and must not be compared with published IEEE-CIS results.**

- **ROC-AUC**: **0.9213**
- **Average Precision (PR-AUC)**: **0.4003**
- **Precision @ 2% Alert Budget**: **49.77%** (Recall: **27.20%**)
- **Validation-Selected Alert Cutoff**: score >= **0.3407**
- **Top Predictive Features**: `ProductCD_C`, `card_credit`, `C1`, `TransactionAmt`, `C13`, `dist1`, `D1`.

---


---

## 4. Repository Structure

```
├── README.md                           # Comprehensive documentation & report
├── requirements.txt                    # Project dependencies
├── data/
│   ├── raw/HI-Small_Trans.csv          # Raw IBM AML dataset (NOT in the repo; optional, see docs/DATA.md)
│   ├── upi_holdout/                    # Independent UPI simulation (git-ignored; regenerate, seed 7)
│   ├── verified_merchants.json         # Verified-merchant registry used by the consumer copilot
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
│   ├── evaluate_amlsim_challenges.py   # Causal GraphSAGE/XGBoost training, alert-policy calibration, challenge evaluation
│   ├── build_verified_merchant_registry.py  # Builds data/verified_merchants.json
│   ├── run_multiseed_ablation.py       # 5-seed ablation (feature sets, blend vs XGBoost)
│   ├── generate_upi_simulation.py      # Independent held-out UPI simulator
│   ├── evaluate_upi_holdout.py         # Zero-shot evaluation on that simulator
│   ├── evaluate_explanations.py        # GNNExplainer fidelity test
│   ├── explainability.py               # GNNExplainer & TreeSHAP explanation module
│   ├── evaluate_fairness_baf.py        # Bank Account Fraud fairness evaluator
│   ├── evaluate_ieee_cis.py            # IEEE-CIS tabular benchmark experiment
│   ├── copilots.py                     # Bank Analyst SAR & Multilingual Consumer Copilots
│   ├── scoring_service.py              # Real-Time Scoring API & Web Application
│   └── benchmark_latency.py            # Automated API latency benchmark suite
├── tests/
│   ├── test_system_components.py       # 23 integration / regression tests
│   └── test_dashboard_e2e.py           # Headless-browser dashboard smoke test (Playwright)
├── docs/DATA.md                        # Data provenance and optional downloads
├── MODEL_CARD.md                       # Intended use, results, failure modes
├── reproduce_all.ps1 / reproduce_all.sh  # One command to rebuild every UPI result
├── requirements-dev.txt                # Playwright (browser test only)
└── .github/workflows/tests.yml         # CI (tests + browser smoke test)
```

---

## 5. Quickstart & Execution Guide

### 0. Setup (Python 3.12 tested)
```bash
python -m venv .venv
# Linux/macOS:  .venv/bin/python -m pip install -r requirements.txt
# Windows:      .venv\Scripts\python -m pip install -r requirements.txt
```
The commands below use the Linux/macOS form. **On Windows (PowerShell)** set the environment once and use `.venv\Scripts\python`:
```powershell
$env:PYTHONPATH = "src"; $env:PYTHONUTF8 = "1"
.venv\Scripts\python tests\test_system_components.py
```
The scoring API, tests and every UPI result need only the committed AMLSim data/models (the held-out simulation is regenerated by script). Supplementary section S1 (IBM AML, non-UPI) needs `data/raw/HI-Small_Trans.csv`, which is not included; see [docs/DATA.md](docs/DATA.md).

### 1. Run Complete Test Suite
Verify that all feature stores, models, copilots, and benchmark reports are verified:
```bash
PYTHONPATH=src .venv/bin/python tests/test_system_components.py
```

### 2. Re-run AMLSim Challenge Scenarios
```bash
PYTHONPATH=src .venv/bin/python src/prepare_amlsim_challenges.py
PYTHONPATH=src .venv/bin/python src/evaluate_amlsim_challenges.py                      # deployed model
PYTHONPATH=src .venv/bin/python src/evaluate_amlsim_challenges.py --feature-set windowed_nopair --output-dir /tmp/r --models-dir /tmp/m   # variant
PYTHONPATH=src .venv/bin/python src/run_multiseed_ablation.py --seeds 42 43 44 45 46
PYTHONPATH=src .venv/bin/python src/generate_upi_simulation.py && PYTHONPATH=src .venv/bin/python src/evaluate_upi_holdout.py
```
Or everything at once: `./reproduce_all.ps1` (Windows) / `./reproduce_all.sh`.

### 3. Re-run Explainability Audit (GNNExplainer + TreeSHAP)
```bash
PYTHONPATH=src .venv/bin/python src/explainability.py
```

### 4. (Supplementary, non-UPI) Re-run BAF-style Fairness Audit & IEEE-CIS-style Benchmark
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
# optional: require an API key  (Windows: $env:UPI_API_KEY = "change-me")
UPI_API_KEY=change-me PYTHONPATH=src .venv/bin/python src/scoring_service.py --port 8000
```
Open **`http://localhost:8000`** in any web browser to access:
- **⚡ Real-Time Live Scoring Feed**: Step-by-step or continuous streaming playback with real-time risk gauges.
- **🧪 3 Challenge Scenarios**: Interactive comparison table and breakdown of disguised, lookalike hub, and partial history tests.
- **🔍 GNN & SHAP Explainability**: Visual topological subgraphs and TreeSHAP waterfall charts.
- **💼 Bank Analyst Copilot**: One-click SAR generation with editable narrative and export.
- **🛡️ Consumer Safety Copilot**: Multilingual UPI ID safety search (English, Hindi, Telugu, Tamil, Marathi, Bengali).
- **📊 Model & Fairness Benchmarks**: IBM AML, BAF-style and IEEE-CIS-style analytics (supplementary, non-UPI).


---

## 6. Known Limitations

Read these before relying on any number above. The full list of failure modes is in [MODEL_CARD.md](MODEL_CARD.md).

- **Simulated data only.** The served model is trained and tested on AMLSim output, and the stress tests and the held-out simulator are generated by this project's own scripts. Nothing has been validated on real UPI traffic, and no public UPI dataset exists.
- **No transfer across traffic mixes (measured).** Zero-shot on the independent UPI simulation the model scores below random guessing (section B). It must be trained on the deployment network's own history.
- **Recurring legitimate payments escalate.** A daily payment to the same payee alerts from about the 6th repeat (repeat-pair features). The `windowed_nopair` variant avoids this but loses Challenge 1 detection power (section A2).
- **Small fan-ins are missed.** A 60-sender collection account did not alert (risk 0.09); the training data only contains a fan-in of 200.
- **Fresh mule accounts look unremarkable.** Accounts with no history are flagged `insufficient_history` and scored low.
- **Benign hubs.** 26% of benign hub events are alerted in the lookalike scenario. Known merchants are exempted from the consumer warning only if listed in `data/verified_merchants.json`, which is seeded from the simulation here.
- **Explanations are not shown to be faithful** (section C). Treat GNNExplainer edges as leads, not evidence.
- **The rolling alert budget lags fast drift.** Measured 5.0-5.2% on the simulation, but a synthetic sweep showed about 11% at 3x and 22% at 10x the simulation's drift. A genuine surge of laundering also cannot raise the alert count.
- **Recipient advisory is a modest signal** (ROC-AUC 0.71) and its translations (6 languages) have not been reviewed by native speakers.
- **Single deployed seed.** The headline deployed numbers (seed 42) are slightly above the 5-seed means; the means are the fairer summary. Bootstrap intervals cover test-set sampling, not training variance.
- **Small train/serve difference.** Offline evaluation scores all transactions of a time step against the graph of previous steps; the live service also sees earlier transactions from the same step.
- **Scale.** Unseen accounts get their own graph node up to a cap of 100,000, after which they share one slot. The graph keeps every committed edge. Hubs with huge in-degree are slower. Not load-tested beyond about 120K edges.
- **Supplementary non-UPI sections** (IBM AML, BAF-style, IEEE-CIS-style): the IBM models are not the served model and degrade out-of-time (validation AP 0.32 vs test 0.20); the BAF-style and IEEE-CIS-style results are protocol checks on generated data, not benchmarks. Their training pipeline was not re-audited for causality.
- **Not done:** a UPI-realistic dataset from an industry partner, independent review of the simulator assumptions, multi-seed runs of the supplementary sections, and native-speaker review.
