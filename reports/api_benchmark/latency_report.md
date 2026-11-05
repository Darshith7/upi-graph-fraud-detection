# Real-Time Risk Scoring API Latency Benchmark Report

## 1. Executive Summary
- **End-to-end API path (`score_transaction`, incl. SHAP explanation), 500 tx**: p50 **2.34 ms**, p95 **3.72 ms**, p99 **4.60 ms**, ~394 tx/s. The verdict below is based on this.
- The staged numbers in sections 2-3 time only feature extraction, XGBoost, GraphSAGE and the graph update (no explanation/result building), so they are a lower bound, not API latency.
- **Tested Transactions (staged)**: 1,000 incoming sequential events.
- **Throughput**: **552.6 transactions / second**.
- **Median Latency (p50)**: **1.33 ms**.
- **95th Percentile Latency (p95)**: **2.58 ms**.
- **Sub-200ms Target Compliance**: **99.90%**.
- **Status**: **PASS: end-to-end p50 2.34 ms, p95 3.72 ms vs 200 ms target.**

## 2. Latency Distribution Breakdown
| Metric | Latency (ms) | Operational Target | Status |
|---|---|---|---|
| **Median (p50)** | `1.33 ms` | &lt; 50 ms | PASS |
| **95th Percentile (p95)** | `2.58 ms` | &lt; 200 ms | PASS |
| **99th Percentile (p99)** | `3.52 ms` | &lt; 300 ms | PASS |
| **Minimum** | `1.04 ms` | - | - |
| **Maximum** | `229.10 ms` | - | - |
| **Mean** | `1.71 ms` | - | - |

## 3. Pipeline Stages & Architectural Optimization (Measured Empirical Timings)
1. **Online Past-Only Feature Extraction**: Incremental updates in Python array buffers avoid full graph traversals at scoring time (Measured Mean: **0.037 ms**).
2. **XGBoost Tree Prediction**: Evaluation of 120 shallow depth-5 trees on tabular features (Measured Mean: **0.467 ms**).
3. **GraphSAGE 2-Hop Embedding & Classifier Head**: Directional neighborhood aggregation over evolving graph topology (Measured Mean: **1.031 ms**).
4. **Graph State Mutation**: Committing the transfer to the dynamic temporal graph occurs strictly *after* risk prediction to guarantee zero future data leakage (Measured Mean: **0.171 ms**).