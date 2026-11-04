# Real-Time Risk Scoring API Latency Benchmark Report

## 1. Executive Summary
- **Tested Transactions**: 1,000 incoming sequential events.
- **Throughput**: **87.8 transactions / second**.
- **Median Latency (p50)**: **10.99 ms**.
- **95th Percentile Latency (p95)**: **12.02 ms**.
- **Sub-200ms Target Compliance**: **100.00%**.
- **Status**: **PASS: Median (10.99 ms) and p95 (12.02 ms) latency conform to SLA.**

## 2. Latency Distribution Breakdown
| Metric | Latency (ms) | Operational Target | Status |
|---|---|---|---|
| **Median (p50)** | `10.99 ms` | &lt; 50 ms | PASS |
| **95th Percentile (p95)** | `12.02 ms` | &lt; 200 ms | PASS |
| **99th Percentile (p99)** | `13.21 ms` | &lt; 300 ms | PASS |
| **Minimum** | `10.25 ms` | - | - |
| **Maximum** | `72.62 ms` | - | - |
| **Mean** | `11.29 ms` | - | - |

## 3. Pipeline Stages & Architectural Optimization (Measured Empirical Timings)
1. **Online Past-Only Feature Extraction**: Incremental updates in Python array buffers avoid full graph traversals at scoring time (Measured Mean: **0.012 ms**).
2. **XGBoost Tree Prediction**: Evaluation of 120 shallow depth-5 trees on tabular features (Measured Mean: **0.198 ms**).
3. **GraphSAGE 2-Hop Embedding & Classifier Head**: Directional neighborhood aggregation over evolving graph topology (Measured Mean: **5.171 ms**).
4. **Graph State Mutation**: Committing the transfer to the dynamic temporal graph occurs strictly *after* risk prediction to guarantee zero future data leakage (Measured Mean: **5.907 ms**).