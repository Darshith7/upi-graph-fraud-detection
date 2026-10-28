# Real-Time Risk Scoring API Latency Benchmark Report

## 1. Executive Summary
- **Tested Transactions**: 1,000 incoming sequential events.
- **Throughput**: **266.9 transactions / second**.
- **Median Latency (p50)**: **3.49 ms**.
- **95th Percentile Latency (p95)**: **4.08 ms**.
- **Sub-200ms Target Compliance**: **100.00%**.
- **Status**: **PASS: Median and p95 latency are well below the 200 ms target.**

## 2. Latency Distribution Breakdown
| Metric | Latency (ms) | Operational Target | Status |
|---|---|---|---|
| **Median (p50)** | `3.49 ms` | &lt; 50 ms | PASS |
| **95th Percentile (p95)** | `4.08 ms` | &lt; 200 ms | PASS |
| **99th Percentile (p99)** | `4.54 ms` | &lt; 300 ms | PASS |
| **Minimum** | `3.15 ms` | - | - |
| **Maximum** | `82.04 ms` | - | - |
| **Mean** | `3.67 ms` | - | - |

## 3. Pipeline Stages & Architectural Optimization
1. **Online Past-Only Feature Extraction**: Incremental updates in Python array buffers avoid full graph traversals at scoring time (~0.12 ms).
2. **Dual Model Scoring**: XGBoost Tree prediction (~0.15 ms) and directional GraphSAGE convolution over local 1-hop subgraph (~0.85 ms).
3. **Graph State Mutation**: Committing the transfer to the temporal graph occurs strictly *after* risk prediction to guarantee zero future data leakage.