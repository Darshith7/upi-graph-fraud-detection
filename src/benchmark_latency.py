"""Benchmark API Real-Time Scoring Latency.

Evaluates Section 10 performance requirements:
- Benchmarks scoring latency on 1,000+ sequential incoming transactions.
- Instruments and empirically measures individual pipeline stages (feature extraction, XGBoost, GraphSAGE, graph state update).
- Computes empirical Median (p50), 95th Percentile (p95), and 99th Percentile (p99) response times.
- Dynamically computes compliance verdicts against sub-200 ms SLA targets.
- Outputs reproducible JSON and Markdown benchmark reports in reports/api_benchmark/.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from evaluate_amlsim_challenges import FEATURE_NAMES
from scoring_service import ENGINE


def run_latency_benchmark(num_transactions: int = 1000, output_dir: Path = Path("reports/api_benchmark")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running real-time scoring latency benchmark on {num_transactions:,} transactions...")
    ENGINE.reset_state()
    ENGINE.load_scenario("baseline")

    total_latencies = []
    stage_feat_times = []
    stage_xgb_times = []
    stage_gnn_times = []
    stage_update_times = []
    alerts = 0
    df = ENGINE.stream_df

    t0_benchmark = time.perf_counter()
    for i in range(num_transactions):
        row = df.iloc[i % len(df)]
        s_upi = str(row["sender_upi"])
        d_upi = str(row["receiver_upi"])
        amt = float(row["amount"])
        t_sec = int(row["timestamp_seconds"])

        s_node = ENGINE.parse_node_id(s_upi)
        d_node = ENGINE.parse_node_id(d_upi)

        # Stage 1: Feature Extraction
        t0 = time.perf_counter()
        features, insufficient_history = ENGINE.tracker.extract_features(s_node, d_node, amt, t_sec)
        t_feat = (time.perf_counter() - t0) * 1000

        # Stage 2: XGBoost Inference
        t0 = time.perf_counter()
        dmatrix = xgb.DMatrix(np.atleast_2d(features), feature_names=ENGINE.feature_names, nthread=1)
        xgb_score = float(ENGINE.bst.predict(dmatrix)[0])
        t_xgb = (time.perf_counter() - t0) * 1000

        # Stage 3: GraphSAGE Convolution
        t0 = time.perf_counter()
        s_idx = ENGINE.node2idx.get(s_node, ENGINE.unknown_node_idx)
        d_idx = ENGINE.node2idx.get(d_node, ENGINE.unknown_node_idx)
        tx_feat = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            s_t = torch.tensor([s_idx], dtype=torch.long)
            d_t = torch.tensor([d_idx], dtype=torch.long)
            # same path as the live service: 2-hop in-neighbourhood subgraph (identical logits, local cost)
            sub_x, sub_ei, (s_loc, d_loc) = ENGINE.graph.khop_inputs([s_idx, d_idx], hops=2)
            gnn_logit = ENGINE.gnn_model(
                sub_x, sub_ei, tx_feat, torch.tensor([s_loc], dtype=torch.long), torch.tensor([d_loc], dtype=torch.long)
            )
            gnn_score = float(torch.sigmoid(gnn_logit).item())
        t_gnn = (time.perf_counter() - t0) * 1000

        # Blended risk
        risk_score = round(ENGINE.w_xgb * xgb_score + ENGINE.w_gnn * gnn_score, 4)
        is_alert = risk_score >= ENGINE.alert_threshold
        if is_alert:
            alerts += 1

        # Stage 4: Graph Mutation strictly after scoring
        t0 = time.perf_counter()
        ENGINE.tracker.update(s_node, d_node, t_sec, amount=amt)
        ENGINE.graph.commit(torch.tensor([s_idx]), torch.tensor([d_idx]), torch.from_numpy(features).unsqueeze(0))
        t_update = (time.perf_counter() - t0) * 1000

        t_total = t_feat + t_xgb + t_gnn + t_update
        total_latencies.append(t_total)
        stage_feat_times.append(t_feat)
        stage_xgb_times.append(t_xgb)
        stage_gnn_times.append(t_gnn)
        stage_update_times.append(t_update)

    total_wall_time_s = time.perf_counter() - t0_benchmark
    arr = np.array(total_latencies)

    # End-to-end pass through the REAL service code path (features + XGBoost + TreeSHAP
    # explanation + GraphSAGE + evidence/result building + state commit). The staged numbers
    # above leave out the explanation/result building, so they understate API latency.
    ENGINE.load_scenario("baseline")
    e2e_n = min(num_transactions, 500)
    e2e_lat = []
    for i in range(e2e_n):
        row = ENGINE.stream_df.iloc[i % len(ENGINE.stream_df)]
        payload = {
            "transaction_id": int(row["transaction_id"]),
            "sender_upi": str(row["sender_upi"]), "receiver_upi": str(row["receiver_upi"]),
            "amount": float(row["amount"]), "timestamp": int(row["timestamp_seconds"]),
        }
        t0 = time.perf_counter()
        ENGINE.score_transaction(payload)
        e2e_lat.append((time.perf_counter() - t0) * 1000)
    e2e = np.array(e2e_lat)
    e2e_stats = {
        "transactions": e2e_n,
        "mean": round(float(np.mean(e2e)), 2),
        "median_p50": round(float(np.median(e2e)), 2),
        "p95": round(float(np.percentile(e2e, 95)), 2),
        "p99": round(float(np.percentile(e2e, 99)), 2),
        "max": round(float(np.max(e2e)), 2),
        "throughput_tx_per_second": round(float(1000.0 / np.mean(e2e)), 1),
    }

    mean_lat = float(np.mean(arr))
    median_lat = float(np.median(arr))
    p95_lat = float(np.percentile(arr, 95))
    p99_lat = float(np.percentile(arr, 99))
    min_lat = float(np.min(arr))
    max_lat = float(np.max(arr))
    sub_200ms_rate = float(np.mean(arr < 200.0))
    throughput_tps = float(num_transactions / total_wall_time_s)

    status_p50 = "PASS" if median_lat < 50.0 else "FAIL"
    status_p95 = "PASS" if p95_lat < 200.0 else "FAIL"
    status_p99 = "PASS" if p99_lat < 300.0 else "FAIL"
    overall_pass = float(np.percentile(e2e, 95)) < 200.0

    report_data = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_transactions_tested": num_transactions,
        "total_alerts_dispatched": alerts,
        "total_wall_time_seconds": round(total_wall_time_s, 3),
        "throughput_tx_per_second": round(throughput_tps, 1),
        "pipeline_stages_measured_ms": {
            "feature_extraction_mean": round(float(np.mean(stage_feat_times)), 3),
            "xgboost_inference_mean": round(float(np.mean(stage_xgb_times)), 3),
            "graphsage_convolution_mean": round(float(np.mean(stage_gnn_times)), 3),
            "graph_mutation_mean": round(float(np.mean(stage_update_times)), 3)
        },
        "end_to_end_score_transaction_ms": e2e_stats,
        "latency_metrics_ms": {
            "mean": round(mean_lat, 2),
            "median_p50": round(median_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "target_evaluation": {
            "target_threshold_ms": 200.0,
            "compliance_percentage": round(sub_200ms_rate * 100, 2),
            "target_achieved": overall_pass,
            "basis": "end_to_end_score_transaction_ms.p95",
            "verdict": f"{'PASS' if overall_pass else 'FAIL'}: end-to-end p50 {e2e_stats['median_p50']:.2f} ms, p95 {e2e_stats['p95']:.2f} ms vs 200 ms target."
        }
    }

    # Save JSON
    json_path = output_dir / "latency_benchmark.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Saved benchmark JSON to {json_path}")

    # Save Markdown
    md_lines = [
        "# Real-Time Risk Scoring API Latency Benchmark Report",
        "",
        "## 1. Executive Summary",
        f"- **End-to-end API path (`score_transaction`, incl. SHAP explanation), {e2e_stats['transactions']} tx**: p50 **{e2e_stats['median_p50']:.2f} ms**, p95 **{e2e_stats['p95']:.2f} ms**, p99 **{e2e_stats['p99']:.2f} ms**, ~{e2e_stats['throughput_tx_per_second']:.0f} tx/s. The verdict below is based on this.",
        "- The staged numbers in sections 2-3 time only feature extraction, XGBoost, GraphSAGE and the graph update (no explanation/result building), so they are a lower bound, not API latency.",
        f"- **Tested Transactions (staged)**: {num_transactions:,} incoming sequential events.",
        f"- **Throughput**: **{throughput_tps:,.1f} transactions / second**.",
        f"- **Median Latency (p50)**: **{median_lat:.2f} ms**.",
        f"- **95th Percentile Latency (p95)**: **{p95_lat:.2f} ms**.",
        f"- **Sub-200ms Target Compliance**: **{sub_200ms_rate * 100:.2f}%**.",
        f"- **Status**: **{report_data['target_evaluation']['verdict']}**",
        "",
        "## 2. Latency Distribution Breakdown",
        "| Metric | Latency (ms) | Operational Target | Status |",
        "|---|---|---|---|",
        f"| **Median (p50)** | `{median_lat:.2f} ms` | &lt; 50 ms | {status_p50} |",
        f"| **95th Percentile (p95)** | `{p95_lat:.2f} ms` | &lt; 200 ms | {status_p95} |",
        f"| **99th Percentile (p99)** | `{p99_lat:.2f} ms` | &lt; 300 ms | {status_p99} |",
        f"| **Minimum** | `{min_lat:.2f} ms` | - | - |",
        f"| **Maximum** | `{max_lat:.2f} ms` | - | - |",
        f"| **Mean** | `{mean_lat:.2f} ms` | - | - |",
        "",
        "## 3. Pipeline Stages & Architectural Optimization (Measured Empirical Timings)",
        f"1. **Online Past-Only Feature Extraction**: Incremental updates in Python array buffers avoid full graph traversals at scoring time (Measured Mean: **{np.mean(stage_feat_times):.3f} ms**).",
        f"2. **XGBoost Tree Prediction**: Evaluation of 120 shallow depth-5 trees on tabular features (Measured Mean: **{np.mean(stage_xgb_times):.3f} ms**).",
        f"3. **GraphSAGE 2-Hop Embedding & Classifier Head**: Directional neighborhood aggregation over evolving graph topology (Measured Mean: **{np.mean(stage_gnn_times):.3f} ms**).",
        f"4. **Graph State Mutation**: Committing the transfer to the dynamic temporal graph occurs strictly *after* risk prediction to guarantee zero future data leakage (Measured Mean: **{np.mean(stage_update_times):.3f} ms**)."
    ]

    md_path = output_dir / "latency_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved benchmark report to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark API latency")
    parser.add_argument("--count", type=int, default=1000, help="Number of transactions to benchmark")
    args = parser.parse_args()
    run_latency_benchmark(args.count)
