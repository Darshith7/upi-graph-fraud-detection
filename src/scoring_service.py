"""Real-Time Risk-Scoring API & Simulated Stream Replayer Service.

Implements Section 10 and the complete interactive demonstration:
- Asynchronous high-throughput REST API.
- Online past-only temporal feature calculation (zero future leakage: score first, commit to graph second).
- Returns: risk_score, risk_level, main_reasons, graph_history_cutoff, insufficient_history.
- Replays simulated streams from AMLSim scenarios.
- Measures latency (median, p95, sub-200ms target).
- Integrated Bank Analyst and Consumer Safety Copilots.
- Serves the Interactive Real-Time Web Dashboard at http://localhost:8000.
"""

from array import array
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, Any, List

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiohttp
from aiohttp import web
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import xgboost as xgb

from evaluate_amlsim_challenges import (
    FEATURE_NAMES, TemporalGraphFeatureTracker, AMLSimGraphSAGE
)
from explainability import TreeSHAPExplainer, GNNExplainer, generate_explanation_package
from copilots import BankAnalystCopilot, ConsumerSafetyCopilot


class RealtimeScoringEngine:
    """Stateful engine maintaining live temporal graph index and scoring models."""

    def __init__(self, models_dir: Path, challenges_dir: Path):
        self.models_dir = Path(models_dir)
        self.challenges_dir = Path(challenges_dir)
        self.tracker = TemporalGraphFeatureTracker()
        self.latencies_ms = deque(maxlen=10_000)
        self.recent_scored = deque(maxlen=100)
        self.total_scored = 0
        self.total_alerts = 0

        # Load XGBoost model
        self.bst = xgb.Booster()
        self.bst.load_model(str(self.models_dir / "xgboost_amlsim.ubj"))
        self.shap_explainer = TreeSHAPExplainer(
            str(self.models_dir / "xgboost_amlsim.ubj"), FEATURE_NAMES
        )

        # Load GraphSAGE model
        gnn_ckpt = torch.load(self.models_dir / "graphsage_amlsim.pt", weights_only=False)
        self.gnn_model = AMLSimGraphSAGE(in_features=len(FEATURE_NAMES), hidden_dim=32, out_dim=1)
        self.gnn_model.load_state_dict(gnn_ckpt["state_dict"])
        self.gnn_model.eval()
        self.node_x = gnn_ckpt["node_x"]
        self.edge_index = gnn_ckpt["edge_index"]
        self.node2idx = gnn_ckpt["node2idx"]
        self.gnn_explainer = GNNExplainer(self.gnn_model, FEATURE_NAMES)

        # Cache scenario streams
        self.scenarios = {
            "baseline": "baseline.csv",
            "challenge_1_disguised": "challenge_1_disguised.csv",
            "challenge_2_lookalike": "challenge_2_lookalike.csv",
            "challenge_3_partial_history": "challenge_3_partial_history.csv"
        }
        self.active_scenario_name = "baseline"
        self.stream_df = None
        self.stream_pointer = 0
        self.load_scenario("baseline")

    def load_scenario(self, scenario_name: str):
        if scenario_name not in self.scenarios:
            scenario_name = "baseline"
        self.active_scenario_name = scenario_name
        csv_file = self.challenges_dir / self.scenarios[scenario_name]
        self.stream_df = pd.read_csv(csv_file)
        self.stream_pointer = 0
        print(f"Loaded scenario '{scenario_name}': {len(self.stream_df):,} transactions.")

    def reset_state(self):
        self.tracker = TemporalGraphFeatureTracker()
        self.latencies_ms.clear()
        self.recent_scored.clear()
        self.total_scored = 0
        self.total_alerts = 0
        self.stream_pointer = 0

    def parse_node_id(self, upi_handle: str) -> int:
        """Extract numeric account index from handle or create hash ID."""
        try:
            if "acc_" in upi_handle:
                part = upi_handle.split("acc_")[1].split("@")[0]
                return int(part)
        except Exception:
            pass
        return abs(hash(upi_handle)) % 20_000

    def score_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """Score incoming transaction with past-only history, then update graph."""
        start_t = time.perf_counter()

        tx_id = tx_data.get("transaction_id") if tx_data.get("transaction_id") is not None else int(time.time() * 1000) % 1_000_000
        sender_upi = str(tx_data.get("sender_upi") or "unknown@upi")
        receiver_upi = str(tx_data.get("receiver_upi") or "unknown@upi")
        try:
            amount = max(0.01, float(tx_data.get("amount", 100.0)))
        except (ValueError, TypeError):
            amount = 100.0
        try:
            timestamp = int(tx_data.get("timestamp", int(time.time())))
        except (ValueError, TypeError):
            timestamp = int(time.time())
        graph_cutoff = timestamp

        s_node = self.parse_node_id(sender_upi)
        d_node = self.parse_node_id(receiver_upi)

        # 1. Past-only feature extraction BEFORE graph mutation
        features, insufficient_history = self.tracker.extract_features(
            s_node, d_node, amount, timestamp
        )

        # 2. XGBoost prediction & TreeSHAP
        dmatrix = xgb.DMatrix(np.atleast_2d(features), feature_names=FEATURE_NAMES, nthread=1)
        xgb_score = float(self.bst.predict(dmatrix)[0])
        shap_res = self.shap_explainer.explain(features, top_k=3)

        # 3. GraphSAGE prediction
        s_idx = self.node2idx.get(s_node, 0)
        d_idx = self.node2idx.get(d_node, 0)
        tx_feat = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            s_t = torch.tensor([s_idx], dtype=torch.long)
            d_t = torch.tensor([d_idx], dtype=torch.long)
            gnn_logit = self.gnn_model(
                self.node_x, self.edge_index[:, :min(self.edge_index.size(1), 50_000)],
                tx_feat, s_t, d_t
            )
            gnn_score = float(torch.sigmoid(gnn_logit).item())

        # Blended risk score (GNN + XGBoost)
        risk_score = round(0.55 * xgb_score + 0.45 * gnn_score, 4)

        if risk_score >= 0.70:
            risk_level = "HIGH"
            alert = True
        elif risk_score >= 0.35:
            risk_level = "MEDIUM"
            alert = False
        else:
            risk_level = "LOW"
            alert = False

        # Build explainable reasons
        reasons = []
        if insufficient_history:
            reasons.append("Sparse prior history: Counterparty has fewer than 3 historical events (cold-start / newly registered handle).")
        for f in shap_res["top_features"]:
            reasons.append(f"{f['feature']} = {f['feature_value']} ({f['direction']}, SHAP: {f['shap_attribution']:+0.2f})")
        if self.tracker.in_counts[d_node] >= 5 and self.tracker.out_counts[d_node] >= 2:
            reasons.append(f"Network flow: Receiver has accumulated {self.tracker.in_counts[d_node]} inflows with pass-through forwarding.")

        # 4. Commit transaction to graph AFTER scoring (no leakage)
        self.tracker.update(s_node, d_node, timestamp)

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        self.latencies_ms.append(latency_ms)
        self.total_scored += 1
        if alert:
            self.total_alerts += 1

        result = {
            "transaction_id": tx_id,
            "sender_upi": sender_upi,
            "receiver_upi": receiver_upi,
            "amount": amount,
            "timestamp": timestamp,
            "graph_history_cutoff": graph_cutoff,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "alert": alert,
            "insufficient_history": insufficient_history,
            "xgb_score": round(xgb_score, 4),
            "gnn_score": round(gnn_score, 4),
            "main_reasons": reasons,
            "top_shap_features": shap_res["top_features"],
            "latency_ms": latency_ms
        }
        self.recent_scored.appendleft(result)
        return result

    def get_latency_stats(self) -> Dict[str, Any]:
        if len(self.latencies_ms) == 0:
            return {"count": 0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "sub_200ms_compliance": 1.0}
        arr = np.array(self.latencies_ms)
        return {
            "count": len(arr),
            "mean_ms": round(float(np.mean(arr)), 2),
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
            "sub_200ms_compliance": round(float(np.mean(arr < 200.0)), 4)
        }


# Global engine instance
ENGINE = RealtimeScoringEngine(
    models_dir=Path("models/amlsim_models"),
    challenges_dir=Path("data/amlsim_challenges")
)


# =========================================================================
# Web API Handlers
# =========================================================================

async def handle_score(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    result = ENGINE.score_transaction(data)
    return web.json_response(result)


async def handle_stream_step(request: web.Request) -> web.Response:
    try:
        data = await request.json() if request.can_read_body else {}
        count = int(data.get("count", 1))
    except Exception:
        count = 1

    scored_batch = []
    for _ in range(count):
        if ENGINE.stream_pointer >= len(ENGINE.stream_df):
            ENGINE.stream_pointer = 0
        row = ENGINE.stream_df.iloc[ENGINE.stream_pointer]
        ENGINE.stream_pointer += 1

        tx_payload = {
            "transaction_id": int(row["transaction_id"]),
            "sender_upi": str(row["sender_upi"]),
            "receiver_upi": str(row["receiver_upi"]),
            "amount": float(row["amount"]),
            "timestamp": int(row["timestamp_seconds"]),
            "is_ground_truth_laundering": int(row.get("is_laundering", 0))
        }
        res = ENGINE.score_transaction(tx_payload)
        res["is_ground_truth_laundering"] = tx_payload["is_ground_truth_laundering"]
        scored_batch.append(res)

    return web.json_response({
        "status": "success",
        "scored_count": len(scored_batch),
        "stream_pointer": ENGINE.stream_pointer,
        "total_stream_rows": len(ENGINE.stream_df),
        "active_scenario": ENGINE.active_scenario_name,
        "recent_batch": scored_batch
    })


async def handle_stream_reset(request: web.Request) -> web.Response:
    try:
        data = await request.json() if request.can_read_body else {}
        scenario = data.get("scenario", "baseline")
    except Exception:
        scenario = "baseline"

    ENGINE.reset_state()
    ENGINE.load_scenario(scenario)
    return web.json_response({
        "status": "reset",
        "scenario": scenario,
        "total_rows": len(ENGINE.stream_df)
    })


async def handle_stream_state(request: web.Request) -> web.Response:
    return web.json_response({
        "active_scenario": ENGINE.active_scenario_name,
        "stream_pointer": ENGINE.stream_pointer,
        "total_stream_rows": len(ENGINE.stream_df),
        "total_scored": ENGINE.total_scored,
        "total_alerts": ENGINE.total_alerts,
        "alert_rate": round(ENGINE.total_alerts / max(1, ENGINE.total_scored), 4),
        "latency_stats": ENGINE.get_latency_stats(),
        "recent_transactions": list(ENGINE.recent_scored)[:20]
    })


async def handle_metrics_latency(request: web.Request) -> web.Response:
    return web.json_response(ENGINE.get_latency_stats())


async def handle_copilot_analyst(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        data = {}

    tx_id = data.get("transaction_id")
    # If transaction_id provided, look up in recent or mock
    case_data = None
    if tx_id:
        for item in ENGINE.recent_scored:
            if item["transaction_id"] == tx_id:
                case_data = item
                break
    if case_data is None:
        case_data = data  # Use provided payload

    sar = BankAnalystCopilot.generate_sar_summary(case_data)
    return web.json_response(sar)


async def handle_copilot_consumer(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        data = {}

    upi_id = data.get("recipient_upi", "unknown@upi")
    lang = data.get("language", "en")

    # Look up real stats for this node in the engine
    node_id = ENGINE.parse_node_id(upi_id)
    in_cnt = ENGINE.tracker.in_counts[node_id]
    out_cnt = ENGINE.tracker.out_counts[node_id]
    total_tx = in_cnt + out_cnt

    # Derive recent risk score
    if total_tx == 0:
        recent_score = 0.05
    else:
        recent_score = 0.85 if (in_cnt > 8 and out_cnt > 2) else 0.08

    advisory = ConsumerSafetyCopilot.check_recipient_safety(
        recipient_upi=upi_id,
        recent_risk_score=recent_score,
        total_prior_events=total_tx,
        language=lang
    )
    return web.json_response(advisory)


async def handle_subgraph(request: web.Request) -> web.Response:
    upi_id = request.query.get("upi_id", "")
    node_id = ENGINE.parse_node_id(upi_id) if upi_id else 8676

    nodes = [{"id": f"acc_{node_id:05d}", "label": upi_id or f"acc_{node_id:05d}@upi", "role": "target"}]
    edges = []

    # Get neighbors
    for src in list(ENGINE.tracker.in_neighbors[node_id])[:10]:
        nodes.append({"id": f"acc_{src:05d}", "label": f"acc_{src:05d}@bank", "role": "inbound"})
        edges.append({"source": f"acc_{src:05d}", "target": f"acc_{node_id:05d}", "type": "inflow"})

    for dst in list(ENGINE.tracker.out_neighbors[node_id])[:10]:
        nodes.append({"id": f"acc_{dst:05d}", "label": f"acc_{dst:05d}@bank", "role": "outbound"})
        edges.append({"source": f"acc_{node_id:05d}", "target": f"acc_{dst:05d}", "type": "outflow"})

    return web.json_response({"nodes": nodes, "edges": edges})


async def handle_benchmark_summary(request: web.Request) -> web.Response:
    """Return consolidated metrics across all experimental modules."""
    ibm_test_file = Path("reports/test_evaluation_v1/test_metrics.json")
    amlsim_file = Path("reports/amlsim_challenges/challenge_metrics.json")
    fairness_file = Path("reports/fairness_baf/fairness_metrics.json")
    ieee_file = Path("reports/ieee_cis_benchmark/ieee_cis_metrics.json")

    summary = {
        "ibm_aml": json.loads(ibm_test_file.read_text()) if ibm_test_file.is_file() else {},
        "amlsim_challenges": json.loads(amlsim_file.read_text()) if amlsim_file.is_file() else {},
        "fairness_baf": json.loads(fairness_file.read_text()) if fairness_file.is_file() else {},
        "ieee_cis": json.loads(ieee_file.read_text()) if ieee_file.is_file() else {},
        "api_latency": ENGINE.get_latency_stats()
    }
    return web.json_response(summary)


async def handle_explainability_cases(request: web.Request) -> web.Response:
    cases_file = Path("reports/explainability/case_studies.json")
    if cases_file.is_file():
        data = json.loads(cases_file.read_text(encoding="utf-8"))
        return web.json_response(data)
    return web.json_response([])


# =========================================================================
# Web Dashboard HTML
# =========================================================================

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UPI Graph Fraud & Mule Detection Platform</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --border: #1f2937;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --red: #ef4444;
      --amber: #f59e0b;
      --green: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text); display: flex; flex-direction: column; min-height: 100vh; }
    header { background: #0f172a; border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; justify-content: space-between; align-items: center; }
    .logo-box { display: flex; align-items: center; gap: 0.75rem; }
    .badge-pill { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
    nav { display: flex; gap: 0.5rem; background: var(--card-bg); padding: 0.5rem 2rem; border-bottom: 1px solid var(--border); }
    nav button { background: transparent; border: none; color: var(--text-muted); padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; cursor: pointer; transition: 0.2s; }
    nav button:hover, nav button.active { background: var(--border); color: var(--text); }
    nav button.active { color: var(--accent); }
    main { flex: 1; padding: 2rem; max-width: 1400px; margin: 0 auto; width: 100%; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; }
    .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; }
    .card-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; display: flex; justify-content: space-between; align-items: center; }
    .stat-val { font-size: 1.8rem; font-weight: 800; color: #fff; margin-top: 0.25rem; }
    .stat-lbl { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
    button.btn-primary { background: var(--accent); color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 600; cursor: pointer; }
    button.btn-primary:hover { background: var(--accent-hover); }
    button.btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border); padding: 0.6rem 1.2rem; border-radius: 6px; font-weight: 600; cursor: pointer; }
    button.btn-outline:hover { background: var(--border); }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { text-align: left; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); }
    th { color: var(--text-muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase; }
    .tag { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .tag-high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .tag-med { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .tag-low { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    input, select, textarea { background: #1e293b; border: 1px solid var(--border); color: var(--text); padding: 0.6rem 0.8rem; border-radius: 6px; width: 100%; margin-bottom: 0.8rem; font-size: 0.9rem; }
    textarea { resize: vertical; min-height: 180px; font-family: monospace; font-size: 0.85rem; line-height: 1.4; }
    .stream-feed { max-height: 400px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; }
    canvas { background: #070b13; border-radius: 8px; border: 1px solid var(--border); }
  </style>
</head>
<body>
  <header>
    <div class="logo-box">
      <h2 style="font-size: 1.25rem;">UPI MuleShield AI</h2>
      <span class="badge-pill">GNN + XGBoost + Copilots</span>
    </div>
    <div style="font-size: 0.85rem; color: var(--text-muted);">
      Team 33 | Anubola Sai Darshith • Male Pranay Yashwin • Krishna Gupta
    </div>
  </header>

  <nav>
    <button class="active" onclick="switchTab('live-stream')">⚡ Real-Time Live Scoring</button>
    <button onclick="switchTab('challenges')">🧪 3 Challenge Scenarios</button>
    <button onclick="switchTab('explainability')">🔍 GNN & SHAP Explainability</button>
    <button onclick="switchTab('bank-copilot')">💼 Bank Analyst Copilot</button>
    <button onclick="switchTab('consumer-copilot')">🛡️ Consumer Safety Copilot</button>
    <button onclick="switchTab('benchmarks')">📊 Model & Fairness Benchmarks</button>
  </nav>

  <main>
    <!-- TAB 1: LIVE STREAM & REAL-TIME SCORING -->
    <div id="live-stream" class="tab-content active">
      <div class="grid-4">
        <div class="card">
          <div class="stat-lbl">Total Scored</div>
          <div class="stat-val" id="stat-total-scored">0</div>
        </div>
        <div class="card">
          <div class="stat-lbl">Alerts Dispatched</div>
          <div class="stat-val" id="stat-total-alerts" style="color: #f87171;">0</div>
        </div>
        <div class="card">
          <div class="stat-lbl">Median Latency (p50)</div>
          <div class="stat-val" id="stat-latency-p50" style="color: #34d399;">-- ms</div>
        </div>
        <div class="card">
          <div class="stat-lbl">p95 Latency (&lt;200ms target)</div>
          <div class="stat-val" id="stat-latency-p95" style="color: #60a5fa;">-- ms</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Simulated UPI Replay Stream</span>
            <div>
              <button class="btn-primary" id="btn-stream-toggle" onclick="toggleStream()">▶ Start Stream</button>
              <button class="btn-outline" onclick="stepStream(1)">Step +1</button>
              <button class="btn-outline" onclick="stepStream(5)">Step +5</button>
              <button class="btn-outline" onclick="resetStream()">Reset</button>
            </div>
          </div>
          <label style="font-size: 0.8rem; color: var(--text-muted);">Active Scenario:</label>
          <select id="select-scenario" onchange="changeScenario()">
            <option value="baseline">Baseline AMLSim Stream (Cycles & Fan-In)</option>
            <option value="challenge_1_disguised">Challenge 1: Disguised Suspicious (Split Amounts/Timing)</option>
            <option value="challenge_2_lookalike">Challenge 2: Benign Lookalike Hubs (Merchant/Payroll)</option>
            <option value="challenge_3_partial_history">Challenge 3: Partial History (50% Masked Edges)</option>
          </select>

          <div style="margin-top: 1rem;">
            <div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">Recent Scored Transactions (Chronological Replay):</div>
            <div class="stream-feed" id="stream-feed-list">
              <div style="color: var(--text-muted); padding: 1rem;">Click 'Start Stream' or 'Step +1' to begin processing incoming UPI transactions.</div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Selected Transaction Real-Time Assessment</span>
            <span id="selected-tx-badge" class="tag tag-low">IDLE</span>
          </div>
          <div id="tx-details-box">
            <div style="color: var(--text-muted); font-size: 0.9rem; padding: 1rem;">Select any transaction in the stream to inspect risk breakdown, GNN score, and SHAP drivers.</div>
          </div>
          <div style="margin-top: 1.5rem;">
            <button class="btn-primary" onclick="escalateToAnalyst()" id="btn-escalate" style="display: none; width: 100%;">
              Escalate to Bank Analyst Copilot (Generate SAR) ➔
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: 3 CHALLENGE SCENARIOS -->
    <div id="challenges" class="tab-content">
      <div class="card">
        <div class="card-title">AMLSim 3 Controlled Challenge Cases Benchmark</div>
        <p style="color: var(--text-muted); margin-bottom: 1.5rem; font-size: 0.95rem;">
          To prevent overly optimistic synthetic scores, the revised proposal introduces three documented challenge cases evaluated using strict past-only online features (zero future data leakage).
        </p>
        <table>
          <thead>
            <tr>
              <th>Evaluation Scenario</th>
              <th>Evaluated Tx</th>
              <th>Fraud Tx</th>
              <th>XGBoost AP</th>
              <th>GraphSAGE AP</th>
              <th>XGB Recall @ 1%</th>
              <th>GNN Recall @ 1%</th>
              <th>Lookalike Hub FPs</th>
            </tr>
          </thead>
          <tbody id="challenges-table-body">
            <tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Loading challenge metrics...</td></tr>
          </tbody>
        </table>
      </div>

      <div class="grid-3">
        <div class="card">
          <div class="card-title">1. Disguised Suspicious</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
            Launderers intentionally alter amounts (micro-ticket sizes ₹115-₹485) and introduce randomized delays (+3 to +13 steps). Tests if detection fails when simple velocity heuristics are evaded.
          </p>
          <div style="margin-top: 1rem; font-weight: 700; color: #f59e0b;">Result: AP drops ~28%; GNN retains structural links.</div>
        </div>
        <div class="card">
          <div class="card-title">2. Benign Lookalike Hubs</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
            Legitimate high fan-in merchant QR hubs and high fan-out corporate salary accounts are injected. Tests if normal busy hubs trigger false alarms.
          </p>
          <div style="margin-top: 1rem; font-weight: 700; color: #10b981;">Result: 0 Hub False Alarms; reciprocal neighbor aggregation filters benign hubs.</div>
        </div>
        <div class="card">
          <div class="card-title">3. Partial / Missing History</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
            50% of earlier history edges are hidden before scoring to test cold-start / unobserved cross-bank context.
          </p>
          <div style="margin-top: 1rem; font-weight: 700; color: #60a5fa;">Result: Insufficient history flag raised; prevents unwarranted false reassurance.</div>
        </div>
      </div>
    </div>

    <!-- TAB 3: EXPLAINABILITY -->
    <div id="explainability" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">GNNExplainer: Influential Local Subgraph</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            Learns continuous edge importance masks optimizing mutual information. Surfaces the exact prior transactions strictly earlier than cutoff that caused the GNN alert.
          </p>
          <div id="gnn-explain-box" style="font-size: 0.9rem;">
            <div style="color: var(--text-muted); padding: 1rem;">Select a high-risk transaction in the Live Scoring tab or below to render topological subgraph.</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">TreeSHAP: Tabular Feature Attributions</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            Exact Shapley value decomposition of the XGBoost baseline showing directional feature impacts on the risk score.
          </p>
          <div id="shap-explain-box" style="font-size: 0.9rem;">
            <div style="color: var(--text-muted); padding: 1rem;">Select a transaction to inspect SHAP waterfall contributions.</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Representative Case Studies (Pre-Computed Audits)</div>
        <div id="case-studies-cards" class="grid-3">
          <!-- Loaded via JS -->
        </div>
      </div>
    </div>

    <!-- TAB 4: BANK ANALYST COPILOT -->
    <div id="bank-copilot" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Investigation Brief & SAR Generator</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            The copilot synthesizes GNN subgraph topology, TreeSHAP feature attributions, and prior transaction history into an evidence-grounded, editable Suspicious Activity Report (SAR).
          </p>
          <label style="font-size: 0.8rem; color: var(--text-muted);">Inspect Transaction ID:</label>
          <input type="text" id="sar-input-tx" value="179904" placeholder="e.g. 179904">
          <button class="btn-primary" onclick="generateSAR()" style="width: 100%; margin-bottom: 1rem;">Generate Evidence-Grounded SAR ➔</button>

          <div style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.4;">
            <strong>Human Review Governance:</strong><br>
            As specified in Section 12, the LLM organizes evidence; the human investigator makes the final decision. The prototype never auto-submits a report.
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Editable SAR Form Narrative</span>
            <button class="btn-outline" onclick="copySAR()">📋 Copy Text</button>
          </div>
          <textarea id="sar-textarea" spellcheck="false" placeholder="SAR brief will appear here..."></textarea>
          <div style="margin-top: 0.5rem; display: flex; justify-content: flex-end; gap: 0.5rem;">
            <button class="btn-outline" onclick="alert('Decision logged: Saved to investigative compliance cache.')">Save Draft</button>
            <button class="btn-primary" onclick="alert('SAR Approved: Exported to compliance staging queue.')">Approve SAR</button>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: CONSUMER SAFETY COPILOT -->
    <div id="consumer-copilot" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Simulated UPI Handle Safety Check</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            Simulates a consumer checking a recipient's VPA before payment. Translates synthetic model risk into simple, plain-language guidance.
          </p>
          <label style="font-size: 0.8rem; color: var(--text-muted);">Enter Recipient UPI ID (VPA):</label>
          <input type="text" id="consumer-upi-input" value="new_reward_cashback@upi" placeholder="e.g. merchant@okhdfcbank or new_user@upi">

          <label style="font-size: 0.8rem; color: var(--text-muted);">Select Language:</label>
          <select id="consumer-lang-select">
            <option value="en">English</option>
            <option value="hi">हिंदी (Hindi)</option>
            <option value="te">తెలుగు (Telugu)</option>
            <option value="ta">தமிழ் (Tamil)</option>
            <option value="mr">मराठी (Marathi)</option>
            <option value="bn">বাংলা (Bengali)</option>
          </select>

          <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
            <button class="btn-primary" onclick="runConsumerCheck()" style="flex: 1;">Check UPI Safety</button>
            <button class="btn-outline" onclick="loadConsumerPreset('sparse')">Try New Account</button>
            <button class="btn-outline" onclick="loadConsumerPreset('mule')">Try Mule Hub</button>
            <button class="btn-outline" onclick="loadConsumerPreset('normal')">Try Verified</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Consumer Advisory Display</div>
          <div id="consumer-result-box" style="padding: 1.5rem; background: #070b13; border-radius: 8px; border: 1px solid var(--border);">
            <div style="color: var(--text-muted);">Enter a UPI ID and click 'Check UPI Safety' to test.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 6: BENCHMARKS & FAIRNESS -->
    <div id="benchmarks" class="tab-content">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">IBM AML Benchmark: GraphSAGE vs XGBoost</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            Main labeled transaction graph comparison evaluated on held-out test split (862,792 transactions) using identical temporal features and 0.5% alert budget.
          </p>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Test AP (PR-AUC)</th>
                <th>Alert Budget</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>FP per 1k</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>XGBoost Baseline</strong></td>
                <td>0.1960</td>
                <td>4,313 alerts (0.5%)</td>
                <td>10.62%</td>
                <td>47.91%</td>
                <td>4.47</td>
              </tr>
              <tr style="background: rgba(59, 130, 246, 0.1);">
                <td><strong>GraphSAGE (Directional)</strong></td>
                <td><strong style="color: #60a5fa;">0.2096</strong></td>
                <td>4,313 alerts (0.5%)</td>
                <td><strong>11.08%</strong></td>
                <td><strong style="color: #34d399;">50.00%</strong></td>
                <td><strong>4.45</strong></td>
              </tr>
            </tbody>
          </table>
          <div style="margin-top: 1rem; font-size: 0.8rem; color: var(--text-muted);">
            GraphSAGE successfully improves recall from 47.91% to 50.00% within the fixed alert budget (+20 true laundering cases captured).
          </div>
        </div>

        <div class="card">
          <div class="card-title">IEEE-CIS Fraud Detection Benchmark</div>
          <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
            Separate conventional tabular fraud detection experiment (not numerically compared with IBM AML).
          </p>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Context</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>ROC-AUC</td>
                <td><strong>0.9238</strong></td>
                <td>E-commerce fraud discrimination</td>
              </tr>
              <tr>
                <td>Average Precision (PR-AUC)</td>
                <td><strong>0.4682</strong></td>
                <td>Tabular features (cards, C/D counts)</td>
              </tr>
              <tr>
                <td>Alert Budget (2%) Precision</td>
                <td><strong>35.33%</strong></td>
                <td>Top 300 alerts evaluated</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Bank Account Fraud (BAF) Demographic Fairness Audit</div>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
          Feedzai BAF suite evaluating error rate disparities across age brackets and income tiers.
        </p>
        <table>
          <thead>
            <tr>
              <th>Age Bracket</th>
              <th>Test Sample</th>
              <th>Population Share</th>
              <th>Base Fraud Rate</th>
              <th>Alert Rate</th>
              <th>False Positive Rate (FPR)</th>
              <th>True Positive Rate (TPR)</th>
            </tr>
          </thead>
          <tbody id="fairness-table-body">
            <!-- Loaded via JS -->
          </tbody>
        </table>
        <div style="margin-top: 1rem; padding: 1rem; background: #070b13; border-radius: 6px; font-size: 0.85rem; color: var(--text-muted);">
          <strong>Fairness Disparity Finding:</strong> Youth applicants (18-24) exhibit a 1.0 percentage point higher False Positive Rate (5.40% vs 4.40% for seniors) due to short credit residency months. Treated honestly as experimental findings; the detector is not claimed as bias-free without threshold calibration.
        </div>
      </div>
    </div>
  </main>

  <script>
    let isStreaming = false;
    let streamTimer = null;
    let currentSelectedTx = null;

    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.target.classList.add('active');
      if (tabId === 'challenges') loadChallengesTable();
      if (tabId === 'explainability') loadCaseStudies();
      if (tabId === 'benchmarks') loadFairnessTable();
    }

    async function stepStream(count = 1) {
      try {
        const res = await fetch('/api/stream/step', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({count})
        });
        const data = await res.json();
        renderRecentFeed(data.recent_batch);
        updateStreamStats();
      } catch (e) {
        console.error(e);
      }
    }

    function toggleStream() {
      const btn = document.getElementById('btn-stream-toggle');
      if (isStreaming) {
        isStreaming = false;
        clearInterval(streamTimer);
        btn.innerText = '▶ Start Stream';
        btn.classList.remove('btn-outline');
        btn.classList.add('btn-primary');
      } else {
        isStreaming = true;
        btn.innerText = '⏸ Pause Stream';
        btn.classList.remove('btn-primary');
        btn.classList.add('btn-outline');
        streamTimer = setInterval(() => stepStream(2), 1200);
      }
    }

    async function resetStream() {
      if (isStreaming) toggleStream();
      const scenario = document.getElementById('select-scenario').value;
      await fetch('/api/stream/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scenario})
      });
      document.getElementById('stream-feed-list').innerHTML = '<div style="color: var(--text-muted); padding: 1rem;">Stream reset.</div>';
      updateStreamStats();
    }

    async function changeScenario() {
      resetStream();
    }

    function renderRecentFeed(items) {
      const container = document.getElementById('stream-feed-list');
      if (container.children[0] && container.children[0].innerText.includes('Click')) {
        container.innerHTML = '';
      }
      items.forEach(tx => {
        const row = document.createElement('div');
        row.style.padding = '0.6rem 0.8rem';
        row.style.borderBottom = '1px solid var(--border)';
        row.style.cursor = 'pointer';
        row.style.display = 'flex';
        row.style.justifyContent = 'space-between';
        row.style.alignItems = 'center';
        row.onmouseover = () => row.style.background = '#1e293b';
        row.onmouseout = () => row.style.background = 'transparent';
        row.onclick = () => selectTransaction(tx);

        const badgeClass = tx.risk_level === 'HIGH' ? 'tag-high' : (tx.risk_level === 'MEDIUM' ? 'tag-med' : 'tag-low');
        const alertFlag = tx.alert ? '🚨 ALERT' : tx.risk_level;

        row.innerHTML = `
          <div>
            <strong>#${tx.transaction_id}</strong>: ${tx.sender_upi} ➔ ${tx.receiver_upi}
            <div style="font-size: 0.75rem; color: var(--text-muted);">INR ${tx.amount.toFixed(2)} | Latency: ${tx.latency_ms}ms</div>
          </div>
          <div>
            <span class="tag ${badgeClass}">${alertFlag} (${tx.risk_score})</span>
          </div>
        `;
        container.insertBefore(row, container.firstChild);
      });
      if (container.children.length > 50) {
        container.removeChild(container.lastChild);
      }
    }

    function selectTransaction(tx) {
      currentSelectedTx = tx;
      const box = document.getElementById('tx-details-box');
      const badge = document.getElementById('selected-tx-badge');
      const escalateBtn = document.getElementById('btn-escalate');

      badge.className = 'tag ' + (tx.risk_level === 'HIGH' ? 'tag-high' : (tx.risk_level === 'MEDIUM' ? 'tag-med' : 'tag-low'));
      badge.innerText = `${tx.risk_level} RISK (${tx.risk_score})`;

      let reasonsHtml = tx.main_reasons.map(r => `<li>${r}</li>`).join('');

      box.innerHTML = `
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin-bottom: 1rem; font-size: 0.85rem;">
          <div><span style="color: var(--text-muted);">Tx ID:</span> <strong>#${tx.transaction_id}</strong></div>
          <div><span style="color: var(--text-muted);">Amount:</span> <strong>₹${tx.amount.toFixed(2)}</strong></div>
          <div><span style="color: var(--text-muted);">Originator:</span> <code>${tx.sender_upi}</code></div>
          <div><span style="color: var(--text-muted);">Beneficiary:</span> <code>${tx.receiver_upi}</code></div>
          <div><span style="color: var(--text-muted);">Graph Cutoff:</span> <code>${tx.graph_history_cutoff} UTC</code></div>
          <div><span style="color: var(--text-muted);">Scoring Latency:</span> <strong>${tx.latency_ms} ms</strong></div>
          <div><span style="color: var(--text-muted);">XGBoost Score:</span> <strong>${tx.xgb_score}</strong></div>
          <div><span style="color: var(--text-muted);">GraphSAGE Score:</span> <strong>${tx.gnn_score}</strong></div>
        </div>
        <div style="margin-top: 1rem;">
          <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 0.4rem;">Model Reasoning & Evidence Drivers:</div>
          <ul style="font-size: 0.85rem; color: var(--text); padding-left: 1.2rem; line-height: 1.4;">
            ${reasonsHtml}
          </ul>
        </div>
      `;

      escalateBtn.style.display = 'block';
    }

    async function updateStreamStats() {
      const res = await fetch('/api/stream/state');
      const data = await res.json();
      document.getElementById('stat-total-scored').innerText = data.total_scored.toLocaleString();
      document.getElementById('stat-total-alerts').innerText = data.total_alerts.toLocaleString();
      document.getElementById('stat-latency-p50').innerText = data.latency_stats.median_ms + ' ms';
      document.getElementById('stat-latency-p95').innerText = data.latency_stats.p95_ms + ' ms';
    }

    function escalateToAnalyst() {
      if (!currentSelectedTx) return;
      switchTab('bank-copilot');
      document.getElementById('sar-input-tx').value = currentSelectedTx.transaction_id;
      generateSAR();
    }

    async function generateSAR() {
      const txId = parseInt(document.getElementById('sar-input-tx').value) || 179904;
      const res = await fetch('/api/copilot/analyst', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(currentSelectedTx || {transaction_id: txId})
      });
      const data = await res.json();
      document.getElementById('sar-textarea').value = data.editable_sar_text;
    }

    function copySAR() {
      const txt = document.getElementById('sar-textarea');
      txt.select();
      document.execCommand('copy');
      alert('SAR Narrative copied to clipboard!');
    }

    async function runConsumerCheck() {
      const upi = document.getElementById('consumer-upi-input').value;
      const lang = document.getElementById('consumer-lang-select').value;
      const res = await fetch('/api/copilot/consumer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({recipient_upi: upi, language: lang})
      });
      const data = await res.json();

      const colorMap = {
        'green': '#10b981',
        'amber': '#f59e0b',
        'red': '#ef4444'
      };
      const color = colorMap[data.badge_color] || '#3b82f6';

      document.getElementById('consumer-result-box').innerHTML = `
        <div style="border-left: 4px solid ${color}; padding-left: 1rem; margin-bottom: 1rem;">
          <h3 style="color: ${color}; font-size: 1.1rem; margin-bottom: 0.4rem;">${data.title}</h3>
          <p style="font-size: 0.95rem; line-height: 1.5; color: #fff;">${data.advisory_message}</p>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 1rem;">
          <div>Recipient VPA: <code>${data.recipient_upi}</code></div>
          <div>Recorded Prior Transfers: <strong>${data.recorded_transactions_count}</strong></div>
          <div>Synthetic Risk Score: <strong>${data.synthetic_risk_score}</strong></div>
          <div>Sparse History Flag: <strong>${data.is_history_sparse ? 'YES' : 'NO'}</strong></div>
        </div>
        <div style="font-size: 0.75rem; color: #64748b; font-style: italic;">
          ${data.disclaimer}
        </div>
      `;
    }

    function loadConsumerPreset(type) {
      if (type === 'sparse') {
        document.getElementById('consumer-upi-input').value = 'brand_new_store@upi';
      } else if (type === 'mule') {
        document.getElementById('consumer-upi-input').value = 'acc_08676@okhdfcbank';
      } else {
        document.getElementById('consumer-upi-input').value = 'dmart_retail_hub@okhdfcbank';
      }
      runConsumerCheck();
    }

    async function loadChallengesTable() {
      const res = await fetch('/api/benchmark/summary');
      const data = await res.json();
      const tbody = document.getElementById('challenges-table-body');
      if (!data.amlsim_challenges || Object.keys(data.amlsim_challenges).length === 0) return;

      tbody.innerHTML = '';
      for (const [name, r] of Object.entries(data.amlsim_challenges)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${name}</strong></td>
          <td>${r.test_events.toLocaleString()}</td>
          <td>${r.test_laundering.toLocaleString()}</td>
          <td>${r.xgboost.average_precision.toFixed(4)}</td>
          <td>${r.graphsage.average_precision.toFixed(4)}</td>
          <td>${(r.xgboost.recall * 100).toFixed(2)}%</td>
          <td>${(r.graphsage.recall * 100).toFixed(2)}%</td>
          <td>${r.xgboost.lookalike_hub_false_positives} / ${r.graphsage.lookalike_hub_false_positives}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    async function loadFairnessTable() {
      const res = await fetch('/api/benchmark/summary');
      const data = await res.json();
      const tbody = document.getElementById('fairness-table-body');
      if (!data.fairness_baf || !data.fairness_baf.subgroups) return;

      const ageGroups = data.fairness_baf.subgroups.customer_age_groups;
      tbody.innerHTML = '';
      for (const [grp, m] of Object.entries(ageGroups)) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${grp}</strong></td>
          <td>${m.sample_count.toLocaleString()}</td>
          <td>${(m.population_share * 100).toFixed(1)}%</td>
          <td>${(m.base_fraud_rate * 100).toFixed(2)}%</td>
          <td>${(m.alert_rate * 100).toFixed(2)}%</td>
          <td><strong>${(m.false_positive_rate * 100).toFixed(2)}%</strong></td>
          <td>${(m.true_positive_rate * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
      }
    }

    async function loadCaseStudies() {
      try {
        let res = await fetch('/api/explainability/cases');
        if (!res.ok) res = await fetch('/reports/explainability/case_studies.json');
        const cases = await res.json();
        const container = document.getElementById('case-studies-cards');
        container.innerHTML = '';
        cases.forEach(cs => {
          const card = document.createElement('div');
          card.className = 'card';
          card.style.cursor = 'pointer';
          card.onclick = () => renderCaseDetail(cs);
          card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
              <strong>Tx #${cs.transaction_id}</strong>
              <span class="tag ${cs.risk_level === 'HIGH' ? 'tag-high' : 'tag-low'}">${cs.risk_level}</span>
            </div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">${cs.sender_upi} ➔ ${cs.receiver_upi}</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #fff;">₹${cs.amount.toFixed(2)}</div>
            <div style="font-size: 0.75rem; color: #60a5fa; margin-top: 0.5rem;">Score: ${cs.risk_score} | Click to view ➔</div>
          `;
          container.appendChild(card);
        });
        if (cases.length > 0) renderCaseDetail(cases[0]);
      } catch (e) {
        console.error(e);
      }
    }

    function renderCaseDetail(cs) {
      // GNN box
      const gnnBox = document.getElementById('gnn-explain-box');
      let gnnEdges = cs.gnn_influential_subgraph.map(e => `
        <div style="padding: 0.4rem 0; border-bottom: 1px solid var(--border);">
          <strong>Tx #${e.prior_transaction_id}</strong>: <code>${e.source_upi}</code> ➔ <code>${e.target_upi}</code><br>
          <span style="color: var(--text-muted); font-size: 0.8rem;">Amount: ₹${e.amount} | Role: ${e.pattern_role} | Weight: <strong>${e.gnn_importance_weight}</strong></span>
        </div>
      `).join('');
      if (!gnnEdges) gnnEdges = '<div style="color: var(--text-muted); padding: 0.5rem;">No prior multi-hop graph edges found.</div>';

      gnnBox.innerHTML = `
        <div style="margin-bottom: 0.8rem;">
          <strong>Target Tx #${cs.transaction_id}</strong> (${cs.risk_level} Risk) | Cutoff: <code>${cs.graph_history_cutoff_timestamp}</code>
        </div>
        <div>${gnnEdges}</div>
      `;

      // SHAP box
      const shapBox = document.getElementById('shap-explain-box');
      let shapRows = cs.tree_shap_explanation.top_features.map(f => `
        <tr>
          <td><code>${f.feature}</code></td>
          <td>${f.feature_value}</td>
          <td><strong style="color: ${f.shap_attribution > 0 ? '#f87171' : '#34d399'};">${f.shap_attribution > 0 ? '+' : ''}${f.shap_attribution}</strong></td>
          <td>${f.direction}</td>
        </tr>
      `).join('');

      shapBox.innerHTML = `
        <table>
          <thead>
            <tr><th>Feature</th><th>Value</th><th>SHAP Impact</th><th>Direction</th></tr>
          </thead>
          <tbody>${shapRows}</tbody>
        </table>
      `;
    }

    // Init on load
    updateStreamStats();
  </script>
</body>
</html>
"""


async def handle_dashboard(request: web.Request) -> web.Response:
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


def create_app() -> web.Application:
    app = web.Application()
    # REST API endpoints
    app.router.add_post("/api/score", handle_score)
    app.router.add_post("/api/stream/step", handle_stream_step)
    app.router.add_post("/api/stream/reset", handle_stream_reset)
    app.router.add_get("/api/stream/state", handle_stream_state)
    app.router.add_get("/api/metrics/latency", handle_metrics_latency)
    app.router.add_post("/api/copilot/analyst", handle_copilot_analyst)
    app.router.add_post("/api/copilot/consumer", handle_copilot_consumer)
    app.router.add_get("/api/graph/subgraph", handle_subgraph)
    app.router.add_get("/api/benchmark/summary", handle_benchmark_summary)
    app.router.add_get("/api/explainability/cases", handle_explainability_cases)

    # Static reports file serving for web UI
    app.router.add_static("/reports/", path=str(Path("reports").resolve()), show_index=True)

    # Web Dashboard UI
    app.router.add_get("/", handle_dashboard)
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run UPI Fraud Detection API Service")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    args = parser.parse_args()

    app = create_app()
    print(f"Starting UPI Fraud Detection Platform & REST API on http://{args.host}:{args.port}...")
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

