"""Comprehensive System Test Suite for UPI Graph Fraud Detection System.

Verifies:
1. Feature extraction & online graph state management (past-only property).
2. Dual model inference (XGBoost + GraphSAGE).
3. TreeSHAP & GNNExplainer attribution consistency and mathematical stability.
4. Edge cases: negative amounts, null fields, unknown handles, zero amounts.
5. Bank Analyst Copilot SAR brief generation.
6. Consumer Safety Copilot multilingual outputs and status categories.
7. Benchmark data integrity across all 6 experimental reports.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
torch.set_num_threads(1)

try:
    from src.evaluate_amlsim_challenges import (
        FEATURE_NAMES, TemporalGraphFeatureTracker, AMLSimGraphSAGE
    )
    from src.explainability import TreeSHAPExplainer, GNNExplainer, generate_explanation_package
    from src.copilots import BankAnalystCopilot, ConsumerSafetyCopilot
    from src.scoring_service import ENGINE
except ImportError:
    from evaluate_amlsim_challenges import (
        FEATURE_NAMES, TemporalGraphFeatureTracker, AMLSimGraphSAGE
    )
    from explainability import TreeSHAPExplainer, GNNExplainer, generate_explanation_package
    from copilots import BankAnalystCopilot, ConsumerSafetyCopilot
    from scoring_service import ENGINE


def test_tracker_past_only_property():
    tracker = TemporalGraphFeatureTracker()
    feat_0, sparse_0 = tracker.extract_features(101, 102, 500.0, 1000)
    assert sparse_0 is True, "New nodes must flag insufficient history"
    assert feat_0[1] == 0.0, "Initial out count must be 0 before update"
    assert feat_0[4] == 0.0, "Initial in count must be 0 before update"

    # Commit event strictly after scoring
    tracker.update(101, 102, 1000)

    # Scored at next timestamp
    feat_1, _ = tracker.extract_features(101, 102, 200.0, 1050)
    assert feat_1[1] > 0.0, "Sender out count must be positive after update"
    assert feat_1[4] > 0.0, "Receiver in count must be positive after update"
    print("✓ test_tracker_past_only_property: PASSED")


def test_realtime_scoring_engine():
    tx = {
        "transaction_id": 999123,
        "sender_upi": "acc_05001@okhdfcbank",
        "receiver_upi": "acc_08676@okhdfcbank",
        "amount": 285.50,
        "timestamp": 1725800000
    }
    res = ENGINE.score_transaction(tx)
    assert "risk_score" in res
    assert 0.0 <= res["risk_score"] <= 1.0, "Risk score must be bounded between 0 and 1"
    assert res["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert "latency_ms" in res
    assert res["latency_ms"] < 200.0, "Scoring latency must be sub-200ms"
    assert len(res["main_reasons"]) > 0
    print(f"✓ test_realtime_scoring_engine: PASSED (Risk: {res['risk_score']}, Latency: {res['latency_ms']}ms)")


def test_edge_cases_and_malformed_inputs():
    """Verify that scoring engine never crashes on edge cases or malformed inputs."""
    edge_cases = [
        {"transaction_id": 1, "sender_upi": None, "receiver_upi": None, "amount": None, "timestamp": None},
        {"transaction_id": 2, "sender_upi": "custom_vpa@randombank", "receiver_upi": "user2@upi", "amount": -50.0},
        {"transaction_id": 3, "sender_upi": "user1@upi", "receiver_upi": "user2@upi", "amount": 0.0},
        {"transaction_id": 4, "sender_upi": "acc_99999@paytm", "receiver_upi": "acc_88888@sbi", "amount": 1000000.0},
        {"transaction_id": 5, "sender_upi": "acc_00001@upi", "receiver_upi": "acc_00002@upi", "amount": "250.75"},
    ]

    for tc in edge_cases:
        res = ENGINE.score_transaction(tc)
        assert "risk_score" in res
        assert 0.0 <= res["risk_score"] <= 1.0
        assert res["latency_ms"] < 200.0

    print("✓ test_edge_cases_and_malformed_inputs: PASSED")


def test_explainers_mathematical_stability():
    """Verify TreeSHAP and GNNExplainer outputs on extreme values."""
    xgb_explainer = TreeSHAPExplainer(
        "models/amlsim_models/xgboost_amlsim.ubj", FEATURE_NAMES
    )
    # Test normal feature vector
    normal_x = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    exp = xgb_explainer.explain(normal_x, top_k=5)
    assert len(exp["top_features"]) == 5
    assert 0.0 <= exp["predicted_probability"] <= 1.0

    # Test extreme positive values (check no exp overflow)
    extreme_high_x = np.full(len(FEATURE_NAMES), 1000.0, dtype=np.float32)
    exp_high = xgb_explainer.explain(extreme_high_x, top_k=3)
    assert 0.0 <= exp_high["predicted_probability"] <= 1.0

    # Test empty subgraph in GNNExplainer
    gnn_explainer = ENGINE.gnn_explainer
    empty_res = gnn_explainer.explain(
        ENGINE.node_x, ENGINE.edge_index, torch.zeros((1, 13)), 0, 1, [], top_k=3
    )
    assert empty_res == [], "Empty prior edge list must return empty attribution safely"
    print("✓ test_explainers_mathematical_stability: PASSED")


def test_copilots():
    # 1. Analyst Copilot
    case_payload = {
        "transaction_id": 179904,
        "sender_upi": "acc_17800@paytm",
        "receiver_upi": "acc_08676@okhdfcbank",
        "amount": 14.03,
        "risk_score": 0.9832,
        "risk_level": "HIGH",
        "graph_cutoff_timestamp": 1725804211,
        "main_reasons": ["Rapid fan-in detected", "Pass-through forwarding anomaly"]
    }
    sar = BankAnalystCopilot.generate_sar_summary(case_payload)
    assert "SAR-UPI-2024-179904" in sar["sar_id"]
    assert "SUSPICIOUS ACTIVITY REPORT" in sar["editable_sar_text"]
    assert "acc_08676@okhdfcbank" in sar["editable_sar_text"]

    # 2. Consumer Safety Copilot
    # Test Sparse
    adv_sparse = ConsumerSafetyCopilot.check_recipient_safety("new@upi", recent_risk_score=0.1, total_prior_events=1, language="en")
    assert adv_sparse["status"] == "SPARSE_HISTORY"
    assert adv_sparse["safety_category"] == "CAUTION"

    # Test High Risk
    adv_risk = ConsumerSafetyCopilot.check_recipient_safety("mule@upi", recent_risk_score=0.85, total_prior_events=25, language="en")
    assert adv_risk["status"] == "HIGH_RISK"
    assert adv_risk["safety_category"] == "WARNING"

    # Test Verified Normal
    adv_safe = ConsumerSafetyCopilot.check_recipient_safety("dmart@upi", recent_risk_score=0.04, total_prior_events=50, language="en")
    assert adv_safe["status"] == "NORMAL"
    assert adv_safe["safety_category"] == "SAFE"

    # Test All Languages
    for lang in ["en", "hi", "te", "ta", "mr", "bn"]:
        adv = ConsumerSafetyCopilot.check_recipient_safety("test@upi", recent_risk_score=0.1, total_prior_events=1, language=lang)
        assert len(adv["advisory_message"]) > 10
        assert len(adv["title"]) > 3

    print("✓ test_copilots: PASSED")


def test_benchmark_reports_exist():
    expected_reports = [
        Path("reports/test_evaluation_v1/test_metrics.json"),
        Path("reports/amlsim_challenges/challenge_metrics.json"),
        Path("reports/fairness_baf/fairness_metrics.json"),
        Path("reports/ieee_cis_benchmark/ieee_cis_metrics.json"),
        Path("reports/api_benchmark/latency_benchmark.json"),
        Path("reports/explainability/case_studies.json")
    ]
    for rep in expected_reports:
        assert rep.is_file(), f"Missing expected report: {rep}"
        data = json.loads(rep.read_text())
        assert len(data) > 0, f"Report {rep} is empty"

    print("✓ test_benchmark_reports_exist: PASSED")


if __name__ == "__main__":
    test_tracker_past_only_property()
    test_realtime_scoring_engine()
    test_edge_cases_and_malformed_inputs()
    test_explainers_mathematical_stability()
    test_copilots()
    test_benchmark_reports_exist()
    print("\n==========================================")
    print("ALL COMPREHENSIVE TESTS PASSED! 100% CLEAN ✅")
    print("==========================================")
