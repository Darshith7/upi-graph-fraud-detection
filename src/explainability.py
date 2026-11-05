"""Explainability module for AI-Based Graph Fraud Detection System.

Implements:
1. GNNExplainer: Learns edge-importance masks over the directional computational graph
   to identify which historical transactions contributed most to the GNN prediction.
2. TreeSHAP Explainer: Uses exact TreeSHAP attributions for the XGBoost baseline.
3. Traceable Case Reports: Packages prediction time, graph cutoff timestamp, supporting transaction IDs,
   and localized subgraph edges for Bank Analyst review.
"""

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Dict, List, Any

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import torch.nn as nn
import xgboost as xgb


class TreeSHAPExplainer:
    """Computes exact Shapley feature attributions for XGBoost predictions."""

    def __init__(self, model_path: str, feature_names: List[str]):
        self.bst = xgb.Booster()
        self.bst.load_model(str(model_path))
        self.feature_names = feature_names

    def explain(self, x: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Compute SHAP values for a single 1D feature vector or 2D array."""
        x_2d = np.atleast_2d(x)
        dm = xgb.DMatrix(x_2d, feature_names=self.feature_names, nthread=1)
        contribs = self.bst.predict(dm, pred_contribs=True)[0]
        base_value = float(contribs[-1])
        shap_values = contribs[:-1]

        # Sort features by absolute attribution
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        top_features = []
        for idx in sorted_indices[:top_k]:
            fname = self.feature_names[idx]
            val = float(x_2d[0, idx])
            shap_val = float(shap_values[idx])
            top_features.append({
                "feature": fname,
                "feature_value": round(val, 4),
                "shap_attribution": round(shap_val, 4),
                "direction": "Pushes Risk HIGHER" if shap_val > 0 else "Lowers Risk"
            })

        predicted_logit = float(np.sum(shap_values) + base_value)
        if predicted_logit < -50.0:
            predicted_prob = 0.0
        elif predicted_logit > 50.0:
            predicted_prob = 1.0
        else:
            predicted_prob = float(1.0 / (1.0 + math.exp(-predicted_logit)))

        return {
            "base_value": round(base_value, 4),
            "predicted_probability": round(predicted_prob, 4),
            "top_features": top_features
        }


class GNNExplainer:
    """GNNExplainer for directional temporal GraphSAGE.

    Learns an edge-importance mask over local candidate historical edges such that
    masking irrelevant edges preserves the model's confidence in the fraud classification,
    optimizing an edge-mask objective directly on the model's message-passing layers.
    """

    def __init__(self, gnn_model: nn.Module, feature_names: List[str], epochs: int = 40, lr: float = 0.05, seed: int = 42):
        self.model = gnn_model
        self.feature_names = feature_names
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def explain(self, node_x: torch.Tensor, edge_index: torch.Tensor,
                tx_feat: torch.Tensor, s_idx: int, d_idx: int,
                local_edge_info: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Identify top influential historical edges by optimizing an edge mask directly on the GNN model."""
        if len(local_edge_info) == 0:
            return []
        final_weights = self.edge_weights(node_x, edge_index, tx_feat, s_idx, d_idx, local_edge_info)
        sorted_indices = np.argsort(final_weights)[::-1]
        top_influential_edges = []
        for idx in sorted_indices[:top_k]:
            edge_meta = {k: v for k, v in local_edge_info[idx].items() if k not in ("src_idx", "dst_idx")}
            edge_meta["gnn_importance_weight"] = round(float(final_weights[idx]), 4)
            top_influential_edges.append(edge_meta)
        return top_influential_edges

    def edge_weights(self, node_x: torch.Tensor, edge_index: torch.Tensor,
                     tx_feat: torch.Tensor, s_idx: int, d_idx: int,
                     local_edge_info: List[Dict[str, Any]]) -> np.ndarray:
        """Learned importance in [0, 1] for every candidate edge (same order as local_edge_info)."""
        if len(local_edge_info) == 0:
            return np.zeros(0, dtype=np.float32)

        torch.manual_seed(self.seed)
        num_edges = len(local_edge_info)

        # Build candidate edge index corresponding to local_edge_info
        local_src = [e["src_idx"] for e in local_edge_info]
        local_dst = [e["dst_idx"] for e in local_edge_info]
        sub_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long)

        self.model.eval()
        s_t = torch.tensor([s_idx], dtype=torch.long)
        d_t = torch.tensor([d_idx], dtype=torch.long)

        # Compute full unmasked model prediction
        with torch.no_grad():
            for conv in (self.model.conv1, self.model.conv2):
                conv.explain = False
                conv._edge_mask = None
            orig_logit = self.model(node_x, sub_edge_index, tx_feat, s_t, d_t)
            orig_prob = torch.sigmoid(orig_logit).item()

        # Learnable mask over the candidate edges, initialized warm
        mask_param = nn.Parameter(torch.ones(num_edges, dtype=torch.float32) * 1.5)
        optimizer = torch.optim.Adam([mask_param], lr=self.lr)

        target = 1.0 if orig_prob > 0.5 else 0.0

        for conv in (self.model.conv1, self.model.conv2):
            conv.explain = True

        for _ in range(self.epochs):
            optimizer.zero_grad()
            mask = torch.sigmoid(mask_param)
            self.model.conv1._edge_mask = mask
            self.model.conv2._edge_mask = mask

            out_logit = self.model(node_x, sub_edge_index, tx_feat, s_t, d_t)
            pred_prob = torch.sigmoid(out_logit)

            # Optimization objective: preserve model confidence while encouraging sparsity
            if target == 1.0:
                pred_loss = -torch.log(pred_prob + 1e-6)
            else:
                pred_loss = -torch.log(1.0 - pred_prob + 1e-6)

            sparsity_loss = torch.mean(mask)
            entropy_loss = -torch.mean(mask * torch.log(mask + 1e-8) + (1.0 - mask) * torch.log(1.0 - mask + 1e-8))
            loss = pred_loss + 0.05 * sparsity_loss + 0.01 * entropy_loss
            loss.backward()
            optimizer.step()

        # Reset explain hooks
        for conv in (self.model.conv1, self.model.conv2):
            conv.explain = False
            conv._edge_mask = None

        with torch.no_grad():
            final_weights = torch.sigmoid(mask_param).numpy()
        return final_weights


def generate_explanation_package(
    transaction_id: int,
    sender_upi: str,
    receiver_upi: str,
    amount: float,
    timestamp_seconds: int,
    graph_cutoff_seconds: int,
    features: np.ndarray,
    xgb_explainer: TreeSHAPExplainer,
    gnn_explainer: GNNExplainer,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    node2idx: Dict[int, int],
    unknown_node_idx: int,
    s_node_id: int,
    d_node_id: int,
    prior_tx_history: List[Dict[str, Any]],
    blend_score: float = None,
    calibrated_threshold: float = 0.4388
) -> Dict[str, Any]:
    """Generates an evidence-grounded explanation package."""
    # 1. SHAP Explanation
    shap_res = xgb_explainer.explain(features, top_k=5)

    # 2. GNN Explanation
    s_idx = node2idx.get(s_node_id, unknown_node_idx)
    d_idx = node2idx.get(d_node_id, unknown_node_idx)
    tx_feat_tensor = torch.from_numpy(features).float().unsqueeze(0)

    # Construct local edge list connecting sender or receiver strictly before cutoff
    local_edges = []
    for ptx in prior_tx_history:
        if int(ptx.get("timestamp_seconds", 0)) >= graph_cutoff_seconds:
            continue  # Strictly enforce past-only property
        ptx_src = int(ptx["sourceNodeId"])
        ptx_dst = int(ptx["targetNodeId"])
        if ptx_src in (s_node_id, d_node_id) or ptx_dst in (s_node_id, d_node_id):
            if ptx_dst == d_node_id:
                role = "Rapid Inflow to Beneficiary"
            elif ptx_src == d_node_id:
                role = "Layering Outflow from Beneficiary"
            elif ptx_dst == s_node_id:
                role = "Inflow to Sender"
            elif ptx_src == s_node_id:
                role = "Outflow from Sender"
            else:
                role = "Prior Network Counterparty Edge"

            local_edges.append({
                "prior_transaction_id": int(ptx["transaction_id"]),
                "source_upi": str(ptx["sender_upi"]),
                "target_upi": str(ptx["receiver_upi"]),
                "amount": round(float(ptx["amount"]), 2),
                "timestamp_seconds": int(ptx["timestamp_seconds"]),
                "pattern_role": role,
                "src_idx": node2idx.get(ptx_src, unknown_node_idx),
                "dst_idx": node2idx.get(ptx_dst, unknown_node_idx)
            })

    gnn_influential = gnn_explainer.explain(
        node_x, edge_index, tx_feat_tensor, s_idx, d_idx, local_edges, top_k=4
    )

    if blend_score is not None:
        score = float(blend_score)
    else:
        score = shap_res["predicted_probability"]

    if score >= calibrated_threshold:
        risk_level = "HIGH"
        action = "Freeze account and escalate to FIU-IND compliance"
    elif score >= (calibrated_threshold * 0.5):
        risk_level = "MEDIUM"
        action = "Flag for manual KYC and velocity verification"
    else:
        risk_level = "LOW"
        action = "Permit standard clearing"

    # Supporting transaction IDs strictly earlier than cutoff
    supporting_tx_ids = [e["prior_transaction_id"] for e in gnn_influential]

    # Human-readable reasons
    reasons = []
    for feat in shap_res["top_features"][:3]:
        reasons.append(f"{feat['feature']} = {feat['feature_value']} ({feat['direction']}, SHAP: {feat['shap_attribution']:+0.2f})")
    if len(gnn_influential) > 0:
        reasons.append(f"GNN detected structural link to {len(gnn_influential)} prior suspicious counterparty transfers")

    return {
        "transaction_id": transaction_id,
        "prediction_time": datetime.now(timezone.utc).isoformat(),
        "graph_cutoff_timestamp": graph_cutoff_seconds,
        "sender_upi": sender_upi,
        "receiver_upi": receiver_upi,
        "amount": amount,
        "risk_score": round(score, 4),
        "risk_level": risk_level,
        "recommended_action": action,
        "main_reasons": reasons,
        "tree_shap_explanation": shap_res,
        "gnn_influential_subgraph": gnn_influential,
        "supporting_transaction_ids": supporting_tx_ids,
        "disclaimer": "AI model explanation generated for investigative assistance. Does not constitute legal proof of financial misconduct."
    }


def run_explainability_study():
    """Generates and saves explainability case studies on selected transactions."""
    output_dir = Path("reports/explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models and data for explainability review...")
    from evaluate_amlsim_challenges import AMLSimGraphSAGE, FEATURE_SETS

    gnn_ckpt = torch.load("models/amlsim_models/graphsage_amlsim.pt", weights_only=False)
    feature_set = gnn_ckpt.get("feature_set", "cumulative")
    FEATURE_NAMES = list(gnn_ckpt.get("feature_names", FEATURE_SETS[feature_set]))

    xgb_explainer = TreeSHAPExplainer(
        "models/amlsim_models/xgboost_amlsim.ubj",
        FEATURE_NAMES
    )

    gnn_model = AMLSimGraphSAGE(in_features=len(FEATURE_NAMES), hidden_dim=32, out_dim=1)
    gnn_model.load_state_dict(gnn_ckpt["state_dict"])
    node_x = gnn_ckpt["node_x"]
    edge_index = gnn_ckpt["edge_index"]
    node2idx = gnn_ckpt["node2idx"]
    unknown_node_idx = gnn_ckpt.get("unknown_node_idx", len(node2idx))

    gnn_explainer = GNNExplainer(gnn_model, FEATURE_NAMES, seed=42)

    # Load baseline transactions
    df = pd.read_csv("data/amlsim_challenges/baseline.csv")
    from evaluate_amlsim_challenges import extract_scenario_features
    _, X = extract_scenario_features(Path("data/amlsim_challenges/baseline.csv"), feature_set=feature_set)

    # Select representative cases from the unseen test period (step > 90) BY MODEL SCORE, not by
    # hard-coded row index (indices silently stop meaning "high risk" whenever the model changes):
    # 1-2. The two highest-scoring true laundering transactions (different receivers)
    # 3.   A benign transaction at the median benign score
    xgb_scores = xgb.Booster(model_file="models/amlsim_models/xgboost_amlsim.ubj").predict(
        xgb.DMatrix(X, feature_names=FEATURE_NAMES))
    in_test = (df["timestamp_step"] > 90).values
    laundering = np.where(in_test & (df["is_laundering"].values == 1))[0]
    ranked = laundering[np.argsort(-xgb_scores[laundering])]
    case_indices = [int(ranked[0])]
    for i in ranked[1:]:
        if df.iloc[int(i)]["targetNodeId"] != df.iloc[case_indices[0]]["targetNodeId"]:
            case_indices.append(int(i))
            break
    benign = np.where(in_test & (df["is_laundering"].values == 0))[0]
    benign = benign[np.argsort(xgb_scores[benign])]
    case_indices.append(int(benign[len(benign) // 2]))

    case_studies = []
    for idx in case_indices:
        row = df.iloc[idx]
        feat = X[idx]
        cutoff = int(row["timestamp_seconds"])

        # Prior history strictly before this transaction
        prior_history = df[(df["timestamp_seconds"] < cutoff) &
                           ((df["sourceNodeId"] == row["sourceNodeId"]) |
                            (df["targetNodeId"] == row["targetNodeId"]) |
                            (df["sourceNodeId"] == row["targetNodeId"]) |
                            (df["targetNodeId"] == row["sourceNodeId"]))].tail(10).to_dict("records")

        case_exp = generate_explanation_package(
            transaction_id=int(row["transaction_id"]),
            sender_upi=str(row["sender_upi"]),
            receiver_upi=str(row["receiver_upi"]),
            amount=float(row["amount"]),
            timestamp_seconds=int(row["timestamp_seconds"]),
            graph_cutoff_seconds=cutoff,
            features=feat,
            xgb_explainer=xgb_explainer,
            gnn_explainer=gnn_explainer,
            node_x=node_x,
            edge_index=edge_index,
            node2idx=node2idx,
            unknown_node_idx=unknown_node_idx,
            s_node_id=int(row["sourceNodeId"]),
            d_node_id=int(row["targetNodeId"]),
            prior_tx_history=prior_history
        )
        case_studies.append(case_exp)

    # Save JSON report
    cases_file = output_dir / "case_studies.json"
    cases_file.write_text(json.dumps(case_studies, indent=2), encoding="utf-8")
    print(f"Saved explainability case studies to {cases_file}")

    # Generate Markdown Report
    md_lines = [
        "# Model Explainability Report: GNNExplainer & TreeSHAP",
        "",
        "## 1. Objectives & Compliance",
        "- **Dual Explanation Architecture**: TreeSHAP for tabular feature attribution and GNNExplainer for local graph topological edge importance.",
        "- **Audit Trail Integrity**: Each explanation strictly logs prediction timestamp, graph history cutoff, and supporting transaction IDs.",
        "- **Non-Prosecutorial Stance**: Explanations are provided as evidentiary aids for compliance analysts, not absolute proof of wrongdoing.",
        "",
        "## 2. Exemplary Case Studies",
        ""
    ]

    for i, cs in enumerate(case_studies, 1):
        md_lines += [
            f"### Case {i}: Transaction #{cs['transaction_id']} ({cs['risk_level']} Risk - Score: {cs['risk_score']})",
            f"- **Transfer**: `{cs['sender_upi']}` ➔ `{cs['receiver_upi']}`",
            f"- **Amount**: ₹{cs['amount']:,.2f}",
            f"- **Graph History Cutoff**: {cs['graph_cutoff_timestamp']} (UTC)",
            f"- **Recommended Action**: {cs['recommended_action']}",
            "",
            "#### TreeSHAP Top Attributions:",
            "| Feature | Feature Value | SHAP Attribution | Direction |",
            "|---|---|---|---|"
        ]
        for f in cs["tree_shap_explanation"]["top_features"]:
            md_lines.append(f"| `{f['feature']}` | {f['feature_value']} | {f['shap_attribution']:+0.3f} | {f['direction']} |")

        md_lines += [
            "",
            "#### GNNExplainer Influential Subgraph Edges:",
            "| Prior Tx ID | Source | Target | Amount | Pattern Role | GNN Importance |",
            "|---|---|---|---|---|---|"
        ]
        for e in cs["gnn_influential_subgraph"]:
            md_lines.append(f"| #{e['prior_transaction_id']} | `{e['source_upi']}` | `{e['target_upi']}` | ₹{e['amount']:,.2f} | {e['pattern_role']} | {e['gnn_importance_weight']} |")

        md_lines.append("")

    report_file = output_dir / "explainability_report.md"
    report_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved explainability report to {report_file}")


if __name__ == "__main__":
    run_explainability_study()
