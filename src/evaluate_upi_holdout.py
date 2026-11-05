"""Zero-shot evaluation of the DEPLOYED AMLSim-trained model on the independent UPI simulation.

Nothing is retrained or re-tuned: weights, blend, static cutoff and the rolling alert-budget policy
are exactly those in models/amlsim_models. The graph starts empty and every account gets its own
node; the first 3 days are a warm-up (scored and committed, but excluded from the metrics) so the
72h feature windows and the graph are populated.

Reports AP / ROC-AUC, the deployed alert policy at the held-out ~0.8% prevalence, recall by
typology (none of which exist in the training simulator), and false alerts on benign hubs
(busy merchants and payroll runs).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_amlsim_challenges import (  # noqa: E402
    AMLSimGraphSAGE, FEATURE_SETS, LiveGraphState, extract_scenario_features,
    rolling_budget_alerts, score_gnn_causal, bootstrap_ap_difference,
)

WARMUP_STEPS = 36   # 3 days at 2h/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "upi_holdout" / "upi_sim_holdout.csv"))
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "upi_holdout"))
    args = ap.parse_args()

    ck = torch.load(ROOT / "models" / "amlsim_models" / "graphsage_amlsim.pt", weights_only=False)
    fs = ck.get("feature_set", "cumulative")
    names = list(ck.get("feature_names", FEATURE_SETS[fs]))
    cal = ck["calibration"]
    w_x, w_g, thr, budget = cal["best_w_xgb"], cal["best_w_gnn"], cal["calibrated_threshold"], cal["val_alert_budget"]
    model = AMLSimGraphSAGE(in_features=len(names), hidden_dim=32, out_dim=1)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    bst = xgb.Booster(model_file=str(ROOT / "models" / "amlsim_models" / "xgboost_amlsim.ubj"))

    df, X = extract_scenario_features(Path(args.csv), feature_set=fs)
    y = df["is_laundering"].values
    xs = bst.predict(xgb.DMatrix(X, feature_names=names))
    nodes = sorted(set(df["sourceNodeId"]) | set(df["targetNodeId"]))
    n2i = {n: i for i, n in enumerate(nodes)}
    s_idx = df["sourceNodeId"].map(n2i).values.astype(np.int64)
    d_idx = df["targetNodeId"].map(n2i).values.astype(np.int64)
    gs = score_gnn_causal(model, LiveGraphState(len(nodes), len(names)), X, s_idx, d_idx, df["timestamp_step"].values)
    blend = w_x * xs + w_g * gs

    ev = (df["timestamp_step"].values > WARMUP_STEPS)
    ye = y[ev]
    print(f"Held-out UPI simulation: {len(df):,} tx, evaluated {int(ev.sum()):,} after warm-up, "
          f"laundering {int(ye.sum()):,} ({ye.mean():.2%})")

    # live alert policy replayed over the evaluated stream, seeded with the AMLSim validation scores
    policy = np.zeros(len(df), dtype=bool)
    policy[ev] = rolling_budget_alerts(blend[ev], ck["calibration"]["val_score_reference"], thr, budget)
    static = ev & (blend >= thr)

    def conf(mask):
        m = mask[ev]
        tp, fp = int((m & (ye == 1)).sum()), int((m & (ye == 0)).sum())
        fn = int((~m & (ye == 1)).sum())
        return {"alerts": int(m.sum()), "alert_rate": round(float(m.mean()), 4),
                "precision": round(tp / max(1, tp + fp), 4), "recall": round(tp / max(1, tp + fn), 4),
                "false_alerts_per_true_alert": round(fp / max(1, tp), 1)}

    per_typology = {}
    for t in sorted(set(df["typology"]) - {"normal"}):
        mt = ev & (df["typology"].values == t)
        per_typology[t] = {"transactions": int(mt.sum()),
                           "recall_deployed_policy": round(float(policy[mt].mean()), 4) if mt.sum() else None,
                           "recall_static_cutoff": round(float(static[mt].mean()), 4) if mt.sum() else None}

    recv_counts = df.loc[ev & (y == 0), "targetNodeId"].value_counts()
    hub_ids = set(recv_counts.index[:25])                       # the 25 busiest legitimate receivers
    hub_mask = ev & (y == 0) & df["targetNodeId"].isin(hub_ids).values
    payroll_mask = ev & (y == 0) & (df["amount"].values >= 15_000) & df["sourceNodeId"].map(df.loc[ev, "sourceNodeId"].value_counts()).ge(40).values

    report = {
        "dataset": "independent UPI simulation (data/upi_holdout), zero-shot",
        "transactions_evaluated": int(ev.sum()), "laundering": int(ye.sum()), "prevalence": round(float(ye.mean()), 5),
        "model": {"feature_set": fs, "blend_weights": [w_x, w_g], "static_cutoff": thr, "alert_budget": budget},
        "average_precision": {"xgboost": round(float(average_precision_score(ye, xs[ev])), 4),
                              "graphsage": round(float(average_precision_score(ye, gs[ev])), 4),
                              "blend": round(float(average_precision_score(ye, blend[ev])), 4),
                              "random_baseline": round(float(ye.mean()), 4)},
        "roc_auc": {"xgboost": round(float(roc_auc_score(ye, xs[ev])), 4), "blend": round(float(roc_auc_score(ye, blend[ev])), 4)},
        "blend_minus_xgboost_ap": bootstrap_ap_difference(ye, blend[ev], xs[ev], n_boot=100, seed=0),
        "deployed_policy": conf(policy),
        "static_cutoff_only": conf(static),
        "per_typology": per_typology,
        "benign_hub_alert_rate": {
            "busiest_25_legit_receivers": {"transactions": int(hub_mask.sum()), "alert_rate": round(float(policy[hub_mask].mean()), 4) if hub_mask.sum() else None},
            "large_payroll_style_senders": {"transactions": int(payroll_mask.sum()), "alert_rate": round(float(policy[payroll_mask].mean()), 4) if payroll_mask.sum() else None},
            "all_benign": round(float(policy[ev & (y == 0)].mean()), 4),
        },
        "caveat": "Generated by this project's own simulator: independent of AMLSim, not of the authors, and not real data.",
    }
    # ---- diagnostics: is the simulator learnable, and where does the transfer break? -------------
    steps = df["timestamp_step"].values
    tr_m = (steps > WARMUP_STEPS) & (steps <= 84)
    te_m = steps > 84
    ref = xgb.train({"objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 5, "learning_rate": 0.05,
                     "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "nthread": 2, "seed": 0,
                     "scale_pos_weight": float((y[tr_m] == 0).sum() / max(1, (y[tr_m] == 1).sum())) ** 0.5},
                    xgb.DMatrix(X[tr_m], label=y[tr_m], feature_names=names), num_boost_round=200)
    ref_scores = ref.predict(xgb.DMatrix(X[te_m], feature_names=names))
    single = {n: round(float(roc_auc_score(y[te_m], X[te_m, names.index(n)])), 3)
              for n in ("receiver_log1p_unique_senders_24h", "receiver_log1p_in_cnt_24h", "insufficient_history_flag", "log1p_amount")}
    # how far is the new world from the training world? median feature values, train sim vs holdout
    tr_df, tr_X = extract_scenario_features(ROOT / "data" / "amlsim_challenges" / "baseline.csv", feature_set=fs)
    shift = {n: {"amlsim_median": round(float(np.median(tr_X[:, names.index(n)])), 3),
                 "holdout_median": round(float(np.median(X[:, names.index(n)])), 3)}
             for n in ("sender_log1p_out_cnt_24h", "receiver_log1p_in_cnt_24h", "receiver_log1p_unique_senders_24h", "log1p_amount")}
    reference = {
        "rows": f"train steps {WARMUP_STEPS + 1}-84, test steps 85-{int(steps.max())}",
        "random_ap": round(float(y[te_m].mean()), 4),
        "xgboost_retrained_on_holdout_sim_ap": round(float(average_precision_score(y[te_m], ref_scores)), 4),
        "deployed_zero_shot_blend_ap_same_rows": round(float(average_precision_score(y[te_m], blend[te_m])), 4),
        "single_feature_roc_auc": single,
        "feature_median_shift_amlsim_vs_holdout": shift,
    }
    report["reference_in_distribution"] = reference

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    dp, sc = report["deployed_policy"], report["static_cutoff_only"]
    md = ["# Zero-shot evaluation on an independent UPI simulation", "",
          f"Deployed AMLSim-trained model, nothing retrained. {report['transactions_evaluated']:,} transactions after a 3-day warm-up, "
          f"{report['laundering']:,} laundering ({report['prevalence']:.2%}; random-guess AP = {report['average_precision']['random_baseline']:.4f}).", "",
          "| Score | XGBoost | GraphSAGE | Blend |", "|---|---|---|---|",
          f"| Average precision | {report['average_precision']['xgboost']:.4f} | {report['average_precision']['graphsage']:.4f} | {report['average_precision']['blend']:.4f} |", "",
          "| Alert rule | Alert rate | Precision | Recall | False alerts per true alert |", "|---|---|---|---|---|",
          f"| Deployed policy (static cutoff + rolling budget) | {dp['alert_rate']:.2%} | {dp['precision']:.1%} | {dp['recall']:.1%} | {dp['false_alerts_per_true_alert']} |",
          f"| Static cutoff only | {sc['alert_rate']:.2%} | {sc['precision']:.1%} | {sc['recall']:.1%} | {sc['false_alerts_per_true_alert']} |", "",
          "| Typology (unseen in training) | Transactions | Recall, deployed policy | Recall, static cutoff |", "|---|---|---|---|"]
    for t, r in per_typology.items():
        md.append(f"| {t} | {r['transactions']} | {r['recall_deployed_policy']:.1%} | {r['recall_static_cutoff']:.1%} |")
    b = report["benign_hub_alert_rate"]
    md += ["", "## Diagnostic: is the simulation learnable, and where does transfer break?", "",
           f"Rows: {reference['rows']}. Random AP {reference['random_ap']}. "
           f"XGBoost retrained on the holdout simulation's own early period: AP **{reference['xgboost_retrained_on_holdout_sim_ap']}**; "
           f"deployed AMLSim-trained blend on the same rows: AP **{reference['deployed_zero_shot_blend_ap_same_rows']}**.",
           f"Single-feature ROC-AUC on the holdout: {reference['single_feature_roc_auc']}.",
           f"Median feature values, AMLSim vs holdout: {reference['feature_median_shift_amlsim_vs_holdout']}."]
    md += ["", f"Benign hubs alerted (deployed policy): busiest 25 legitimate receivers {b['busiest_25_legit_receivers']['alert_rate']}, "
               f"large payroll-style senders {b['large_payroll_style_senders']['alert_rate']}; all benign traffic {b['all_benign']:.2%}.", "",
           "Caveat: " + report["caveat"]]
    (out / "holdout_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
