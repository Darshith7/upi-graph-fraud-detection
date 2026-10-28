"""Train the first XGBoost baseline; report VALIDATION results only.

Run from the project folder: python src/train_baseline.py
Requires numpy, scikit-learn, and xgboost (tested with XGBoost 3.4.1).
Test and tail arrays are never opened by this script.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, auc, precision_recall_curve
import xgboost as xgb


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classification_counts(labels, alerts):
    positive = labels == 1
    tp = int(np.count_nonzero(alerts & positive))
    fp = int(np.count_nonzero(alerts & ~positive))
    fn = int(np.count_nonzero(~alerts & positive))
    tn = int(np.count_nonzero(~alerts & ~positive))
    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "alerts": tp + fp,
        "alert_fraction": (tp + fp) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positives_per_1000_non_laundering": 1000 * fp / (fp + tn) if fp + tn else 0.0,
    }


def evaluate_validation(labels, scores, transaction_ids, alert_fraction):
    """Return rank-budget metrics plus a conservative, fixed score cutoff.

    The budget is a preset operating choice, not optimized using outcomes.
    Labels are used to assess alerts, never to select them or break score ties.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(labels)
    k = math.floor(n * alert_fraction)
    if k < 1:
        raise ValueError("Validation is too small for this alert fraction")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Unexpected prediction scores")
    order = np.lexsort((transaction_ids, -scores))
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    boundary = float(scores[order[k - 1]])
    # A fixed score rule can exceed a top-k budget if many rows tie at the cutoff.
    # Exclude boundary ties in that case; keep the explicit comparator for serving.
    if int(np.count_nonzero(scores >= boundary)) > k:
        comparator = ">"
        threshold_alerts = scores > boundary
    else:
        comparator = ">="
        threshold_alerts = scores >= boundary
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = float(average_precision_score(labels, scores))
    prevalence = float(np.mean(labels))
    return {
        "evaluation_split": "validation",
        "rows": n,
        "laundering_rows": int(np.count_nonzero(labels)),
        "positive_prevalence": prevalence,
        "average_precision": ap,
        "pr_auc_trapezoidal": float(auc(recall, precision)),
        "constant_score_average_precision": prevalence,
        "ap_divided_by_prevalence": ap / prevalence,
        "rank_budget": {
            "requested_alert_fraction": alert_fraction,
            "budget_rows": k,
            "tie_break": "Descending score, then ascending transaction_id; no labels used.",
            "boundary_score": boundary,
            "rows_tied_at_boundary": int(np.count_nonzero(scores == boundary)),
            **classification_counts(labels, selected),
        },
        "fixed_score_rule": {
            "cutoff": boundary,
            "comparator": comparator,
            "selection": "Chosen from validation scores alone to avoid exceeding the preset "
                "validation alert budget; boundary ties may reduce the number of alerts.",
            **classification_counts(labels, threshold_alerts),
        },
        "notes": [
            "Development results: validation also selects the number of trees, so these are not final test results.",
            "Average precision is sklearn's recall-weighted precision, not trapezoidal PR area.",
            "XGBoost aucpr used during early stopping uses its own continuous PR interpolation.",
            "Rank-budget evaluation selects top scores across this completed period; it is an offline ranking metric.",
            "The fixed score rule can produce a different alert rate on later periods; it is not a production capacity guarantee.",
            "Scores have not been probability-calibrated. No account-risk or mule classification is evaluated here.",
        ],
    }


def load_split(folder, split, schema):
    features = np.load(folder / f"{split}_X.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(folder / f"{split}_y.npy", mmap_mode="r", allow_pickle=False)
    expected_rows = schema["rows_by_split"][split]
    if features.shape != (expected_rows, schema["feature_count"]) or labels.shape != (expected_rows,):
        raise ValueError(f"{split}: unexpected feature/label shapes")
    if features.dtype != np.float32:
        raise ValueError(f"{split}: expected float32 features")
    positives = 0
    for start in range(0, len(labels), 100_000):
        y = labels[start:start + 100_000]
        if not np.isin(y, [0, 1]).all():
            raise ValueError(f"{split}: nonbinary labels")
        if not np.isfinite(features[start:start + 100_000]).all():
            raise ValueError(f"{split}: nonfinite feature values")
        positives += int(np.count_nonzero(y))
    if positives != schema["laundering_by_split"][split] or not 0 < positives < len(labels):
        raise ValueError(f"{split}: inconsistent labels or only one class")
    return features, labels


def train(args):
    started = time.perf_counter()
    output = args.output
    pending = output.with_name(output.name + ".building")
    if output.exists() or pending.exists():
        raise SystemExit(
            f"{output} or {pending} already exists. Nothing was overwritten. "
            "For a separate experiment use --output models/xgboost_baseline_v2. "
            "If a run failed, send the error before retrying."
        )
    schema_bytes = (args.features / "feature_schema.json").read_bytes()
    schema = json.loads(schema_bytes)
    names = schema["feature_names"]
    if len(names) != schema["feature_count"] or len(set(names)) != len(names):
        raise ValueError("Invalid feature names")

    print("Checking training and validation arrays...", flush=True)
    X_train, y_train = load_split(args.features, "train", schema)
    X_validation, y_validation = load_split(args.features, "validation", schema)
    ids = np.load(args.features / "validation_transaction_ids.npy", mmap_mode="r", allow_pickle=False)
    if ids.shape != y_validation.shape:
        raise ValueError("Validation transaction IDs do not align")
    if math.floor(len(y_validation) * args.alert_fraction) < 1:
        raise ValueError("Validation data is too small for the requested alert budget")
    print(f"Training rows: {len(y_train):,}; validation rows: {len(y_validation):,}")
    print(f"Features: {len(names)}; CPU threads: {args.threads}")
    print("Creating compact training matrix (this stage may be quiet)...", flush=True)
    dtrain = xgb.QuantileDMatrix(
        X_train, label=y_train, feature_names=names,
        max_bin=args.max_bin, nthread=args.threads, missing=np.nan,
    )
    print("Creating validation matrix using the training bin boundaries...", flush=True)
    dvalidation = xgb.QuantileDMatrix(
        X_validation, label=y_validation, feature_names=names,
        ref=dtrain, max_bin=args.max_bin, nthread=args.threads, missing=np.nan,
    )
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "tree_method": "hist",
        "device": "cpu",
        "nthread": args.threads,
        "max_depth": 4,
        "max_bin": args.max_bin,
        "eta": 0.1,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "scale_pos_weight": 1.0,
        "base_score": float(np.mean(y_train)),
        "seed": 42,
        "verbosity": 1,
        "validate_parameters": True,
    }
    pending.mkdir(parents=True)
    run = {
        "project_title": "AI-Based Graph Analysis for Suspicious Transaction Detection and Account Risk Scoring",
        "experiment": "XGBoost transaction baseline v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": params,
        "maximum_boosting_rounds": args.rounds,
        "early_stopping_patience": args.patience,
        "selection_metric": "validation aucpr (XGBoost continuous PR interpolation)",
        "preset_validation_alert_fraction": args.alert_fraction,
        "feature_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "source_csv_sha256": schema["source_csv_sha256"],
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "label_balance": "Natural class distribution retained; no class weighting or resampling in this initial baseline.",
        "test_and_tail_used": False,
        "status": "running",
    }
    write_json(pending / "training_run.json", run)
    (pending / "feature_schema.json").write_bytes(schema_bytes)
    evaluations = {}
    early_stop = xgb.callback.EarlyStopping(
        rounds=args.patience, metric_name="aucpr", data_name="validation",
        maximize=True, save_best=True,
    )
    print("Training begins. Reporting validation metrics every 10 rounds...", flush=True)
    fit_start = time.perf_counter()
    booster = xgb.train(
        params, dtrain, num_boost_round=args.rounds,
        evals=[(dvalidation, "validation")], evals_result=evaluations,
        callbacks=[early_stop], verbose_eval=10,
    )
    fit_seconds = time.perf_counter() - fit_start
    # save_best=True discards trees after the best validation iteration.
    booster.save_model(pending / "model.ubj")
    scores = booster.predict(dvalidation)
    report = evaluate_validation(y_validation, scores, ids, args.alert_fraction)
    write_json(pending / "validation_metrics.json", report)
    np.save(pending / "validation_scores.npy", scores, allow_pickle=False)
    np.save(pending / "validation_transaction_ids.npy", ids, allow_pickle=False)
    write_json(pending / "alert_policy.json", {
        "score_cutoff": report["fixed_score_rule"]["cutoff"],
        "comparator": report["fixed_score_rule"]["comparator"],
        "selected_on": "validation scores",
        "validation_alert_fraction_target": args.alert_fraction,
        "model": "model.ubj",
        "calibrated_probabilities": False,
    })
    with (pending / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round_1_based", "validation_logloss", "validation_xgboost_aucpr"])
        for i, (loss, area) in enumerate(zip(evaluations["validation"]["logloss"],
                                           evaluations["validation"]["aucpr"]), 1):
            writer.writerow([i, loss, area])
    gains = booster.get_score(importance_type="gain")
    with (pending / "feature_importance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "average_split_gain"])
        writer.writerows(sorted(((name, float(gains.get(name, 0))) for name in names),
                                key=lambda pair: (-pair[1], pair[0])))
    rounds_run = len(evaluations["validation"]["aucpr"])
    run.update({
        "status": "complete",
        "best_round_1_based": int(booster.best_iteration) + 1,
        "saved_boosting_rounds": booster.num_boosted_rounds(),
        "rounds_evaluated": rounds_run,
        "best_validation_xgboost_aucpr": float(booster.best_score),
        "round_limit_reached": rounds_run == args.rounds,
        "training_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "importance_note": "Split-gain importance is descriptive; it is not SHAP or causal evidence.",
    })
    write_json(pending / "training_run.json", run)
    pending.rename(output)

    budget = report["rank_budget"]
    print("\nBASELINE TRAINING COMPLETE")
    print("Results below are VALIDATION results, not final test performance.")
    print(f"Best round: {run['best_round_1_based']} / {rounds_run} evaluated")
    print(f"Validation average precision (AP): {report['average_precision']:.6f}")
    print(f"Validation trapezoidal PR-AUC: {report['pr_auc_trapezoidal']:.6f}")
    print(f"Constant-score AP reference: {report['constant_score_average_precision']:.6f}")
    print(f"Top {100 * args.alert_fraction:g}% alert budget: {budget['alerts']:,} alerts")
    print(f"Precision at budget: {budget['precision']:.4%}")
    print(f"Recall at budget: {budget['recall']:.4%}")
    print(f"True positives: {budget['true_positives']:,}; false positives: {budget['false_positives']:,}")
    print(f"False positives / 1,000 non-laundering: {budget['false_positives_per_1000_non_laundering']:.4f}")
    rule = report["fixed_score_rule"]
    print(f"Separate fixed score rule: score {rule['comparator']} {rule['cutoff']:.9g}")
    print(f"Fixed-rule validation alerts: {rule['alerts']:,}; recall: {rule['recall']:.4%}")
    print(f"Training time: {fit_seconds / 60:.1f} minutes")
    if run["round_limit_reached"]:
        print("Reached the round limit; inspect the validation history before deciding whether to extend training.")
    print(f"Model and reports saved to: {output}")
    print("Test and tail arrays were not opened. The original feature files were not modified.")
    return run, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--output", type=Path, default=Path("models/xgboost_baseline_v1"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-bin", type=int, default=64)
    parser.add_argument("--alert-fraction", type=float, default=0.005)
    args = parser.parse_args()
    if min(args.threads, args.rounds, args.patience) < 1 or args.max_bin < 2:
        parser.error("threads, rounds, patience must be positive; max-bin must be at least 2")
    if not 0 < args.alert_fraction <= 1:
        parser.error("alert-fraction must be in (0, 1]")
    train(args)
