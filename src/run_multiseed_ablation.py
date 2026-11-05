"""Multi-seed + feature-set ablation for the AMLSim experiments.

Runs src/evaluate_amlsim_challenges.py once per (feature_set, seed) in an isolated scratch
directory (the deployed model in models/amlsim_models is NOT touched) and aggregates the results
into mean +/- std, so conclusions do not rest on a single training run.

    python src/run_multiseed_ablation.py --seeds 42 43 44 45 46 --feature-sets windowed cumulative
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ["Baseline (Standard AMLSim)", "Challenge 1 (Disguised Suspicious)",
             "Challenge 2 (Benign Lookalike Hubs)", "Challenge 3 (Partial History 50%)"]


def run_one(feature_set: str, seed: int, workdir: Path) -> dict:
    out = workdir / f"{feature_set}_seed{seed}"
    cmd = [sys.executable, "-u", str(ROOT / "src" / "evaluate_amlsim_challenges.py"),
           "--feature-set", feature_set, "--seed", str(seed),
           "--output-dir", str(out / "reports"), "--models-dir", str(out / "models")]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", PYTHONWARNINGS="ignore")
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  {feature_set} seed {seed}: done in {time.time() - t0:.0f}s", flush=True)
    return json.loads((out / "reports" / "challenge_metrics.json").read_text(encoding="utf-8"))


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--feature-sets", nargs="+", default=["windowed", "cumulative"])
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "amlsim_challenges"))
    ap.add_argument("--name", default="multiseed_ablation", help="output file stem (so extra variants do not overwrite)")
    args = ap.parse_args()

    results = {}
    with tempfile.TemporaryDirectory() as td:
        for fs in args.feature_sets:
            results[fs] = [run_one(fs, s, Path(td)) for s in args.seeds]

    summary = {}
    for fs, runs in results.items():
        summary[fs] = {}
        for sc in SCENARIOS:
            rows = [r["scenarios"][sc] for r in runs]
            pol = [r["deployed_blend"]["deployed_alert_policy"] for r in rows]
            stat = [r["deployed_blend"]["deployed_threshold_policy"] for r in rows]
            summary[fs][sc] = {
                "xgboost_ap": ms([r["xgboost"]["average_precision"] for r in rows]),
                "graphsage_ap": ms([r["graphsage"]["average_precision"] for r in rows]),
                "blend_ap": ms([r["deployed_blend"]["average_precision"] for r in rows]),
                "blend_minus_xgb_ap": ms([r["blend_minus_xgboost_ap"]["mean"] for r in rows]),
                "policy_precision": ms([p["precision"] for p in pol]),
                "policy_recall": ms([p["recall"] for p in pol]),
                "policy_alert_rate": ms([p["alert_rate"] for p in pol]),
                "static_cutoff_alert_rate": ms([p["alert_rate"] for p in stat]),
                "benign_hub_alert_fraction": ms([p["lookalike_hub_false_positives"] / max(1, r["lookalike_hub_events"])
                                                 for p, r in zip(pol, rows)]) if rows[0]["lookalike_hub_events"] else None,
                "prevalence_1pct_precision": ms([r["deployed_blend"]["prevalence_adjusted"]["0.01"]["policy_precision"] for r in rows]),
                "prevalence_0.1pct_precision": ms([r["deployed_blend"]["prevalence_adjusted"]["0.001"]["policy_precision"] for r in rows]),
            }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(
        json.dumps({"seeds": args.seeds, "feature_sets": args.feature_sets, "summary": summary}, indent=2), encoding="utf-8")

    def cell(d, pct=False):
        if d is None:
            return "n/a"
        return f"{d['mean']:.1%} +/- {d['std']:.1%}" if pct else f"{d['mean']:.4f} +/- {d['std']:.4f}"

    md = ["# AMLSim multi-seed ablation (mean +/- std over seeds " + ", ".join(map(str, args.seeds)) + ")", "",
          "Each row retrains XGBoost and GraphSAGE from scratch; thresholds and blend weights are re-tuned on validation each time.", "",
          "| Features | Scenario | XGBoost AP | GraphSAGE AP | Blend AP | Blend - XGB | Alert rate (policy) | Precision | Recall | Static-cutoff alert rate | Benign hubs alerted |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fs in args.feature_sets:
        for sc in SCENARIOS:
            s = summary[fs][sc]
            md.append(f"| {fs} | {sc} | {cell(s['xgboost_ap'])} | {cell(s['graphsage_ap'])} | {cell(s['blend_ap'])} | "
                      f"{cell(s['blend_minus_xgb_ap'])} | {cell(s['policy_alert_rate'], True)} | {cell(s['policy_precision'], True)} | "
                      f"{cell(s['policy_recall'], True)} | {cell(s['static_cutoff_alert_rate'], True)} | {cell(s['benign_hub_alert_fraction'], True)} |")
    md += ["", "## Precision at realistic base rates (deployed policy, negatives re-weighted)", "",
           "| Features | Scenario | Precision @ 1% prevalence | Precision @ 0.1% prevalence |", "|---|---|---|---|"]
    for fs in args.feature_sets:
        for sc in SCENARIOS:
            s = summary[fs][sc]
            md.append(f"| {fs} | {sc} | {cell(s['prevalence_1pct_precision'], True)} | {cell(s['prevalence_0.1pct_precision'], True)} |")
    (out_dir / f"{args.name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Saved {out_dir / (args.name + '.json')}")


if __name__ == "__main__":
    main()
