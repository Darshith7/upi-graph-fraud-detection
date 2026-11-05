"""Explanation FIDELITY test for the GNNExplainer (AMLSim UPI model).

A good explanation must point at edges that actually matter to the model. For high-scoring true
laundering transactions in the unseen test period we take the candidate prior transactions touching
the sender/receiver and measure how much the GraphSAGE probability DROPS when we delete

  * the edges the explainer ranks most important (top-k),
  * k random candidate edges (averaged over many draws),
  * the k most recent candidate edges (a trivial heuristic).

Faithful explanations => top-k deletion hurts the score clearly more than random deletion.
Reported with a paired Wilcoxon signed-rank test.  Uses only the deployed checkpoint.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_amlsim_challenges import AMLSimGraphSAGE, FEATURE_SETS, extract_scenario_features  # noqa: E402
from explainability import GNNExplainer  # noqa: E402


def logit(model, node_x, ei, feat, s, d):
    """Raw logit: probabilities saturate near 1 for confident cases and hide real changes."""
    with torch.no_grad():
        return float(model(node_x, ei, feat, torch.tensor([s]), torch.tensor([d])).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--random-draws", type=int, default=30)
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "explainability"))
    args = ap.parse_args()

    ckpt = torch.load(ROOT / "models" / "amlsim_models" / "graphsage_amlsim.pt", weights_only=False)
    fs = ckpt.get("feature_set", "cumulative")
    names = list(ckpt.get("feature_names", FEATURE_SETS[fs]))
    model = AMLSimGraphSAGE(in_features=len(names), hidden_dim=32, out_dim=1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    node_x, n2i, unk = ckpt["node_x"], ckpt["node2idx"], ckpt["unknown_node_idx"]
    explainer = GNNExplainer(model, names, seed=42)

    csv = ROOT / "data" / "amlsim_challenges" / "baseline.csv"
    df = pd.read_csv(csv)
    _, X = extract_scenario_features(csv, feature_set=fs)
    bst = xgb.Booster(model_file=str(ROOT / "models" / "amlsim_models" / "xgboost_amlsim.ubj"))
    xs = bst.predict(xgb.DMatrix(X, feature_names=names))
    cand = np.where((df["timestamp_step"].values > 90) & (df["is_laundering"].values == 1))[0]
    # sample cases across the whole score range (not only the most confident ones, which saturate)
    cand = np.random.default_rng(1).permutation(cand)

    rng = np.random.default_rng(0)
    rows = []
    for idx in cand:
        if len(rows) >= args.cases:
            break
        r = df.iloc[int(idx)]
        t = int(r["timestamp_seconds"])
        prior = df[(df["timestamp_seconds"] < t) & ((df["sourceNodeId"] == r["sourceNodeId"]) | (df["targetNodeId"] == r["targetNodeId"])
                                                    | (df["sourceNodeId"] == r["targetNodeId"]) | (df["targetNodeId"] == r["sourceNodeId"]))].tail(10)
        if len(prior) < max(4, args.k + 2):
            continue
        info = [{"src_idx": n2i.get(int(a), unk), "dst_idx": n2i.get(int(b), unk), "ts": int(c)}
                for a, b, c in zip(prior["sourceNodeId"], prior["targetNodeId"], prior["timestamp_seconds"])]
        s_i, d_i = n2i.get(int(r["sourceNodeId"]), unk), n2i.get(int(r["targetNodeId"]), unk)
        feat = torch.from_numpy(X[int(idx)]).float().unsqueeze(0)
        src = torch.tensor([e["src_idx"] for e in info])
        dst = torch.tensor([e["dst_idx"] for e in info])
        full_ei = torch.stack([src, dst])
        p_full = logit(model, node_x, full_ei, feat, s_i, d_i)

        w = explainer.edge_weights(node_x, full_ei, feat, s_i, d_i, info)

        def drop(removed):
            keep = [i for i in range(len(info)) if i not in set(removed)]
            ei = torch.stack([src[keep], dst[keep]]) if keep else torch.empty((2, 0), dtype=torch.long)
            return p_full - logit(model, node_x, ei, feat, s_i, d_i)

        top = list(np.argsort(-w)[: args.k])
        recent = list(np.argsort([-e["ts"] for e in info])[: args.k])
        rand = np.mean([drop(list(rng.choice(len(info), args.k, replace=False))) for _ in range(args.random_draws)])
        rows.append({"p_full": p_full, "drop_top_k": drop(top), "drop_random_k": float(rand), "drop_most_recent_k": drop(recent)})

    d = pd.DataFrame(rows)
    top, rnd, rec = d["drop_top_k"].values, d["drop_random_k"].values, d["drop_most_recent_k"].values
    stat_r = wilcoxon(top, rnd, alternative="greater") if np.any(top != rnd) else None
    stat_m = wilcoxon(top, rec, alternative="greater") if np.any(top != rec) else None
    report = {
        "cases": int(len(d)), "k": args.k, "mean_full_logit": round(float(d["p_full"].mean()), 4),
        "mean_abs_logit_change_random_k": round(float(np.abs(rnd).mean()), 5),
        "mean_logit_drop": {"explainer_top_k": round(float(top.mean()), 4), "random_k": round(float(rnd.mean()), 4),
                                  "most_recent_k": round(float(rec.mean()), 4)},
        "cases_where_top_k_beats_random": round(float((top > rnd).mean()), 4),
        "wilcoxon_top_vs_random_p": None if stat_r is None else float(stat_r.pvalue),
        "wilcoxon_top_vs_most_recent_p": None if stat_m is None else float(stat_m.pvalue),
        "faithful": bool(top.mean() > rnd.mean() and (stat_r is not None and stat_r.pvalue < 0.05)),
        "note": "drop = logit(all candidate edges) - logit(after removing edges); larger means the removed edges mattered more"
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "explanation_fidelity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
