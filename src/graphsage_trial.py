"""Small CPU GraphSAGE training trial: speed/memory check, not evaluation.

Run from the project folder: python src/graphsage_trial.py
Keep prepare_graph.py beside this file in src/. Requires the packages already
installed: numpy, scikit-learn, torch, torch-geometric.
Default: 100 batches of 256 uniformly sampled training transactions.
Validation/test/tail labels and feature matrices are never opened.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import SAGEConv

from prepare_graph import TemporalGraphIndex


HISTORY_SUFFIXES = ["log1p_prior_out_count", "log1p_prior_in_count",
                    "log1p_seconds_since_out", "log1p_seconds_since_in",
                    "has_prior_out", "has_prior_in"]
BASE_NAMES = ["log1p_amount_paid", "log1p_amount_received", "same_bank", "self_transfer",
              "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
BASE_NAMES += [f"{role}_{suffix}" for role in ("sender", "receiver") for suffix in HISTORY_SUFFIXES]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def peak_memory_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 ** 2) if platform.system() == "Darwin" else peak / 1024


def fit_normalizer(X, names):
    if names[:20] != BASE_NAMES:
        raise ValueError("Unexpected base feature layout; keep the original feature schema")
    transaction_scaler, node_scaler = StandardScaler(), StandardScaler()
    print("Fitting input scaling on training rows only...", flush=True)
    for start in range(0, len(X), 100_000):
        block = np.asarray(X[start:start + 100_000, :20])
        if not np.isfinite(block).all():
            raise ValueError("Training contains nonfinite values")
        transaction_scaler.partial_fit(block)
        # One shared scaler for account histories; sender and receiver share the same units.
        node_scaler.partial_fit(block[:, 8:14])
        node_scaler.partial_fit(block[:, 14:20])
        if start and start % 1_000_000 == 0:
            print(f"Scaling: scanned {start:,} training rows...", flush=True)
    return {
        "transaction_mean": transaction_scaler.mean_.tolist(),
        "transaction_scale": transaction_scaler.scale_.tolist(),
        "node_mean": node_scaler.mean_.tolist(),
        "node_scale": node_scaler.scale_.tolist(),
        "clip": 10.0,
        "fit_split": "train",
        "policy": "First 20 transaction features and six node-history features standardized "
                  "using training statistics, then clipped to [-10,10]. One-hot columns unchanged.",
    }


class GraphBatchBuilder:
    """One-hop graphs whose virtual nodes are keyed by (account, cutoff).

    Sharing an account across DIFFERENT cutoffs is forbidden. Roots that have
    the SAME account and cutoff may share their identical history and messages.
    Incoming/outgoing branches separately aggregate counterparties of prior
    transfers. Neighbor node features are also measured strictly before cutoff.
    """

    def __init__(self, graph, X, edges, times, normalizer, fanout=5):
        self.graph, self.X, self.edges, self.times = graph, X, edges, times
        self.fanout = fanout
        self.tx_mean = np.array(normalizer["transaction_mean"], dtype=np.float32)
        self.tx_scale = np.array(normalizer["transaction_scale"], dtype=np.float32)
        self.node_mean = np.array(normalizer["node_mean"], dtype=np.float32)
        self.node_scale = np.array(normalizer["node_scale"], dtype=np.float32)
        self.clip = float(normalizer["clip"])

    def build(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        nodes, accounts, cutoffs = {}, [], []

        def virtual_node(account, cutoff):
            key = (int(account), int(cutoff))
            if key not in nodes:
                nodes[key] = len(accounts)
                accounts.append(key[0])
                cutoffs.append(key[1])
            return nodes[key]

        root_senders, root_receivers = [], []
        for row in rows:
            cutoff = int(self.times[row])
            sender, receiver = self.edges[row]
            root_senders.append(virtual_node(sender, cutoff))
            root_receivers.append(virtual_node(receiver, cutoff))
        root_count = len(accounts)
        messages = {direction: [] for direction in ("incoming", "outgoing")}
        evidence_times = {direction: [] for direction in messages}
        evidence_ids = {direction: [] for direction in messages}
        for root in range(root_count):
            account, cutoff = accounts[root], cutoffs[root]
            for direction in messages:
                prior = self.graph.recent_neighbors(account, cutoff, limit=self.fanout,
                                                    direction=direction, exclude_self=True)
                if not np.all(prior["times"] < cutoff):
                    raise RuntimeError("Graph message contains a current/future transaction")
                for neighbor, edge_time, tx_id in zip(prior["neighbors"], prior["times"],
                                                     prior["transaction_ids"]):
                    node = virtual_node(int(neighbor), cutoff)
                    # PyG messages flow neighbor -> root. Direction remains explicit via branches.
                    messages[direction].append((node, root))
                    evidence_times[direction].append(int(edge_time))
                    evidence_ids[direction].append(int(tx_id))

        by_cutoff = defaultdict(list)
        for i, cutoff in enumerate(cutoffs):
            by_cutoff[cutoff].append(i)
        history = np.zeros((len(accounts), 6), dtype=np.float32)
        account_array = np.asarray(accounts, dtype=np.int64)
        for cutoff, positions in by_cutoff.items():
            positions = np.asarray(positions, dtype=np.int64)
            history[positions] = self.graph.history_features(account_array[positions], cutoff)

        # Independent check against the original endpoint features in every trial batch.
        transaction_features = np.array(self.X[rows], dtype=np.float32, copy=True)
        np.testing.assert_allclose(history[root_senders], transaction_features[:, 8:14],
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(history[root_receivers], transaction_features[:, 14:20],
                                   rtol=1e-5, atol=1e-5)
        node_features = np.clip((history - self.node_mean) / self.node_scale,
                               -self.clip, self.clip).astype(np.float32)
        transaction_features[:, :20] = np.clip(
            (transaction_features[:, :20] - self.tx_mean) / self.tx_scale,
            -self.clip, self.clip,
        )
        if not np.isfinite(node_features).all() or not np.isfinite(transaction_features).all():
            raise ValueError("Nonfinite model inputs")

        def edge_tensor(direction):
            edge = np.asarray(messages[direction], dtype=np.int64).reshape(-1, 2).T.copy()
            return torch.from_numpy(edge)

        return {
            "node_features": torch.from_numpy(node_features),
            "incoming_edges": edge_tensor("incoming"),
            "outgoing_edges": edge_tensor("outgoing"),
            "root_senders": torch.tensor(root_senders, dtype=torch.long),
            "root_receivers": torch.tensor(root_receivers, dtype=torch.long),
            "transaction_features": torch.from_numpy(transaction_features),
            # Audit fields below are NOT model inputs.
            "node_accounts": account_array,
            "node_cutoffs": np.asarray(cutoffs, dtype=np.int64),
            "evidence_times": evidence_times,
            "evidence_transaction_ids": evidence_ids,
        }


class TransactionGraphSAGE(nn.Module):
    """One-hop, directional GraphSAGE plus a transaction classification head."""

    def __init__(self, feature_count, branch_width=16, head_width=64, prior=0.001):
        super().__init__()
        self.incoming = SAGEConv(6, branch_width, aggr="mean")
        self.outgoing = SAGEConv(6, branch_width, aggr="mean")
        self.head = nn.Sequential(
            nn.Linear(feature_count + 4 * branch_width, head_width),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(head_width, 1),
        )
        with torch.no_grad():
            self.head[-1].bias.fill_(math.log(prior / (1 - prior)))

    def forward(self, batch):
        history = batch["node_features"]
        incoming = self.incoming(history, batch["incoming_edges"])
        outgoing = self.outgoing(history, batch["outgoing_edges"])
        embeddings = torch.relu(torch.cat([incoming, outgoing], dim=1))
        edge_input = torch.cat([
            batch["transaction_features"], embeddings[batch["root_senders"]],
            embeddings[batch["root_receivers"]],
        ], dim=1)
        return self.head(edge_input).squeeze(1)


def run_trial(args):
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output, pending = args.output, args.output.with_name(args.output.name + ".building")
    if output.exists() or pending.exists():
        raise SystemExit(
            f"{output} or {pending} already exists. Nothing was overwritten. "
            "If a run failed, send the error before retrying. "
            "--output can select a new folder for a separate trial."
        )
    schema_bytes = (args.features / "feature_schema.json").read_bytes()
    schema = json.loads(schema_bytes)
    feature_hash = hashlib.sha256(schema_bytes).hexdigest()
    n = int(schema["rows_by_split"]["train"])
    X = np.load(args.features / "train_X.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(args.features / "train_y.npy", mmap_mode="r", allow_pickle=False)
    edges = np.load(args.features / "train_edges.npy", mmap_mode="r", allow_pickle=False)
    times = np.load(args.features / "train_time_seconds.npy", mmap_mode="r", allow_pickle=False)
    ids = np.load(args.features / "train_transaction_ids.npy", mmap_mode="r", allow_pickle=False)
    if (X.shape != (n, schema["feature_count"]) or labels.shape != (n,) or
            edges.shape != (n, 2) or times.shape != (n,) or ids.shape != (n,)):
        raise ValueError("Training arrays are inconsistent with the schema")
    positives = int(np.count_nonzero(labels))
    if not np.isin(labels, [0, 1]).all() or positives != schema["laundering_by_split"]["train"]:
        raise ValueError("Training labels are inconsistent")
    if not 0 < positives < n:
        raise ValueError("Training needs examples of both classes")

    with TemporalGraphIndex(args.graph) as graph:
        if graph.metadata["feature_schema_sha256"] != feature_hash:
            raise ValueError("Graph index and features come from different builds")
        pending.mkdir(parents=True)
        scaler_started = time.perf_counter()
        normalizer = fit_normalizer(X, schema["feature_names"])
        scaler_seconds = time.perf_counter() - scaler_started
        normalizer["feature_schema_sha256"] = feature_hash
        write_json(pending / "normalization.json", normalizer)

        count = min(n, args.trial_batches * args.batch_size)
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(n, size=count, replace=False)
        # Selection uses row count and seed only, never labels.
        np.save(pending / "sampled_training_rows.npy", selected, allow_pickle=False)
        np.save(pending / "sampled_transaction_ids.npy", ids[selected], allow_pickle=False)
        builder = GraphBatchBuilder(graph, X, edges, times, normalizer, args.fanout)
        model = TransactionGraphSAGE(schema["feature_count"], prior=positives / n)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        objective = nn.BCEWithLogitsLoss()
        before = model.head[-1].weight.detach().clone()
        model.train()
        timings = []
        graph_seconds = learning_seconds = 0.0
        largest_nodes = largest_messages = 0
        completed = 0
        print(f"Trial: {count:,} training transactions, batch size {args.batch_size}.", flush=True)
        print(f"At most {args.fanout} incoming and {args.fanout} outgoing prior edges per root.")
        print("CPU training begins; this is a resource check, not a model-quality result.", flush=True)
        with (pending / "trial_batches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["batch", "rows", "laundering_rows", "loss", "graph_seconds",
                             "learning_seconds", "virtual_nodes", "messages"])
            for step, low in enumerate(range(0, count, args.batch_size), 1):
                batch_rows = selected[low:low + args.batch_size]
                batch_started = time.perf_counter()
                batch = builder.build(batch_rows)
                built = time.perf_counter()
                y = torch.from_numpy(np.array(labels[batch_rows], dtype=np.float32, copy=True))
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                loss = objective(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                learned = time.perf_counter()
                prepare_time, train_time = built - batch_started, learned - built
                graph_seconds += prepare_time
                learning_seconds += train_time
                completed += len(batch_rows)
                nodes = len(batch["node_accounts"])
                messages = batch["incoming_edges"].shape[1] + batch["outgoing_edges"].shape[1]
                largest_nodes = max(largest_nodes, nodes)
                largest_messages = max(largest_messages, messages)
                timings.append((len(batch_rows), learned - batch_started))
                writer.writerow([step, len(batch_rows), int(y.sum()), float(loss.detach()),
                                 prepare_time, train_time, nodes, messages])
                if step == 1 or step % 10 == 0 or completed == count:
                    elapsed = graph_seconds + learning_seconds
                    print(f"Batch {step}: {completed:,}/{count:,} rows; "
                          f"loss={float(loss.detach()):.6f}; "
                          f"average speed={completed / elapsed:.0f} rows/s", flush=True)

        if torch.equal(before, model.head[-1].weight.detach()):
            raise RuntimeError("Classifier parameters did not change during the trial")
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite model parameters")
        useful_timings = timings[10:] if len(timings) > 10 else timings
        speed = sum(size for size, _ in useful_timings) / sum(seconds for _, seconds in useful_timings)
        report = {
            "purpose": "Resource and correctness trial only; not a model comparison or final evaluation.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project_title": "AI-Based Graph Analysis for Suspicious Transaction Detection and Account Risk Scoring",
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "torch_geometric": torch_geometric.__version__, "numpy": np.__version__},
            "feature_schema_sha256": feature_hash,
            "source_csv_sha256": schema["source_csv_sha256"],
            "architecture": {"hops": 1, "incoming_branch_width": 16, "outgoing_branch_width": 16,
                             "head_width": 64, "dropout": 0.1, "transaction_features": schema["feature_count"],
                             "node_history_features": 6, "trainable_account_id_embeddings": False},
            "training": {"optimizer": "Adam", "learning_rate": 0.001,
                         "loss": "Unweighted BCEWithLogitsLoss", "class_resampling": False,
                         "gradient_norm_limit": 5.0, "device": "cpu", "threads": args.threads,
                         "seed": args.seed, "batch_size": args.batch_size, "fanout_per_direction": args.fanout},
            "all_training_rows": n,
            "sampled_training_rows": count,
            "sampled_laundering_rows": int(np.count_nonzero(labels[selected])),
            "batches_completed": len(timings),
            "sampling": "Uniform training-row sample without replacement; independent of labels.",
            "scaler_fit_seconds": scaler_seconds,
            "graph_construction_seconds": graph_seconds,
            "learning_seconds": learning_seconds,
            "steady_rows_per_second": speed,
            "rough_full_training_epoch_minutes": n / speed / 60,
            "estimate_limitations": "Linear extrapolation from this trial, excluding validation, "
                "checkpointing, and initial scaling. Actual full-run speed can differ.",
            "peak_process_rss_mib": peak_memory_mib(),
            "memory_note": "Process peak resident memory includes imports, scaling, and mapped file pages; "
                           "this is not total Mac memory usage or a guaranteed full-run bound.",
            "largest_batch_virtual_nodes": largest_nodes,
            "largest_batch_messages": largest_messages,
            "last_batch_training_loss": float(loss.detach()),
            "parameter_update_check": "passed",
            "strict_past_edge_checks": "passed for every sampled graph message",
            "endpoint_history_checks": "passed for every sampled training transaction",
            "temporal_isolation": "Virtual nodes keyed by (account, cutoff), preventing cross-time sharing.",
            "validation_test_tail_labels_loaded": False,
            "total_seconds": time.perf_counter() - started,
        }
        torch.save({"model_state_dict": model.state_dict(), "architecture": report["architecture"],
                    "normalizer": normalizer, "trial_only": True}, pending / "trial_model.pt")
        write_json(pending / "trial_report.json", report)

    pending.rename(output)
    print("\nGRAPHSAGE TRIAL COMPLETE")
    print(f"Training transactions sampled: {count:,}")
    print(f"Laundering transactions in sample: {report['sampled_laundering_rows']:,}")
    print(f"Batches completed: {len(timings)}")
    print(f"Steady processing speed: {speed:.1f} transactions/second")
    print(f"Peak process memory: {report['peak_process_rss_mib']:.1f} MiB")
    print(f"Largest batch: {largest_nodes:,} virtual nodes; {largest_messages:,} messages")
    print(f"Rough full training epoch estimate: {report['rough_full_training_epoch_minutes']:.1f} minutes")
    print("Estimate excludes validation and may change during a full run.")
    print("Parameter updates, strict-past edges, and endpoint history checks: passed")
    print("No validation/test/tail labels loaded. No detection-quality metrics computed.")
    print(f"Trial checkpoint and report saved to: {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--graph", type=Path, default=Path("data/temporal_graph"))
    parser.add_argument("--output", type=Path, default=Path("models/graphsage_trial_v1"))
    parser.add_argument("--trial-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fanout", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.trial_batches, args.batch_size, args.threads) < 1 or args.fanout < 0:
        parser.error("batches, batch-size, threads must be positive; fanout must be nonnegative")
    run_trial(args)
