"""Build a compact temporal graph index for later GraphSAGE training.

Run from the project folder: python src/prepare_graph.py
Requires numpy only. No labels are loaded. One existing validation feature
row is read to check that graph histories agree with the baseline features.
The index stores all periods, but every history query requires a cutoff and
returns ONLY edges with timestamp strictly less than that cutoff.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test", "tail")
DIRECTIONS = ("outgoing", "incoming")


def close_maps(arrays):
    for value in arrays:
        mapping = getattr(value, "_mmap", None)
        if mapping is not None and not mapping.closed:
            if value.mode != "r":
                value.flush()
            mapping.close()


class TemporalGraphIndex:
    """Read-only, timestamp-filtered adjacency and past account statistics.

    Node IDs use account_index.csv from build_features.py. Bank/account pairs
    remain distinct. Repeated transactions are retained as separate edges.
    history_features columns exactly match each six-column account-history
    block in build_features.py: log outgoing/incoming counts, log recency for
    outgoing/incoming events, and two history-presence flags.
    """

    def __init__(self, folder):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder / "graph_manifest.json").read_text())
        self.account_count = self.metadata["account_count"]
        self.arrays = {}
        for direction in DIRECTIONS:
            self.arrays[direction] = {
                name: np.load(self.folder / f"{direction}_{name}.npy",
                              mmap_mode="r", allow_pickle=False)
                for name in ("ptr", "neighbors", "times", "transaction_ids")
            }

    def _window(self, account, cutoff, direction):
        if direction not in DIRECTIONS:
            raise ValueError("direction must be incoming or outgoing")
        if not isinstance(account, (int, np.integer)) or not 0 <= account < self.account_count:
            raise ValueError("Unknown account index")
        if not isinstance(cutoff, (int, np.integer)):
            raise TypeError("cutoff must be integer seconds since the feature time origin")
        values = self.arrays[direction]
        start, end = int(values["ptr"][account]), int(values["ptr"][account + 1])
        # side='left' excludes every event at cutoff, including lower-ID events.
        stop = start + int(np.searchsorted(values["times"][start:end], cutoff, side="left"))
        return start, stop

    def recent_neighbors(self, account, cutoff, limit=5, direction="incoming", exclude_self=True):
        """Return up to limit recent prior edges, newest first.

        Parallel edges remain parallel; this is recent-transaction sampling,
        not uniform sampling of unique neighboring accounts. Self-transfers
        are excluded from sampled neighbors by default, but still count in
        history_features. A self node is handled by GraphSAGE's root pathway.
        """
        if not isinstance(limit, (int, np.integer)) or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        start, stop = self._window(account, cutoff, direction)
        values = self.arrays[direction]
        chosen = []
        high = stop
        while high > start and len(chosen) < limit:
            low = max(start, high - 128)
            positions = np.arange(high - 1, low - 1, -1, dtype=np.int64)
            if exclude_self:
                positions = positions[values["neighbors"][positions] != account]
            chosen.extend(positions[:limit - len(chosen)].tolist())
            high = low
        selected = np.asarray(chosen, dtype=np.int64)
        return {name: np.asarray(values[name][selected])
                for name in ("neighbors", "times", "transaction_ids")}

    def history_features(self, accounts, cutoff):
        accounts = np.asarray(accounts)
        if accounts.ndim != 1 or accounts.dtype.kind not in "iu":
            raise ValueError("accounts must be a one-dimensional integer array")
        result = np.zeros((len(accounts), 6), dtype=np.float32)
        for i, account in enumerate(accounts):
            for j, direction in enumerate(DIRECTIONS):
                start, stop = self._window(int(account), cutoff, direction)
                count = stop - start
                result[i, j] = np.log1p(count)
                if count:
                    previous = int(self.arrays[direction]["times"][stop - 1])
                    result[i, j + 2] = np.log1p(cutoff - previous)
                    result[i, j + 4] = 1.0
        return result

    def close(self):
        close_maps([a for direction in self.arrays.values() for a in direction.values()])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def build(features, output, batch_size=250_000):
    features, output = Path(features), Path(output)
    pending = output.with_name(output.name + ".building")
    if output.exists() or pending.exists():
        raise SystemExit(
            f"{output} or {pending} already exists; nothing was overwritten. "
            "If a run failed, send the error before retrying. "
            "For an independent build, --output can select a new folder."
        )
    schema_bytes = (features / "feature_schema.json").read_bytes()
    schema = json.loads(schema_bytes)
    account_count = int(schema["account_count"])
    if not 0 < account_count < np.iinfo(np.int32).max:
        raise ValueError("Account count is outside this compact index's supported range")
    counts = {s: int(schema["rows_by_split"][s]) for s in SPLITS}
    if any(n < 0 for n in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("Invalid transaction counts")
    total = sum(counts.values())
    pending.mkdir(parents=True)
    opened = []

    def new_array(name, dtype, shape):
        result = np.lib.format.open_memmap(pending / name, mode="w+", dtype=dtype, shape=shape)
        opened.append(result)
        return result

    try:
        edges = new_array("_work_edges.npy", np.int32, (total, 2))
        times = new_array("_work_times.npy", np.int64, (total,))
        ids = new_array("_work_ids.npy", np.int64, (total,))
        offset = 0
        previous_time = previous_id = None
        print("Reading chronological edge records; no labels are loaded...", flush=True)
        for split in SPLITS:
            n = counts[split]
            source = {}
            for name in ("edges", "time_seconds", "transaction_ids"):
                source[name] = np.load(features / f"{split}_{name}.npy", mmap_mode="r", allow_pickle=False)
                opened.append(source[name])
            if source["edges"].shape != (n, 2) or any(
                    source[name].shape != (n,) for name in ("time_seconds", "transaction_ids")):
                raise ValueError(f"{split}: unexpected edge array shape")
            if any(a.dtype.kind not in "iu" for a in source.values()):
                raise ValueError(f"{split}: graph arrays must contain integers")
            for low in range(0, n, batch_size):
                high = min(n, low + batch_size)
                edge = source["edges"][low:high]
                timestamp = source["time_seconds"][low:high]
                tx_ids = source["transaction_ids"][low:high]
                if np.any(edge < 0) or np.any(edge >= account_count):
                    raise ValueError(f"{split}: account index out of range")
                if np.any(timestamp < 0) or np.any(tx_ids <= 0):
                    raise ValueError(f"{split}: invalid time or transaction ID")
                backward = timestamp[1:] < timestamp[:-1]
                unordered_ties = (timestamp[1:] == timestamp[:-1]) & (tx_ids[1:] <= tx_ids[:-1])
                if np.any(backward | unordered_ties):
                    raise ValueError(f"{split}: records are not ordered by time then transaction ID")
                if previous_time is not None and (
                        int(timestamp[0]) < previous_time or
                        (int(timestamp[0]) == previous_time and int(tx_ids[0]) <= previous_id)):
                    raise ValueError("Chronological order failed at a batch/split boundary")
                previous_time, previous_id = int(timestamp[-1]), int(tx_ids[-1])
                destination = slice(offset + low, offset + high)
                edges[destination] = edge
                times[destination] = timestamp
                ids[destination] = tx_ids
            offset += n
            print(f"Loaded {split}: {n:,} edges", flush=True)
        if offset != total:
            raise RuntimeError("Combined edge count mismatch")

        for endpoint, direction in enumerate(DIRECTIONS):
            print(f"Indexing {direction} connections...", flush=True)
            # Stable grouping preserves the verified timestamp/ID ordering within accounts.
            order = np.argsort(edges[:, endpoint], kind="stable")
            degree = np.bincount(edges[:, endpoint], minlength=account_count)
            pointer = np.empty(account_count + 1, dtype=np.int64)
            pointer[0] = 0
            np.cumsum(degree, out=pointer[1:])
            if int(pointer[-1]) != total:
                raise RuntimeError("Graph pointer count mismatch")
            np.save(pending / f"{direction}_ptr.npy", pointer, allow_pickle=False)
            neighbor = new_array(f"{direction}_neighbors.npy", np.int32, (total,))
            edge_time = new_array(f"{direction}_times.npy", np.int64, (total,))
            tx_id = new_array(f"{direction}_transaction_ids.npy", np.int64, (total,))
            for low in range(0, total, batch_size):
                high = min(total, low + batch_size)
                positions = order[low:high]
                neighbor[low:high] = edges[positions, 1 - endpoint]
                edge_time[low:high] = times[positions]
                tx_id[low:high] = ids[positions]
            close_maps([neighbor, edge_time, tx_id])
            del order, degree, pointer

        metadata = {
            "version": 1,
            "source_csv_sha256": schema["source_csv_sha256"],
            "feature_schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
            "account_count": account_count,
            "transaction_count": total,
            "rows_by_split": counts,
            "time_origin_in_source_timezone": schema["time_origin_in_source_timezone"],
            "time_units": "Integer seconds since the feature time origin; source timezone is unspecified.",
            "account_mapping": "Same integer IDs as data/features/account_index.csv, based on (bank, account).",
            "labels_loaded": False,
            "cutoff_rule": "Every query returns only events with time < cutoff; equal-time events are excluded.",
            "stored_periods": "All four periods are indexed for reuse. Never pass raw unfiltered adjacency "
                "to training; use TemporalGraphIndex with each scored transaction's cutoff.",
            "directions": "outgoing neighbors are earlier recipients; incoming neighbors are earlier senders.",
            "sampling_policy": "recent_neighbors returns recent prior transactions, newest first. "
                "Repeated counterparties are allowed. Self-transfers are excluded by default from "
                "sampled neighbors but included in historical transaction counts.",
            "history_feature_order": ["log1p_prior_out_count", "log1p_prior_in_count",
                "log1p_seconds_since_out", "log1p_seconds_since_in", "has_prior_out", "has_prior_in"],
            "status": "Prepared graph index only; no graph model has been trained.",
        }
        (pending / "graph_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        close_maps(opened)

    for name in ("_work_edges.npy", "_work_times.npy", "_work_ids.npy"):
        (pending / name).unlink()

    # Check one real validation event against its already-built account features.
    # Validation labels are not read. This verifies consistency of both history implementations.
    check_arrays = []
    try:
        val_edges = np.load(features / "validation_edges.npy", mmap_mode="r", allow_pickle=False)
        val_times = np.load(features / "validation_time_seconds.npy", mmap_mode="r", allow_pickle=False)
        val_features = np.load(features / "validation_X.npy", mmap_mode="r", allow_pickle=False)
        check_arrays.extend([val_edges, val_times, val_features])
        if len(val_edges) == 0:
            raise ValueError("Validation split is empty")
        cutoff = int(val_times[0])
        sender, receiver = map(int, val_edges[0])
        names = schema["feature_names"]
        suffixes = metadata["history_feature_order"]
        with TemporalGraphIndex(pending) as index:
            histories = index.history_features(np.array([sender, receiver], dtype=np.int64), cutoff)
            for i, role in enumerate(("sender", "receiver")):
                columns = [names.index(f"{role}_{suffix}") for suffix in suffixes]
                np.testing.assert_allclose(histories[i], val_features[0, columns], rtol=1e-6, atol=1e-6)
                for direction in DIRECTIONS:
                    sampled = index.recent_neighbors((sender, receiver)[i], cutoff, direction=direction)
                    if not np.all(sampled["times"] < cutoff):
                        raise RuntimeError("A sampled edge violates the strict-past rule")
    finally:
        close_maps(check_arrays)

    metadata["first_validation_history_consistency_check"] = "passed"
    (pending / "graph_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pending.rename(output)
    size = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    print("\nTEMPORAL GRAPH PREPARATION COMPLETE")
    print(f"Accounts: {account_count:,}")
    print(f"Transactions: {total:,}")
    for split in SPLITS:
        print(f"{split}: {counts[split]:,} transactions")
    print("Incoming and outgoing history indexes: ready")
    print("First validation history consistency check: passed")
    print("Strict-past sample check: passed")
    print(f"Graph index size: {size / 1024**2:.1f} MiB")
    print(f"Saved to: {output}")
    print("No labels loaded and no graph model trained.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/features"))
    parser.add_argument("--output", type=Path, default=Path("data/temporal_graph"))
    parser.add_argument("--batch-size", type=int, default=250_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    build(args.features, args.output, args.batch_size)
