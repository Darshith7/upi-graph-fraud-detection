"""Create past-only transaction features from the prepared SQLite database.

Run from the project folder: python src/build_features.py
Requires numpy. Writes disk-backed .npy arrays; no model is trained.
All events at a timestamp are scored before that timestamp updates history.
"""

import argparse
from array import array
from collections import Counter
import csv
from datetime import datetime
from itertools import groupby
import json
import math
from pathlib import Path
import sqlite3

import numpy as np


SPLITS = ("train", "validation", "test", "tail")
CATEGORICAL = ("receiving_currency", "payment_currency", "payment_format")
BASE_NAMES = [
    "log1p_amount_paid", "log1p_amount_received", "same_bank", "self_transfer",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
]
for role in ("sender", "receiver"):
    BASE_NAMES += [
        f"{role}_log1p_prior_out_count", f"{role}_log1p_prior_in_count",
        f"{role}_log1p_seconds_since_out", f"{role}_log1p_seconds_since_in",
        f"{role}_has_prior_out", f"{role}_has_prior_in",
    ]


def canonical(value, column):
    # Merge spelling differences such as 'US Dollar' and 'USDollar'.
    # This is text normalization only, never an exchange-rate conversion.
    text = value.strip().upper()
    return "".join(text.split()) if column.endswith("currency") else " ".join(text.split())


class AccountHistory:
    def __init__(self, writer):
        self.lookup = {}
        self.out_count = array("q")
        self.in_count = array("q")
        self.last_out = array("q")
        self.last_in = array("q")
        self.writer = writer

    def account_id(self, bank, account):
        key = (bank, account)
        existing = self.lookup.get(key)
        if existing is not None:
            return existing
        index = len(self.lookup)
        self.lookup[key] = index
        self.out_count.append(0)
        self.in_count.append(0)
        self.last_out.append(-1)
        self.last_in.append(-1)
        self.writer.writerow([index, bank, account])
        return index

    def features(self, index, seconds):
        outgoing, incoming = self.out_count[index], self.in_count[index]
        return [
            math.log1p(outgoing), math.log1p(incoming),
            math.log1p(seconds - self.last_out[index]) if outgoing else 0.0,
            math.log1p(seconds - self.last_in[index]) if incoming else 0.0,
            float(outgoing > 0), float(incoming > 0),
        ]

    def apply_timestamp(self, updates, seconds):
        for index, (outgoing, incoming) in updates.items():
            self.out_count[index] += outgoing
            self.in_count[index] += incoming
            if outgoing:
                self.last_out[index] = seconds
            if incoming:
                self.last_in[index] = seconds


def build(database, output):
    database, output = Path(database), Path(output)
    if not database.is_file():
        raise SystemExit(f"Database not found: {database}")
    pending = output.with_name(output.name + ".building")
    if output.exists() or pending.exists():
        raise SystemExit(
            f"{output} or {pending} already exists. Nothing was overwritten. "
            "If a previous run failed, send its error before retrying; "
            "--output can select a new folder for an independent run."
        )

    # A read-only connection protects the prepared transaction database.
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA cache_size = -32768")
    connection.execute("PRAGMA temp_store = FILE")
    maps, positions = {}, Counter()
    try:
        source_manifest = json.loads(connection.execute(
            "SELECT value FROM metadata WHERE key='manifest'"
        ).fetchone()[0])
        counts = dict(connection.execute("SELECT split, COUNT(*) FROM transactions GROUP BY split"))
        if set(counts) - set(SPLITS) or any(not counts.get(s) for s in SPLITS[:3]):
            raise ValueError("Unexpected or empty main data split")
        if sum(counts.values()) != source_manifest["total_rows"]:
            raise ValueError("Database counts do not match its preparation record")

        print("Learning category names from training transactions only...", flush=True)
        vocabularies, encoders, offsets = {}, {}, {}
        names = BASE_NAMES.copy()
        for column in CATEGORICAL:
            # column names come solely from the fixed CATEGORICAL tuple above.
            values = connection.execute(
                f"SELECT DISTINCT {column} FROM transactions WHERE split='train'"
            )
            vocabulary = sorted({canonical(row[0], column) for row in values})
            if len(vocabulary) > 500:
                raise ValueError(f"Unexpectedly many categories in {column}; review data first")
            vocabularies[column] = vocabulary
            encoders[column] = {value: i + 1 for i, value in enumerate(vocabulary)}
            offsets[column] = len(names)
            names.append(f"{column}__unknown")
            names.extend(f"{column}__known_{i}" for i in range(1, len(vocabulary) + 1))

        pending.mkdir(parents=True)
        estimated = sum(counts.values()) * (len(names) * 4 + 1 + 8 + 8 + 16)
        print(f"Feature columns: {len(names)}")
        print(f"Array storage: approximately {estimated / 1024**2:.0f} MiB plus account map.")
        for split in SPLITS:
            n = counts.get(split, 0)
            specifications = {
                "X": (np.float32, (n, len(names))),
                "y": (np.uint8, (n,)),
                "transaction_ids": (np.int64, (n,)),
                "time_seconds": (np.int64, (n,)),
                "edges": (np.int64, (n, 2)),
            }
            maps[split] = {
                name: np.lib.format.open_memmap(
                    pending / f"{split}_{name}.npy", mode="w+", dtype=dtype, shape=shape
                ) for name, (dtype, shape) in specifications.items()
            }

        unknown = {s: Counter({c: 0 for c in CATEGORICAL}) for s in SPLITS}
        positive_counts = Counter()
        query = """
            SELECT transaction_id, timestamp, sender_bank, sender_account,
                   receiver_bank, receiver_account, amount_received, receiving_currency,
                   amount_paid, payment_currency, payment_format, label, split
            FROM transactions ORDER BY timestamp, transaction_id
        """
        cursor = connection.execute(query)
        total = 0
        origin = None
        with (pending / "account_index.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["account_index", "bank", "account"])
            history = AccountHistory(writer)
            for timestamp, group in groupby(cursor, key=lambda row: row[1]):
                dt = datetime.fromisoformat(timestamp)
                if origin is None:
                    origin = dt
                seconds = int((dt - origin).total_seconds())
                hour_angle = 2 * math.pi * (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400
                weekday_angle = 2 * math.pi * dt.weekday() / 7
                time_features = [math.sin(hour_angle), math.cos(hour_angle),
                                 math.sin(weekday_angle), math.cos(weekday_angle)]
                # Aggregate updates by account, without retaining all rows in a minute.
                # History remains frozen until every row at this timestamp is emitted.
                updates = {}
                for row in group:
                    (tx_id, _, sb, sa, rb, ra, received, receiving_currency,
                     paid, payment_currency, payment_format, label, split) = row
                    sender = history.account_id(sb, sa)
                    receiver = history.account_id(rb, ra)
                    values = [math.log1p(paid), math.log1p(received), float(sb == rb),
                              float(sender == receiver)] + time_features
                    values += history.features(sender, seconds)
                    values += history.features(receiver, seconds)
                    vector = np.zeros(len(names), dtype=np.float32)
                    vector[:len(BASE_NAMES)] = values
                    categories = (receiving_currency, payment_currency, payment_format)
                    for column, value in zip(CATEGORICAL, categories):
                        index = encoders[column].get(canonical(value, column), 0)
                        vector[offsets[column] + index] = 1.0
                        if index == 0:
                            unknown[split][column] += 1

                    position = positions[split]
                    target = maps[split]
                    target["X"][position] = vector
                    # Outcome labels and graph identifiers are separate from X.
                    target["y"][position] = label
                    target["transaction_ids"][position] = tx_id
                    target["time_seconds"][position] = seconds
                    target["edges"][position] = (sender, receiver)
                    positions[split] += 1
                    positive_counts[split] += label
                    updates.setdefault(sender, [0, 0])[0] += 1
                    updates.setdefault(receiver, [0, 0])[1] += 1
                    total += 1
                    if total % 250_000 == 0:
                        print(f"Built features for {total:,} transactions...", flush=True)
                    if total % 500_000 == 0:
                        for arrays in maps.values():
                            for matrix in arrays.values():
                                matrix.flush()
                history.apply_timestamp(updates, seconds)

        for split in SPLITS:
            if positions[split] != counts.get(split, 0):
                raise RuntimeError(f"Feature count mismatch for {split}")
            expected = source_manifest["splits"].get(split, {}).get("laundering", 0)
            if positive_counts[split] != expected:
                raise RuntimeError(f"Label count mismatch for {split}")

        schema = {
            "version": 1,
            "source_csv_sha256": source_manifest["source_sha256"],
            "split_manifest": source_manifest,
            "feature_names": names,
            "feature_count": len(names),
            "dtype": "float32",
            "rows_by_split": {s: positions[s] for s in SPLITS},
            "laundering_by_split": {s: positive_counts[s] for s in SPLITS},
            "account_count": len(history.lookup),
            "category_vocabularies": vocabularies,
            "category_encoding": "One-hot: unknown=0, known_i uses vocabulary[i-1]. Fitted on training only.",
            "unknown_category_counts": {s: dict(unknown[s]) for s in SPLITS},
            "time_origin_in_source_timezone": str(origin),
            "time_units": "Seconds since time_origin; source timezone is unspecified.",
            "history_policy": "Counts and recency use strictly earlier timestamps. All rows at the "
                "same timestamp see the same prior history. Updates never use labels.",
            "evaluation_policy": "Sequential scoring: prior unlabeled validation/test events may "
                "update history for later events. The model itself is not refitted on those events.",
            "currency_policy": "Whitespace removed and text uppercased for currency categories. "
                "Amounts stay in their original currencies; no cross-currency totals are computed.",
            "amount_availability": "Both paid and received amounts are used for retrospective "
                "transaction scoring. A pre-transfer demo must omit any value not yet known.",
            "features_exclude": ["labels", "raw account IDs", "transaction IDs", "split", "future history"],
            "history_scope": "Cumulative incoming/outgoing transaction counts and recency. "
                "Not unique-neighbor degrees; no rolling windows, pair counts, or graph embeddings yet.",
            "cold_start": "Zero counts and has_prior_* flags identify accounts without prior events.",
            "label_balance": "Every source transaction retained; no oversampling or downsampling.",
            "files": "For each split: X inputs, y targets, transaction_ids, time_seconds, edges. "
                "Row i aligns across these five arrays. edges[:,0] is sender and edges[:,1] receiver.",
        }
        (pending / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    finally:
        for arrays in maps.values():
            for matrix in arrays.values():
                matrix.flush()
                matrix._mmap.close()
        connection.close()

    pending.rename(output)
    print("\nFEATURE BUILD COMPLETE")
    print(f"Features per transaction: {len(names)}")
    print(f"Accounts indexed: {len(history.lookup):,}")
    for split in SPLITS:
        print(f"{split}: {positions[split]:,} rows; {positive_counts[split]:,} laundering")
    print("Unknown category counts:", {s: dict(unknown[s]) for s in SPLITS})
    print(f"Saved to: {output}")
    print("Labels are stored separately from inputs. No model trained yet.")
    return schema


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/processed/transactions.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("data/features"))
    args = parser.parse_args()
    build(args.database, args.output)
