"""Build a disk-backed transaction database and fixed chronological splits.

Run from the project folder: python src/prepare_data.py
Uses only Python's standard library. The input CSV is never modified.
This prepares data only: it does not train a model or generate graph features.
"""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path


RAW_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
PANDAS_HEADER = RAW_HEADER.copy()
PANDAS_HEADER[4] = "Account.1"
START = datetime(2022, 9, 1)
VALIDATION_START = datetime(2022, 9, 7)
TEST_START = datetime(2022, 9, 9)
TAIL_START = datetime(2022, 9, 11)
END = datetime(2022, 9, 19)
SPLITS = ("train", "validation", "test", "tail")


def choose_split(timestamp):
    if not START <= timestamp < END:
        raise ValueError("Timestamp is outside the audited September 1-18 period")
    if timestamp < VALIDATION_START:
        return "train"
    if timestamp < TEST_START:
        return "validation"
    if timestamp < TAIL_START:
        return "test"
    return "tail"


def source_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, output, batch_size=25_000):
    if not source.is_file():
        raise SystemExit(f"File not found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    database = output / "transactions.sqlite"
    manifest = output / "split_manifest.json"
    pending = output / "transactions.building.sqlite"
    if database.exists() or manifest.exists() or pending.exists():
        raise SystemExit(
            "Preparation output already exists. No files were overwritten. "
            "If a previous run failed, send its error for help; "
            "use --output with a new folder for an independent rebuild."
        )

    print("Recording the source file fingerprint...", flush=True)
    fingerprint = source_hash(source)
    initial_stat = source.stat()
    counts = Counter()
    positives = Counter()
    total = 0
    connection = sqlite3.connect(pending)
    try:
        # Bound SQLite's page cache to about 32 MiB and put sort work on disk.
        connection.execute("PRAGMA cache_size = -32768")
        connection.execute("PRAGMA temp_store = FILE")
        connection.execute("""
            CREATE TABLE transactions (
                transaction_id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                sender_bank TEXT NOT NULL,
                sender_account TEXT NOT NULL,
                receiver_bank TEXT NOT NULL,
                receiver_account TEXT NOT NULL,
                amount_received REAL NOT NULL CHECK(amount_received >= 0),
                receiving_currency TEXT NOT NULL,
                amount_paid REAL NOT NULL CHECK(amount_paid >= 0),
                payment_currency TEXT NOT NULL,
                payment_format TEXT NOT NULL,
                label INTEGER NOT NULL CHECK(label IN (0, 1)),
                split TEXT NOT NULL CHECK(split IN ('train','validation','test','tail'))
            )
        """)
        insert = "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        batch = []
        with source.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if header not in (RAW_HEADER, PANDAS_HEADER):
                raise ValueError(f"Unexpected columns: {header}")
            for record_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                try:
                    if len(row) != 11 or any(not value.strip() for value in row):
                        raise ValueError("Missing value or incorrect number of columns")
                    timestamp = datetime.fromisoformat(row[0].strip().replace("/", "-"))
                    if timestamp.tzinfo is not None:
                        raise ValueError("Unexpected timezone in timestamp")
                    split = choose_split(timestamp)
                    received, paid = float(row[5]), float(row[7])
                    if any(not math.isfinite(v) or v < 0 for v in (received, paid)):
                        raise ValueError("Invalid or negative amount")
                    if row[10].strip() not in ("0", "1"):
                        raise ValueError("Label must be 0 or 1")
                    label = int(row[10])
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"CSV data record {record_number}: {exc}") from exc

                # Preserve bank/account strings, including leading zeros.
                # IDs are source record positions, stable for this source fingerprint.
                batch.append((record_number, timestamp.isoformat(sep=" "),
                              row[1], row[2], row[3], row[4], received, row[6],
                              paid, row[8], row[9], label, split))
                counts[split] += 1
                positives[split] += label
                total += 1
                if len(batch) >= batch_size:
                    connection.executemany(insert, batch)
                    connection.commit()
                    batch.clear()
                    if total % 250_000 == 0:
                        print(f"Prepared {total:,} transactions...", flush=True)
            if batch:
                connection.executemany(insert, batch)
                connection.commit()

        if not total:
            raise ValueError("CSV contains no transactions")
        if any(counts[name] == 0 for name in ("train", "validation", "test")):
            raise ValueError("A main split is empty. The data needs review.")
        if (source.stat().st_size, source.stat().st_mtime_ns) != (
                initial_stat.st_size, initial_stat.st_mtime_ns):
            raise ValueError("Source changed while being processed")

        print("Building the timestamp index; this may take a few minutes...", flush=True)
        connection.execute("CREATE INDEX idx_time ON transactions(timestamp, transaction_id)")
        connection.execute("CREATE INDEX idx_split_time ON transactions(split, timestamp, transaction_id)")
        rows = connection.execute("""
            SELECT split, COUNT(*), SUM(label), MIN(timestamp), MAX(timestamp)
            FROM transactions GROUP BY split
        """).fetchall()
        by_split = {name: {"rows": n, "laundering": p, "non_laundering": n-p,
                           "first_timestamp": lo, "last_timestamp": hi}
                    for name, n, p, lo, hi in rows}
        if sum(info["rows"] for info in by_split.values()) != total:
            raise RuntimeError("Database row-count verification failed")
        for name in SPLITS:
            if by_split.get(name, {}).get("laundering", 0) != positives[name]:
                raise RuntimeError("Database label-count verification failed")

        metadata = {
            "schema_version": 1,
            "source_file": source.name,
            "source_sha256": fingerprint,
            "source_size_bytes": initial_stat.st_size,
            "total_rows": total,
            "splits": by_split,
            "cutoffs_in_source_time": {
                "train": [str(START), str(VALIDATION_START)],
                "validation": [str(VALIDATION_START), str(TEST_START)],
                "test": [str(TEST_START), str(TAIL_START)],
                "tail": [str(TAIL_START), str(END)],
            },
            "boundary_rule": "Start inclusive, end exclusive; timestamps have no supplied timezone.",
            "selection_reason": "Main benchmark covers dense September 1-10 activity. "
                "September 11-18 has sparse activity and a sharply different label rate "
                "in the full-file audit, so it is retained as a separate tail evaluation.",
            "scope": "Split creation only. No resampling, duplicate removal, feature fitting, or training.",
            "currency_policy": "Original currency strings retained. No exchange conversion applied.",
            "account_identity": "Use (bank, account) pairs; transaction IDs are not model features.",
            "target": "label is derived from Is Laundering; never use it as a predictor.",
            "ordered_read": "SELECT * FROM transactions ORDER BY timestamp, transaction_id",
            "history_rule": "History features must use events strictly earlier than each scored timestamp. "
                "A transaction_id tie-break is only for reproducible output; it does not establish "
                "causal order among same-time events. Fit transforms on training rows only.",
            "evaluation_rule": "Tune on validation only. Reserve test and tail labels for final evaluation. "
                "Report tail results separately; do not pool them into the main headline metrics.",
        }
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("manifest", json.dumps(metadata)))
        connection.commit()
    finally:
        connection.close()

    pending.rename(database)
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\nPREPARATION COMPLETE")
    print(f"Total transactions: {total:,}")
    for name in SPLITS:
        info = by_split.get(name, {"rows": 0, "laundering": 0})
        print(f"{name}: {info['rows']:,} transactions; {info['laundering']:,} laundering")
    print(f"Database size: {database.stat().st_size / 1024**2:.1f} MiB")
    print(f"Database: {database}")
    print(f"Split record: {manifest}")
    print("Original CSV unchanged. No model has been trained yet.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/HI-Small_Trans.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=25_000)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    prepare(args.input, args.output, args.batch_size)
