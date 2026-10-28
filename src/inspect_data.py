"""Audit IBM HI-Small transactions in batches, without changing the source CSV.

Run from the project folder: python src/inspect_data.py
Requires pandas and numpy. Reports describe the full source, not a training split.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


RAW_HEADER = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
COLUMNS = RAW_HEADER.copy()
COLUMNS[4] = "Account.1"


def audit(path, output, chunk_size):
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    # pandas may have already added .1 if a CSV was exported again.
    if header not in (RAW_HEADER, COLUMNS):
        raise SystemExit(f"Unexpected column layout. Send this output for review: {header}")

    missing = Counter({name: 0 for name in COLUMNS})
    labels = Counter()
    currencies = {name: Counter() for name in ("Receiving Currency", "Payment Currency")}
    amount_issues = {name: {"invalid_or_missing": 0, "negative": 0}
                     for name in ("Amount Received", "Amount Paid")}
    daily_rows, daily_positive, daily_bad_label = Counter(), Counter(), Counter()
    total = bad_times = inversions = 0
    earliest = latest = previous_time = None

    # Read IDs as text so leading zeros remain intact. Blank detection is explicit.
    reader = pd.read_csv(path, dtype=str, keep_default_na=False,
                         encoding="utf-8-sig", chunksize=chunk_size)
    for batch_number, frame in enumerate(reader, 1):
        total += len(frame)
        for name in COLUMNS:
            missing[name] += int(frame[name].str.strip().eq("").sum())

        target = frame["Is Laundering"].str.strip()
        labels.update(target.replace("", "<blank>").tolist())
        valid_label = target.isin(["0", "1"])
        times = pd.to_datetime(frame["Timestamp"].str.strip(),
                               format="%Y/%m/%d %H:%M", errors="coerce")
        bad_times += int(times.isna().sum())
        valid_times = times.dropna()
        if not valid_times.empty:
            first, last = valid_times.iloc[0], valid_times.iloc[-1]
            inversions += int(valid_times.diff().lt(pd.Timedelta(0)).sum())
            if previous_time is not None and first < previous_time:
                inversions += 1
            previous_time = last
            lo, hi = valid_times.min(), valid_times.max()
            earliest = lo if earliest is None else min(earliest, lo)
            latest = hi if latest is None else max(latest, hi)

        days = times.dt.strftime("%Y-%m-%d").fillna("INVALID_TIMESTAMP")
        daily_rows.update(days.tolist())
        daily_positive.update(days[target.eq("1")].tolist())
        daily_bad_label.update(days[~valid_label].tolist())

        for name, issues in amount_issues.items():
            values = pd.to_numeric(frame[name], errors="coerce")
            finite = np.isfinite(values)
            issues["invalid_or_missing"] += int((~finite).sum())
            issues["negative"] += int((finite & values.lt(0)).sum())
        for name in currencies:
            currencies[name].update(frame[name].tolist())
        if batch_number == 1 or batch_number % 10 == 0:
            print(f"Scanned {total:,} transactions...", flush=True)

    if total == 0:
        raise SystemExit("The CSV has a header but no transactions.")

    invalid_labels = total - labels["0"] - labels["1"]
    summary = {
        "source_file": path.name,
        "source_size_bytes": path.stat().st_size,
        "rows": total,
        "non_laundering_rows": labels["0"],
        "laundering_rows": labels["1"],
        "laundering_percent_of_all_rows": 100 * labels["1"] / total,
        "invalid_or_missing_labels": invalid_labels,
        "label_value_counts": dict(labels),
        "earliest_valid_timestamp": str(earliest) if earliest is not None else None,
        "latest_valid_timestamp": str(latest) if latest is not None else None,
        "invalid_or_missing_timestamps": bad_times,
        "backward_steps_among_valid_timestamps": inversions,
        "valid_timestamps_in_nondecreasing_order": inversions == 0,
        "blank_cells_by_column": dict(missing),
        "amount_checks": amount_issues,
        "currency_values": {name: dict(counts) for name, counts in currencies.items()},
        "notes": [
            "Full-file audit only; no sampling, sorting, cleaning, or training performed.",
            "Account and Account.1 identify the sender and receiver respectively.",
            "Use bank ID together with account ID when constructing graph identities.",
            "Is Laundering is a target and must not be included in model inputs.",
            "No duplicate-transaction or unique-account count is computed in this audit.",
            "Currency values are recorded as supplied; no conversion or normalization applied.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "data_audit.json"
    daily_path = output / "transactions_by_day.csv"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with daily_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "transactions", "non_laundering", "laundering", "invalid_label"])
        for day in sorted(daily_rows):
            count, positive, bad = daily_rows[day], daily_positive[day], daily_bad_label[day]
            writer.writerow([day, count, count - positive - bad, positive, bad])

    print("\nFULL DATASET AUDIT")
    for key in ("rows", "non_laundering_rows", "laundering_rows",
                "laundering_percent_of_all_rows", "invalid_or_missing_labels",
                "earliest_valid_timestamp", "latest_valid_timestamp",
                "invalid_or_missing_timestamps", "backward_steps_among_valid_timestamps"):
        print(f"{key}: {summary[key]}")
    print("Blank cells:", dict(missing))
    print("Amount checks:", amount_issues)
    print("\nTRANSACTIONS BY DAY")
    print(daily_path.read_text(encoding="utf-8"))
    print(f"Reports saved: {summary_path} and {daily_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/HI-Small_Trans.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    audit(args.input, args.output, args.chunk_size)
