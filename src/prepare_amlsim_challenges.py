"""Generate synthetic UPI-like transaction datasets with 3 controlled challenge cases.

Implements the experimental scenarios from the project specification:
1. Baseline: Standard AMLSim synthetic UPI stream (cycles, fan-in, normal transfers).
2. Challenge 1 (Disguised Suspicious): Disguised amounts (ticket-size splitting) and randomized delay intervals.
3. Challenge 2 (Benign Lookalike): Legitimate high-volume merchant QR hub (high fan-in) and payroll hub (high fan-out).
4. Challenge 3 (Partial History): 50% suppressed / unobserved historical edges to test cold-start / sparse context.

Rules strictly adhered to:
- Retain generator labels (is_laundering).
- Scenario names are NEVER model inputs.
- Configurations and random seeds are explicitly saved.
- Outputs saved in data/amlsim_challenges/.
"""

import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd


UPI_BANKS = ["okhdfcbank", "oksbi", "okaxis", "icici", "paytm", "upi"]


def format_upi_handle(node_id: int, bank_id: int = None) -> str:
    """Format an account node into a realistic UPI Virtual Payment Address (VPA)."""
    if bank_id is None:
        bank = UPI_BANKS[node_id % len(UPI_BANKS)]
    else:
        bank = UPI_BANKS[bank_id % len(UPI_BANKS)]
    return f"acc_{node_id:05d}@{bank}"


def prepare_challenges(raw_dir: Path, output_dir: Path, seed: int = 42):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    print(f"Loading raw AMLSim files from {raw_dir}...")
    nodes_df = pd.read_csv(raw_dir / "nodes.csv").set_index("nodeid")
    tx_df = pd.read_csv(raw_dir / "transactions.csv")

    # Sort strictly by timestamp, then transaction order
    tx_df = tx_df.sort_values(by=["time", "sourceNodeId", "targetNodeId"]).reset_index(drop=True)
    tx_df["transaction_id"] = np.arange(100_000, 100_000 + len(tx_df))

    # Identify fraud transactions: both source and target are fraud nodes in laundering patterns
    tx_df["src_fraud"] = tx_df["sourceNodeId"].map(nodes_df["isFraud"]).fillna(0).astype(int)
    tx_df["dst_fraud"] = tx_df["targetNodeId"].map(nodes_df["isFraud"]).fillna(0).astype(int)
    tx_df["is_laundering"] = ((tx_df["src_fraud"] == 1) & (tx_df["dst_fraud"] == 1)).astype(int)

    # Base columns
    tx_df["sender_upi"] = tx_df["sourceNodeId"].apply(format_upi_handle)
    tx_df["receiver_upi"] = tx_df["targetNodeId"].apply(format_upi_handle)
    tx_df["amount"] = tx_df["value"].round(2)
    tx_df["timestamp_step"] = tx_df["time"]

    # Map time step (1 to 149) to synthetic UPI timestamp (seconds since epoch)
    base_epoch = 1725148800  # 2024-09-01 00:00:00 UTC
    # Each time step corresponds to a 2-hour window with jitter
    time_jitters = rng.integers(0, 7200, size=len(tx_df))
    tx_df["timestamp_seconds"] = base_epoch + (tx_df["timestamp_step"] * 7200) + time_jitters
    tx_df = tx_df.sort_values(by=["timestamp_seconds", "transaction_id"]).reset_index(drop=True)

    keep_cols = [
        "transaction_id", "timestamp_seconds", "timestamp_step",
        "sender_upi", "receiver_upi", "sourceNodeId", "targetNodeId",
        "amount", "is_laundering"
    ]

    base_df = tx_df[keep_cols].copy()
    base_file = output_dir / "baseline.csv"
    base_df.to_csv(base_file, index=False)
    print(f"Saved baseline: {len(base_df):,} transactions ({base_df['is_laundering'].sum():,} laundering) to {base_file}")

    # =========================================================================
    # Challenge 1: Disguised Suspicious
    # =========================================================================
    # Launderers disguise activity:
    # 1. Amount perturbation: Round amounts are split into typical daily ticket sizes (₹100-₹450).
    # 2. Timing perturbation: Introduce randomized delays (+3 to +12 steps) to break velocity rules.
    # Non-fraudulent transfers are preserved intact.
    disguised_df = base_df.copy()
    fraud_mask = disguised_df["is_laundering"] == 1

    # Apply amount perturbation
    disguised_amounts = disguised_df.loc[fraud_mask, "amount"].values.copy()
    # Replace large/round laundering amounts with micro-ticket amounts (disguised splitting)
    split_amounts = rng.uniform(115.0, 485.0, size=fraud_mask.sum()).round(2)
    disguised_df.loc[fraud_mask, "amount"] = split_amounts

    # Apply timing perturbation: delay laundering transfers
    step_delays = rng.integers(3, 14, size=fraud_mask.sum())
    disguised_df.loc[fraud_mask, "timestamp_step"] = np.minimum(
        149, disguised_df.loc[fraud_mask, "timestamp_step"] + step_delays
    )
    disguised_df.loc[fraud_mask, "timestamp_seconds"] = (
        base_epoch + (disguised_df.loc[fraud_mask, "timestamp_step"] * 7200) + rng.integers(0, 7200, size=fraud_mask.sum())
    )

    disguised_df = disguised_df.sort_values(by=["timestamp_seconds", "transaction_id"]).reset_index(drop=True)
    disguised_file = output_dir / "challenge_1_disguised.csv"
    disguised_df.to_csv(disguised_file, index=False)
    print(f"Saved Challenge 1 (Disguised): {len(disguised_df):,} rows to {disguised_file}")

    # =========================================================================
    # Challenge 2: Benign Lookalike Hubs
    # =========================================================================
    # Inject legitimate, busy accounts that structurally look like laundering hubs:
    # 1. Merchant QR Hub: massive fan-in from hundreds of distinct customers (is_laundering = 0).
    # 2. Payroll Disburser Hub: massive fan-out to hundreds of distinct employees (is_laundering = 0).
    lookalike_df = base_df.copy()
    merchant_node_id = 99001
    payroll_node_id = 99002
    merchant_upi = "dmart_retail_hub@okhdfcbank"
    payroll_upi = "infosys_payroll@icici"

    new_rows = []
    current_tx_id = 200_000

    # Generate 1,500 benign customer transactions paying merchant QR hub across steps 15..140
    distinct_customers = rng.choice(nodes_df.index[nodes_df["isFraud"] == 0], size=300, replace=False)
    for _ in range(1500):
        c_node = int(rng.choice(distinct_customers))
        t_step = int(rng.integers(15, 145))
        t_sec = base_epoch + (t_step * 7200) + int(rng.integers(0, 7200))
        amt = round(float(rng.uniform(45.0, 1850.0)), 2)
        new_rows.append({
            "transaction_id": current_tx_id,
            "timestamp_seconds": t_sec,
            "timestamp_step": t_step,
            "sender_upi": format_upi_handle(c_node),
            "receiver_upi": merchant_upi,
            "sourceNodeId": c_node,
            "targetNodeId": merchant_node_id,
            "amount": amt,
            "is_laundering": 0  # Legitimate merchant
        })
        current_tx_id += 1

    # Generate 1,000 benign salary disburser transactions paying employees on steps 30, 60, 90, 120
    distinct_employees = rng.choice(nodes_df.index[nodes_df["isFraud"] == 0], size=250, replace=False)
    salary_steps = [30, 60, 90, 120]
    for s_step in salary_steps:
        for emp_node in distinct_employees:
            t_sec = base_epoch + (s_step * 7200) + int(rng.integers(100, 3600))
            amt = round(float(rng.uniform(25000.0, 85000.0)), 2)
            new_rows.append({
                "transaction_id": current_tx_id,
                "timestamp_seconds": t_sec,
                "timestamp_step": s_step,
                "sender_upi": payroll_upi,
                "receiver_upi": format_upi_handle(int(emp_node)),
                "sourceNodeId": payroll_node_id,
                "targetNodeId": int(emp_node),
                "amount": amt,
                "is_laundering": 0  # Legitimate corporate payroll
            })
            current_tx_id += 1

    extra_df = pd.DataFrame(new_rows)
    lookalike_df = pd.concat([lookalike_df, extra_df], ignore_index=True)
    lookalike_df = lookalike_df.sort_values(by=["timestamp_seconds", "transaction_id"]).reset_index(drop=True)
    lookalike_file = output_dir / "challenge_2_lookalike.csv"
    lookalike_df.to_csv(lookalike_file, index=False)
    print(f"Saved Challenge 2 (Lookalike Hubs): {len(lookalike_df):,} rows ({len(new_rows):,} benign hub events) to {lookalike_file}")

    # =========================================================================
    # Challenge 3: Partial / Missing History
    # =========================================================================
    # Simulates missing cross-bank context or newly boarded accounts:
    # Randomly suppress 50% of earlier edges for both classes before scoring.
    partial_df = base_df.copy()
    # Retain all transactions, but mark an edge_suppression_mask to simulate unobserved history
    suppressed_mask = rng.uniform(0.0, 1.0, size=len(partial_df)) < 0.50
    partial_df["history_suppressed"] = suppressed_mask.astype(int)
    partial_file = output_dir / "challenge_3_partial_history.csv"
    partial_df.to_csv(partial_file, index=False)
    print(f"Saved Challenge 3 (Partial History): {len(partial_df):,} rows (50% masked) to {partial_file}")

    # Metadata & Manifest
    manifest = {
        "generator": "AMLSim UPI Scenario Generator",
        "random_seed": seed,
        "raw_source": str(raw_dir),
        "total_baseline_transactions": len(base_df),
        "baseline_laundering_count": int(base_df["is_laundering"].sum()),
        "baseline_laundering_rate": float(base_df["is_laundering"].mean()),
        "scenarios": {
            "baseline": {
                "file": "baseline.csv",
                "rows": len(base_df),
                "laundering": int(base_df["is_laundering"].sum()),
                "description": "Base synthetic AMLSim stream with cycles and fan-in laundering."
            },
            "challenge_1_disguised": {
                "file": "challenge_1_disguised.csv",
                "rows": len(disguised_df),
                "laundering": int(disguised_df["is_laundering"].sum()),
                "amount_perturbation": "Split amounts to daily micro-transactions (₹115-₹485)",
                "timing_perturbation": "Added +3 to +13 delay steps to evade velocity rules"
            },
            "challenge_2_lookalike": {
                "file": "challenge_2_lookalike.csv",
                "rows": len(lookalike_df),
                "benign_hub_transactions_added": len(new_rows),
                "merchant_hub": merchant_upi,
                "payroll_hub": payroll_upi,
                "description": "Legitimate high-degree merchant (fan-in) and corporate payroll (fan-out) hubs."
            },
            "challenge_3_partial_history": {
                "file": "challenge_3_partial_history.csv",
                "rows": len(partial_df),
                "suppression_fraction": 0.50,
                "description": "50% unobserved earlier transaction history edges to test sparse context / cold-start."
            }
        },
        "policy": "Scenario names and generator metadata are NEVER model inputs. Transactions are scored strictly using past-only history before committing events into the graph."
    }

    manifest_file = output_dir / "challenges_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare AMLSim challenge datasets")
    parser.add_argument("--raw-dir", default="data/amlsim_raw/20K_fanin200cycle200", help="Raw AMLSim directory")
    parser.add_argument("--output-dir", default="data/amlsim_challenges", help="Output directory for challenge datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    prepare_challenges(args.raw_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()

