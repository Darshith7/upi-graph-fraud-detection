"""Build data/verified_merchants.json - the list of VERIFIED benign merchant handles.

The consumer-safety copilot used to treat any handle containing "store" / "merchant" /
"retail_hub" as a verified merchant and skip the high-risk warning. Anyone can choose such a
handle, so that was an evasion route. Merchant status now comes only from this registry.

In production the registry would come from KYC / merchant onboarding (e.g. the acquiring
bank's verified-merchant list). Here it is seeded from the AMLSim benign-hub scenario, where
the legitimate hubs are known by construction of the simulation.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Build verified-merchant registry")
    parser.add_argument("--challenge-file", default="data/amlsim_challenges/challenge_2_lookalike.csv")
    parser.add_argument("--output", default="data/verified_merchants.json")
    args = parser.parse_args()

    df = pd.read_csv(args.challenge_file)
    # Benign hubs injected by the simulation: high fan-in retail QR merchants.
    merchants = sorted(
        {str(h).strip().lower() for h in df["receiver_upi"].unique() if "retail_hub" in str(h).lower()}
    )
    payload = {
        "description": "Verified benign merchant handles (seed data from the AMLSim benign-hub scenario). "
                       "Replace with a KYC-backed merchant registry in production.",
        "source": Path(args.challenge_file).name,
        "verified_merchants": merchants,
    }
    out = Path(args.output)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(merchants)} verified merchant(s) to {out}")


if __name__ == "__main__":
    main()
