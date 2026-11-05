#!/usr/bin/env bash
# Reproduce every UPI result in the README (Linux / macOS).
#   ./reproduce_all.sh            full run (about 30-40 min: includes the 5-seed ablation)
#   ./reproduce_all.sh --quick    skip the multi-seed ablation (about 10 min)
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"; [ -x "$PY" ] || PY="python"
export PYTHONPATH=src PYTHONUTF8=1 PYTHONWARNINGS=ignore
step() { echo; echo "=== $1 ==="; shift; "$PY" "$@"; }

step "1/8 Challenge datasets (seeded)"            src/prepare_amlsim_challenges.py
step "2/8 Train deployed model + evaluate"        src/evaluate_amlsim_challenges.py
if [ "${1:-}" != "--quick" ]; then
  step "3/8 Multi-seed ablation (windowed vs cumulative)" src/run_multiseed_ablation.py --seeds 42 43 44 45 46
  step "3b/8 Multi-seed ablation (no pair features)"      src/run_multiseed_ablation.py --seeds 42 43 44 45 46 --feature-sets windowed_nopair --name multiseed_ablation_nopair
fi
step "4/8 Verified-merchant registry"             src/build_verified_merchant_registry.py
step "5/8 Independent UPI simulation (held-out)"  src/generate_upi_simulation.py
step "6/8 Zero-shot evaluation on the simulation" src/evaluate_upi_holdout.py
step "7/8 Explainability case studies"            src/explainability.py
step "7b/8 Explanation fidelity"                  src/evaluate_explanations.py
step "8/8 Latency benchmark"                      src/benchmark_latency.py --count 1000
step "Tests"                                      tests/test_system_components.py
echo; echo "Done. Optional browser test: pip install -r requirements-dev.txt; python -m playwright install chromium; python tests/test_dashboard_e2e.py"
