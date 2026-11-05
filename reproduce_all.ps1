# Reproduce every UPI result in the README (Windows / PowerShell).
#   .\reproduce_all.ps1            full run (about 30-40 min: includes the 5-seed ablation)
#   .\reproduce_all.ps1 -Quick     skip the multi-seed ablation (about 10 min)
param([switch]$Quick)

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$env:PYTHONUTF8 = "1"
$env:PYTHONWARNINGS = "ignore"
Set-Location $PSScriptRoot

function Step($name, $args_) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $py @args_
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $name" }
}

Step "1/8 Challenge datasets (seeded)"            @("src\prepare_amlsim_challenges.py")
Step "2/8 Train deployed model + evaluate"        @("src\evaluate_amlsim_challenges.py")
if (-not $Quick) {
    Step "3/8 Multi-seed ablation (windowed vs cumulative)" @("src\run_multiseed_ablation.py", "--seeds", "42", "43", "44", "45", "46")
    Step "3b/8 Multi-seed ablation (no pair features)"     @("src\run_multiseed_ablation.py", "--seeds", "42", "43", "44", "45", "46", "--feature-sets", "windowed_nopair", "--name", "multiseed_ablation_nopair")
}
Step "4/8 Verified-merchant registry"             @("src\build_verified_merchant_registry.py")
Step "5/8 Independent UPI simulation (held-out)"  @("src\generate_upi_simulation.py")
Step "6/8 Zero-shot evaluation on the simulation" @("src\evaluate_upi_holdout.py")
Step "7/8 Explainability case studies + fidelity" @("src\explainability.py")
Step "7b/8 Explanation fidelity"                  @("src\evaluate_explanations.py")
Step "8/8 Latency benchmark"                      @("src\benchmark_latency.py", "--count", "1000")
Step "Tests"                                      @("tests\test_system_components.py")
Write-Host "`nAll steps finished. Optional browser test: pip install -r requirements-dev.txt; python -m playwright install chromium; python tests\test_dashboard_e2e.py" -ForegroundColor Green
