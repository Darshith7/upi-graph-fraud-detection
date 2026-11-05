# Model card - UPI mule-account detector (research prototype)

## Intended use
Rank UPI-style transactions by laundering / mule risk to help an analyst triage alerts, and give a
payer a screening hint about a recipient. **Not** for automated blocking, credit decisions or any
use on real customers: it has never seen real UPI traffic.

## Model
* **XGBoost** on 23 past-only windowed features (6h / 24h / 72h counts, unique counterparties, amounts,
  flow-through, repeat-pair, recency).
* **Directional 2-layer GraphSAGE** over a causal transaction graph (running-mean node features).
* **Blend** 0.25 x XGBoost + 0.75 x GraphSAGE (weights tuned on validation).
* **Alert rule:** score >= max(0.302, rolling top-5% cutoff over the last 2,000 scores).
* **Training data:** AMLSim 20K simulation, steps 1-75 train / 76-90 validation / 91-149 test.

## Measured performance (all on simulated data)
| Evidence | Result |
|---|---|
| AMLSim test, deployed model (seed 42) | blend AP 0.841; policy precision 95.3%, recall 58.6%, 5.1% of traffic alerted |
| 5 seeds (mean +/- std) | baseline blend AP 0.834 +/- 0.007; blend beats XGBoost-only by +0.046 to +0.068 AP depending on the scenario (mean over 5 seeds; std 0.007-0.013) |
| Realistic base rate (re-weighted to 0.1% prevalence) | policy precision about 18% (4.5 false alerts per true alert) on the baseline scenario |
| Cumulative features (legacy) vs windowed | windowed is better in all scenarios; Challenge 1 AP 0.40 -> 0.75 (5 seeds) |
| **Zero-shot on the independent UPI simulation** | **fails**: AP 0.0046 vs 0.0084 random; the static cutoff alerts on 56% of traffic |
| Explanation fidelity (GNNExplainer) | not demonstrated: top-ranked edges change the logit 0.018 vs 0.011 for random edges (p = 0.44) |
| Recipient advisory (consumer copilot) | separates future laundering receivers from benign ones with AUC 0.71 |
| Latency (CPU, one request at a time) | end-to-end p50 2.3 ms, p95 3.7 ms (about 390 tx/s) |

See `reports/` for the raw metrics behind each line.

## Known failure modes
1. **Does not transfer across traffic mixes.** Trained on AMLSim's sparse, mostly person-to-person
   traffic it mis-scores a UPI-like mix dominated by busy merchants (receivers handle ~8 payments/day
   vs ~1). A model must be (re)trained on the target network's own history; retrained on the
   simulation's own early period it reaches AP 0.995 (that simulation is easy; see `docs/DATA.md`).
2. **Recurring legitimate payments escalate.** A daily INR 5,000 payment to the same payee alerts from
   about the 6th repeat. The repeat-pair features that fix Challenge 1 (AP 0.77 vs 0.51 without them)
   cause this; a `windowed_nopair` variant avoids it at the cost of that detection power
   (`--feature-set windowed_nopair`).
3. **Small fan-ins are missed.** A 60-sender collection account does not alert (the training data only
   contains a fan-in of 200).
4. **Benign hubs.** About a quarter of benign hub events are alerted in the benign-lookalike scenario;
   known merchants are exempted only from the consumer warning, via a verified-merchant registry.
5. **Fresh mule accounts** look like "insufficient history" and are scored as low risk.
6. **Explanations are indicative only** (see above). The SAR draft says so.

## Ethical and operational notes
* Alerts are about accounts and transactions, not people; any real deployment needs human review,
  appeal and data-protection controls (DPDP Act 2023), which this prototype does not provide.
* The consumer warning can harm a legitimate payee (wrongly flagged). Warnings are advisory text only.
* The API has an optional API key (`UPI_API_KEY`) and per-client rate limiting, refuses to bind a
  non-local interface without a key, and has no user accounts, audit log or encryption at rest.

## Reproduce
`./reproduce_all.ps1` (Windows) or `./reproduce_all.sh`; data provenance in `docs/DATA.md`.
