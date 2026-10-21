# Budget-Constrained Drift-Triggered Federated Learning on Simulated Clinical Tabular Streams

*Draft skeleton. Every number must come from `results/*/tables/` (regenerate with `python -m src.analyze ...`).
Sections marked **not yet measured** must not be filled by hand.*

## Abstract
Simulated clinical streams; four artificial clinics; candidate scheduling rule (severity + waiting time under an
explicit byte budget). Results: see Table 2. Clinical results: **not yet measured** (run in Colab).

## 1 Introduction
Motivation; scope limited to simulated streams (no verified timestamps or hospital IDs). Do not claim "first".

## 2 Related work
Fill from `paper/related_work.csv` (>= 6 primary papers, each claimed difference tied to a section/equation).
FedConD [1] and FedDrift [2] already combine drift handling with communication strategies; event-triggered FL [3]
predates this work. The drift-only control in this repository is an internal control, **not** FedConD/FedDrift.

## 3 Method
- Frozen-sentinel one-sided CUSUM per clinic (engineering detector; no false-alarm guarantee).
- Pending requests: 3 service credits, 15-step expiry, coalescing.
- Priority = min(severity, 3) + eta * min(wait / 10, 1); largest affordable prefix admitted.
- Token bucket: starts empty, refill b x full-round cost per step, capacity 2 full rounds, hard cap.
- Transaction cost (2m + 4)P + 8m bytes; monitoring 16 B/clinic/step; broadcasts to all four clinics charged.
- Budget-feasibility lemma (induction; unit-tested in `tests/test_accounting.py`).
- Figure 1 (method flow): **to draw**.

## 4 Experimental protocol
Table 1 (datasets, clients, prevalence, splits, scenarios): synthetic from `configs/core.yaml`;
clinical counts from `results/cache/clinical_*.json` — **not yet measured**.
Immediate-label assumption; development-only calibration (`results/calibration_*.json`).

## 5 Results
- Table 2: `results/synthetic/tables/table2_main.csv`, `results.md`
- Paired differences: `paired_differences.csv` (descriptive; five seeds)
- Table 3 (drift outcomes, ablations): `table3_drift.csv`
- Figure 2: `figures/fig2_loss_over_time_S3.png` (C1: **not yet measured**)
- Figure 3: `figures/fig3_quality_vs_bytes_S3.png`
- Figure 4: `figures/fig4_per_client_S3.png`

## 6 Limitations
Single global model; synchronous aggregation (staggered drift, not asynchronous FL); simulated payload bytes only;
immediate labels (delayed-label sensitivity **not yet measured**); no published comparator reproduced yet;
five seeds; one clinical dataset with artificial Dirichlet clinics.

## 7 Conclusion
Write only after the submission gates in README are closed.
