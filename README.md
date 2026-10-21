# Drift-aware budgeted federated learning — four simulated clinics

Reproducible single-process simulator testing whether a **drift-aware client scheduler** (severity + waiting time,
under an explicit byte budget) uses a fixed communication budget better than periodic, random or drift-only
scheduling. Implements the *Single Lead Four Clinic Implementation Plan* (sections 3–10).

> Scope: **simulated clinical streams**. Clinics are artificial (Dirichlet label allocation), stream order is
> synthetic, labels are revealed immediately (optimistic for 30-day readmission). This is not a deployed or
> privacy-preserving system and makes no clinical-validity claim.

## Status (21 Oct 2026)

| Item | State |
|---|---|
| Core modules (data, streams, drift, model, federation, detector, policies, accounting, metrics, runner, analysis) | done |
| Correctness gates (section 10) | 39 tests pass (`pytest -q`) — leakage, prequential order, aggregation, deployment, accounting, detector, oracle isolation, seed isolation, resume, metric edge cases |
| Synthetic dev calibration | done → `results/calibration_synthetic.json` |
| Synthetic tiers: full core (S0–S5 × 6 methods × 5 seeds), ablation (S3), budget sweep (S3) | done → `results/synthetic/` |
| Clinical (UCI 296) runs | **not yet measured** — the build sandbox could not reach archive.ics.uci.edu; run in Colab (below) |
| Published comparator (FedConD/FedDrift), delayed labels, ≥10 seeds, related-work matrix | **open submission gates** |

## Quick start (local)

```bash
pip install torch && pip install -r requirements.txt
python -m pytest -q                                   # correctness gates
python -m src.run --dataset synthetic --scenario S3 --method proposed --seed 11
python -m src.matrix --tier smoke --workers 2         # 12 runs
python -m src.analyze --manifests results/manifest_smoke.json --out results/smoke
```

Clinical: `python -m src.data --out data/raw` → `python -m src.calibrate --dataset clinical` → freeze the selected
threshold/interval under `dataset_overrides.clinical` in `configs/core.yaml` → `python -m src.matrix --tier minimum --only-dataset clinical`.

## Google Colab

Open `notebooks/colab_launcher.ipynb` in Colab (File → Open notebook → GitHub → `DhanviMohanraj/drift-aware-FL`)
and run the cells top to bottom. A CPU runtime is enough. The notebook mounts Drive for run outputs (so a
disconnect loses nothing — re-running a tier skips completed runs and resumes checkpointed ones), clones the repo,
installs requirements, runs the tests, downloads UCI 296, calibrates on development patients, freezes the
clinical overrides, then runs the clinical tiers and the analysis.

## Layout

```
src/data.py, preprocess.py   UCI download + manifest; one seeded encounter/patient; 20/20/60 patient pools;
                             disjoint sentinel-calibration subset; warm-start-only fitting; Dirichlet clinics; chunks
src/streams.py, drift.py     Stream.next(client, t) -> Batch; scenarios S0–S5, C0–C2; evaluator-only event manifest
src/model.py, federation.py  MLP 64-32-1, local training (fresh Adam), weighted FedAvg, FIFO(256), deployed versions
src/detector.py, policies.py frozen-sentinel CUSUM; pending credits/expiry; token bucket; 6 methods + 2 ablations
src/accounting.py, metrics.py byte ledger + transaction_cost(m); log loss/Brier/AUROC/AUPRC, detection, recovery
src/simulation.py            per-step order of operations; checkpoint/resume
src/run.py, matrix.py        immutable run dirs (run.json, metrics/events/communication csv, summary.json); tiers
src/calibrate.py             dev-only threshold & periodic-interval selection
src/analyze.py               tables/figures from manifests of declared complete runs, pairing checks
tests/                       correctness gates     configs/core.yaml   frozen configuration
notebooks/colab_launcher.ipynb   paper/draft.md, related_work.csv
```

## Protocol decisions recorded during the build (all on development streams/seeds 101, 102)

1. **Warm-start local epochs = 10** (plan fixes 5 rounds, not epochs). With 1 epoch the warm model sat near the
   prior (loss ≈ 0.56), so the frozen sentinel barely reacted to any drift (dev detection 40%). Tried {1, 5, 10, 20}.
2. **Generator variant rejected:** zeroing clinic offsets on the two rotated coordinates (kept as
   `offset_exclude_rotated: false`). It weakened the drift signal once warm-start was fit.
3. **CUSUM threshold = 6** from {3, 6, 12}: 0.38 false alarms / 100 client-steps on no-drift dev streams (target ≤ 1),
   100% dev detection, mean delay 2.3 steps. Threshold 3 broke the target (1.12).
4. **Periodic interval = 2** from {2, 5, 10} (plan's starting default was 5).
5. `eta = 1.0` frozen without tuning.

Not changed: target, splits, metrics, event schedules, 3 service credits, 15-step expiry, b = 0.25, budget rule.

## Communication model

P = full-model payload (numel × element size; synthetic MLP = 13,828 B). Admitted transaction with m clinics:
`(2m + 4)·P + 8m` bytes (m downloads, m uploads, broadcast to all 4, two int32 selection fields per clinic),
deducted once. Monitoring: 16 B per clinic per step for every budgeted method. Token bucket starts empty, refills
`b × cost(4)` per step, capacity `2 × cost(4)`, hard cap `b × H × cost(4)`. Always-on is unbudgeted and sends no
monitoring records. Warm-start traffic is reported separately and inside end-to-end totals. These are simulated
application payloads (no TCP/TLS); `bytes_end_to_end_with_overhead` adds 64 B per message as a sensitivity check.

## Results

See `results/synthetic/tables/results.md` (generated) and `results/synthetic/figures/`. Summary in
`RESULTS_SUMMARY.md`. Headline: at b = 0.25 the proposed scheduler uses ~2.5–7× fewer adaptation bytes
than periodic but has slightly worse post-change log loss in every synthetic scenario, and its priority rule
never changes a decision (identical to drift-only) because the budget is not binding.

## Known limitations / open gates

- Clinical evidence not yet measured; Dirichlet(α=0.5) on a binary target gives strong label skew between clinics,
  so expect large worst-clinic losses for a single global model (visible in a schema-only fixture check).
- Drift-triggered methods spend far less than the budget: three credits per alarm caps adaptation at three
  local updates per event. Report as is; any change to credits must be decided on dev streams.
- Five seeds → descriptive paired differences only; no significance tests.
- No published comparator reproduced; the drift-only control is internal, not FedConD/FedDrift.
- Immediate labels; delayed-label sensitivity not implemented.
- Data citation/licence in `src/data.py` must be re-checked on the UCI page before release.
