"""Single-process federated simulator implementing the plan's per-step order of operations.

Per step t:
 1. every clinic predicts its fresh chunk with the model DEPLOYED at that clinic (before t)
 2. labels are revealed (immediate-label assumption)
 3. frozen sentinel scores the chunk -> CUSUM update -> 16-byte monitoring record; FIFO append
 4. token bucket refills; policy builds candidates; largest affordable prefix admitted
 5. admitted clinics download the server model, train on <=256 recent examples, upload
 6. aggregate (weighted by examples trained) and broadcast to all clinics (charged);
    new version is used only from step t+1. No admission -> nothing changes.
"""
from __future__ import annotations

import copy
import pickle
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .accounting import MODEL_KINDS, Accounting, Ledger
from .detector import CusumDetector
from .drift import get_scenario
from .federation import Deployment, FIFOBuffer, aggregate, warm_start
from .metrics import Evaluator, detection_summary, recovery_summary
from .model import get_state, make_model, mean_bce, payload_bytes, predict_proba, train_local
from .policies import TokenBucket, UnlimitedBudget, make_policy
from .streams import ClinicalWorld, Stream, SyntheticWorld
from .utils import atomic_write_pickle, deep_update, sha256_array

torch.set_num_threads(1)


class Simulation:
    def __init__(self, cfg: dict, dataset: str, scenario: str, method: str, seed: int,
                 split: str = "final", prepared: dict | None = None, trace: bool = False,
                 policy_seed: int | None = None, manifest_override: list[dict] | None = None):
        cfg = deep_update(cfg, cfg.get("dataset_overrides", {}).get(dataset, {}))
        self.cfg, self.dataset, self.method, self.seed, self.split = cfg, dataset, method, seed, split
        self.sc = get_scenario(scenario)
        if self.sc.dataset != dataset:
            raise ValueError(f"scenario {scenario} belongs to dataset {self.sc.dataset}")
        self.K = cfg["num_clients"]
        if dataset == "synthetic":
            self.world = SyntheticWorld(cfg, seed, self.sc)
            self.H = cfg["horizon"]
        else:
            if prepared is None:
                from .preprocess import prepare_clinical
                prepared = prepare_clinical(cfg, seed)
            self.world = ClinicalWorld(cfg, seed, self.sc, prepared)
            self.H = self.world.horizon
        self.stream = Stream(self.world, split)
        # evaluator-only event manifest (known_time is the single privileged policy)
        self.manifest = manifest_override if manifest_override is not None else self.sc.manifest()
        self.trace = [] if trace else None

        warm = [self.world.warm_data(c) for c in range(self.K)]
        in_dim = warm[0].x.shape[1]
        self.model = make_model(in_dim, cfg["model"]["hidden"], seed)
        self.ledger = Ledger()
        state, self.warm_info = warm_start(self.model, get_state(self.model), warm, cfg, seed, self.ledger)
        self.warm_checksum = sha256_array(*[v.numpy() for v in state.values()])
        self.P = payload_bytes(state)
        self.acct = Accounting(cfg, self.P)
        self.server_state = state
        self.deploy = Deployment(self.K, state)
        self.sentinel = make_model(in_dim, cfg["model"]["hidden"], seed)
        self.sentinel.load_state_dict(copy.deepcopy(state))      # frozen copy, never trained
        self.eval_model = make_model(in_dim, cfg["model"]["hidden"], seed)
        self.train_model = make_model(in_dim, cfg["model"]["hidden"], seed)
        self.detectors = []
        self.calib_ref = []
        for c in range(self.K):
            losses = [mean_bce(self.sentinel, b.x, b.y) for b in self.world.calib_chunks(c)]
            d = CusumDetector.from_calibration(losses, cfg["detector"])
            self.detectors.append(d)
            self.calib_ref.append((d.ref_mean, d.ref_sd, len(losses)))
        self.policy = make_policy(method, cfg, seed if policy_seed is None else policy_seed,
                                  self.manifest if method == "known_time" else [])
        full = self.acct.full_round()
        b = cfg["budget"]["b"]
        if self.policy.budgeted:
            self.budget = TokenBucket(b * full, cfg["budget"]["bucket_full_rounds"] * full,
                                      b * self.H * full, self.acct.transaction_cost)
        else:
            self.budget = UnlimitedBudget(self.acct.transaction_cost)
        self.buffers = [FIFOBuffer(cfg["model"]["buffer_size"]) for _ in range(self.K)]
        self.evaluator = Evaluator(self.manifest)
        self.event_rows: list[dict] = []
        self.n_transactions = 0
        self.n_client_trainings = 0
        self.n_train_steps = 0
        self.t = 0
        self.wall = 0.0

    def _tr(self, phase, c, t):
        if self.trace is not None:
            self.trace.append((t, phase, c))

    # ------------------------------------------------------------------
    def step(self):
        t = self.t
        t0 = _time.perf_counter()
        batches = []
        for c in range(self.K):                                   # 1. predict with deployed model
            b = self.stream.next(c, t)
            self.eval_model.load_state_dict(self.deploy.states[c])
            p = predict_proba(self.eval_model, b.x)
            self._tr("predict", c, t)
            self.evaluator.record(p, b.y, c, t, b.sample_ids, self.deploy.versions[c], self.ledger.cumulative)
            batches.append(b)
        signals = []
        for c, b in enumerate(batches):                           # 2-3. reveal labels, detect, buffer
            self._tr("label", c, t)
            loss = mean_bce(self.sentinel, b.x, b.y)
            signals.append(self.detectors[c].update(loss, t))
            if self.policy.budgeted:
                self.ledger.record(t, c, "server", "monitor", self.acct.monitor_bytes)
            self.buffers[c].append(b.x, b.y)
            self._tr("buffer", c, t)
        tokens_before = self.budget.tokens
        self.budget.refill()                                      # 4. refill and admit
        tokens_refilled = self.budget.tokens
        admitted = self.policy.select(signals, self.budget, t)
        expected = self.acct.transaction_cost(len(admitted))
        if admitted:                                              # 5. local training
            states, ns = [], []
            for c in admitted:
                self.ledger.record(t, "server", c, "selection_meta", self.acct.selection_bytes)
                self.ledger.record(t, "server", c, "model_download", self.P)
                x, y = self.buffers[c].arrays()
                s, n, k = train_local(self.train_model, self.server_state, x, y, self.cfg, (self.seed, t, c))
                self._tr("train", c, t)
                self.ledger.record(t, c, "server", "model_upload", self.P)
                states.append(s)
                ns.append(n)
                self.n_client_trainings += 1
                self.n_train_steps += k
            self.server_state = aggregate(states, ns)              # 6. aggregate + charged broadcast
            self.deploy.broadcast(self.server_state, self.ledger, t, self.P)
            self.n_transactions += 1
        recorded = self.ledger.bytes_at(t, MODEL_KINDS + ("selection_meta",))
        if recorded != expected:
            raise AssertionError(f"t={t}: ledger {recorded} != predicted transaction cost {expected}")
        for c in range(self.K):
            s = signals[c]
            self.event_rows.append(dict(
                client_id=c, time=t, chunk_loss=s.loss, z=s.z, detector_score=s.score, flag=int(s.flag),
                severity=s.severity, pending_credits=self.policy.credits(c), selected=int(c in admitted),
                expiry=int(c in self.policy.just_expired), cooldown=int(s.in_cooldown),
                tokens_before=tokens_before, tokens_refilled=tokens_refilled, tokens_after=self.budget.tokens,
                deployed_version=self.deploy.versions[c]))
        self.t += 1
        self.wall += _time.perf_counter() - t0

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        return dict(t=self.t, server_state=self.server_state, versions=list(self.deploy.versions),
                    server_version=self.deploy.server_version,
                    buffers=[b.state() for b in self.buffers], detectors=[d.state() for d in self.detectors],
                    policy=self.policy.state(), budget=self.budget.state(),
                    ledger=(list(self.ledger.rows), self.ledger.cumulative),
                    eval_rows=list(self.evaluator.rows), eval_step_rows=list(self.evaluator.step_rows),
                    event_rows=list(self.event_rows),
                    counters=(self.n_transactions, self.n_client_trainings, self.n_train_steps, self.wall))

    def restore(self, st: dict):
        self.t = st["t"]
        self.server_state = st["server_state"]
        self.deploy.versions = list(st["versions"])
        self.deploy.server_version = st["server_version"]
        self.deploy.states = [self.server_state] * self.K if self.deploy.server_version else self.deploy.states
        for b, s in zip(self.buffers, st["buffers"]):
            b.load(s)
        for d, s in zip(self.detectors, st["detectors"]):
            d.load(s)
        self.policy.load(st["policy"])
        self.budget.load(st["budget"])
        self.ledger.rows, self.ledger.cumulative = list(st["ledger"][0]), st["ledger"][1]
        self.evaluator.rows, self.evaluator.step_rows = list(st["eval_rows"]), list(st["eval_step_rows"])
        self.event_rows = list(st["event_rows"])
        self.n_transactions, self.n_client_trainings, self.n_train_steps, self.wall = st["counters"]

    def run(self, checkpoint_path: Path | None = None, stop_at: int | None = None):
        every = self.cfg.get("checkpoint_every", 10)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self.restore(pickle.loads(Path(checkpoint_path).read_bytes()))
        while self.t < self.H:
            if stop_at is not None and self.t >= stop_at:
                return False
            self.step()
            if checkpoint_path is not None and self.t % every == 0 and self.t < self.H:
                atomic_write_pickle(checkpoint_path, self.snapshot())
        return True

    # ------------------------------------------------------------------
    def results(self) -> dict:
        ev = self.cfg["evaluation"]
        metrics = self.evaluator.metrics_frame(ev["window"])
        events = pd.DataFrame(self.event_rows)
        comm = self.ledger.frame()
        post_start = min((e["onset"] for e in self.manifest), default=self.cfg["null_post_start"])
        summ = self.evaluator.predictive_summary(post_start, self.K)
        summ.update(self.ledger.totals())
        det = detection_summary(events, self.manifest, self.H, self.K, ev["match_window"])
        rec = recovery_summary(metrics, self.manifest, ev)
        summ.update({k: v for k, v in det.items() if k != "detection_delays"})
        summ.update({k: v for k, v in rec.items() if k != "recovery_detail"})
        # alarm -> first service delay (detector-driven methods only)
        delays = []
        if self.policy.uses_detector:
            for _, r in events[events.flag == 1].iterrows():
                s = events[(events.client_id == r.client_id) & (events.time >= r.time) & (events.selected == 1)]
                if len(s):
                    delays.append(int(s.time.min() - r.time))
        summ["mean_alarm_to_service"] = float(np.mean(delays)) if delays else np.nan
        summ["expired_requests"] = len(self.policy.pending.expired_log)
        summ.update(n_transactions=self.n_transactions, n_client_trainings=self.n_client_trainings,
                    n_train_steps=self.n_train_steps, wall_clock_s=self.wall, model_payload_bytes=self.P,
                    horizon=self.H, warm_checksum=self.warm_checksum,
                    budget_cap=None if self.budget.unlimited else self.budget.cap_total,
                    budget_spent=self.budget.spent)
        overhead = self.cfg["accounting"]["overhead_sensitivity_bytes"]
        summ["bytes_end_to_end_with_overhead"] = summ["bytes_end_to_end"] + overhead * summ["n_messages"]
        return dict(metrics=metrics, events=events, communication=comm,
                    predictions=self.evaluator.predictions(), summary=summ,
                    detail=dict(detection_delays=det["detection_delays"], recovery=rec["recovery_detail"],
                                calibration_reference=self.calib_ref, warm_info=self.warm_info,
                                expired=self.policy.pending.expired_log, world=self.world.info,
                                manifest=self.manifest))
