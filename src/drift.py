"""Scenario definitions and the evaluator-only event manifest.

The manifest (true event times) is consumed ONLY by the evaluator and the known-time
comparator. The proposed policy and every other baseline receive no event metadata.
The stream environment uses the scenario internally to generate data, but policies
never receive the stream object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

ALL = (0, 1, 2, 3)


@dataclass(frozen=True)
class Event:
    client: int
    onset: int
    end: int          # == onset for abrupt changes; last ramp step for gradual
    kind: str         # covariate | concept | gradual_concept | prior | measurement


@dataclass(frozen=True)
class Scenario:
    sid: str
    dataset: str      # synthetic | clinical
    description: str
    events: tuple = field(default_factory=tuple)
    params: dict = field(default_factory=dict)

    def manifest(self) -> list[dict]:
        return [asdict(e) for e in self.events]


def _abrupt(kind, clients, onset):
    return tuple(Event(c, onset, onset, kind) for c in clients)


SCENARIOS: dict[str, Scenario] = {
    "S0": Scenario("S0", "synthetic", "Synthetic no drift"),
    "S1": Scenario("S1", "synthetic", "Covariate shift: +1 SD on features 0-3, all clinics at t=40",
                   _abrupt("covariate", ALL, 40), {"shift": 1.0, "features": [0, 1, 2, 3]}),
    "S2": Scenario("S2", "synthetic", "Concept shift: rotate first two coefficients 60 deg, clinics 1-2 at t=40",
                   _abrupt("concept", (0, 1), 40), {"angle_deg": 60.0}),
    "S3": Scenario("S3", "synthetic", "Staggered concept shift: clinics 1,2,3 at t=30,50,70",
                   (Event(0, 30, 30, "concept"), Event(1, 50, 50, "concept"), Event(2, 70, 70, "concept")),
                   {"angle_deg": 60.0}),
    "S4": Scenario("S4", "synthetic", "Gradual concept shift: 0->60 deg over t=35-55, all clinics",
                   tuple(Event(c, 35, 55, "gradual_concept") for c in ALL), {"angle_deg": 60.0}),
    "S5": Scenario("S5", "synthetic", "Prior shift: prevalence 0.3->0.6 at t=40, all clinics",
                   _abrupt("prior", ALL, 40), {"pre": 0.3, "post": 0.6}),
    "C0": Scenario("C0", "clinical", "Clinical stationary control (seeded order)"),
    "C1": Scenario("C1", "clinical", "Measurement shift +1 warm-start SD in num_lab_procedures, clinics 1-2 at t=40",
                   _abrupt("measurement", (0, 1), 40), {"shift": 1.0}),
    "C2": Scenario("C2", "clinical", "Staggered measurement shift: clinics 1,2,3 at t=30,50,70",
                   (Event(0, 30, 30, "measurement"), Event(1, 50, 50, "measurement"),
                    Event(2, 70, 70, "measurement")), {"shift": 1.0}),
}


def get_scenario(sid: str) -> Scenario:
    if sid not in SCENARIOS:
        raise KeyError(f"unknown scenario {sid}; choose from {sorted(SCENARIOS)}")
    return SCENARIOS[sid]


# ---- drift state queries used by the data environment (not by policies) ----

def rotation_angle(sc: Scenario, client: int, t: int) -> float:
    """Concept-rotation angle in degrees applied to `client` at time `t`."""
    total = sc.params.get("angle_deg", 0.0)
    for e in sc.events:
        if e.client != client:
            continue
        if e.kind == "concept" and t >= e.onset:
            return total
        if e.kind == "gradual_concept":
            if t < e.onset:
                return 0.0
            if t >= e.end:
                return total
            return total * (t - e.onset) / (e.end - e.onset)
    return 0.0


def is_active(sc: Scenario, kind: str, client: int, t: int) -> bool:
    return any(e.kind == kind and e.client == client and t >= e.onset for e in sc.events)


def prevalence(sc: Scenario, t: int) -> float | None:
    if not any(e.kind == "prior" for e in sc.events):
        return None
    onset = min(e.onset for e in sc.events if e.kind == "prior")
    return sc.params["post"] if t >= onset else sc.params["pre"]
