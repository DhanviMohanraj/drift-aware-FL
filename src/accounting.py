"""Byte ledger and the transaction-cost callback used for admission.

Simulated application payloads only (no TCP/TLS, retries or container overhead).
"""
from __future__ import annotations

import pandas as pd

MODEL_KINDS = ("model_download", "model_upload", "model_broadcast")
WARM_KINDS = ("warm_download", "warm_upload", "warm_broadcast")


class Accounting:
    def __init__(self, cfg: dict, model_bytes: int):
        a = cfg["accounting"]
        self.K = cfg["num_clients"]
        self.P = int(model_bytes)
        self.monitor_bytes = int(a["monitor_bytes"])
        self.selection_bytes = int(a["selection_bytes"])

    def transaction_cost(self, m: int) -> int:
        """(2m + K) * P model traffic + selection metadata for m selected clinics."""
        if m <= 0:
            return 0
        return (2 * m + self.K) * self.P + self.selection_bytes * m

    def full_round(self) -> int:
        return self.transaction_cost(self.K)

    def monitoring_per_step(self) -> int:
        return self.monitor_bytes * self.K


class Ledger:
    def __init__(self):
        self.rows: list[tuple] = []
        self.cumulative = 0

    def record(self, time, sender, receiver, kind, nbytes) -> None:
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("negative payload")
        self.cumulative += nbytes
        self.rows.append((int(time), str(sender), str(receiver), kind, nbytes, self.cumulative))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["time", "sender", "receiver", "message_kind",
                                                "payload_bytes", "cumulative_bytes"])

    def bytes_at(self, t: int, kinds=None) -> int:
        return sum(r[4] for r in self.rows if r[0] == t and (kinds is None or r[3] in kinds))

    def totals(self) -> dict:
        tot: dict[str, int] = {}
        for r in self.rows:
            tot[r[3]] = tot.get(r[3], 0) + r[4]
        out = {f"bytes_{k}": v for k, v in tot.items()}
        out["bytes_monitoring"] = tot.get("monitor", 0)
        out["bytes_selection_meta"] = tot.get("selection_meta", 0)
        out["bytes_model_adaptation"] = sum(tot.get(k, 0) for k in MODEL_KINDS)
        out["bytes_adaptation"] = out["bytes_model_adaptation"] + out["bytes_selection_meta"]
        out["bytes_warm_start"] = sum(tot.get(k, 0) for k in WARM_KINDS)
        out["bytes_end_to_end"] = sum(tot.values())
        out["n_messages"] = len(self.rows)
        return out
