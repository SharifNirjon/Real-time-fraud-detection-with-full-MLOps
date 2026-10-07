"""Replay the held-out "live" stream to the API in TransactionDT order, as if live.

- Ground truth is revealed to POST /labels only after --label-delay-hours of
  SIMULATED time (chargebacks arrive late), so the monitor can compute live
  PR-AUC / recall on labelled traffic only.
- --drift shifts the amount and device distributions from --drift-start
  (share of the stream) onward, to demonstrate drift detection and retraining.
  Labels are left untouched: this is covariate shift, not a new fraud pattern.

Usage:
  python scripts/simulate.py --api-url http://localhost:8000 [--limit N] [--drift]
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.config import load_config, path  # noqa: E402

NEW_DEVICES = ["Pixel 9 Pro Build/AP4A", "SM-S931U Build/UP1A", "iPhone17,3", "Moto G Play 2026"]


def apply_drift(
    df: pd.DataFrame, start_frac: float, amount_mult: float, device_share: float, seed: int
) -> pd.DataFrame:
    """Shift amount and device distributions for the tail of the stream."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    start = int(len(df) * start_frac)
    idx = df.index[start:]
    noise = rng.lognormal(0.0, 0.25, len(idx))
    df.loc[idx, "TransactionAmt"] = np.round(
        df.loc[idx, "TransactionAmt"].to_numpy() * amount_mult * noise, 2
    )
    swap = rng.random(len(idx)) < device_share
    sw_idx = idx[swap]
    df["DeviceInfo"] = df["DeviceInfo"].astype(object)
    df["DeviceType"] = df["DeviceType"].astype(object)
    df.loc[sw_idx, "DeviceInfo"] = rng.choice(NEW_DEVICES, len(sw_idx))
    df.loc[sw_idx, "DeviceType"] = "mobile"
    return df


def to_payload(rec: dict, fields: list[str]) -> dict:
    out = {}
    for k in fields:
        v = rec.get(k)
        if v is None:
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            continue
        out[k] = v
    return out


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=None, help="only the first N live transactions")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N live transactions")
    ap.add_argument(
        "--batch-size", type=int, default=1, help="1 = POST /score per txn, >1 = /score/batch"
    )
    ap.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="simulated seconds per wall second (0 = as fast as the API allows)",
    )
    ap.add_argument("--label-delay-hours", type=float, default=cfg["monitor"]["label_delay_hours"])
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--drift-start", type=float, default=0.0)
    ap.add_argument("--drift-amount-mult", type=float, default=2.5)
    ap.add_argument("--drift-device-share", type=float, default=0.6)
    ap.add_argument(
        "--flush-labels",
        action="store_true",
        help="at the end, reveal labels whose delay has passed by the stream end + delay",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    live = pd.read_parquet(path("processed_dir") / "live.parquet")
    live = live.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    live = live.iloc[args.offset :]
    if args.limit:
        live = live.iloc[: args.limit]
    live = live.reset_index(drop=True)
    if args.drift:
        live = apply_drift(
            live, args.drift_start, args.drift_amount_mult, args.drift_device_share, args.seed
        )

    client = httpx.Client(base_url=args.api_url, timeout=30.0)
    fields = client.get("/metadata").json()["input_fields"]
    delay = args.label_delay_hours * 3600
    pending: list[tuple[int, int, int, int]] = []  # (reveal_dt, txid, label, dt)
    stats = {
        "sent": 0,
        "errors": 0,
        "labels_sent": 0,
        "decisions": {"approve": 0, "review": 0, "block": 0},
    }
    latencies: list[float] = []
    t_start = time.perf_counter()
    sim_start = int(live["TransactionDT"].iloc[0])

    def reveal(until_dt: float) -> None:
        batch = []
        while pending and pending[0][0] <= until_dt:
            _, tid, lab, dt = heapq.heappop(pending)
            batch.append({"TransactionID": tid, "isFraud": lab, "TransactionDT": dt})
        for i in range(0, len(batch), 5000):
            client.post("/labels", json={"labels": batch[i : i + 5000]}).raise_for_status()
            stats["labels_sent"] += len(batch[i : i + 5000])

    records = live.to_dict("records")
    for i in range(0, len(records), args.batch_size):
        chunk = records[i : i + args.batch_size]
        now_dt = int(chunk[-1]["TransactionDT"])
        if args.speed > 0:
            target = (now_dt - sim_start) / args.speed
            sleep = target - (time.perf_counter() - t_start)
            if sleep > 0:
                time.sleep(sleep)
        payloads = [to_payload(r, fields) for r in chunk]
        t0 = time.perf_counter()
        try:
            if args.batch_size == 1:
                r = client.post("/score", json=payloads[0])
                results = [r.json()] if r.status_code == 200 else []
            else:
                r = client.post("/score/batch", json={"transactions": payloads})
                results = r.json()["results"] if r.status_code == 200 else []
            if r.status_code != 200:
                stats["errors"] += len(chunk)
                print(f"error {r.status_code}: {r.text[:300]}", file=sys.stderr)
        except httpx.HTTPError as e:
            stats["errors"] += len(chunk)
            print(f"http error: {e}", file=sys.stderr)
            results = []
        latencies.append((time.perf_counter() - t0) * 1000 / len(chunk))
        for res in results:
            stats["decisions"][res["decision"]] += 1
        stats["sent"] += len(chunk)
        for rec in chunk:
            heapq.heappush(
                pending,
                (
                    int(rec["TransactionDT"] + delay),
                    int(rec["TransactionID"]),
                    int(rec["isFraud"]),
                    int(rec["TransactionDT"]),
                ),
            )
        reveal(now_dt)
        if stats["sent"] % 2000 < args.batch_size:
            el = time.perf_counter() - t_start
            print(
                f"sent={stats['sent']:,} sim_day={(now_dt - sim_start) / 86400:.1f} "
                f"labels={stats['labels_sent']:,} rps={stats['sent'] / el:.0f} "
                f"decisions={stats['decisions']}",
                flush=True,
            )
    if args.flush_labels and len(records):
        reveal(records[-1]["TransactionDT"])
    stats["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    stats["client_latency_ms_per_txn"] = (
        {f"p{q}": round(float(np.percentile(latencies, q)), 2) for q in (50, 95, 99)}
        if latencies
        else {}
    )
    stats["pending_labels"] = len(pending)
    stats["drift"] = args.drift
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
