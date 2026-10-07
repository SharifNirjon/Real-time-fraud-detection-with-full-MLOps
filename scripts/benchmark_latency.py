"""Async load test for POST /score.

Sends N real transactions (from the live stream, in time order) with a fixed
number of concurrent in-flight requests and reports client-side p50/p95/p99
latency, throughput and the server-reported model latency.

Note: requests update the online feature store and the prediction log like
real traffic. Re-run `make warm` (and `make clean`) before a simulation.

Usage: python scripts/benchmark_latency.py --api-url http://localhost:8000 --requests 2000 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
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

from fraud.config import path  # noqa: E402


def payloads(n: int, fields: list[str]) -> list[dict]:
    live = pd.read_parquet(path("processed_dir") / "live.parquet").head(n)
    out = []
    for rec in live.to_dict("records"):
        p = {}
        for k in fields:
            v = rec.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            p[k] = v.item() if hasattr(v, "item") else v
        out.append(p)
    return out


async def run(api_url: str, n: int, concurrency: int, warmup: int) -> dict:
    async with httpx.AsyncClient(
        base_url=api_url, timeout=30.0, limits=httpx.Limits(max_connections=concurrency)
    ) as client:
        fields = (await client.get("/metadata")).json()["input_fields"]
        data = payloads(n + warmup, fields)
        for p in data[:warmup]:
            await client.post("/score", json=p)
        queue: asyncio.Queue = asyncio.Queue()
        for p in data[warmup:]:
            queue.put_nowait(p)
        client_ms: list[float] = []
        server_ms: list[float] = []
        errors = 0

        async def worker() -> None:
            nonlocal errors
            while not queue.empty():
                p = queue.get_nowait()
                t = time.perf_counter()
                r = await client.post("/score", json=p)
                client_ms.append((time.perf_counter() - t) * 1000)
                if r.status_code == 200:
                    server_ms.append(r.json()["latency_ms"])
                else:
                    errors += 1

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        elapsed = time.perf_counter() - t0

    def pct(a: list[float]) -> dict[str, float]:
        return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)}

    return {
        "requests": len(client_ms),
        "concurrency": concurrency,
        "errors": errors,
        "throughput_rps": round(len(client_ms) / elapsed, 1),
        "client_latency_ms": pct(client_ms),
        "server_latency_ms": pct(server_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default=str(ROOT / "reports" / "latency.json"))
    args = ap.parse_args()
    results = []
    for c in args.concurrency:
        res = asyncio.run(run(args.api_url, args.requests, c, args.warmup))
        print(json.dumps(res), flush=True)
        results.append(res)
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
