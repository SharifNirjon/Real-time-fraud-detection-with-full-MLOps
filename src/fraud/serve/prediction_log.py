"""Append-only parquet logs for predictions and delayed labels.

Records are buffered in memory and flushed to a new parquet file every
`max_rows` records or `max_seconds`, so the hot path never touches disk.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


class ParquetLog:
    def __init__(self, directory: Path, prefix: str, max_rows: int = 500, max_seconds: float = 5.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_rows = max_rows
        self.max_seconds = max_seconds
        self._buf: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def append(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            self._buf.extend(records)
            full = len(self._buf) >= self.max_rows
        if full:
            self.flush()

    def flush(self) -> int:
        with self._lock:
            buf, self._buf = self._buf, []
            self._last_flush = time.monotonic()
        if not buf:
            return 0
        name = f"{self.prefix}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = self.dir / f".{name}.tmp"
        pd.DataFrame(buf).to_parquet(tmp, index=False)
        tmp.rename(self.dir / name)  # atomic: readers never see half-written files
        return len(buf)

    def _loop(self) -> None:
        while not self._stop.wait(1.0):
            if time.monotonic() - self._last_flush >= self.max_seconds:
                try:
                    self.flush()
                except Exception:  # never kill the thread
                    log.exception("prediction log flush failed")

    def close(self) -> None:
        self._stop.set()
        self.flush()


def read_log(directory: Path) -> pd.DataFrame:
    files = sorted(Path(directory).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
