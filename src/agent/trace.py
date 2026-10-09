"""Minimal per-request span timer. Step 12 adds JSONL output on top of this."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Trace:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0.0) + (time.perf_counter() - t) * 1000

    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000