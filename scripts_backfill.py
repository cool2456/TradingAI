"""Resumable 1-minute SIP backfill into the feature store.

Chunked by month and by symbol group so a failure loses at most one chunk, and
so progress is observable. Completed chunks are recorded in a manifest; re-running
skips them, which matters because the store is append-only and a re-run would
otherwise duplicate bars.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quantlab.live.config import Adjustment, Feed
from quantlab.live.datafeed import fetch_history
from quantlab.live.featurestore import FeatureStore

STORE = Path("data/features")
MANIFEST = Path("data/backfill_manifest.json")
SYMBOL_GROUP = 20


def main(months_limit: int | None = None) -> None:
    universe = pd.read_csv("data_universe.csv")["symbol"].tolist()
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=365)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    done = set(json.loads(MANIFEST.read_text())) if MANIFEST.exists() else set()
    store = FeatureStore(STORE, buffer_size=10_000_000)

    edges = pd.date_range(start, end, freq="MS", tz="UTC").tolist()
    bounds = [start] + edges + [end]
    bounds = sorted({b for b in bounds if start <= b <= end})
    windows = list(zip(bounds[:-1], bounds[1:]))
    if months_limit:
        windows = windows[:months_limit]

    groups = [universe[i:i + SYMBOL_GROUP] for i in range(0, len(universe), SYMBOL_GROUP)]
    total = len(windows) * len(groups)
    completed = bars_total = 0
    t_start = time.time()

    for w_start, w_end in windows:
        for gi, group in enumerate(groups):
            key = f"{w_start.date()}|{gi}"
            completed += 1
            if key in done:
                continue
            t0 = time.time()
            bars = fetch_history(
                group, w_start, w_end, feed=Feed.SIP,
                adjustment=Adjustment.ALL, timeframe_minutes=1,
                symbol_chunk=SYMBOL_GROUP, regular_hours_only=True,
            )
            fetched = time.time() - t0
            n = store.log_bars(bars) if not bars.empty else 0
            bars_total += n
            done.add(key)
            MANIFEST.write_text(json.dumps(sorted(done)))
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed else 0
            eta = (total - completed) / rate / 60 if rate else 0
            print(
                f"[{completed:4d}/{total}] {w_start.date()} g{gi:02d} "
                f"{n:7d} bars  fetch {fetched:5.1f}s  write {time.time()-t0-fetched:5.1f}s "
                f"| total {bars_total:9,d}  ETA {eta:5.1f}m",
                flush=True,
            )
    print(f"DONE: {bars_total:,} bars in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
