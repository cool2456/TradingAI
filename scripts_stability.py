"""Split-half stability. Diagnostic, not logged as trials.

A single-period IC is one number from one year. If the effect is real it must
appear in both halves; if it appears in one and not the other, the full-sample
figure is an average over a period that had it and a period that did not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from quantlab.live.featurestore import forward_returns
from quantlab.live.ic import compute_ic
from quantlab.signals import SIGNALS

panel = pd.read_parquet("data/panel_close.parquet")
params_by_signal = {h["signal"]: h["params"] for h in yaml.safe_load(open("hypotheses.yaml"))["hypotheses"]}
survivors = pd.read_csv("ic_report.csv")
survivors = survivors[survivors["p_value_by"] < 0.05]

mid = len(panel) // 2
halves = {"H1 " + str(panel.index[0].date()): panel.iloc[:mid],
          "H2 " + str(panel.index[mid].date()): panel.iloc[mid:]}
quarters = {f"Q{i+1}": panel.iloc[i * len(panel) // 4:(i + 1) * len(panel) // 4] for i in range(4)}

def ic_for(sub: pd.DataFrame, name: str, h: int):
    sig = SIGNALS[name].bind(**params_by_signal[name])(sub)
    within = h + 1 <= 389
    fwd = forward_returns(sub, h, within_session=within, entry_lag=1)
    return compute_ic(sig, fwd, h, residualise=True)

print("=" * 100)
print("SPLIT-HALF")
print("=" * 100)
print(f"{'signal':22s} {'h':>4s} {'full year':>18s} " + " ".join(f"{k:>18s}" for k in halves))
print("-" * 100)
for _, r in survivors.iterrows():
    name, h = r["signal"], int(r["horizon_bars"])
    cells = [f"{r['ic']:+.4f}(t{r['t_hac']:+5.2f})"]
    for sub in halves.values():
        res = ic_for(sub, name, h)
        cells.append(f"{res.ic:+.4f}(t{res.t_hac:+5.2f})")
    print(f"{name:22s} {h:4d} " + " ".join(f"{c:>18s}" for c in cells))

print()
print("=" * 100)
print("QUARTERS")
print("=" * 100)
print(f"{'signal':22s} {'h':>4s} " + " ".join(f"{k:>18s}" for k in quarters))
print("-" * 100)
signs = {}
for _, r in survivors.iterrows():
    name, h = r["signal"], int(r["horizon_bars"])
    cells, ics = [], []
    for sub in quarters.values():
        res = ic_for(sub, name, h)
        cells.append(f"{res.ic:+.4f}(t{res.t_hac:+5.2f})")
        ics.append(res.ic)
    signs[f"{name}@{h}"] = ics
    print(f"{name:22s} {h:4d} " + " ".join(f"{c:>18s}" for c in cells))

print()
full_sign = {f"{r['signal']}@{int(r['horizon_bars'])}": np.sign(r["ic"]) for _, r in survivors.iterrows()}
for key, ics in signs.items():
    agree = sum(1 for x in ics if np.sign(x) == full_sign[key])
    print(f"  {key:28s} sign agrees with full sample in {agree}/4 quarters")
