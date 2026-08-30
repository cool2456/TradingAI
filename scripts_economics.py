"""Is the measured IC economically meaningful, and is it a microstructure artifact?

Step 5 established statistical detectability. The phase question asks for a
horizon "where costs do not eat it", which is a different and harder bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.live.featurestore import FeatureStore, forward_returns, session_id
from quantlab.live.ic import compute_ic, cross_sectional_demean, mark_tested, ICReport, ICResult
from quantlab.signals import SIGNALS

PANEL_CACHE = "data/panel_close.parquet"


def load_panel() -> pd.DataFrame:
    import os
    if os.path.exists(PANEL_CACHE):
        return pd.read_parquet(PANEL_CACHE)
    store = FeatureStore("data/features")
    bars = store.load(columns=["timestamp", "symbol", "close"])
    panel = bars.pivot_table(index="timestamp", columns="symbol", values="close",
                             aggfunc="last").sort_index()
    sessions = pd.Series(session_id(pd.Series(panel.index)).to_numpy(), index=panel.index)
    panel = panel.groupby(sessions.to_numpy()).ffill()
    panel.to_parquet(PANEL_CACHE)
    return panel


panel = load_panel()
report = pd.read_csv("ic_report.csv")
survivors = report[report["p_value_by"] < 0.05].copy()

print("=" * 104)
print("ECONOMIC SIGNIFICANCE")
print("=" * 104)
print("A dollar-neutral book sized proportional to the standardised signal z earns")
print("  gross per round trip  = IC * sigma_r        (sigma_r = sd of the residual h-bar return)")
print("  cost  per round trip  = 2 * c * E|z|        (c = one-way cost rate, E|z| ~ 0.80)")
print("so it breaks even at    c* = IC * sigma_r / (2 * E|z|).")
print()

rows = []
for _, r in survivors.iterrows():
    h = int(r["horizon_bars"])
    within = h + 1 <= 389
    fwd = forward_returns(panel, h, within_session=within, entry_lag=1)
    resid = cross_sectional_demean(fwd)
    sigma_r = float(np.nanstd(resid.to_numpy()))

    sig = SIGNALS[r["signal"]].bind(**eval(str(report.loc[_, "signal"]) and "{}") if False else {})
    rows.append((r, h, sigma_r))

# Signal params come from the registration, not from the report csv.
import yaml
params_by_signal = {h["signal"]: h["params"] for h in yaml.safe_load(open("hypotheses.yaml"))["hypotheses"]}

print(f"{'signal':22s} {'h':>4s} {'IC':>9s} {'sigma_r':>9s} {'E|z|':>6s} {'gross/RT':>10s} {'breakeven c*':>13s}")
print("-" * 104)
out = []
for r, h, sigma_r in rows:
    name = r["signal"]
    sig_panel = SIGNALS[name].bind(**params_by_signal[name])(panel)
    z = sig_panel.to_numpy()
    z = z[np.isfinite(z)]
    z = (z - z.mean()) / z.std()
    e_abs_z = float(np.abs(z).mean())
    ic = float(r["ic"])
    gross = abs(ic) * sigma_r
    c_star = gross / (2 * e_abs_z)
    out.append({"signal": name, "h": h, "ic": ic, "sigma_r": sigma_r,
                "e_abs_z": e_abs_z, "gross_bps": gross * 1e4, "c_star_bps": c_star * 1e4})
    print(f"{name:22s} {h:4d} {ic:+9.4f} {sigma_r*100:8.3f}% {e_abs_z:6.2f} "
          f"{gross*1e4:9.3f}bp {c_star*1e4:12.3f}bp")

econ = pd.DataFrame(out)
print()
print("Reference one-way costs for US large-cap equities:")
print("  half-spread on the most liquid names   ~0.25 - 0.50 bp")
print("  plus impact for a small clip           ~0.25 - 1.0  bp")
print("  realistic one-way all-in                ~0.5  - 1.5  bp")
print()
best = econ.loc[econ["c_star_bps"].idxmax()]
print(f"Highest break-even cost across all survivors: {best['c_star_bps']:.3f} bp one-way "
      f"({best['signal']} at h={int(best['h'])})")
verdict = "ABOVE" if best["c_star_bps"] > 0.5 else "BELOW"
print(f"-> {verdict} the optimistic 0.5bp one-way floor")
econ.to_csv("ic_economics.csv", index=False)

print()
print("=" * 104)
print("ROBUSTNESS: does the effect survive a later entry? (diagnostic, not logged as trials)")
print("=" * 104)
print("A microstructure artifact decays within a few bars. A real effect persists.")
print()
print(f"{'signal':22s} {'h':>4s} " + " ".join(f"{'lag=' + str(l):>16s}" for l in (1, 2, 5, 10, 30)))
print("-" * 104)
for _, r in survivors.iterrows():
    name, h = r["signal"], int(r["horizon_bars"])
    sig_panel = SIGNALS[name].bind(**params_by_signal[name])(panel)
    cells = []
    for lag in (1, 2, 5, 10, 30):
        within = h + lag <= 389
        fwd = forward_returns(panel, h, within_session=within, entry_lag=lag)
        res = compute_ic(sig_panel, fwd, h, residualise=True)
        cells.append(f"{res.ic:+.4f}(t{res.t_hac:+5.2f})")
    print(f"{name:22s} {h:4d} " + " ".join(f"{c:>16s}" for c in cells))
