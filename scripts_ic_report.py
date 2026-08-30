"""Step 5: the experiment. Run the pre-registered IC grid on real data.

This is the whole point of Phase 2. Whatever it prints is the answer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.live.featurestore import FeatureStore, session_id
from quantlab.live.ic import (
    cross_sectional_demean,
    load_hypotheses,
    mark_tested,
    run_ic_grid,
)
from quantlab.validate import current_threshold, trial_count

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def build_panel(store: FeatureStore) -> pd.DataFrame:
    bars = store.load(columns=["timestamp", "symbol", "close"])
    panel = bars.pivot_table(index="timestamp", columns="symbol",
                             values="close", aggfunc="last").sort_index()

    sessions = pd.Series(session_id(pd.Series(panel.index)).to_numpy(), index=panel.index)
    # Forward-fill within a session only. A symbol that did not trade in a given
    # minute still has a last price, and that is what a trader would see. Filling
    # ACROSS sessions would carry a stale price over the overnight gap, which is
    # not a price anyone could have traded.
    filled = panel.groupby(sessions.to_numpy()).ffill()
    return filled, panel, sessions


def data_quality(raw: pd.DataFrame, filled: pd.DataFrame, sessions: pd.Series) -> None:
    n_sessions = sessions.nunique()
    bars_per_session = raw.groupby(sessions.to_numpy()).size()
    print("=" * 110)
    print("DATA")
    print("=" * 110)
    print(f"  symbols            {raw.shape[1]}")
    print(f"  timestamps         {len(raw):,}")
    print(f"  sessions           {n_sessions}   ({bars_per_session.min()}-{bars_per_session.max()} bars each, median {int(bars_per_session.median())})")
    print(f"  span               {raw.index.min().date()} .. {raw.index.max().date()}")
    print(f"  panel cells        {raw.size:,}")
    print(f"  missing before ffill {raw.isna().to_numpy().mean():.2%}")
    print(f"  missing after  ffill {filled.isna().to_numpy().mean():.2%}")

    returns = np.log(filled).diff()
    residual = cross_sectional_demean(returns)
    def mean_pair(frame):
        c = frame.corr().to_numpy()
        off = c[~np.eye(c.shape[0], dtype=bool)]
        return float(np.nanmean(off))
    raw_rho, res_rho = mean_pair(returns), mean_pair(residual)
    n = raw.shape[1]
    n_eff = lambda rho: n / (1 + (n - 1) * rho) if (1 + (n - 1) * rho) > 0 else float("inf")
    print()
    print(f"  mean pairwise corr, raw returns       {raw_rho:+.4f}  -> N_eff {n_eff(raw_rho):6.1f} of {n}")
    print(f"  mean pairwise corr, residual returns  {res_rho:+.4f}  -> N_eff {n_eff(res_rho):6.1f} of {n}")
    print("  (residualising is what converts breadth into effective sample size)")


def main() -> None:
    store = FeatureStore("data/features")
    filled, raw, sessions = build_panel(store)
    data_quality(raw, filled, sessions)

    session_length = int(pd.Series(session_id(pd.Series(filled.index))).value_counts().median())
    print()
    print("=" * 110)
    print(f"IC GRID  (pre-registered, session length {session_length} bars)")
    print("=" * 110)
    report = run_ic_grid(filled, session_length=session_length, residualise=True)

    print()
    print("=" * 110)
    print("RESULT")
    print("=" * 110)
    frame = report.ranking()
    show = frame.copy()
    for col in ("ic", "p_value", "p_value_bh", "p_value_by"):
        show[col] = show[col].map(lambda v: "--" if pd.isna(v) else f"{v:.4f}")
    for col in ("t_iid", "t_hac"):
        show[col] = show[col].map(lambda v: "--" if pd.isna(v) else f"{v:+.2f}")
    for col in ("n_raw", "n_effective", "n_effective_naive"):
        show[col] = show[col].map(lambda v: "--" if pd.isna(v) else f"{v:,.0f}")
    show["variance_inflation"] = show["variance_inflation"].map(
        lambda v: "--" if pd.isna(v) else f"{v:.1f}")
    show["note"] = show["note"].str.slice(0, 28)
    print(show.to_string(index=False))

    print()
    print(report.summary_line())
    survivors = report.survivors()
    if not survivors.empty:
        print()
        print("SURVIVORS (Benjamini-Yekutieli corrected, alpha=0.05):")
        print(survivors[["hypothesis_id", "signal", "horizon_bars", "ic", "t_hac",
                         "n_effective", "p_value_by"]].to_string(index=False))

    print()
    verdicts = mark_tested(report)
    print("hypotheses.yaml status updated:")
    for hid, verdict in verdicts.items():
        print(f"  {hid}: {verdict[:96]}")

    print()
    print(f"trials logged: {trial_count()}")
    th = current_threshold(2500)
    print(f"DSR luck threshold: {th['luck_threshold_ann']:.3f} annualised Sharpe")
    frame.to_csv("ic_report.csv", index=False)
    print("written: ic_report.csv")


if __name__ == "__main__":
    main()
