"""Regenerate the IC report with the raw-vs-residual comparison.

log=False on purpose: these are the SAME 21 pre-registered hypotheses already
charged to research_log.jsonl. Re-measuring one hypothesis is not a new trial,
and logging it again would inflate the deflated-Sharpe threshold for work that
entertained no new hypothesis.
"""
import pandas as pd
from quantlab.live.ic import run_ic_grid, mark_tested
from quantlab.validate import current_threshold, trial_count

panel = pd.read_parquet("data/panel_close.parquet")
report = run_ic_grid(panel, session_length=390, residualise=True, log=False, verbose=False)
frame = report.ranking()
frame.to_csv("ic_report.csv", index=False)

show = frame.copy()
show["beta_share"] = (1 - (show["ic"].abs() / show["ic_raw"].abs())).map(
    lambda v: "--" if pd.isna(v) else f"{v:+.0%}")
for c in ("ic_raw", "ic", "p_value_bh", "p_value_by"):
    show[c] = show[c].map(lambda v: "--" if pd.isna(v) else f"{v:+.4f}".replace("+0.", " 0."))
for c in ("t_hac_raw", "t_iid", "t_hac"):
    show[c] = show[c].map(lambda v: "--" if pd.isna(v) else f"{v:+.2f}")
for c in ("n_effective",):
    show[c] = show[c].map(lambda v: "--" if pd.isna(v) else f"{v:,.0f}")
cols = ["hypothesis_id", "signal", "horizon_bars", "ic_raw", "ic", "beta_share",
        "t_iid", "t_hac_raw", "t_hac", "n_effective", "p_value_bh", "p_value_by"]
print(show[cols].to_string(index=False))
print()
print(report.summary_line())
print(f"trials logged: {trial_count()} (unchanged -- re-measurement is not a new trial)")
print(f"DSR luck threshold: {current_threshold(2500)['luck_threshold_ann']:.3f}")
mark_tested(report)
