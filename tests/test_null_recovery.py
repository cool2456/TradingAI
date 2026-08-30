"""The null test: the pipeline must find nothing in Brownian motion.

Geometric Brownian motion has independent increments. No causal function of
the past has any correlation with the future, so the correct answer for every
strategy is a gross Sharpe of zero and a net Sharpe of *minus the cost*.

If this file fails, nothing else in the repository means anything: a pipeline
that manufactures edge from noise will manufacture it from real data too, and
no downstream statistic can tell the difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import run_strategy, t_stat_of_mean
from quantlab.metrics import deflated_sharpe, sharpe, turnover
from quantlab.signals import default_signal_set
from quantlab.simulate import gbm

N_SEEDS = 200
N_STEPS = 1200
COST_BPS = 2.0

#: The suite tests 7 signals at once and therefore faces exactly the multiple-
#: testing problem it exists to police. A per-signal threshold of 2.0 would
#: fail roughly 30% of the time on 7 independent nulls. Bonferroni at a family
#: -wise 5% gives |t| < 2.69.
N_SIGNALS = 7
BONFERRONI_T = float(stats.norm.ppf(1 - 0.025 / N_SIGNALS))


@pytest.fixture(scope="module")
def null_panel() -> pd.DataFrame:
    """Per-seed, per-signal results over ``N_SEEDS`` independent GBM paths."""
    rows = []
    signals = default_signal_set()
    for seed in range(N_SEEDS):
        prices = gbm(n_steps=N_STEPS, n_assets=1, mu_ann=0.0, sigma_ann=0.20, seed=1000 + seed)
        for signal in signals:
            result = run_strategy(prices, signal, cost_bps=COST_BPS)
            rows.append(
                {
                    "seed": seed,
                    "signal": signal.name,
                    "net": sharpe(result.net_returns),
                    "gross": sharpe(result.gross_returns),
                    # n_trials=1 on purpose: this measures the *uncorrected*
                    # false-positive rate. Deflating by the global trial count
                    # from research_log.jsonl would crush every probability and
                    # let the test pass without testing anything.
                    "psr_net": deflated_sharpe(result.net_returns, n_trials=1),
                    "psr_gross": deflated_sharpe(result.gross_returns, n_trials=1),
                    "turnover": turnover(result.positions),
                }
            )
    return pd.DataFrame(rows)


def test_gross_sharpe_is_centred_on_zero(null_panel: pd.DataFrame) -> None:
    """Before costs, every strategy must average zero Sharpe on a random walk."""
    failures = {}
    for name, group in null_panel.groupby("signal"):
        t = t_stat_of_mean(group["gross"].to_numpy())
        if abs(t) > BONFERRONI_T:
            failures[name] = (group["gross"].mean(), t)
    assert not failures, (
        f"gross Sharpe significantly non-zero on GBM (|t| > {BONFERRONI_T:.2f}): {failures}. "
        "The pipeline is extracting signal from noise."
    )


def test_net_sharpe_is_negative_not_zero(null_panel: pd.DataFrame) -> None:
    """With zero skill a strategy does not break even -- it bleeds cost.

    This is the correction to the specification in the project brief, which
    asks for the mean out-of-sample Sharpe to be within two standard errors of
    zero. That is the right test for *gross* returns and the wrong test for
    net: a system whose net Sharpe on Brownian motion were genuinely zero
    would be failing to charge for its own trading.
    """
    per_signal = null_panel.groupby("signal")[["net", "gross"]].mean()
    assert (per_signal["net"] < per_signal["gross"] + 1e-12).all(), (
        "some signal earned more net than gross; costs are not being charged"
    )

    # The pooled mean across every strategy and seed must be strictly negative.
    assert t_stat_of_mean(null_panel["net"].to_numpy()) < -2.0, (
        "pooled net Sharpe on GBM is not significantly negative; "
        f"mean = {null_panel['net'].mean():.4f}"
    )


def test_cost_bleed_is_proportional_to_turnover(null_panel: pd.DataFrame) -> None:
    """The gross-to-net gap must be explained by trading, and only by trading.

    A near-perfect correlation is the mechanistic check that the cost model is
    wired to turnover rather than applied as an arbitrary haircut.
    """
    per_signal = null_panel.groupby("signal").agg(
        gap=("gross", "mean"), net=("net", "mean"), turnover=("turnover", "mean")
    )
    gap = per_signal["gap"] - per_signal["net"]
    assert np.corrcoef(gap, per_signal["turnover"])[0, 1] > 0.98

    # The highest-turnover strategy must be significantly unprofitable.
    busiest = per_signal["turnover"].idxmax()
    busiest_net = null_panel.loc[null_panel["signal"] == busiest, "net"].to_numpy()
    assert t_stat_of_mean(busiest_net) < -3.0, (
        f"{busiest} turns over {per_signal.loc[busiest, 'turnover']:.0f}x per year "
        "on pure noise and is not significantly losing money"
    )


def test_false_positive_rate_is_at_its_nominal_level(null_panel: pd.DataFrame) -> None:
    """A 95% test must fire on about 5% of nulls -- not 30%, and not 0%.

    Both directions matter. Over-rejection means the library declares noise
    significant. Under-rejection to zero would mean the test has no power and
    would never detect a real edge either.
    """
    gross_rate = float((null_panel["psr_gross"] > 0.95).mean())
    net_rate = float((null_panel["psr_net"] > 0.95).mean())

    n = len(null_panel)
    tolerance = 3 * np.sqrt(0.05 * 0.95 / n)
    assert gross_rate < 0.05 + tolerance, (
        f"gross false-positive rate {gross_rate:.1%} exceeds nominal 5%"
    )
    assert 0.01 < gross_rate, (
        f"gross rejection rate {gross_rate:.1%} is implausibly low; the test may have no power"
    )
    # Net returns carry a negative cost drift, so the rate must be *below* 5%.
    assert net_rate <= gross_rate + tolerance


def test_deflation_removes_the_survivor(null_panel: pd.DataFrame) -> None:
    """Take the best of 200 null seeds; the DSR must not call it significant.

    This is the multiple-testing scenario in miniature: 200 worthless
    strategies, keep the winner, report it. Undeflated it looks excellent.
    """
    best_seed = int(null_panel.loc[null_panel["net"].idxmax(), "seed"])
    best_signal = null_panel.loc[null_panel["net"].idxmax(), "signal"]
    prices = gbm(n_steps=N_STEPS, n_assets=1, sigma_ann=0.20, seed=1000 + best_seed)
    signal = next(s for s in default_signal_set() if s.name == best_signal)
    result = run_strategy(prices, signal, cost_bps=COST_BPS)

    n_trials = len(null_panel)
    trial_variance = float(
        (null_panel["net"].to_numpy() / np.sqrt(252)).var(ddof=1)
    )  # per-period units
    deflated = deflated_sharpe(result.net_returns, n_trials=n_trials,
                               sharpe_variance=trial_variance)
    assert deflated < 0.95, (
        f"the best of {n_trials} pure-noise trials (Sharpe "
        f"{null_panel['net'].max():.2f}) survived deflation with DSR={deflated:.3f}"
    )
