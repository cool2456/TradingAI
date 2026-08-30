"""The power test: the pipeline must find an edge that was deliberately planted.

The null test proves the system does not hallucinate. On its own that is
satisfied by a system that always reports zero. This file proves the other
half: on an Ornstein-Uhlenbeck process with a known half-life there is a real,
exploitable mean reversion, and the pipeline must recover it out of sample and
net of costs.

Every test here is run against a matched GBM control using the identical
procedure, because "found an edge" is only meaningful relative to "found
nothing when there was nothing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import run_strategy, t_stat_of_mean
from quantlab.metrics import sharpe
from quantlab.signals import zscore_reversion
from quantlab.simulate import gbm, ornstein_uhlenbeck

N_SEEDS = 40
N_STEPS = 2500
HALF_LIFE = 10.0
COST_BPS = 2.0
WINDOW_GRID = (5, 10, 20, 40, 80)


def _select_then_evaluate(prices: pd.DataFrame, cost_bps: float = COST_BPS) -> tuple[float, int]:
    """Pick the z-score window on the first half, score it on the second.

    A single split rather than full walk-forward -- ``validate.py`` does not
    exist yet at this point in the build order -- but it is a genuine
    out-of-sample evaluation: the test half is untouched during selection.
    """
    mid = len(prices) // 2
    train, test = prices.iloc[:mid], prices.iloc[mid:]
    scores = {
        w: sharpe(run_strategy(train, zscore_reversion.bind(window=w), cost_bps).net_returns)
        for w in WINDOW_GRID
    }
    best = max(scores, key=lambda w: scores[w])
    oos = sharpe(run_strategy(test, zscore_reversion.bind(window=best), cost_bps).net_returns)
    return oos, best


def _panel(generator, n_seeds: int = N_SEEDS, cost_bps: float = COST_BPS) -> pd.DataFrame:
    rows = [_select_then_evaluate(generator(2000 + s), cost_bps) for s in range(n_seeds)]
    return pd.DataFrame(rows, columns=["oos_sharpe", "chosen_window"])


def _ou(seed: int, half_life: float = HALF_LIFE) -> pd.DataFrame:
    return ornstein_uhlenbeck(n_steps=N_STEPS, n_assets=1, half_life=half_life,
                              sigma_ann=0.20, seed=seed)


def _gbm(seed: int) -> pd.DataFrame:
    return gbm(n_steps=N_STEPS, n_assets=1, mu_ann=0.0, sigma_ann=0.20, seed=seed)


@pytest.fixture(scope="module")
def ou_panel() -> pd.DataFrame:
    return _panel(_ou)


@pytest.fixture(scope="module")
def gbm_panel() -> pd.DataFrame:
    return _panel(_gbm)


def test_planted_edge_is_recovered_out_of_sample(ou_panel: pd.DataFrame) -> None:
    """Positive out-of-sample net Sharpe with a t-statistic above 2."""
    oos = ou_panel["oos_sharpe"].to_numpy()
    t = t_stat_of_mean(oos)
    assert oos.mean() > 0.5, f"mean OOS Sharpe {oos.mean():.3f} is too weak to call an edge"
    assert t > 2.0, f"t-statistic {t:.2f} does not clear 2"
    assert (oos > 0).mean() > 0.85, (
        f"only {(oos > 0).mean():.0%} of seeds were profitable out of sample"
    )


def test_matched_control_finds_nothing(gbm_panel: pd.DataFrame) -> None:
    """The identical procedure on a random walk must not clear the same bar."""
    oos = gbm_panel["oos_sharpe"].to_numpy()
    assert t_stat_of_mean(oos) < 2.0, (
        "the selection procedure found a significant edge in Brownian motion; "
        "the split is leaking"
    )
    assert oos.mean() < 0.2


def test_edge_and_control_are_clearly_separated(
    ou_panel: pd.DataFrame, gbm_panel: pd.DataFrame
) -> None:
    """A two-sample comparison, which is the claim that actually matters."""
    from scipy import stats

    t, p = stats.ttest_ind(
        ou_panel["oos_sharpe"], gbm_panel["oos_sharpe"], equal_var=False
    )
    assert t > 5 and p < 1e-6, f"OU vs GBM separation is weak: t={t:.2f}, p={p:.3g}"


def test_parameter_selection_is_stable_only_when_an_edge_exists(
    ou_panel: pd.DataFrame, gbm_panel: pd.DataFrame
) -> None:
    """Parameter instability across seeds is itself a finding.

    When a real edge exists the selected window converges: the same parameter
    wins on almost every path. Under the null there is nothing to select, so
    the choice is driven by noise and scatters roughly uniformly across the
    grid. Normalised entropy of the selection distribution separates the two
    cleanly, and is the diagnostic ``walk_forward`` surfaces per fold.
    """

    def normalised_entropy(chosen: pd.Series) -> float:
        p = chosen.value_counts(normalize=True).reindex(WINDOW_GRID).fillna(0.0)
        nonzero = p[p > 0]
        return float(-(nonzero * np.log(nonzero)).sum() / np.log(len(WINDOW_GRID)))

    ou_entropy = normalised_entropy(ou_panel["chosen_window"])
    gbm_entropy = normalised_entropy(gbm_panel["chosen_window"])

    assert ou_entropy < 0.45, f"parameter choice on a real edge should concentrate, got H={ou_entropy:.2f}"
    assert gbm_entropy > 0.70, f"parameter choice under the null should scatter, got H={gbm_entropy:.2f}"
    assert gbm_entropy > ou_entropy + 0.25


def test_faster_mean_reversion_is_a_stronger_edge() -> None:
    """A shorter half-life is a larger, more exploitable edge.

    A monotonicity check on the whole pipeline: if a mechanically stronger
    signal does not produce a higher Sharpe, the pipeline is not measuring the
    thing it claims to measure.
    """
    fast = _panel(lambda s: _ou(s, half_life=5.0), n_seeds=20)["oos_sharpe"].mean()
    slow = _panel(lambda s: _ou(s, half_life=40.0), n_seeds=20)["oos_sharpe"].mean()
    assert fast > slow > 0, f"half-life 5 gave {fast:.2f}, half-life 40 gave {slow:.2f}"


def test_costs_monotonically_erode_the_edge() -> None:
    """Raising costs must lower net Sharpe, and enough of them must kill it."""
    sharpes = [_panel(_ou, n_seeds=15, cost_bps=c)["oos_sharpe"].mean()
               for c in (0.0, 5.0, 50.0)]
    assert sharpes[0] > sharpes[1] > sharpes[2], f"non-monotone in cost: {sharpes}"
    assert sharpes[2] < sharpes[0] / 2, (
        "a 50bp cost barely dented a high-turnover reversion strategy; "
        "costs are probably not being charged per trade"
    )
