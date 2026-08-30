"""Trade budget acceptance tests.

The budget is a hard constraint, so these are not statistical tests -- each one
either holds exactly or the constraint is not a constraint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.allocator import (
    DEFAULT_MIN_TRADE_SIZE,
    apply_trade_budget,
    constant_edge,
    regime_weights,
    select_trades,
)
from quantlab.signals import zscore_reversion
from quantlab.simulate import regime_switching

ASSETS = [f"asset_{i}" for i in range(6)]


def _row(values: dict[str, float]) -> pd.Series:
    return pd.Series({a: float(values.get(a, 0.0)) for a in ASSETS})


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return regime_switching(n_steps=800, n_assets=6, correlation=0.2, seed=3)


@pytest.fixture(scope="module")
def desired(panel: pd.DataFrame) -> pd.DataFrame:
    return zscore_reversion.bind(window=20)(panel).fillna(0.0)


# ------------------------------------------------------------ the hard cap


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 5])
def test_daily_trade_count_never_exceeds_budget(desired: pd.DataFrame, budget: int) -> None:
    result = apply_trade_budget(
        desired, None, budget=budget, cost_bps=2.0, expected_edge=constant_edge(desired, 5.0)
    )
    assert result.budget_used.max() <= budget
    # And the audit trail agrees with the position path.
    actual_changes = (result.positions.diff().abs() > 1e-12).sum(axis=1)
    assert actual_changes.groupby(actual_changes.index.normalize()).sum().max() <= budget


def test_budget_resets_each_day(panel: pd.DataFrame) -> None:
    """With several bars per day the cap applies per day, not per bar."""
    index = pd.date_range("2020-01-01 00:00", periods=240, freq="h")
    rng = np.random.default_rng(0)
    targets = pd.DataFrame(rng.uniform(-1, 1, size=(240, len(ASSETS))), index=index, columns=ASSETS)

    result = apply_trade_budget(
        targets, None, budget=3, cost_bps=1.0, expected_edge=constant_edge(targets, 10.0)
    )
    assert result.budget_used.max() <= 3
    assert len(result.budget_used) == 10          # ten calendar days
    assert result.budget_used.sum() == len(result.executed)
    # A budget that binds should be fully spent on most days.
    assert (result.budget_used == 3).mean() > 0.8


def test_zero_budget_freezes_the_book(desired: pd.DataFrame) -> None:
    result = apply_trade_budget(
        desired, None, budget=0, cost_bps=1.0, expected_edge=constant_edge(desired, 10.0)
    )
    assert (result.positions == 0.0).all().all()
    assert result.budget_used.sum() == 0


# ------------------------------------------------------ top-K by expected value


def test_executes_exactly_the_top_k_by_value() -> None:
    """Hand-built case: six candidates, budget of three, known ordering."""
    current = _row({})
    desired = _row({"asset_0": 1.0, "asset_1": 0.8, "asset_2": 0.6,
                    "asset_3": 0.4, "asset_4": 0.2, "asset_5": 0.1})
    edge = _row({a: 0.0010 for a in ASSETS})       # 10 bps everywhere
    cost_bps = 2.0                                  # 2 bps

    # value_i = |delta| * (edge - cost) = |delta| * 0.0008, so the ranking is
    # exactly by |delta|: asset_0 > asset_1 > ... > asset_5.
    new, decisions = select_trades(desired, current, budget=3, cost_bps=cost_bps,
                                   expected_edge=edge, min_trade_size=0.05)

    executed = {d.asset for d in decisions if d.executed}
    assert executed == {"asset_0", "asset_1", "asset_2"}
    assert new["asset_0"] == 1.0 and new["asset_3"] == 0.0

    frame = pd.DataFrame([d.__dict__ for d in decisions])
    top3 = set(frame.nlargest(3, "value")["asset"])
    assert executed == top3


def test_edge_not_size_can_determine_the_ranking() -> None:
    """A small trade with a large edge must outrank a large trade with none."""
    current = _row({})
    desired = _row({"asset_0": 1.0, "asset_1": 0.2})
    edge = _row({"asset_0": 0.00025, "asset_1": 0.0100})   # 2.5bps vs 100bps
    new, decisions = select_trades(desired, current, budget=1, cost_bps=2.0,
                                   expected_edge=edge, min_trade_size=0.05)
    executed = {d.asset for d in decisions if d.executed}
    assert executed == {"asset_1"}, "budget was spent on trade size rather than value"
    assert new["asset_0"] == 0.0


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_top_k_property_holds_on_random_rows(budget: int) -> None:
    """Randomised: the executed set is always the top-K of the positive-value set."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        current = pd.Series(rng.uniform(-1, 1, len(ASSETS)), index=ASSETS)
        desired = pd.Series(rng.uniform(-1, 1, len(ASSETS)), index=ASSETS)
        edge = pd.Series(rng.uniform(-0.002, 0.004, len(ASSETS)), index=ASSETS)

        _, decisions = select_trades(desired, current, budget, cost_bps=2.0,
                                     expected_edge=edge, min_trade_size=DEFAULT_MIN_TRADE_SIZE)
        frame = pd.DataFrame([d.__dict__ for d in decisions])
        eligible = frame[(frame["delta"].abs() > DEFAULT_MIN_TRADE_SIZE) & (frame["value"] > 0)]
        expected = set(eligible.nlargest(budget, "value")["asset"])
        assert set(frame.loc[frame["executed"], "asset"]) == expected


def test_negative_value_trades_are_never_taken_even_with_free_budget() -> None:
    """The budget is a ceiling, not a quota."""
    current = _row({})
    desired = _row({a: 1.0 for a in ASSETS})
    edge = _row({a: 0.00005 for a in ASSETS})   # 0.5 bps of edge
    # 10 bps of cost against 0.5 bps of edge: every trade is value-destroying.
    new, decisions = select_trades(desired, current, budget=99, cost_bps=10.0,
                                   expected_edge=edge, min_trade_size=0.05)
    assert not any(d.executed for d in decisions)
    assert (new == 0.0).all()
    assert all(d.reason == "negative_expected_value" for d in decisions)


def test_mixed_signs_takes_only_the_profitable_ones() -> None:
    current = _row({})
    desired = _row({a: 1.0 for a in ASSETS})
    edge = _row({"asset_0": 0.005, "asset_1": 0.004,
                 "asset_2": -0.001, "asset_3": -0.002,
                 "asset_4": 0.0, "asset_5": 0.00001})
    _, decisions = select_trades(desired, current, budget=6, cost_bps=2.0,
                                 expected_edge=edge, min_trade_size=0.05)
    executed = {d.asset for d in decisions if d.executed}
    assert executed == {"asset_0", "asset_1"}


# --------------------------------------------------------- the dead band


def test_sub_threshold_changes_are_not_trades() -> None:
    """Volatility targeting nudges every position; that must not spend budget."""
    current = _row({a: 0.5 for a in ASSETS})
    desired = current + 0.001                       # far below min_trade_size
    _, decisions = select_trades(desired, current, budget=3, cost_bps=1.0,
                                 expected_edge=_row({a: 0.01 for a in ASSETS}),
                                 min_trade_size=DEFAULT_MIN_TRADE_SIZE)
    assert not any(d.executed for d in decisions)
    assert all(d.reason == "below_min_trade_size" for d in decisions)


# ------------------------------------------------ cost structure properties


def test_proportional_cost_cannot_reorder_under_a_uniform_edge() -> None:
    """When every asset shares one edge estimate, cost cannot reorder anything.

    ``value_i = |delta_i| (e - c)`` is ``|delta_i|`` times a constant, so the
    queue is ordered by trade size alone and raising ``c`` can only gate trades
    out, never promote one above another. This is the regime the library
    defaults to, since :func:`constant_edge` is the honest default.
    """
    rng = np.random.default_rng(5)
    current = pd.Series(rng.uniform(-1, 1, len(ASSETS)), index=ASSETS)
    desired = pd.Series(rng.uniform(-1, 1, len(ASSETS)), index=ASSETS)
    edge = pd.Series(0.01, index=ASSETS)

    orderings = []
    for cost_bps in (0.0, 1.0, 5.0, 20.0):
        _, decisions = select_trades(desired, current, budget=6, cost_bps=cost_bps,
                                     expected_edge=edge, min_trade_size=0.01)
        frame = pd.DataFrame([d.__dict__ for d in decisions])
        orderings.append(tuple(frame.sort_values("value", ascending=False)["asset"]))
    assert len(set(orderings)) == 1, f"cost reordered a uniform-edge queue: {orderings}"


def test_proportional_cost_reorders_when_edges_differ() -> None:
    """A proportional cost tilts the queue toward small, high-edge trades.

    ``|delta| (edge - c)`` penalises a candidate in proportion to its size, so
    raising ``c`` hurts the big trade more in absolute terms. Here a full-size
    trade at 50bps of edge outranks a third-size trade at 100bps when trading
    is free, and loses to it once trading costs 40bps.
    """
    current = _row({})
    desired = _row({"asset_0": 1.0, "asset_1": 0.3})
    edge = _row({"asset_0": 0.005, "asset_1": 0.010})

    _, free = select_trades(desired, current, budget=1, cost_bps=0.0,
                            expected_edge=edge, min_trade_size=0.05)
    assert {d.asset for d in free if d.executed} == {"asset_0"}

    _, costly = select_trades(desired, current, budget=1, cost_bps=40.0,
                              expected_edge=edge, min_trade_size=0.05)
    assert {d.asset for d in costly if d.executed} == {"asset_1"}
    # Both remain profitable; the change is in ranking, not in viability.
    # (The other four assets have zero delta and are filtered by the dead band.)
    assert all(d.value > 0 for d in costly if d.asset in {"asset_0", "asset_1"})


def test_fixed_cost_never_reorders_but_imposes_a_minimum_trade_size() -> None:
    """A uniform ticket charge is a constant subtraction: it gates, never ranks.

    This is also the precise sense in which the brief's value function is
    mis-specified -- it subtracts a constant while calling it a proportional
    basis-point cost, so it behaves as a ticket charge.
    """
    current = _row({})
    desired = _row({"asset_0": 0.10, "asset_1": 1.00})
    edge = _row({"asset_0": 0.01, "asset_1": 0.01})

    orderings = []
    for fixed_cost_bps in (0.0, 5.0, 20.0, 100.0):
        _, decisions = select_trades(desired, current, budget=2, cost_bps=0.0,
                                     expected_edge=edge, min_trade_size=0.05,
                                     fixed_cost_bps=fixed_cost_bps)
        frame = pd.DataFrame([d.__dict__ for d in decisions])
        orderings.append(tuple(frame.sort_values("value", ascending=False)["asset"]))
    assert len(set(orderings)) == 1, "a uniform ticket charge reordered the queue"

    # 20bps of ticket charge exceeds the small trade's 10bps of gross benefit
    # (0.10 * 100bps) but not the large trade's 100bps, so it gates one and not
    # the other: |delta| > f / edge is the minimum viable size.
    _, gated = select_trades(desired, current, budget=2, cost_bps=0.0,
                             expected_edge=edge, min_trade_size=0.05,
                             fixed_cost_bps=20.0)
    assert {d.asset for d in gated if d.executed} == {"asset_1"}


# ------------------------------------------------------- path dependence


def test_budget_makes_the_position_path_genuinely_path_dependent(
    desired: pd.DataFrame,
) -> None:
    """The reason ``current`` is an initial condition rather than a panel.

    A blocked trade on one bar changes the book on every later bar, so the
    constrained path is not a row-wise function of the unconstrained one.
    """
    edge = constant_edge(desired, 5.0)
    tight = apply_trade_budget(desired, None, budget=1, cost_bps=2.0, expected_edge=edge)
    loose = apply_trade_budget(desired, None, budget=6, cost_bps=2.0, expected_edge=edge)

    assert not tight.positions.equals(loose.positions)
    # The constrained book must lag the unconstrained one, not merely differ.
    assert tight.positions.diff().abs().sum().sum() < loose.positions.diff().abs().sum().sum()

    # Starting from a different book yields a different path under the same targets.
    seeded = apply_trade_budget(
        desired, pd.Series(0.9, index=desired.columns), budget=1, cost_bps=2.0, expected_edge=edge
    )
    assert not seeded.positions.equals(tight.positions)


def test_decisions_are_deterministic(desired: pd.DataFrame) -> None:
    edge = constant_edge(desired, 5.0)
    a = apply_trade_budget(desired, None, budget=3, cost_bps=2.0, expected_edge=edge)
    b = apply_trade_budget(desired, None, budget=3, cost_bps=2.0, expected_edge=edge)
    pd.testing.assert_frame_equal(a.positions, b.positions)


def test_budget_used_matches_executed_decisions(desired: pd.DataFrame) -> None:
    result = apply_trade_budget(
        desired, None, budget=3, cost_bps=2.0, expected_edge=constant_edge(desired, 5.0)
    )
    assert result.budget_used.sum() == len(result.executed)
    assert set(result.decisions["reason"].unique()) <= {
        "executed", "budget_exhausted", "negative_expected_value", "below_min_trade_size"
    }


# --------------------------------------------------------- regime weights


def test_regime_weights_maps_labels_to_vectors() -> None:
    labels = pd.Series(
        pd.Categorical(["trend", "chop", "trend", None], categories=["trend", "chop"]),
        index=pd.date_range("2020-01-01", periods=4, freq="B"),
    )
    weight_map = {"trend": {"mom": 0.8, "rev": 0.2}, "chop": {"mom": 0.1, "rev": 0.9}}
    weights = regime_weights(labels, weight_map)

    assert list(weights.columns) == ["mom", "rev"]
    assert weights.iloc[0].tolist() == [0.8, 0.2]
    assert weights.iloc[1].tolist() == [0.1, 0.9]
    # Unlabelled bars fall back to equal weight: stop discriminating, not stop trading.
    assert weights.iloc[3].tolist() == [0.5, 0.5]


def test_regime_weights_honours_an_explicit_default() -> None:
    labels = pd.Series(pd.Categorical([None, "trend"], categories=["trend", "chop"]))
    weights = regime_weights(
        labels, {"trend": {"mom": 1.0, "rev": 0.0}}, default={"mom": 0.0, "rev": 0.0}
    )
    assert weights.iloc[0].tolist() == [0.0, 0.0]


def test_compound_regime_labels_resolve_to_their_components() -> None:
    """Tier-3 labels must find the trend/chop priors they inherit from.

    Regression test for a silent failure. Tier-3 labels are compound
    (``"high_vol|trend|ac-"``) while the priors are keyed by ``"trend"`` and
    ``"chop"``. A direct lookup matches none of them, every bar falls back to
    equal weight, and the result is a fixed-weight book still reported as
    regime-conditional -- every number valid, the label a lie.
    """
    from quantlab.allocator import resolve_weight_map

    base = {"trend": {"mom": 0.8, "rev": 0.2}, "chop": {"mom": 0.1, "rev": 0.9}}
    labels = pd.Series(["high_vol|trend|ac-", "low_vol|chop|ac+", "normal_vol|trend|ac+"])

    expanded, coverage = resolve_weight_map(labels, base)
    assert coverage == 1.0
    assert expanded["high_vol|trend|ac-"] == {"mom": 0.8, "rev": 0.2}
    assert expanded["low_vol|chop|ac+"] == {"mom": 0.1, "rev": 0.9}


def test_uncovered_weight_map_is_refused_rather_than_silently_ignored() -> None:
    """A map that matches nothing must raise, not quietly become equal weight."""
    from quantlab.allocator import resolve_weight_map
    from quantlab.engine import run_regime_portfolio
    from quantlab.simulate import regime_switching as rs

    _, coverage = resolve_weight_map(pd.Series(["alpha", "beta"]), {"trend": {"a": 1.0}})
    assert coverage == 0.0

    prices = rs(n_steps=900, n_assets=2, seed=1)
    with pytest.raises(ValueError, match="covers none of the tier"):
        run_regime_portfolio(prices, tier=2, weight_map={"nonexistent_regime": {"x": 1.0}},
                             regime_min_periods=60)


def test_tier3_map_is_actually_differentiated() -> None:
    """Twelve regimes must produce twelve distinct books, or they are pure cost."""
    from quantlab.engine import run_regime_portfolio, tier3_weight_map
    from quantlab.metrics import sharpe
    from quantlab.simulate import regime_switching as rs

    weight_map = tier3_weight_map()
    assert len(weight_map) == 12
    assert len({tuple(sorted(v.items())) for v in weight_map.values()}) == 12
    for weights in weight_map.values():
        assert sum(weights.values()) == pytest.approx(1.0)

    # High volatility concentrates the book; low volatility flattens it.
    concentrated = max(weight_map["high_vol|chop|ac-"].values())
    flattened = max(weight_map["low_vol|chop|ac-"].values())
    assert concentrated > flattened

    prices = rs(n_steps=2000, n_assets=3, correlation=0.2, seed=4)
    tier2 = run_regime_portfolio(prices, tier=2)
    tier3 = run_regime_portfolio(prices, tier=3, allow_tier3=True)
    assert tier3.config["n_regimes"] == 12
    assert sharpe(tier3.regime_conditional.net_returns) != pytest.approx(
        sharpe(tier2.regime_conditional.net_returns), abs=1e-9
    ), "tier 3 produced the same book as tier 2; the extra regimes are pure cost"


def test_tier1_uses_equal_strategy_weights_by_design() -> None:
    """Tier 1 conditions sizing on volatility, not strategy weights.

    The two books being identical here is correct, and is reached explicitly
    rather than by a lookup that happened to miss.
    """
    from quantlab.engine import run_regime_portfolio
    from quantlab.metrics import sharpe
    from quantlab.simulate import regime_switching as rs

    result = run_regime_portfolio(rs(n_steps=1500, n_assets=3, seed=2), tier=1)
    assert sharpe(result.regime_conditional.net_returns) == pytest.approx(
        sharpe(result.fixed_weight.net_returns)
    )
    assert result.config["weight_map_coverage"] == 1.0


# ==========================================================================
# Phase 2: silent identity substitution audit
#
# The Tier 3 bug was arithmetically valid, wrongly labelled, and raised
# nothing. These lock the same failure class out of the remaining lookups.
# ==========================================================================


def test_misspelled_strategy_in_weight_map_is_refused() -> None:
    """A typo must not silently become a zero weight."""
    labels = pd.Series(pd.Categorical(["trend", "chop"], categories=["trend", "chop"]))
    with pytest.raises(ValueError, match="do not exist"):
        regime_weights(
            labels,
            {"trend": {"mom": 0.5, "revrsion": 0.5}, "chop": {"mom": 0.5, "rev": 0.5}},
            strategies=["mom", "rev"],
        )


def test_strategy_omitted_from_a_regime_warns() -> None:
    """Held at zero is a defensible choice, but it must be a stated one."""
    labels = pd.Series(pd.Categorical(["trend", "chop"], categories=["trend", "chop"]))
    with pytest.warns(RuntimeWarning, match="assigns no weight"):
        weights = regime_weights(
            labels, {"trend": {"mom": 1.0}, "chop": {"mom": 0.5, "rev": 0.5}},
            strategies=["mom", "rev"],
        )
    assert weights.iloc[0].tolist() == [1.0, 0.0]


def test_backtest_refuses_positions_that_share_no_symbols_with_prices() -> None:
    """Reindexing mismatched columns yields an all-NaN book and a flat curve."""
    from quantlab.engine import run_backtest
    from quantlab.simulate import gbm as _gbm

    prices = _gbm(n_steps=200, n_assets=3, seed=1)
    positions = prices.rename(columns=lambda c: c + "_typo") * 0 + 1.0
    with pytest.raises(ValueError, match="share no columns"):
        run_backtest(prices, positions)


def test_backtest_warns_when_positions_cover_only_some_symbols() -> None:
    from quantlab.engine import run_backtest
    from quantlab.simulate import gbm as _gbm

    prices = _gbm(n_steps=200, n_assets=3, seed=1)
    partial = (prices * 0 + 1.0).iloc[:, :2]
    with pytest.warns(RuntimeWarning, match="will be held flat"):
        run_backtest(prices, partial)


def test_partial_weight_map_coverage_warns() -> None:
    """Covering some labels and not others is a partly fixed-weight book."""
    from quantlab.engine import run_regime_portfolio
    from quantlab.simulate import regime_switching as rs

    prices = rs(n_steps=1200, n_assets=2, seed=3)
    # Cover 'trend' but not 'chop'; chop bars silently fall back to equal weight.
    weight_map = {
        "trend": {name: 0.2 for name in
                  ["zscore_reversion", "zscore_momentum", "timeseries_momentum",
                   "ma_crossover", "vol_breakout"]}
    }
    with pytest.warns(RuntimeWarning, match="covers only"):
        run_regime_portfolio(prices, tier=2, weight_map=weight_map, regime_min_periods=60)


# --------------------------------------------------- tier 0 is the default


def test_tier_zero_is_the_default_and_means_fixed_weights() -> None:
    """Phase 1 found Tier 2 unsupported; unconditional allocation is the default.

    Tier 2 stays implemented and tested. It is simply not what runs unless
    somebody asks for it and the upper-bound test has cleared.
    """
    from quantlab.engine import run_regime_portfolio
    from quantlab.metrics import sharpe
    from quantlab.regime import classify
    from quantlab.simulate import regime_switching as rs

    prices = rs(n_steps=1500, n_assets=3, seed=5)
    assert classify(prices).unique().tolist() == ["unconditional"]

    default = run_regime_portfolio(prices)
    assert default.config["tier"] == 0
    assert default.config["n_regimes"] == 1
    assert sharpe(default.regime_conditional.net_returns) == pytest.approx(
        sharpe(default.fixed_weight.net_returns)
    )


def test_tier_two_still_works_when_asked_for_explicitly() -> None:
    from quantlab.engine import run_regime_portfolio
    from quantlab.metrics import sharpe
    from quantlab.simulate import regime_switching as rs

    prices = rs(n_steps=1500, n_assets=3, seed=5)
    tier2 = run_regime_portfolio(prices, tier=2)
    assert tier2.config["n_regimes"] == 2
    assert sharpe(tier2.regime_conditional.net_returns) != pytest.approx(
        sharpe(tier2.fixed_weight.net_returns), abs=1e-9
    )
