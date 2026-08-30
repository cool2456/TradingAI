"""The backtest loop.

The accounting is a dozen lines and every one of them is a place to be wrong,
so the timing convention is stated once here and used nowhere else.

Timing convention
-----------------
Let ``target_t`` be the position a strategy *wants*, computed from prices at
indices ``<= t``.  Then with execution lag ``lag >= 1``::

    ret_t      = P_t / P_{t-1} - 1        # return realised over (t-1, t]
    held_t     = target_{t - lag}         # position actually held over (t-1, t]
    gross_t    = held_t * ret_t
    turnover_t = |held_t - held_{t-1}|
    cost_t     = turnover_t * cost_bps / 1e4
    net_t      = gross_t - cost_t

In pandas this is ``held = target.shift(lag)`` and ``gross = held * ret``.

This differs from the specification in the project brief, which writes both
``position_t = target.shift(lag)`` *and* ``gross_{t+1} = position_t *
ret_{t+1}``.  Those compose to ``lag + 1`` periods of delay in pandas, because
``pct_change()`` at index ``t`` already refers to the interval ``(t-1, t]``.
The brief's version is not dangerous -- it is over-conservative rather than
optimistic -- but it means a stated ``lag=1`` is silently a two-bar lag, and
sensitivity analysis on ``lag`` would be reported against the wrong axis.  One
shift, applied here, is the convention.

``lag=1`` is the minimum honest value: a signal computed from the close at
``t-1`` is executed into the interval that starts at ``t-1``.  ``lag=0`` is
rejected outright, because it would earn the return of the bar whose close was
used to make the decision.

Cost timing
-----------
``cost_t`` is charged in the same period in which the new position first earns.
The alternative -- charging it at the moment of the trade, one period earlier
-- differs only in which period the first and last trades land in, and shifts
no totals.  What matters is that it is charged exactly once, which the
turnover-of-``held`` formulation guarantees.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import metrics

__all__ = [
    "BacktestResult",
    "run_backtest",
    "PortfolioResult",
    "run_regime_portfolio",
    "DEFAULT_WEIGHT_MAP",
    "tier3_weight_map",
]

#: A position change smaller than this is not counted as a trade. Continuous
#: sizing rules nudge every position every bar; without a dead band the trade
#: count is unbounded and the trade budget binds on rounding noise.
DEFAULT_MIN_TRADE_SIZE = 1e-4


@dataclass
class BacktestResult:
    """Everything a backtest produced, with nothing recomputed on the way out.

    Attributes
    ----------
    net_returns
        Portfolio return per period, **after** costs. The headline series.
    gross_returns
        Before costs. A diagnostic only -- reporting it as a result is the
        error the whole library is arranged to prevent.
    positions
        The positions actually *held*, already lagged. Not the targets.
    trade_counts
        Number of assets whose held position changed by more than
        ``min_trade_size``, aggregated per calendar day.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame
    asset_net_returns: pd.DataFrame
    trade_counts: pd.Series
    config: dict[str, Any] = field(default_factory=dict)
    regime: pd.Series | None = None

    @property
    def equity(self) -> pd.Series:
        """Compounded net equity curve, starting at 1.0."""
        return (1.0 + self.net_returns.fillna(0.0)).cumprod()

    @property
    def total_cost(self) -> float:
        """Total transaction cost paid, as a fraction of notional."""
        return float(self.costs.sum())

    def summary(self, periods_per_year: float = 252.0, n_trials: int = 1) -> dict[str, float]:
        """Net-of-cost metric block, plus the gross Sharpe for diagnosis only."""
        out = metrics.summary(
            self.net_returns, self.positions, periods_per_year, n_trials=n_trials
        )
        out["gross_sharpe_ann_DIAGNOSTIC"] = metrics.sharpe(self.gross_returns, periods_per_year)
        out["total_cost"] = self.total_cost
        out["mean_daily_trades"] = float(self.trade_counts.mean())
        out["max_daily_trades"] = float(self.trade_counts.max()) if len(self.trade_counts) else 0.0
        return out


def run_backtest(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    cost_bps: float = 1.0,
    lag: int = 1,
    warmup: int = 0,
    min_trade_size: float = DEFAULT_MIN_TRADE_SIZE,
    regime: pd.Series | None = None,
    config: dict[str, Any] | None = None,
) -> BacktestResult:
    """Run the accounting for one set of target positions.

    Parameters
    ----------
    prices
        Price panel, one column per asset.
    positions
        **Target** positions, aligned to ``prices``. These are what the
        strategy wants at each timestamp; the lag is applied here, once.
    cost_bps
        Proportional transaction cost in basis points of notional traded.
    lag
        Execution delay in bars. Must be ``>= 1``.
    warmup
        Leading bars to discard before evaluating. Use the signal's
        ``.lookback`` so undefined values never enter the reported sample.
    min_trade_size
        Dead band for counting trades. See :data:`DEFAULT_MIN_TRADE_SIZE`.

    Returns
    -------
    BacktestResult
        Portfolio returns are the **sum** across assets, since positions are
        notional weights rather than portfolio shares. A book holding +1 in
        each of three assets is levered 3x, and this function reports it that
        way rather than quietly normalising.
    """
    if lag < 1:
        raise ValueError(
            f"lag={lag} is not causal: a position taken at t would earn the return "
            "of the bar whose close produced the signal. lag must be >= 1."
        )
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")

    # reindex fills unmatched columns with NaN, so a positions frame whose
    # symbols do not match the price panel produces an all-NaN book, a
    # zero-return backtest and no error at all. Refuse rather than reindex.
    shared = positions.columns.intersection(prices.columns)
    if len(shared) == 0:
        raise ValueError(
            f"positions and prices share no columns. positions has "
            f"{sorted(positions.columns)[:5]}, prices has {sorted(prices.columns)[:5]}. "
            "Reindexing would silently produce an empty book and a flat equity curve."
        )
    if len(shared) < len(prices.columns):
        warnings.warn(
            f"positions cover {len(shared)} of {len(prices.columns)} priced symbols; "
            f"{sorted(set(prices.columns) - set(shared))[:5]} will be held flat.",
            RuntimeWarning,
            stacklevel=2,
        )
    positions = positions.reindex(index=prices.index, columns=prices.columns)

    returns = prices.pct_change()
    held = positions.shift(lag)

    gross_by_asset = held * returns
    # Turnover of the *held* book, so each trade is charged exactly once.
    turnover_by_asset = held.diff().abs()
    cost_by_asset = turnover_by_asset * (cost_bps / 1e4)
    net_by_asset = gross_by_asset - cost_by_asset

    trades = (turnover_by_asset > min_trade_size).sum(axis=1)

    if warmup > 0:
        keep = prices.index[warmup:]
        held, gross_by_asset = held.loc[keep], gross_by_asset.loc[keep]
        turnover_by_asset, cost_by_asset = turnover_by_asset.loc[keep], cost_by_asset.loc[keep]
        net_by_asset, trades = net_by_asset.loc[keep], trades.loc[keep]

    # min_count=1 so an all-NaN row stays NaN instead of summing to a fake 0.0.
    gross = gross_by_asset.sum(axis=1, min_count=1).rename("gross")
    costs = cost_by_asset.sum(axis=1, min_count=1).rename("cost")
    turnover = turnover_by_asset.sum(axis=1, min_count=1).rename("turnover")
    net = (gross - costs.fillna(0.0)).rename("net")

    trade_counts = _daily_trade_counts(trades)
    if regime is not None:
        regime = regime.reindex(net.index)

    return BacktestResult(
        net_returns=net,
        gross_returns=gross,
        costs=costs,
        turnover=turnover,
        positions=held,
        asset_net_returns=net_by_asset,
        trade_counts=trade_counts,
        regime=regime,
        config={
            "cost_bps": cost_bps,
            "lag": lag,
            "warmup": warmup,
            "min_trade_size": min_trade_size,
            "n_assets": int(prices.shape[1]),
            "n_periods": int(len(net)),
            **(config or {}),
        },
    )


def _daily_trade_counts(trades: pd.Series) -> pd.Series:
    """Aggregate per-bar trade counts to calendar days.

    The trade budget is a *daily* constraint. On daily bars this is the
    identity; it matters only if the engine is fed intraday data, where many
    bars share one budget.
    """
    if isinstance(trades.index, pd.DatetimeIndex):
        return trades.groupby(trades.index.normalize()).sum().rename("trades")
    return trades.rename("trades")


# ==========================================================================
# Regime-conditional portfolio composition
# ==========================================================================

#: Default Tier-2 weight map: trend-following in trending markets, mean
#: reversion in choppy ones.
#:
#: These are **priors, not fitted parameters**. Nothing in this library
#: estimated them, and that is deliberate -- fitting them on the same data they
#: are evaluated on is the failure mode the whole repository exists to avoid.
#: They encode a hypothesis, and the question the system answers is whether
#: that hypothesis beats an equal fixed allocation out of sample. Frequently it
#: does not, and when it does not the honest report says so.
DEFAULT_WEIGHT_MAP: dict[str, dict[str, float]] = {
    "trend": {
        "zscore_reversion": 0.05,
        "zscore_momentum": 0.25,
        "timeseries_momentum": 0.35,
        "ma_crossover": 0.25,
        "vol_breakout": 0.10,
    },
    "chop": {
        "zscore_reversion": 0.55,
        "zscore_momentum": 0.05,
        "timeseries_momentum": 0.10,
        "ma_crossover": 0.10,
        "vol_breakout": 0.20,
    },
}


#: Strategies classified by what they bet on, used to build the Tier-3 map.
MOMENTUM_STRATEGIES = ("zscore_momentum", "timeseries_momentum", "ma_crossover", "vol_breakout")
REVERSION_STRATEGIES = ("zscore_reversion",)


def tier3_weight_map(
    base_map: dict[str, dict[str, float]] | None = None,
    autocorr_tilt: float = 0.6,
    vol_concentration: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Build 12 regime weight vectors from two priors and two rules.

    Tier 3 is meant to drive *selection*, so its map must actually differ across
    the twelve regimes -- otherwise the extra regimes raise the hypothesis count
    without changing a single position, which is pure cost.

    Rather than hand-writing twelve vectors (twelve opportunities to encode a
    result already seen in the data), each is derived mechanically:

    1. Start from the trend/chop prior in :data:`DEFAULT_WEIGHT_MAP`.
    2. **Autocorrelation tilt.** Positive trailing autocorrelation multiplies the
       momentum strategies by ``1 + autocorr_tilt``; negative does the same for
       the reversion strategies.
    3. **Volatility concentration.** Weights are raised to the power
       ``1 + vol_concentration`` in high volatility -- concentrating the book on
       its strongest convictions -- and ``1 - vol_concentration`` in low
       volatility, flattening it. Normal volatility is left alone.

    Then renormalised to sum to 1.

    Every number here is a **prior**. Nothing was fitted, and that is the only
    thing keeping this defensible. Twelve fitted regime vectors over 2500 bars
    would be roughly 200 bars per decision, which is not enough to estimate a
    sign, let alone a weight -- see :func:`quantlab.regime.regime_sample_counts`.
    """
    from .regime import TREND_LABELS, VOL_LABELS

    base_map = base_map or DEFAULT_WEIGHT_MAP
    exponents = {
        VOL_LABELS[0]: 1.0 - vol_concentration,
        VOL_LABELS[1]: 1.0,
        VOL_LABELS[2]: 1.0 + vol_concentration,
    }

    out: dict[str, dict[str, float]] = {}
    for vol in VOL_LABELS:
        for trend in TREND_LABELS:
            for sign in ("ac+", "ac-"):
                favoured = MOMENTUM_STRATEGIES if sign == "ac+" else REVERSION_STRATEGIES
                weights = {
                    name: w * (1.0 + autocorr_tilt if name in favoured else 1.0)
                    for name, w in base_map[trend].items()
                }
                weights = {name: w ** exponents[vol] for name, w in weights.items()}
                total = sum(weights.values()) or 1.0
                out[f"{vol}|{trend}|{sign}"] = {k: v / total for k, v in weights.items()}
    return out


@dataclass
class PortfolioResult:
    """A regime-conditional portfolio, its fixed-weight control, and its parts.

    ``regime_conditional`` and ``fixed_weight`` differ in exactly one respect:
    whether strategy weights vary with the detected regime. Everything else --
    signals, sizing, budget, costs, lag -- is identical. Any difference between
    them is therefore attributable to the regime logic and to nothing else,
    which is the only way to answer whether the regime logic earns its
    statistical cost.
    """

    regime_conditional: BacktestResult
    fixed_weight: BacktestResult
    per_strategy: dict[str, BacktestResult]
    regime: pd.Series
    weights: pd.DataFrame
    budget_used: pd.Series
    decisions: pd.DataFrame
    warmup: int
    config: dict[str, Any] = field(default_factory=dict)

    def ledger(self, periods_per_year: float = 252.0, n_trials: int = 1) -> pd.DataFrame:
        """Ranked table of every strategy plus both controls plus both portfolios."""
        rows = {}
        for name, result in self.per_strategy.items():
            rows[name] = result.summary(periods_per_year, n_trials)
        rows["PORTFOLIO regime-conditional"] = self.regime_conditional.summary(
            periods_per_year, n_trials
        )
        rows["PORTFOLIO fixed-weight"] = self.fixed_weight.summary(periods_per_year, n_trials)
        frame = pd.DataFrame(rows).T
        return frame.sort_values("sharpe_ann", ascending=False)

    def regime_verdict(self, periods_per_year: float = 252.0) -> dict[str, Any]:
        """Did regime conditioning beat the fixed allocation, after its own cost?

        Compares the two portfolios and charges the regime version for the
        hypothesis inflation it caused, using
        :func:`quantlab.regime.hypothesis_cost`.
        """
        from .regime import hypothesis_cost

        conditional = metrics.sharpe(self.regime_conditional.net_returns, periods_per_year)
        fixed = metrics.sharpe(self.fixed_weight.net_returns, periods_per_year)
        n_regimes = int(self.regime.dropna().nunique()) or 1
        cost = hypothesis_cost(
            len(self.per_strategy), n_regimes, len(self.regime_conditional.net_returns)
        )
        uplift = conditional - fixed
        earns_it = uplift > (
            cost["luck_threshold_conditioned_ann"] - cost["luck_threshold_ann"]
        )
        return {
            "regime_conditional_sharpe": conditional,
            "fixed_weight_sharpe": fixed,
            "uplift": uplift,
            "extra_luck_threshold": cost["luck_threshold_conditioned_ann"]
            - cost["luck_threshold_ann"],
            "regime_logic_earns_its_cost": bool(earns_it),
            **cost,
        }


def run_regime_portfolio(
    prices: pd.DataFrame,
    signals: Sequence[Any] | None = None,
    weight_map: dict[str, dict[str, float]] | None = None,
    tier: int = 0,
    budget: int = 3,
    cost_bps: float = 2.0,
    target_ann_vol: float = 0.10,
    max_leverage: float = 3.0,
    vol_lookback: int = 60,
    regime_min_periods: int = 126,
    edge_bps: float = 5.0,
    min_trade_size: float = 0.05,
    lag: int = 1,
    allow_tier3: bool = False,
) -> PortfolioResult:
    """Compose signals into a regime-weighted, volatility-targeted, budgeted book.

    The pipeline, in order::

        signals -> regime weights -> combine -> volatility target
                -> trade budget -> lagged execution -> costs

    Volatility targeting is applied **before** the trade budget, so the budget
    sees the positions the book will actually hold. Doing it the other way
    round would let the sizing layer silently undo the budget's decisions.

    **The default is ``tier=0``: fixed weights, no regime conditioning.** The
    Phase 1 evaluation found Tier 2 conditioning worked only on the generator
    constructed to contain its assumption and lost significantly on GBM and
    GARCH, and the trend/chop label's own IC failed at alpha=0.05 with
    HAC t=1.81. Tier 2 remains implemented and tested but is not deployed;
    re-enabling it requires the upper-bound test to clear first.

    Tier 1 (volatility-driven sizing) is applied at every tier including 0 --
    it governs ``vol_target`` rather than strategy weights, and is the one
    tier the evidence supports. Tier 3 requires ``allow_tier3=True``.
    """
    from .allocator import apply_trade_budget, constant_edge, regime_weights, resolve_weight_map
    from .regime import classify
    from .signals import CONTROLS, SIGNALS
    from .sizing import vol_target

    if signals is None:
        signals = [SIGNALS[name].bind() for name in SIGNALS]
    research = [s for s in signals if s.name not in CONTROLS]
    if not research:
        raise ValueError("at least one non-control strategy is required")
    # Tier 3 needs its own twelve-regime map; falling back to the trend/chop
    # priors would make it identical to Tier 2 while costing more statistically.
    if weight_map is None:
        weight_map = tier3_weight_map() if tier == 3 else DEFAULT_WEIGHT_MAP

    returns = prices.pct_change()
    regime = classify(prices, tier=tier, min_periods=regime_min_periods,
                      allow_tier3=allow_tier3).rename("regime")

    names = [s.name for s in research]
    targets = {s.name: s(prices).fillna(0.0) for s in research}
    warmup = max(s.lookback for s in signals) + regime_min_periods + vol_lookback + 1

    # Tier 1 conditions *sizing* on volatility, not strategy weights, so equal
    # strategy weights are correct there and are chosen explicitly rather than
    # arrived at by a failed lookup. Tiers 2 and 3 must actually cover the
    # labels the classifier emits.
    if tier in (0, 1):
        # Tier 0 has one regime; Tier 1 conditions sizing rather than weights.
        # Either way the strategy weights are uniform, chosen explicitly rather
        # than arrived at by a lookup that happened to miss.
        resolved, coverage = {}, 1.0
    else:
        resolved, coverage = resolve_weight_map(regime, weight_map)
        if 0.0 < coverage < 1.0:
            warnings.warn(
                f"the weight map covers only {coverage:.1%} of labelled bars at "
                f"tier {tier}; the remainder fall back to equal weight. That is a "
                "partially fixed-weight book being reported as regime-conditional.",
                RuntimeWarning,
                stacklevel=2,
            )
        if coverage == 0.0:
            raise ValueError(
                f"the weight map covers none of the tier-{tier} regime labels "
                f"({sorted(str(x) for x in regime.dropna().unique())[:4]}...). Every bar "
                "would fall back to equal weight, producing a fixed-weight book "
                "labelled as regime-conditional. Supply a weight_map keyed by these "
                "labels, or by a '|'-separated component of them."
            )

    weights = regime_weights(regime, resolved, strategies=names)
    equal = pd.DataFrame(1.0 / len(names), index=prices.index, columns=names)

    def _compose(w: pd.DataFrame) -> pd.DataFrame:
        book = sum(targets[name].mul(w[name], axis=0) for name in names)
        return vol_target(book, returns, target_ann_vol, vol_lookback, max_leverage)

    edge = constant_edge(prices, edge_bps)
    conditional_budgeted = apply_trade_budget(
        _compose(weights), None, budget, cost_bps, edge, min_trade_size
    )
    fixed_budgeted = apply_trade_budget(
        _compose(equal), None, budget, cost_bps, edge, min_trade_size, record_decisions=False
    )

    config = {
        "tier": tier, "budget": budget, "cost_bps": cost_bps,
        "target_ann_vol": target_ann_vol, "max_leverage": max_leverage,
        "edge_bps": edge_bps, "min_trade_size": min_trade_size, "lag": lag,
        "strategies": names, "allow_tier3": allow_tier3,
        "weight_map_coverage": coverage, "n_regimes": int(regime.dropna().nunique()),
    }

    def _run(positions: pd.DataFrame, label: str) -> BacktestResult:
        return run_backtest(prices, positions, cost_bps, lag, warmup,
                            min_trade_size=min_trade_size, regime=regime,
                            config={**config, "book": label})

    # Individual strategies and both controls are run standalone and unbudgeted:
    # they are diagnostics showing what each would do on its own, and the budget
    # is a property of the combined portfolio rather than of any one strategy.
    per_strategy = {
        s.name: _run(
            vol_target(s(prices).fillna(0.0), returns, target_ann_vol, vol_lookback, max_leverage),
            f"standalone:{s.name}",
        )
        for s in signals
    }

    return PortfolioResult(
        regime_conditional=_run(conditional_budgeted.positions, "regime_conditional"),
        fixed_weight=_run(fixed_budgeted.positions, "fixed_weight"),
        per_strategy=per_strategy,
        regime=regime,
        weights=weights,
        budget_used=conditional_budgeted.budget_used,
        decisions=conditional_budgeted.decisions,
        warmup=warmup,
        config=config,
    )
