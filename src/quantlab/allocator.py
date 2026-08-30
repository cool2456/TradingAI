"""Regime-conditional weights and hard trade-budget enforcement.

The trade budget is the interesting engineering problem in this system.  A cap
of ``K`` trades per day across the whole portfolio turns position management
from a continuous optimisation into a discrete selection: when more signals
fire than the budget allows, something must be ranked and something must be
left undone.

The value function
------------------
Each candidate trade is scored by its expected net value::

    value_i = |desired_i - current_i| * (expected_edge_i - cost_bps / 1e4)
              - fixed_cost_bps / 1e4

This differs from the formula in the project brief, which is

``value_i = |desired_i - current_i| * expected_edge_i - cost_bps / 1e4``.

That version is dimensionally inconsistent: the benefit term scales with trade
size while the cost term does not, so a 0.02 position tweak is charged the same
as a full -1 to +1 flip.  Under a proportional cost model -- which is what
``cost_bps`` means -- the cost of trading ``|delta|`` of notional is
``|delta| * cost_bps / 1e4``, and it belongs inside the same factor as the edge.

The two cost terms act on the queue in completely different ways, and the
difference is not the intuitive one.

- The **fixed** term ``f`` is subtracted from every candidate equally, so it can
  never reorder them.  What it does is impose a minimum economically viable
  trade size: ``value_i > 0`` requires ``|delta_i| > f / (edge_i - c)``.  A
  per-ticket charge is a reason not to bother with small trades, never a reason
  to prefer one trade over another.

- The **proportional** term ``c`` penalises candidates in proportion to their
  size, so it *does* reorder the queue -- but only when the edge estimates
  differ across assets.  Raising ``c`` tilts the ranking away from large trades
  and toward small ones carrying high edge.  If every asset shares the same
  edge, ``value_i = |delta_i| (e - c)`` is ``|delta_i|`` times a constant, the
  ranking collapses to trade size alone, and cost cannot change it at all.

Seen this way, the brief's formula is the first of these wearing the name of
the second: it subtracts a constant while calling it a proportional
basis-point cost.  What it actually implements is a ticket charge.

A warning about ``expected_edge``
---------------------------------
``expected_edge`` is an **estimate**, and every trade selected under a hard
budget is selected on the strength of that estimate.  If the edge is
mis-estimated, the ranking is wrong, the budget is spent on the wrong trades,
and every metric downstream degrades silently -- the backtest still runs, the
Sharpe is still computed, and nothing in the output announces that the
selection was garbage.  Under a binding budget the system is *more* sensitive
to edge estimation error than an unconstrained one, because it acts only on the
estimate's ordering rather than on its whole vector.  Treat a flat, constant
edge as the honest default unless you can demonstrate a better one out of
sample.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "regime_weights",
    "resolve_weight_map",
    "apply_trade_budget",
    "select_trades",
    "constant_edge",
    "BudgetResult",
    "TradeDecision",
]

#: Position changes at or below this are not trades. Volatility targeting
#: perturbs every position on every bar; without a dead band the trade count is
#: unbounded and a 3-per-day budget is exhausted by rounding noise before any
#: real signal is considered.
DEFAULT_MIN_TRADE_SIZE = 0.05


def regime_weights(
    regime_labels: pd.Series,
    weight_map: Mapping[str, Mapping[str, float]],
    default: Mapping[str, float] | None = None,
    strategies: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Map each regime label to a vector of strategy weights.

    Parameters
    ----------
    regime_labels
        Categorical Series from :func:`quantlab.regime.classify`.
    weight_map
        ``{regime_label: {strategy_name: weight}}``.
    default
        Weights used where the label is missing -- during the expanding-quantile
        warm-up, or for a regime absent from ``weight_map``. Defaults to equal
        weight across ``strategies``. Falling back to equal weight rather than
        to zero is deliberate: an unknown regime is a reason to stop
        *discriminating*, not a reason to stop trading.

    Returns
    -------
    DataFrame indexed like ``regime_labels``, one column per strategy.

    Note the shape of the claim being made here. ``weight_map`` encodes ``R x S``
    decisions, all of which are hypotheses. Each is estimated from only the bars
    in its regime, and all of them are charged against the deflated Sharpe.
    """
    if strategies is None:
        seen: list[str] = []
        for row in weight_map.values():
            seen.extend(k for k in row if k not in seen)
        strategies = seen
    strategies = list(strategies)
    if not strategies:
        raise ValueError("no strategies given and weight_map is empty")

    # A strategy named in weight_map but absent from `strategies` would be
    # silently dropped, and one present in `strategies` but absent from a
    # regime's row would silently receive weight 0.0. Both are typos that
    # produce a valid, wrong allocation with no error -- the same failure class
    # as the Tier 3 compound-label bug. Named misspellings are refused.
    named = {s for row in weight_map.values() for s in row}
    unknown = named - set(strategies)
    if unknown:
        raise ValueError(
            f"weight_map names strategies that do not exist: {sorted(unknown)}. "
            f"Known strategies are {sorted(strategies)}. A misspelled name would "
            "otherwise be dropped silently and its intended weight lost."
        )
    for regime, row in weight_map.items():
        absent = set(strategies) - set(row)
        if absent:
            warnings.warn(
                f"regime {regime!r} assigns no weight to {sorted(absent)}; they will "
                "be held at 0.0. If that is deliberate, state it explicitly by "
                "listing them with a weight of 0.",
                RuntimeWarning,
                stacklevel=2,
            )

    if default is None:
        default = {s: 1.0 / len(strategies) for s in strategies}

    table = pd.DataFrame(
        {s: {r: float(w.get(s, 0.0)) for r, w in weight_map.items()} for s in strategies}
    )
    labels = regime_labels.astype("object")
    weights = table.reindex(labels.to_numpy(), columns=strategies)
    weights.index = regime_labels.index

    fallback = pd.Series({s: float(default.get(s, 0.0)) for s in strategies})
    missing = weights.isna().all(axis=1)
    weights.loc[missing, :] = fallback.to_numpy()
    return weights.fillna(0.0)


def resolve_weight_map(
    regime_labels: pd.Series,
    base_map: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, float]], float]:
    """Expand a weight map to cover the regime labels actually present.

    Tier 3 labels are compound -- ``"high_vol|trend|ac-"`` -- while the priors in
    :data:`quantlab.engine.DEFAULT_WEIGHT_MAP` are keyed by ``"trend"`` and
    ``"chop"``. A direct lookup misses every compound label, ``regime_weights``
    falls back to equal weight on 100% of bars, and the result is a
    fixed-weight book that still calls itself regime-conditional. That is a
    silent failure of exactly the kind this library exists to prevent: the
    numbers are all valid, the label is a lie.

    This resolves each observed label to the base entry whose key appears as one
    of its ``|``-separated components, so ``"high_vol|trend|ac-"`` inherits the
    ``"trend"`` weights.

    Returns ``(expanded_map, coverage)``, where ``coverage`` is the fraction of
    labelled bars that matched something. A coverage of 0 means the map and the
    classifier disagree entirely.
    """
    labels = regime_labels.dropna().astype(str)
    if labels.empty:
        return dict(base_map), 0.0

    expanded: dict[str, dict[str, float]] = {}
    for label in labels.unique():
        if label in base_map:
            expanded[label] = dict(base_map[label])
            continue
        for component in label.split("|"):
            if component in base_map:
                expanded[label] = dict(base_map[component])
                break

    matched = labels.isin(expanded.keys()).mean()
    return expanded, float(matched)


def constant_edge(
    like: pd.DataFrame, edge_bps: float = 2.0, periods_per_year: float = 252.0
) -> pd.DataFrame:
    """A flat expected edge of ``edge_bps`` per trade, for every asset and bar.

    The honest default. It expresses "I believe my signals have some edge but I
    cannot rank them against each other", which is usually the truth. Under a
    flat edge the trade budget ranks purely by trade size ``|delta|``, i.e. it
    spends the budget on the positions that are furthest from where they should
    be -- a defensible policy that requires no forecast at all.
    """
    return pd.DataFrame(edge_bps / 1e4, index=like.index, columns=like.columns)


@dataclass(frozen=True)
class TradeDecision:
    """One candidate trade and what became of it."""

    timestamp: pd.Timestamp
    asset: str
    current: float
    desired: float
    delta: float
    expected_edge: float
    value: float
    executed: bool
    reason: str


@dataclass
class BudgetResult:
    """Positions actually held after budget enforcement, plus the audit trail."""

    positions: pd.DataFrame
    budget_used: pd.Series
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def executed(self) -> pd.DataFrame:
        if self.decisions.empty:
            return self.decisions
        return self.decisions[self.decisions["executed"]]

    @property
    def rejection_reasons(self) -> pd.Series:
        if self.decisions.empty:
            return pd.Series(dtype=int)
        return self.decisions.loc[~self.decisions["executed"], "reason"].value_counts()


def _select_kernel(
    desired: np.ndarray,
    current: np.ndarray,
    budget: int,
    cost_bps: float,
    edge: np.ndarray,
    min_trade_size: float,
    fixed_cost_bps: float,
    name_rank: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised core of the selection rule.

    Split out from :func:`select_trades` purely for speed: the sequential day
    loop in :func:`apply_trade_budget` calls this once per bar, and building
    pandas objects at that rate dominates the runtime of every evaluation. The
    selection logic itself is identical, and :func:`select_trades` remains the
    documented, tested entry point.

    Returns ``(new_positions, executed_mask, values, eligible_mask)``.
    """
    delta = desired - current
    magnitude = np.abs(delta)
    with np.errstate(invalid="ignore"):
        value = magnitude * (edge - cost_bps / 1e4) - fixed_cost_bps / 1e4
        eligible = (
            np.isfinite(magnitude)
            & (magnitude > min_trade_size)
            & np.isfinite(value)
            & (value > 0)
        )

    new_positions = current.copy()
    executed = np.zeros(desired.size, dtype=bool)
    if budget > 0 and eligible.any():
        candidates = np.flatnonzero(eligible)
        if candidates.size > budget:
            # Highest value first; ties broken by asset name so the outcome does
            # not depend on the order the columns happen to arrive in.
            order = candidates[
                np.lexsort((name_rank[candidates], -value[candidates]))
            ][:budget]
        else:
            order = candidates
        new_positions[order] = desired[order]
        executed[order] = True
    return new_positions, executed, value, eligible


def _name_rank(columns: Sequence[str]) -> np.ndarray:
    """Rank of each column by name, for deterministic tie-breaking."""
    names = np.asarray([str(c) for c in columns])
    return np.argsort(np.argsort(names))


def select_trades(
    desired: pd.Series,
    current: pd.Series,
    budget: int,
    cost_bps: float,
    expected_edge: pd.Series,
    min_trade_size: float = DEFAULT_MIN_TRADE_SIZE,
    fixed_cost_bps: float = 0.0,
    timestamp: pd.Timestamp | None = None,
) -> tuple[pd.Series, list[TradeDecision]]:
    """Choose at most ``budget`` trades for one bar. Pure, and the unit of testing.

    Ranks candidates by ``value`` (see the module docstring), executes the top
    ``budget`` with ``value > 0``, and leaves everything else exactly where it
    was.  A trade with negative expected value is never taken **even when
    budget remains** -- the budget is a ceiling, not a quota.

    Returns the new position vector and one :class:`TradeDecision` per asset
    that was considered.
    """
    if budget < 0:
        raise ValueError("budget must be non-negative")

    columns = list(desired.index)
    desired_v = desired.to_numpy(dtype=float)
    current_v = current.reindex(desired.index).to_numpy(dtype=float)
    edge_v = expected_edge.reindex(desired.index).fillna(0.0).to_numpy(dtype=float)

    new_v, executed, value, eligible = _select_kernel(
        desired_v, current_v, budget, cost_bps, edge_v,
        min_trade_size, fixed_cost_bps, _name_rank(columns),
    )
    decisions = _decisions_from(
        columns, timestamp, desired_v, current_v, edge_v, value, executed, eligible, min_trade_size
    )
    return pd.Series(new_v, index=desired.index), decisions


def _decisions_from(
    columns: Sequence[str],
    timestamp: pd.Timestamp | None,
    desired: np.ndarray,
    current: np.ndarray,
    edge: np.ndarray,
    value: np.ndarray,
    executed: np.ndarray,
    eligible: np.ndarray,
    min_trade_size: float,
) -> list[TradeDecision]:
    """Build the audit trail for one bar, explaining every asset's outcome."""
    magnitude = np.abs(desired - current)
    decisions: list[TradeDecision] = []
    for i, asset in enumerate(columns):
        if executed[i]:
            reason = "executed"
        elif eligible[i]:
            reason = "budget_exhausted"
        elif not np.isfinite(magnitude[i]) or magnitude[i] <= min_trade_size:
            reason = "below_min_trade_size"
        else:
            reason = "negative_expected_value"
        decisions.append(
            TradeDecision(
                timestamp, str(asset), float(current[i]), float(desired[i]),
                float(desired[i] - current[i]), float(edge[i]), float(value[i]),
                bool(executed[i]), reason,
            )
        )
    return decisions


def apply_trade_budget(
    desired: pd.DataFrame,
    current: pd.DataFrame | pd.Series | None,
    budget: int,
    cost_bps: float,
    expected_edge: pd.DataFrame,
    min_trade_size: float = DEFAULT_MIN_TRADE_SIZE,
    fixed_cost_bps: float = 0.0,
    record_decisions: bool = True,
) -> BudgetResult:
    """Enforce a portfolio-wide budget of ``budget`` trades per day.

    Why ``current`` cannot be a full panel
    --------------------------------------
    The project brief types ``current`` as a DataFrame alongside ``desired``,
    which suggests the whole panel can be evaluated at once.  It cannot, and
    the reason is the point of the exercise: the book you hold today is the
    result of which trades the budget let you make yesterday.  If the budget
    blocks a trade on Monday, Tuesday's ``current`` differs from what an
    unconstrained path would have held, which changes Tuesday's deltas, its
    ranking, and its budget consumption.  ``current`` is therefore an
    **initial condition**, not an input series, and the process is simulated
    forward.

    A DataFrame is accepted for signature compatibility and its first row is
    used as the starting book; ``None`` starts flat.

    The budget resets at the start of each calendar day.  ``budget_used`` is
    reported per day for display.
    """
    if not desired.index.equals(expected_edge.index):
        expected_edge = expected_edge.reindex(desired.index)
    expected_edge = expected_edge.reindex(columns=desired.columns).fillna(0.0)

    if current is None:
        book = pd.Series(0.0, index=desired.columns)
    elif isinstance(current, pd.DataFrame):
        book = current.iloc[0].reindex(desired.columns).fillna(0.0).astype(float)
    else:
        book = current.reindex(desired.columns).fillna(0.0).astype(float)

    if isinstance(desired.index, pd.DatetimeIndex):
        day_keys = desired.index.normalize()
    else:
        day_keys = pd.Index(desired.index)

    columns = list(desired.columns)
    held = np.empty((len(desired), desired.shape[1]))
    used = np.zeros(len(desired), dtype=int)
    all_decisions: list[TradeDecision] = []

    current_day = None
    remaining = budget
    desired_values = desired.to_numpy(dtype=float)
    edge_values = expected_edge.to_numpy(dtype=float)
    book_v = book.to_numpy(dtype=float)
    rank = _name_rank(columns)

    # Genuinely sequential: today's book is the output of yesterday's budgeted
    # decisions, and the day's remaining budget carries across bars within a
    # day. Neither can be expressed as an array operation over the panel.
    for i in range(len(desired)):
        day = day_keys[i]
        if day != current_day:
            current_day, remaining = day, budget

        row_desired = desired_values[i]
        if np.isnan(row_desired).any():
            # An undefined target is not a reason to trade to zero; hold.
            row_desired = np.where(np.isnan(row_desired), book_v, row_desired)

        # _select_kernel copies rather than mutates, so this stays valid as the
        # pre-trade book for the audit trail.
        pre_trade = book_v
        book_v, executed, value, eligible = _select_kernel(
            row_desired, book_v, remaining, cost_bps, edge_values[i],
            min_trade_size, fixed_cost_bps, rank,
        )
        n_executed = int(executed.sum())
        remaining -= n_executed
        used[i] = n_executed
        held[i] = book_v
        if record_decisions:
            all_decisions.extend(
                _decisions_from(columns, desired.index[i], row_desired, pre_trade,
                                edge_values[i], value, executed, eligible, min_trade_size)
            )

    positions = pd.DataFrame(held, index=desired.index, columns=columns)
    per_bar = pd.Series(used, index=desired.index, name="trades")
    budget_used = (
        per_bar.groupby(day_keys).sum().rename("budget_used")
        if isinstance(desired.index, pd.DatetimeIndex)
        else per_bar.rename("budget_used")
    )

    decisions_frame = (
        pd.DataFrame([d.__dict__ for d in all_decisions])
        if all_decisions
        else pd.DataFrame(columns=[f.name for f in TradeDecision.__dataclass_fields__.values()])
    )
    return BudgetResult(positions=positions, budget_used=budget_used, decisions=decisions_frame)
