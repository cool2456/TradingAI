"""Out-of-sample validation and honest trial accounting.

Trial accounting
----------------
Every backtest configuration that is evaluated appends one line to
``research_log.jsonl``, and :func:`deflated_sharpe_now` reads its trial count
from that file.  The point is mechanical rather than moral: it must not be
*possible* to run a parameter sweep without paying for it statistically,
because the alternative -- remembering to declare your trial count honestly
after the fact -- is a discipline nobody sustains.

A sweep over an 8-point grid writes 8 lines and raises the bar every future
result must clear.  That is the intended cost, not a side effect.

The log is append-only.  Editing it by hand is the same act as deleting the
failed experiments from a lab notebook.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import metrics
from .engine import BacktestResult, run_backtest

__all__ = [
    "research_log_path",
    "log_trial",
    "trial_count",
    "trial_sharpes",
    "deflated_sharpe_now",
    "current_threshold",
    "walk_forward",
    "purged_kfold",
    "parameter_stability",
    "WalkForwardResult",
]

DEFAULT_LOG_NAME = "research_log.jsonl"


def research_log_path() -> Path:
    """Location of the append-only trial log.

    Overridable with ``QUANTLAB_RESEARCH_LOG`` so that tests -- which run
    thousands of configurations -- do not pollute the real research record.
    """
    return Path(os.environ.get("QUANTLAB_RESEARCH_LOG", DEFAULT_LOG_NAME))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def log_trial(
    config: Mapping[str, Any],
    result: BacktestResult | Mapping[str, Any],
    path: Path | None = None,
    kind: str = "backtest",
) -> dict[str, Any]:
    """Append one trial to the research log and return the record written.

    Called automatically by every evaluation entry point. ``result`` may be a
    :class:`~quantlab.engine.BacktestResult` or a plain metrics mapping.
    """
    path = path or research_log_path()
    if isinstance(result, BacktestResult):
        payload = result.summary()
    else:
        payload = dict(result)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "config": _jsonable(dict(config)),
        "metrics": _jsonable(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def _read_log(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or research_log_path()
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a corrupted line must not silently reduce the count
    return records


def trial_count(path: Path | None = None) -> int:
    """Number of configurations ever evaluated, read from the research log.

    This is the ``n_trials`` that feeds :func:`quantlab.metrics.deflated_sharpe`.
    It only ever goes up.
    """
    return len(_read_log(path))


def trial_sharpes(path: Path | None = None, periods_per_year: float = 252.0) -> np.ndarray:
    """Per-period Sharpe of every logged trial, for the deflation variance.

    Using the observed scatter of trial Sharpes is materially better than the
    ``1/T`` default, which assumes the trials differ only by sampling noise and
    is therefore anti-conservative whenever the search was genuinely diverse.
    """
    values = []
    for record in _read_log(path):
        sr = record.get("metrics", {}).get("sharpe_per_period")
        if sr is None:
            annual = record.get("metrics", {}).get("sharpe_ann")
            sr = None if annual is None else annual / np.sqrt(periods_per_year)
        if sr is not None and np.isfinite(sr):
            values.append(float(sr))
    return np.asarray(values, dtype=float)


def deflated_sharpe_now(
    returns: pd.Series, path: Path | None = None, extra_trials: int = 0
) -> float:
    """Deflated Sharpe using the trial count and scatter from the research log."""
    n_trials = max(trial_count(path) + extra_trials, 1)
    observed = trial_sharpes(path)
    variance = (
        metrics.sharpe_variance_across_trials(observed) if observed.size >= 2 else None
    )
    if variance is not None and not np.isfinite(variance):
        variance = None
    return metrics.deflated_sharpe(returns, n_trials=n_trials, sharpe_variance=variance)


def current_threshold(
    n_periods: int, path: Path | None = None, periods_per_year: float = 252.0
) -> dict[str, float]:
    """The Sharpe a new result must clear, given everything tried so far.

    Reported in annualised units for display. This is the number the README is
    required to state, and it rises every time anyone runs anything.
    """
    n_trials = max(trial_count(path), 1)
    observed = trial_sharpes(path)
    variance = (
        metrics.sharpe_variance_across_trials(observed)
        if observed.size >= 2
        else 1.0 / max(n_periods - 1, 1)
    )
    if not np.isfinite(variance) or variance <= 0:
        variance = 1.0 / max(n_periods - 1, 1)
    threshold = metrics.expected_max_sharpe(n_trials, variance)
    return {
        "n_trials": float(n_trials),
        "sharpe_variance_per_period": float(variance),
        "luck_threshold_per_period": float(threshold),
        "luck_threshold_ann": float(threshold * np.sqrt(periods_per_year)),
        "n_periods_assumed": float(n_periods),
    }


# --------------------------------------------------------------------------
# Walk-forward
# --------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """Stitched out-of-sample returns plus the per-fold selection record."""

    oos_returns: pd.Series
    folds: pd.DataFrame
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def stability(self) -> dict[str, Any]:
        return parameter_stability(self.folds, self.param_grid)

    def summary(self, periods_per_year: float = 252.0, path: Path | None = None) -> dict[str, Any]:
        out = metrics.summary(self.oos_returns, periods_per_year=periods_per_year,
                              n_trials=max(trial_count(path), 1))
        out["deflated_sharpe"] = deflated_sharpe_now(self.oos_returns, path)
        out["n_folds"] = int(len(self.folds))
        out.update({f"stability_{k}": v for k, v in self.stability.items()
                    if isinstance(v, (int, float, str))})
        return out


def parameter_stability(
    folds: pd.DataFrame, param_grid: Mapping[str, Sequence[Any]] | None = None
) -> dict[str, Any]:
    """Quantify how much the selected parameters move between folds.

    Instability is a finding, not a nuisance. If the best window is 10 on one
    fold and 80 on the next, the procedure has not found a parameter -- it has
    found noise, and the out-of-sample Sharpe it produces is the average of a
    lottery rather than an estimate of anything.

    Normalised entropy of the selection distribution is the summary statistic:
    0 means the same parameter won every fold, 1 means the choice was uniform
    over the grid and carried no information at all.
    """
    if folds.empty or "params" not in folds:
        return {"entropy": float("nan"), "modal_share": float("nan"), "verdict": "no folds"}

    chosen = folds["params"].map(lambda p: json.dumps(p, sort_keys=True))
    shares = chosen.value_counts(normalize=True)
    grid_size = (
        int(np.prod([len(v) for v in param_grid.values()])) if param_grid else len(shares)
    )
    grid_size = max(grid_size, 2)
    entropy = float(-(shares * np.log(shares)).sum() / np.log(grid_size))
    modal_share = float(shares.iloc[0])

    if entropy < 0.3:
        verdict = "stable: one configuration dominates"
    elif entropy < 0.7:
        verdict = "mixed: selection moves between folds, treat the OOS Sharpe with caution"
    else:
        verdict = (
            "UNSTABLE: parameter choice is near-uniform across the grid, which is "
            "what selection on noise looks like. The out-of-sample result is an "
            "average over a lottery, not an estimate of a parameter."
        )
    return {
        "entropy": entropy,
        "modal_share": modal_share,
        "modal_params": shares.index[0],
        "n_distinct": int(len(shares)),
        "grid_size": grid_size,
        "verdict": verdict,
    }


def _grid_points(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, combo)) for combo in product(*(param_grid[k] for k in keys))]


def _evaluate(
    prices: pd.DataFrame,
    strategy: Callable[[pd.DataFrame], pd.DataFrame],
    start_pos: int,
    end_pos: int,
    warmup: int,
    cost_bps: float,
    lag: int,
) -> BacktestResult:
    """Evaluate ``strategy`` over ``[start_pos, end_pos)`` with prior bars as context.

    A signal with a 60-bar lookback recomputed from scratch on a 126-bar test
    block spends half that block undefined, so a naive implementation discards
    most of its own out-of-sample data and reports the strategy's cold-start
    behaviour rather than its steady state.

    Feeding the ``warmup`` bars immediately *preceding* the block as context and
    then discarding them fixes both problems, and it is causal: those bars are
    past data at every point in the evaluation window. It is also what actually
    happens live -- a strategy does not forget its history at the start of each
    quarter.

    When fewer than ``warmup`` prior bars exist (the very first fold), the
    shortfall is taken out of the evaluation block instead, which is the honest
    outcome: the signal genuinely is not warm yet.
    """
    context_start = max(0, start_pos - warmup)
    window = prices.iloc[context_start:end_pos]
    return run_backtest(window, strategy(window), cost_bps, lag, warmup=warmup)


def walk_forward(
    prices: pd.DataFrame,
    strategy_factory: Callable[..., Callable[[pd.DataFrame], pd.DataFrame]],
    param_grid: Mapping[str, Sequence[Any]],
    train: int = 504,
    test: int = 126,
    step: int | None = None,
    cost_bps: float = 2.0,
    lag: int = 1,
    warmup: int = 0,
    objective: Callable[[BacktestResult], float] | None = None,
    log: bool = True,
    log_path: Path | None = None,
) -> WalkForwardResult:
    """Rolling train/test with parameter re-selection on every fold.

    Each fold fits nothing beyond choosing a grid point on ``train`` bars, then
    evaluates that choice on the following ``test`` bars, which are never seen
    during selection.  The test segments are concatenated into one
    out-of-sample series.

    Trial accounting: one line is written to the research log **per grid
    point**, not per fold.  The grid is the set of hypotheses entertained; the
    folds are one procedure applied to it.  Logging per fold would over-charge
    a careful validation scheme relative to a careless single split, which is
    the wrong incentive.

    Parameters
    ----------
    strategy_factory
        ``factory(**params)`` returns a callable mapping prices to target
        positions -- a :class:`~quantlab.signals.BoundSignal` satisfies this.
    train, test
        Fold lengths in bars. ``step`` defaults to ``test``, giving contiguous
        non-overlapping out-of-sample segments.
    objective
        Fold selection criterion. Defaults to **net** Sharpe. Selecting on
        gross Sharpe would pick the highest-turnover configuration every time.
    """
    step = step or test
    if train < 2 or test < 1:
        raise ValueError("train >= 2 and test >= 1 required")
    if len(prices) < train + test:
        raise ValueError(
            f"need at least train+test = {train + test} bars, got {len(prices)}"
        )
    objective = objective or (lambda r: metrics.sharpe(r.net_returns))
    points = _grid_points(param_grid)

    fold_rows: list[dict[str, Any]] = []
    oos_segments: list[pd.Series] = []
    per_point_oos: dict[str, list[float]] = {json.dumps(p, sort_keys=True): [] for p in points}

    start = 0
    fold = 0
    while start + train + test <= len(prices):
        train_slice = prices.iloc[start : start + train]
        test_slice = prices.iloc[start + train : start + train + test]

        train_lo, train_hi = start, start + train
        test_lo, test_hi = start + train, start + train + test

        scores: dict[int, float] = {}
        for i, params in enumerate(points):
            result = _evaluate(prices, strategy_factory(**params), train_lo, train_hi,
                               warmup, cost_bps, lag)
            score = objective(result)
            scores[i] = -np.inf if not np.isfinite(score) else score

        best_index = max(scores, key=lambda i: (scores[i], -i))
        best_params = points[best_index]

        # The test block is evaluated with the preceding ``warmup`` bars as
        # context, so none of it is lost to the signal's own start-up.
        oos = _evaluate(prices, strategy_factory(**best_params), test_lo, test_hi,
                        warmup, cost_bps, lag)
        oos_segments.append(oos.net_returns)

        # Track every grid point's OOS behaviour, so the log records what each
        # hypothesis actually did rather than only what the winner did.
        for i, params in enumerate(points):
            r_i = (
                oos if i == best_index
                else _evaluate(prices, strategy_factory(**params), test_lo, test_hi,
                               warmup, cost_bps, lag)
            )
            per_point_oos[json.dumps(params, sort_keys=True)].append(
                metrics.sharpe(r_i.net_returns, annualize=False)
            )

        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_slice.index[0],
                "train_end": train_slice.index[-1],
                "test_start": test_slice.index[0],
                "test_end": test_slice.index[-1],
                "params": best_params,
                "train_score": float(scores[best_index]),
                "test_sharpe": metrics.sharpe(oos.net_returns),
                "test_return": float(oos.net_returns.sum()),
            }
        )
        start += step
        fold += 1

    if not oos_segments:
        raise ValueError("no complete folds were produced")

    stitched = pd.concat(oos_segments).rename("oos_net_return")
    folds = pd.DataFrame(fold_rows)
    config = {
        "train": train, "test": test, "step": step, "cost_bps": cost_bps,
        "lag": lag, "warmup": warmup, "n_folds": len(folds),
        "n_grid_points": len(points), "n_periods": int(len(prices)),
    }

    if log:
        for params in points:
            key = json.dumps(params, sort_keys=True)
            oos_sharpes = np.asarray(per_point_oos[key], dtype=float)
            mean_sr = float(np.nanmean(oos_sharpes)) if oos_sharpes.size else float("nan")
            log_trial(
                {**config, "params": params, "selected": params == folds["params"].iloc[0]},
                {
                    "sharpe_per_period": mean_sr,
                    "sharpe_ann": mean_sr * np.sqrt(252.0),
                    "n_periods": int(test * len(folds)),
                },
                path=log_path,
                kind="walk_forward_grid_point",
            )

    return WalkForwardResult(stitched, folds, {k: list(v) for k, v in param_grid.items()}, config)


# --------------------------------------------------------------------------
# Purged k-fold
# --------------------------------------------------------------------------


def purged_kfold(
    n_samples: int,
    n_splits: int = 5,
    embargo: int = 0,
    label_horizon: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold splits with purging and an embargo, for time-series labels.

    Plain k-fold is invalid on time series in a way that is easy to miss.  If a
    label at ``t`` is built from returns over ``t .. t + h``, then a training
    sample at ``t - 1`` shares ``h - 1`` periods of realised return with a test
    sample at ``t``.  The two are not independent, so the test set is partly a
    restatement of the training set and the measured performance is inflated.

    Two corrections, both applied here:

    - **Purging** removes training samples whose label window overlaps the test
      block -- the ``label_horizon`` bars immediately before it.
    - **Embargo** additionally drops ``embargo`` bars immediately *after* the
      test block. Serial correlation in features means a training sample just
      after the test block still carries information about it, even with no
      label overlap.

    Returns a list of ``(train_indices, test_indices)`` integer arrays. Test
    blocks are contiguous and in chronological order.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_samples < n_splits:
        raise ValueError("need at least one sample per split")
    if embargo < 0 or label_horizon < 1:
        raise ValueError("embargo >= 0 and label_horizon >= 1 required")

    indices = np.arange(n_samples)
    bounds = np.linspace(0, n_samples, n_splits + 1).astype(int)

    splits = []
    for k in range(n_splits):
        lo, hi = bounds[k], bounds[k + 1]
        test_idx = indices[lo:hi]
        # Purge the label_horizon bars before the block, embargo the bars after.
        purge_lo = max(0, lo - (label_horizon - 1))
        embargo_hi = min(n_samples, hi + embargo)
        mask = np.ones(n_samples, dtype=bool)
        mask[purge_lo:embargo_hi] = False
        splits.append((indices[mask], test_idx))
    return splits
