"""Trial accounting and out-of-sample validation.

The claim under test is the one the project brief calls non-negotiable: it must
not be *possible* to run a parameter sweep without paying for it statistically.
That is a mechanical property of the code, so it can be checked mechanically.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantlab.signals import zscore_reversion
from quantlab.simulate import gbm, ornstein_uhlenbeck
from quantlab.validate import (
    current_threshold,
    deflated_sharpe_now,
    log_trial,
    parameter_stability,
    purged_kfold,
    trial_count,
    trial_sharpes,
    walk_forward,
)

GRID = {"window": [5, 10, 20, 40, 80]}


@pytest.fixture
def log(tmp_path, monkeypatch):
    """Redirect the research log so tests never touch the real record."""
    path = tmp_path / "research_log.jsonl"
    monkeypatch.setenv("QUANTLAB_RESEARCH_LOG", str(path))
    return path


# ------------------------------------------------------- the log itself


def test_log_is_append_only_and_counted(log) -> None:
    assert trial_count() == 0
    for i in range(3):
        log_trial({"window": i}, {"sharpe_ann": 0.5, "sharpe_per_period": 0.03})
    assert trial_count() == 3

    lines = log.read_text().strip().split("\n")
    assert len(lines) == 3
    assert all(json.loads(line)["kind"] == "backtest" for line in lines)
    assert json.loads(lines[0])["config"] == {"window": 0}


def test_log_survives_non_finite_metrics(log) -> None:
    """NaN is not valid JSON; it must be stored as null rather than corrupting the line."""
    log_trial({"a": 1}, {"sharpe_ann": float("nan"), "sharpe_per_period": float("inf")})
    record = json.loads(log.read_text().strip())
    assert record["metrics"]["sharpe_ann"] is None
    assert trial_count() == 1


def test_corrupted_line_does_not_reduce_the_count(log) -> None:
    log_trial({"a": 1}, {"sharpe_per_period": 0.01})
    with log.open("a") as handle:
        handle.write("{not valid json\n")
    log_trial({"a": 2}, {"sharpe_per_period": 0.02})
    assert trial_count() == 2  # the unparseable line is skipped, not counted


# ------------------------------------------- a sweep must cost something


def test_walk_forward_logs_one_trial_per_grid_point(log) -> None:
    prices = ornstein_uhlenbeck(n_steps=1400, half_life=10, seed=1)
    walk_forward(prices, zscore_reversion.bind, GRID, train=504, test=126, warmup=80)
    assert trial_count() == len(GRID["window"])

    walk_forward(prices, zscore_reversion.bind, GRID, train=504, test=126, warmup=80)
    assert trial_count() == 2 * len(GRID["window"])


def test_a_sweep_raises_the_bar_for_every_later_result(log) -> None:
    """The enforcement claim, stated as an inequality.

    Searching a wider grid must strictly raise the Sharpe a future result has
    to clear. If it did not, the log would be decorative.
    """
    prices = ornstein_uhlenbeck(n_steps=1400, half_life=10, seed=2)

    walk_forward(prices, zscore_reversion.bind, {"window": [20]}, train=504, test=126, warmup=80)
    narrow = current_threshold(1400)["luck_threshold_ann"]

    walk_forward(prices, zscore_reversion.bind,
                 {"window": [5, 8, 12, 16, 20, 30, 45, 60, 90]}, train=504, test=126, warmup=80)
    wide = current_threshold(1400)["luck_threshold_ann"]

    assert trial_count() == 10
    assert wide > narrow, f"a 9-point sweep did not raise the threshold ({narrow} -> {wide})"


def test_deflated_sharpe_falls_as_trials_accumulate(log) -> None:
    """The same returns become less believable the more you have searched."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0006, 0.01, 1500))

    before = deflated_sharpe_now(returns)
    for i in range(60):
        log_trial({"i": i}, {"sharpe_per_period": float(rng.normal(0.02, 0.03))})
    after = deflated_sharpe_now(returns)

    assert after < before
    assert trial_count() == 60


def test_trial_sharpes_round_trip(log) -> None:
    for sr in (0.01, 0.02, 0.03):
        log_trial({}, {"sharpe_per_period": sr})
    assert np.allclose(np.sort(trial_sharpes()), [0.01, 0.02, 0.03])


# --------------------------------------------------------- walk-forward


def test_walk_forward_folds_are_chronological_and_disjoint(log) -> None:
    prices = ornstein_uhlenbeck(n_steps=2000, half_life=10, seed=3)
    result = walk_forward(prices, zscore_reversion.bind, GRID, train=504, test=126, warmup=80)
    folds = result.folds

    assert (folds["train_end"] < folds["test_start"]).all(), "a fold trains on its own test data"
    assert folds["test_start"].is_monotonic_increasing
    # Non-overlapping test segments stitch into a series with unique timestamps.
    assert not result.oos_returns.index.duplicated().any()
    assert len(result.oos_returns) == len(folds) * 126


def test_walk_forward_separates_signal_from_noise(log) -> None:
    ou = walk_forward(ornstein_uhlenbeck(n_steps=2500, half_life=10, seed=4),
                      zscore_reversion.bind, GRID, train=504, test=126, warmup=80)
    noise = walk_forward(gbm(n_steps=2500, seed=4),
                         zscore_reversion.bind, GRID, train=504, test=126, warmup=80)

    from quantlab.metrics import sharpe

    assert sharpe(ou.oos_returns) > 0.8
    assert sharpe(noise.oos_returns) < 0.4
    assert ou.stability["entropy"] < noise.stability["entropy"]


def test_parameter_instability_is_surfaced_as_a_verdict() -> None:
    grid = {"window": [5, 10, 20, 40, 80]}
    stable = pd.DataFrame({"params": [{"window": 20}] * 10})
    scattered = pd.DataFrame({"params": [{"window": w} for w in [5, 10, 20, 40, 80] * 2]})

    assert parameter_stability(stable, grid)["entropy"] == pytest.approx(0.0)
    assert "stable" in parameter_stability(stable, grid)["verdict"]

    assert parameter_stability(scattered, grid)["entropy"] == pytest.approx(1.0)
    assert "UNSTABLE" in parameter_stability(scattered, grid)["verdict"]


def test_walk_forward_rejects_a_series_too_short_for_one_fold() -> None:
    with pytest.raises(ValueError, match="at least"):
        walk_forward(gbm(n_steps=100, seed=0), zscore_reversion.bind, GRID,
                     train=504, test=126, log=False)


# ---------------------------------------------------------- purged k-fold


@pytest.mark.parametrize("embargo,horizon", [(0, 1), (5, 1), (0, 10), (10, 5)])
def test_purged_kfold_train_and_test_never_intersect(embargo: int, horizon: int) -> None:
    for train_idx, test_idx in purged_kfold(500, n_splits=5, embargo=embargo,
                                            label_horizon=horizon):
        assert not set(train_idx) & set(test_idx)
        assert len(test_idx) > 0 and len(train_idx) > 0


def test_purged_kfold_leaves_the_requested_gaps() -> None:
    """Purge ``label_horizon - 1`` bars before the block, embargo ``embargo`` after."""
    splits = purged_kfold(100, n_splits=5, embargo=5, label_horizon=3)
    train_idx, test_idx = splits[2]

    before = train_idx[train_idx < test_idx.min()]
    after = train_idx[train_idx > test_idx.max()]
    assert test_idx.min() - before.max() == 3      # 2 purged bars + 1
    assert after.min() - test_idx.max() == 6       # 5 embargoed bars + 1


def test_purged_kfold_covers_every_sample_exactly_once_in_test() -> None:
    splits = purged_kfold(500, n_splits=5)
    covered = np.concatenate([test for _, test in splits])
    assert np.array_equal(np.sort(covered), np.arange(500))


def test_embargo_of_zero_still_purges_the_label_horizon() -> None:
    train_idx, test_idx = purged_kfold(100, n_splits=4, embargo=0, label_horizon=5)[1]
    before = train_idx[train_idx < test_idx.min()]
    assert test_idx.min() - before.max() == 5
