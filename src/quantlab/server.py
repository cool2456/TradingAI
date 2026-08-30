"""FastAPI backend for the research frontend.

A note on how the live session works
------------------------------------
Each session generates its whole price path up front, runs the pipeline once
over the full path, and then *reveals* the result bar by bar as the frontend
ticks.

That is only legitimate because every component is causal: the value computed
at bar ``t`` from the full path is identical to the value computed from
``prices[:t+1]``.  That is precisely the property ``tests/test_causality.py``
asserts, so this optimisation is a direct payoff of the test suite -- and if
those tests ever fail, this design silently starts showing the frontend
information from the future.  The alternative, recomputing the pipeline on a
growing prefix at every tick, is quadratic and no more correct.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import simulate
from .engine import DEFAULT_WEIGHT_MAP, PortfolioResult, run_regime_portfolio
from .metrics import sharpe
from .regime import hypothesis_cost, realized_vol, regime_sample_counts
from .signals import CONTROLS, SIGNALS
from .validate import current_threshold, log_trial, trial_count, _read_log

app = FastAPI(
    title="quantlab",
    version="0.1.0",
    description=(
        "Research backend. Every figure returned is net of transaction costs. "
        "A backtest is a hypothesis, not a result."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GENERATORS = {
    "gbm": simulate.gbm,
    "ornstein_uhlenbeck": simulate.ornstein_uhlenbeck,
    "garch_t": simulate.garch_t,
    "regime_switching": simulate.regime_switching,
}


class SessionConfig(BaseModel):
    """Everything the frontend can change. Any change re-runs the pipeline."""

    generator: Literal["gbm", "ornstein_uhlenbeck", "garch_t", "regime_switching"] = (
        "regime_switching"
    )
    n_steps: int = Field(1500, ge=400, le=10000)
    n_assets: int = Field(4, ge=1, le=12)
    correlation: float = Field(0.25, gt=-1.0, le=1.0)
    seed: int | None = 7
    cost_bps: float = Field(2.0, ge=0.0, le=100.0)
    budget: int = Field(3, ge=0, le=50)
    target_ann_vol: float = Field(0.10, gt=0.0, le=1.0)
    max_leverage: float = Field(3.0, gt=0.0, le=10.0)
    tier: Literal[1, 2, 3] = 2
    allow_tier3: bool = False
    edge_bps: float = Field(5.0, ge=-50.0, le=200.0)
    min_trade_size: float = Field(0.05, ge=0.0, le=1.0)


class BacktestRequest(SessionConfig):
    """A full historical backtest. Same knobs, no live cursor."""

    log: bool = True


@dataclass
class Session:
    id: str
    config: SessionConfig
    prices: pd.DataFrame
    portfolio: PortfolioResult
    true_states: pd.Series | None
    cursor: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


SESSIONS: dict[str, Session] = {}
MAX_SESSIONS = 32


def _build(config: SessionConfig) -> tuple[pd.DataFrame, pd.Series | None, PortfolioResult]:
    kwargs: dict[str, Any] = {
        "n_steps": config.n_steps,
        "n_assets": config.n_assets,
        "correlation": config.correlation,
        "seed": config.seed,
    }
    true_states = None
    if config.generator == "regime_switching":
        prices, true_states = simulate.regime_switching(**kwargs, return_states=True)
    else:
        prices = GENERATORS[config.generator](**kwargs)

    try:
        portfolio = run_regime_portfolio(
            prices,
            tier=config.tier,
            budget=config.budget,
            cost_bps=config.cost_bps,
            target_ann_vol=config.target_ann_vol,
            max_leverage=config.max_leverage,
            edge_bps=config.edge_bps,
            min_trade_size=config.min_trade_size,
            allow_tier3=config.allow_tier3,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return prices, true_states, portfolio


def _finite(value: Any) -> Any:
    """JSON has no NaN. Convert non-finite floats to None rather than emitting invalid JSON."""
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 8)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if value is pd.NaT or (isinstance(value, float) and pd.isna(value)):
        return None
    return value


def _downsample(series: pd.Series, max_points: int) -> list[dict[str, Any]]:
    if series.empty:
        return []
    step = max(1, len(series) // max_points)
    sampled = series.iloc[::step]
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return [
        {"t": ts.isoformat(), "v": _finite(v)}
        for ts, v in sampled.items()
    ]


def _regime_bands(labels: pd.Series) -> list[dict[str, Any]]:
    """Compress a label series into contiguous runs, for shaded bands."""
    clean = labels.dropna()
    if clean.empty:
        return []
    values = clean.astype(str)
    change = values.ne(values.shift())
    group = change.cumsum()
    bands = []
    for _, run in values.groupby(group):
        bands.append(
            {
                "regime": run.iloc[0],
                "start": run.index[0].isoformat(),
                "end": run.index[-1].isoformat(),
                "bars": int(len(run)),
            }
        )
    return bands


def _round_trips(positions: pd.DataFrame, returns: pd.DataFrame, limit: int = 40) -> list[dict]:
    """Closed round trips: a position opened, held, and returned to flat or flipped.

    P&L is the sum of that asset's realised net contribution while the position
    was open, so the tape reconciles with the equity curve rather than being a
    separate, prettier accounting.
    """
    trips: list[dict[str, Any]] = []
    for asset in positions.columns:
        pos = positions[asset].fillna(0.0)
        ret = returns[asset].fillna(0.0)
        side = np.sign(pos.to_numpy())
        open_at: int | None = None
        for i in range(1, len(side)):
            if side[i] != side[i - 1]:
                if open_at is not None and side[i - 1] != 0:
                    window = slice(open_at + 1, i + 1)
                    pnl = float((pos.to_numpy()[window.start - 1 : window.stop - 1]
                                 * ret.to_numpy()[window]).sum())
                    trips.append(
                        {
                            "asset": asset,
                            "side": "long" if side[i - 1] > 0 else "short",
                            "opened": pos.index[open_at].isoformat(),
                            "closed": pos.index[i].isoformat(),
                            "bars": int(i - open_at),
                            "pnl": _finite(pnl),
                        }
                    )
                open_at = i if side[i] != 0 else None
    trips.sort(key=lambda trip: trip["closed"], reverse=True)
    return trips[:limit]


def _strategy_block(portfolio: PortfolioResult, upto: pd.Timestamp, max_points: int) -> list[dict]:
    """Per-strategy equity, stats and control flag, truncated at the cursor."""
    blocks = []
    entries: list[tuple[str, Any, str]] = [
        (name, result, "control" if name in CONTROLS else "strategy")
        for name, result in portfolio.per_strategy.items()
    ]
    entries.append(("PORTFOLIO regime-conditional", portfolio.regime_conditional, "portfolio"))
    entries.append(("PORTFOLIO fixed-weight", portfolio.fixed_weight, "portfolio"))

    for name, result, kind in entries:
        net = result.net_returns.loc[:upto]
        if net.empty:
            continue
        equity = (1.0 + net.fillna(0.0)).cumprod()
        costs = result.costs.loc[:upto]
        trades = result.trade_counts.loc[:upto]
        wins = int((net > 0).sum())
        losses = int((net < 0).sum())
        drawdown = float((equity / equity.cummax() - 1.0).min())
        held = result.positions.loc[:upto]
        blocks.append(
            {
                "name": name,
                "kind": kind,
                "positions": (
                    {k: _finite(v) for k, v in held.iloc[-1].to_dict().items()}
                    if len(held) else {}
                ),
                "equity": _downsample(equity, max_points),
                "stats": _finite(
                    {
                        "pnl": float(equity.iloc[-1] - 1.0),
                        "sharpe_ann": sharpe(net),
                        "win_rate": wins / (wins + losses) if wins + losses else None,
                        "max_drawdown": drawdown,
                        "cost_paid": float(costs.sum()),
                        "trades": int(trades.sum()),
                        "n_periods": int(len(net)),
                    }
                ),
            }
        )
    return blocks


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "sessions": len(SESSIONS), "trials_logged": trial_count()}


@app.get("/config/defaults")
def defaults() -> dict[str, Any]:
    """Defaults, plus the metadata the frontend needs to render controls honestly."""
    return {
        "config": SessionConfig().model_dump(),
        "strategies": [n for n in SIGNALS if n not in CONTROLS],
        "controls": list(CONTROLS),
        "weight_map": DEFAULT_WEIGHT_MAP,
        "tiers": {
            "1": "volatility regime drives position sizing - well supported",
            "2": "trend/chop regime drives strategy weights - testable but unproven",
            "3": "12 fine-grained regimes drive selection - HIGH OVERFITTING RISK",
        },
    }


@app.post("/session")
def create_session(config: SessionConfig) -> dict[str, Any]:
    """Start a simulation and return its id."""
    if len(SESSIONS) >= MAX_SESSIONS:
        oldest = min(SESSIONS.values(), key=lambda s: s.created_at)
        SESSIONS.pop(oldest.id, None)

    prices, true_states, portfolio = _build(config)
    session_id = uuid.uuid4().hex[:12]
    # Start the cursor at the end of warm-up: before that there is no regime,
    # no volatility estimate and no position, and showing a flat line there
    # invites the reader to mistake warm-up for a result.
    cursor = min(portfolio.warmup + 1, len(prices) - 1)
    SESSIONS[session_id] = Session(session_id, config, prices, portfolio, true_states, cursor)
    return {
        "session_id": session_id,
        "config": config.model_dump(),
        "n_steps": len(prices),
        "warmup": portfolio.warmup,
        "cursor": cursor,
        "assets": list(prices.columns),
        "note": "All returned figures are net of costs.",
    }


@app.patch("/session/{session_id}")
def update_session(session_id: str, config: SessionConfig) -> dict[str, Any]:
    """Change the configuration and re-run the pipeline, preserving the cursor."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    prices, true_states, portfolio = _build(config)
    session.config = config
    session.prices = prices
    session.portfolio = portfolio
    session.true_states = true_states
    session.cursor = min(max(session.cursor, portfolio.warmup + 1), len(prices) - 1)
    return {"session_id": session_id, "config": config.model_dump(), "cursor": session.cursor}


@app.get("/session/{session_id}/tick")
def tick(
    session_id: str,
    steps: int = 1,
    window: int = 250,
    max_points: int = 220,
) -> dict[str, Any]:
    """Advance the session by ``steps`` bars and return the visible state."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    if steps < 0:
        raise HTTPException(status_code=400, detail="steps must be non-negative")

    session.cursor = min(session.cursor + steps, len(session.prices) - 1)
    cursor = session.cursor
    now = session.prices.index[cursor]
    portfolio = session.portfolio

    visible = session.prices.iloc[max(0, cursor - window + 1) : cursor + 1]
    positions = portfolio.regime_conditional.positions.loc[:now]
    regime_history = portfolio.regime.loc[visible.index[0] : now]
    vol = realized_vol(session.prices.iloc[: cursor + 1], window=20).mean(axis=1)

    current_regime = portfolio.regime.loc[:now].dropna()
    budget_used = portfolio.budget_used.loc[: now.normalize()]
    today = int(budget_used.iloc[-1]) if len(budget_used) else 0

    latest_positions = (
        positions.iloc[-1].to_dict() if len(positions) else
        {asset: 0.0 for asset in session.prices.columns}
    )

    return _finite(
        {
            "session_id": session_id,
            "cursor": cursor,
            "n_steps": len(session.prices),
            "finished": cursor >= len(session.prices) - 1,
            "timestamp": now.isoformat(),
            "prices": [
                {"t": ts.isoformat(), **{c: float(v) for c, v in row.items()}}
                for ts, row in visible.iterrows()
            ],
            "regime": {
                "current": str(current_regime.iloc[-1]) if len(current_regime) else None,
                "tier": session.config.tier,
                "bands": _regime_bands(regime_history),
                "true_state": (
                    ["trend", "chop"][int(session.true_states.iloc[cursor])]
                    if session.true_states is not None
                    else None
                ),
            },
            "realized_vol": _downsample(vol.loc[visible.index[0] :], max_points),
            "positions": {k: _finite(v) for k, v in latest_positions.items()},
            "budget": {
                "used_today": today,
                "limit": session.config.budget,
                "date": now.normalize().date().isoformat(),
            },
            "strategies": _strategy_block(portfolio, now, max_points),
            "trade_tape": _round_trips(
                positions, session.prices.pct_change().loc[:now]
            ),
        }
    )


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    if SESSIONS.pop(session_id, None) is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return {"deleted": session_id}


@app.post("/backtest")
def backtest(request: BacktestRequest) -> dict[str, Any]:
    """Run a full historical backtest and return metrics for every book.

    Logs one trial per evaluated book, so calling this endpoint repeatedly with
    different settings raises the deflated-Sharpe threshold exactly as a manual
    parameter sweep would.
    """
    prices, _, portfolio = _build(request)
    n_trials = max(trial_count(), 1)
    ledger = portfolio.ledger(n_trials=n_trials)

    if request.log:
        for name, row in ledger.iterrows():
            log_trial(
                {**portfolio.config, "book": name, "generator": request.generator,
                 "seed": request.seed, "n_steps": request.n_steps},
                row.to_dict(),
                kind="server_backtest",
            )

    verdict = portfolio.regime_verdict()
    counts = regime_sample_counts(portfolio.regime)
    return _finite(
        {
            "config": request.model_dump(),
            "ledger": [{"name": name, **row.to_dict()} for name, row in ledger.iterrows()],
            "regime_verdict": verdict,
            "regime_sample_counts": [
                {"regime": str(idx), **row.to_dict()} for idx, row in counts.iterrows()
            ],
            "hypothesis_cost": hypothesis_cost(
                len(portfolio.per_strategy),
                max(int(portfolio.regime.dropna().nunique()), 1),
                len(prices),
            ),
            "threshold": current_threshold(len(prices)),
            "note": (
                "Every figure is net of transaction costs. The two controls "
                "(always_long, random_signal) are listed in the same table with the "
                "same prominence; a strategy that does not beat both has failed."
            ),
        }
    )


@app.get("/research-log")
def research_log(limit: int = 200, n_periods: int = 2500) -> dict[str, Any]:
    """Trial history and the current deflated-Sharpe threshold."""
    records = _read_log()
    threshold = current_threshold(n_periods)
    return _finite(
        {
            "n_trials": len(records),
            "threshold": threshold,
            "trials": records[-limit:],
            "note": (
                "Every configuration ever evaluated appears here. The luck threshold "
                "rises with the count: it is the Sharpe the best of this many "
                "worthless strategies would be expected to reach by chance alone."
            ),
        }
    )
