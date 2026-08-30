"""Settings, credentials and risk limits.

Two principles govern this module.

**Every risk limit is required.** None has a default. A default risk limit is a
decision made by whoever wrote the library on behalf of whoever runs it, and it
will be wrong. If the operator has not stated a daily loss limit, the correct
behaviour is to refuse to start, not to invent one.

**Credentials never touch disk and never reach a log.** They are read from the
environment, held in a type whose ``repr`` is redacted, and never serialised.
``.env`` is in ``.gitignore`` from the first commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "Feed",
    "Adjustment",
    "DisconnectPolicy",
    "Credentials",
    "RiskLimits",
    "LiveConfig",
    "load_credentials",
    "load_env_file",
    "SIP_DELAY",
    "PAPER_BASE_URL",
]

#: Free-tier historical SIP data is available only for queries whose ``end`` is
#: at least this far in the past. Research is not real-time, so this costs
#: nothing; live decisions must use IEX.
SIP_DELAY = timedelta(minutes=15)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
#: Deliberately recorded so the paper-only check can name what it is refusing.
#: Nothing in this package may construct a client against it.
_LIVE_BASE_URL = "https://api.alpaca.markets"


class Feed(str, Enum):
    """Which consolidated tape a bar came from.

    ``IEX`` is roughly 2% of US equity volume with quotes wider than the NBBO.
    It is what the free tier provides in real time, and it is not
    representative of executable prices. ``SIP`` is the full consolidated tape.

    This is required at every call site rather than defaulted, because a
    research result computed on IEX bars and reported as if it were SIP is a
    silent substitution of the same kind as Phase 1's regime-label bug: every
    number valid, the label wrong.
    """

    IEX = "iex"
    SIP = "sip"


class Adjustment(str, Enum):
    """Corporate-action handling.

    ``ALL`` is the only correct choice for research. Unadjusted prices put a
    phantom gap at every split and dividend, which every momentum and
    breakout signal will happily trade.
    """

    RAW = "raw"
    SPLIT = "split"
    DIVIDEND = "dividend"
    ALL = "all"


class DisconnectPolicy(str, Enum):
    """What to do when the data feed or broker connection is lost.

    There is deliberately **no default**. ``HOLD`` risks carrying an unmanaged
    position through a move you cannot see; ``FLATTEN`` risks selling into a
    gap because a socket dropped. Which is worse depends on the strategy and
    the operator's tolerance, and neither this library nor its author can know
    that. The operator chooses, and the choice is logged.
    """

    HOLD = "hold"
    FLATTEN = "flatten"


@dataclass(frozen=True)
class Credentials:
    """Alpaca API credentials, redacted in every string representation.

    ``__repr__`` and ``__str__`` are overridden so that an accidental
    ``print(config)``, an exception traceback, or a structured log call cannot
    leak the secret. This is not paranoia -- tracebacks are the most common way
    credentials end up in a log aggregator.
    """

    key_id: str
    secret_key: str

    def __repr__(self) -> str:
        tail = self.key_id[-4:] if len(self.key_id) >= 4 else "?"
        return f"Credentials(key_id='...{tail}', secret_key='<redacted>')"

    __str__ = __repr__

    def __post_init__(self) -> None:
        if not self.key_id or not self.secret_key:
            raise ValueError("both key_id and secret_key must be non-empty")


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Read ``KEY=value`` lines from a dotenv file into ``os.environ``.

    Existing environment variables win, so an explicitly exported value is
    never silently overridden by a stale file. Returns the names loaded --
    names only, never values.
    """
    path = Path(path)
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        loaded[key] = "<set>"
    return loaded


def load_credentials(env_file: str | Path | None = ".env") -> Credentials:
    """Read credentials from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``.

    Raises with an actionable message when either is missing. The message names
    the variables and never echoes a partial value.
    """
    if env_file is not None:
        load_env_file(env_file)
    key_id = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    missing = [
        name
        for name, value in (("APCA_API_KEY_ID", key_id), ("APCA_API_SECRET_KEY", secret))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"missing Alpaca credentials: {', '.join(missing)}. Export them, or put "
            "them in a .env file (which is gitignored). Paper-trading keys only -- "
            "this package refuses live endpoints."
        )
    return Credentials(key_id, secret)


class RiskLimits(BaseModel):
    """Hard limits checked before every order and at the top of every loop.

    Every field is required. See the module docstring for why.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    daily_loss_limit: float = Field(
        ..., gt=0, description="Absolute account loss, in currency, that halts trading for the day. No override exists."
    )
    max_gross_leverage: float = Field(
        ..., gt=0, description="Sum of |position notional| divided by equity."
    )
    max_position_notional: float = Field(
        ..., gt=0, description="Largest absolute notional permitted in any single symbol."
    )
    trade_budget_per_day: int = Field(
        ..., ge=0, description="Discrete order submissions permitted per session, portfolio-wide."
    )
    min_trade_notional: float = Field(
        ...,
        gt=0,
        description=(
            "The definition of 'a trade'. An order below this notional is not "
            "submitted and does not consume budget. Volatility targeting gives "
            "every bar nonzero turnover, so without this threshold the trade "
            "count is unbounded and a 3-per-day budget is exhausted by rounding."
        ),
    )
    max_orders_per_minute: int = Field(..., gt=0)
    data_staleness_seconds: float = Field(
        ...,
        gt=0,
        description=(
            "Halt if the newest bar is older than this. A frozen feed with an "
            "open position is the most dangerous state the system can occupy: "
            "it looks calm and is not."
        ),
    )
    disconnect_policy: DisconnectPolicy = Field(
        ..., description="No default. The operator must decide; see DisconnectPolicy."
    )


class LiveConfig(BaseModel):
    """Top-level runtime configuration. Paper only, dry-run by default."""

    model_config = {"frozen": True, "extra": "forbid"}

    risk: RiskLimits
    universe: list[str] = Field(..., min_length=1)
    research_feed: Feed = Feed.SIP
    live_feed: Feed = Feed.IEX
    adjustment: Adjustment = Adjustment.ALL
    timeframe_minutes: int = Field(1, gt=0)
    base_url: str = PAPER_BASE_URL
    dry_run: bool = True
    paper: Literal[True] = True
    data_root: Path = Path("data")

    @field_validator("base_url")
    @classmethod
    def _refuse_live_endpoint(cls, value: str) -> str:
        if value.rstrip("/") == _LIVE_BASE_URL:
            raise ValueError(
                "live trading is an explicit anti-goal of this project. "
                f"base_url must be the paper endpoint ({PAPER_BASE_URL})."
            )
        return value

    @model_validator(mode="after")
    def _research_feed_must_be_sip(self) -> "LiveConfig":
        if self.research_feed is not Feed.SIP:
            raise ValueError(
                "research_feed must be SIP. IEX is ~2% of volume with quotes wider "
                "than the NBBO; an information coefficient measured on it does not "
                "describe the market. Override per-call with allow_iex=True and a "
                "documented reason if you really mean it."
            )
        return self

    @property
    def credentials(self) -> Credentials:
        """Loaded on access, never stored on the model, never serialised."""
        return load_credentials()
