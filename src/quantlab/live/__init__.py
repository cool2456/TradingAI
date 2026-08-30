"""Live-data extensions to quantlab.

Phase 2 exists to answer one question: do any of the signals in
:mod:`quantlab.signals` have an information coefficient reliably above zero on
real market data, at a horizon where costs do not eat it?

:mod:`quantlab.live.ic` is the deliverable. Everything else is plumbing that
only matters if the answer is yes.
"""

from __future__ import annotations

__all__ = ["config", "datafeed", "featurestore", "ic"]
