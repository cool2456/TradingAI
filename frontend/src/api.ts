/**
 * Client for the quantlab research backend.
 *
 * Every figure returned by these endpoints is net of transaction costs. The
 * frontend does not compute performance numbers of its own -- it displays what
 * the validated engine produced, so there is exactly one implementation of the
 * accounting and no chance of the screen and the test suite disagreeing.
 */

export type Point = { t: string; v: number | null }
export type StrategyKind = 'strategy' | 'control' | 'portfolio'

export interface StrategyStats {
  pnl: number | null
  sharpe_ann: number | null
  win_rate: number | null
  max_drawdown: number | null
  cost_paid: number | null
  trades: number
  n_periods: number
}

export interface StrategyBlock {
  name: string
  kind: StrategyKind
  /** Current per-asset position held by this strategy standalone. */
  positions: Record<string, number>
  equity: Point[]
  stats: StrategyStats
}

export interface RegimeBand {
  regime: string
  start: string
  end: string
  bars: number
}

export interface RoundTrip {
  asset: string
  side: 'long' | 'short'
  opened: string
  closed: string
  bars: number
  pnl: number | null
}

export interface Tick {
  session_id: string
  cursor: number
  n_steps: number
  finished: boolean
  timestamp: string
  prices: Array<Record<string, string | number>>
  regime: {
    current: string | null
    tier: number
    bands: RegimeBand[]
    true_state: string | null
  }
  realized_vol: Point[]
  positions: Record<string, number>
  budget: { used_today: number; limit: number; date: string }
  strategies: StrategyBlock[]
  trade_tape: RoundTrip[]
}

export interface SessionConfig {
  generator: 'gbm' | 'ornstein_uhlenbeck' | 'garch_t' | 'regime_switching'
  n_steps: number
  n_assets: number
  correlation: number
  seed: number | null
  cost_bps: number
  budget: number
  target_ann_vol: number
  max_leverage: number
  tier: 1 | 2 | 3
  allow_tier3: boolean
  edge_bps: number
  min_trade_size: number
}

export interface SessionInfo {
  session_id: string
  config: SessionConfig
  n_steps: number
  warmup: number
  cursor: number
  assets: string[]
}

export interface Defaults {
  config: SessionConfig
  strategies: string[]
  controls: string[]
  weight_map: Record<string, Record<string, number>>
  tiers: Record<string, string>
}

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`${response.status} ${path}: ${detail.slice(0, 300)}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  defaults: () => request<Defaults>('/config/defaults'),
  createSession: (config: SessionConfig) =>
    request<SessionInfo>('/session', { method: 'POST', body: JSON.stringify(config) }),
  updateSession: (id: string, config: SessionConfig) =>
    request<{ cursor: number }>(`/session/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(config),
    }),
  tick: (id: string, steps: number) =>
    request<Tick>(`/session/${id}/tick?steps=${steps}`),
  deleteSession: (id: string) => request<unknown>(`/session/${id}`, { method: 'DELETE' }),
}

/** Formatting helpers. All numeric display goes through these so alignment is uniform. */
export const fmt = {
  pct: (v: number | null | undefined, digits = 2) =>
    v == null || !isFinite(v) ? '--' : `${(v * 100).toFixed(digits)}%`,
  num: (v: number | null | undefined, digits = 2) =>
    v == null || !isFinite(v) ? '--' : v.toFixed(digits),
  int: (v: number | null | undefined) => (v == null ? '--' : Math.round(v).toLocaleString()),
  date: (iso: string) => iso.slice(0, 10),
}
