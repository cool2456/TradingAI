import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type SessionConfig, type Tick } from './api'
import { BudgetIndicator, RegimeBadge } from './components/Header'
import { Controls } from './components/Controls'
import { MarketTab } from './MarketTab'
import { StrategiesTab } from './StrategiesTab'

type TabName = 'market' | 'strategies'

const DEFAULT_CONFIG: SessionConfig = {
  generator: 'regime_switching',
  n_steps: 1500,
  n_assets: 4,
  correlation: 0.25,
  seed: 7,
  cost_bps: 2.0,
  budget: 3,
  target_ann_vol: 0.1,
  max_leverage: 3.0,
  tier: 2,
  allow_tier3: false,
  edge_bps: 5.0,
  min_trade_size: 0.05,
}

export default function App() {
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [assets, setAssets] = useState<string[]>([])
  const [tick, setTick] = useState<Tick | null>(null)
  const [tab, setTab] = useState<TabName>('market')
  const [running, setRunning] = useState(true)
  const [speed, setSpeed] = useState(4)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // Guards against overlapping requests: at 20x the poll interval can be
  // shorter than the round trip, and without this the cursor advances in
  // bursts and the chart stutters.
  const inFlight = useRef(false)

  const start = useCallback(async (next: SessionConfig) => {
    setBusy(true)
    setError(null)
    try {
      const info = await api.createSession(next)
      setSessionId(info.session_id)
      setAssets(info.assets)
      setTick(await api.tick(info.session_id, 0))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    void start(DEFAULT_CONFIG)
  }, [start])

  // Reconfigure in place so the cursor and the visible history survive a
  // change to cost, budget, volatility target or regime tier.
  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    const handle = setTimeout(async () => {
      setBusy(true)
      try {
        await api.updateSession(sessionId, config)
        if (!cancelled) setTick(await api.tick(sessionId, 0))
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (!cancelled) setBusy(false)
      }
    }, 220)
    return () => {
      cancelled = true
      clearTimeout(handle)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config])

  useEffect(() => {
    if (!running || !sessionId) return
    const handle = setInterval(async () => {
      if (inFlight.current) return
      inFlight.current = true
      try {
        const next = await api.tick(sessionId, 1)
        setTick(next)
        if (next.finished) setRunning(false)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        setRunning(false)
      } finally {
        inFlight.current = false
      }
    }, Math.max(1000 / speed, 40))
    return () => clearInterval(handle)
  }, [running, sessionId, speed])

  const patch = useCallback((delta: Partial<SessionConfig>) => {
    setConfig((prev) => ({ ...prev, ...delta }))
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>quantlab</h1>
          <span className="tag">research · never traded real money</span>
        </div>
        <div className="tabs" role="tablist">
          <button
            className="tab"
            role="tab"
            aria-selected={tab === 'market'}
            onClick={() => setTab('market')}
          >
            Market
          </button>
          <button
            className="tab"
            role="tab"
            aria-selected={tab === 'strategies'}
            onClick={() => setTab('strategies')}
          >
            Strategies
          </button>
        </div>
        <div className="header-right">
          <RegimeBadge tick={tick} />
          <BudgetIndicator tick={tick} />
        </div>
      </header>

      <Controls
        config={config}
        onChange={patch}
        running={running}
        onToggleRun={() => setRunning((r) => !r)}
        onRestart={() => void start(config)}
        speed={speed}
        onSpeed={setSpeed}
        busy={busy}
      />

      {error && <div className="error">{error}</div>}

      <main className="main">
        {tab === 'market' ? (
          <MarketTab tick={tick} assets={assets} />
        ) : (
          <StrategiesTab tick={tick} />
        )}
      </main>
    </div>
  )
}
