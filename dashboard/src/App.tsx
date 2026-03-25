import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Play, Square, Zap, RefreshCw, Trash2, ExternalLink, Activity, Waves, LayoutDashboard, CheckCircle, AlertCircle, Loader2 } from "lucide-react"

const DASHBOARDS = [
  { id: "ddm",    label: "DDM Dashboard",  url: "http://localhost/grafana/d/advz874/ddm-dashboard?orgId=1&kiosk&refresh=5s&theme=dark" },
  { id: "mlops",  label: "MLOps Monitor",  url: "http://localhost/grafana/d/ddm_pipeline/ddm-e28093-mlops-monitor?orgId=1&kiosk&refresh=5s&theme=dark" },
  { id: "airflow",label: "Airflow",         url: "http://localhost/airflow/dags/retrain_dag" },
  { id: "mlflow", label: "MLflow",          url: "http://localhost/mlflow/" },
  { id: "minio",  label: "MinIO",           url: "http://localhost/minio/" },
]

type Toast = { id: number; msg: string; ok: boolean }

export default function App() {
  const [simStatus, setSimStatus]     = useState({ running: false })
  const [driftStatus, setDriftStatus] = useState({ is_drifted: false, drift_score: null as number | null })
  const [driftActive, setDriftActive] = useState(false)
  const [activeDash, setActiveDash]   = useState("ddm")
  const [toasts, setToasts]           = useState<Toast[]>([])
  // Track which action is in-flight to prevent duplicate requests
  const [loading, setLoading]         = useState<Record<string, boolean>>({})
  const API_BASE = "/api"

  const toast = useCallback((msg: string, ok = true) => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, msg, ok }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 3500)
  }, [])

  const checkStatus = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/health`)
      const data = await res.json()
      setSimStatus({ running: data.sim_running })
    } catch { /* API offline */ }
  }, [])

  const checkDrift = useCallback(async () => {
    try {
      const res  = await fetch(`${API_BASE}/drift/status`)
      const data = await res.json()
      setDriftStatus({ is_drifted: data.is_drifted, drift_score: data.drift_score })
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    checkStatus()
    checkDrift()
    const i1 = setInterval(checkStatus, 5000)
    const i2 = setInterval(checkDrift,  30000)
    return () => { clearInterval(i1); clearInterval(i2) }
  }, [checkStatus, checkDrift])

  /** Generic POST action with loading-lock to prevent duplicate clicks. */
  const handleAction = useCallback(async (
    key: string,
    endpoint: string,
    params: Record<string, string> = {},
    confirmMsg?: string,
  ) => {
    if (loading[key]) return                                    // already in-flight
    if (confirmMsg && !window.confirm(confirmMsg)) return       // user cancelled

    setLoading(prev => ({ ...prev, [key]: true }))
    try {
      const url = new URL(`${window.location.origin}${API_BASE}${endpoint}`)
      Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v))
      const res  = await fetch(url.toString(), { method: "POST" })
      let data: Record<string, unknown> = {}
      data = await res.json()
      const responseMsg = typeof data.message === "string" ? data.message : "Done"
      toast(responseMsg, res.ok)
      await checkStatus()
    } catch {
      toast("Failed to connect to API server", false)
    } finally {
      setLoading(prev => ({ ...prev, [key]: false }))
    }
  }, [loading, toast, checkStatus])

  const toggleDrift = useCallback(() => {
    const next = !driftActive
    setDriftActive(next)
    handleAction("drift", "/sim/drift", { active: String(next) })
  }, [driftActive, handleAction])

  const isLoading = (key: string) => !!loading[key]

  return (
    <div className="flex h-screen bg-background text-foreground overflow-hidden font-sans dark">

      {/* ── TOAST LAYER ── */}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map(t => (
          <div key={t.id}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg shadow-lg text-xs font-medium border
              animate-in slide-in-from-right-4 fade-in duration-200
              ${t.ok
                ? "bg-green-900/90 border-green-700 text-green-200"
                : "bg-red-900/90  border-red-700  text-red-200"}`}>
            {t.ok
              ? <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
              : <AlertCircle  className="w-3.5 h-3.5 flex-shrink-0" />}
            {t.msg}
          </div>
        ))}
      </div>

      {/* ── LEFT SIDEBAR ── */}
      <aside className="w-60 flex-shrink-0 bg-card border-r border-border flex flex-col overflow-y-auto">

        {/* Brand */}
        <div className="p-4 border-b border-border flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary flex-shrink-0" />
          <span className="font-bold text-sm tracking-tight text-primary">CWRU Diagnostics</span>
        </div>

        <div className="flex flex-col gap-2 p-4 flex-1">

          {/* ── Dashboard Switcher ── */}
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-1 flex items-center gap-1">
            <LayoutDashboard className="w-3 h-3" /> View
          </p>
          <div className="flex flex-col gap-1">
            {DASHBOARDS.map(d => (
              <button
                key={d.id}
                onClick={() => setActiveDash(d.id)}
                className={`text-left text-xs px-3 py-2 rounded-md transition-colors ${
                  activeDash === d.id
                    ? "bg-primary/15 text-primary font-semibold border border-primary/30"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                {d.label}
              </button>
            ))}
          </div>

          <div className="border-t border-border my-2" />

          {/* Status Badges */}
          <div className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-bold border
            ${simStatus.running
              ? "bg-green-500/10 text-green-400 border-green-500/30"
              : "bg-muted text-muted-foreground border-border"}`}>
            <span className={`w-2 h-2 rounded-full ${simStatus.running ? "bg-green-400 animate-pulse" : "bg-muted-foreground"}`} />
            {simStatus.running ? "ONLINE" : "STANDBY"}
          </div>

          {driftStatus.is_drifted && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-bold border bg-red-500/10 text-red-400 border-red-500/30 animate-pulse">
              <span className="w-2 h-2 rounded-full bg-red-400" />
              DRIFT DETECTED
              {driftStatus.drift_score !== null && (
                <span className="ml-auto opacity-70">{(driftStatus.drift_score * 100).toFixed(0)}%</span>
              )}
            </div>
          )}

          {/* Start / Stop */}
          <div className="flex gap-2 mt-1">
            <Button
              onClick={() => handleAction("start", "/sim/start")}
              disabled={simStatus.running || isLoading("start")}
              size="sm" className="flex-1 gap-1 bg-primary text-primary-foreground hover:bg-primary/90 text-xs"
            >
              {isLoading("start") ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
              Start
            </Button>
            <Button
              onClick={() => handleAction("stop", "/sim/stop")}
              disabled={!simStatus.running || isLoading("stop")}
              variant="outline" size="sm" className="flex-1 gap-1 text-xs"
            >
              {isLoading("stop") ? <Loader2 className="w-3 h-3 animate-spin" /> : <Square className="w-3 h-3" />}
              Stop
            </Button>
          </div>

          <div className="border-t border-border my-2" />

          {/* Fault Injection */}
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-1 mb-1">Fault Injection</p>
          <Button
            onClick={() => handleAction("inject", "/sim/inject", { fault_type: "B" })}
            disabled={!simStatus.running || isLoading("inject")}
            variant="destructive" size="sm" className="gap-1 w-full text-xs"
          >
            {isLoading("inject") ? <Loader2 className="w-3 h-3 animate-spin" /> : <Zap className="w-3 h-3" />}
            Inject Ball Fault (B, trained)
          </Button>

          <div className="border-t border-border my-2" />

          {/* Controls */}
          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-1">Controls</p>
          <Button
            onClick={() => handleAction("reset", "/sim/reset")}
            disabled={!simStatus.running || isLoading("reset")}
            variant="secondary" size="sm" className="gap-1 w-full text-xs justify-start"
          >
            {isLoading("reset") ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
            Reset to Normal
          </Button>

          {/* Drift Toggle */}
          <div className="flex flex-col gap-1 mt-2">
            <label className={`flex items-center justify-between bg-muted/50 rounded-lg px-3 py-2 border border-amber-500/30
              ${simStatus.running ? "cursor-pointer hover:bg-muted/80" : "opacity-40 cursor-not-allowed"}`}>
              <div className="flex flex-col">
                <div className="flex items-center gap-2 text-xs font-bold text-amber-500">
                  <Waves className="w-3 h-3" />
                  <span>Simulate Drift</span>
                </div>
                <span className="text-[10px] text-muted-foreground mt-0.5">Injects IR-style drift (untrained data)</span>
              </div>
              <input
                type="checkbox"
                checked={driftActive}
                onChange={toggleDrift}
                disabled={!simStatus.running || isLoading("drift")}
                className="w-4 h-4 rounded text-amber-500 disabled:cursor-not-allowed"
              />
            </label>
          </div>

          <div className="border-t border-border my-2" />

          {/* Danger / Utility */}
          <Button
            onClick={() => handleAction("clear", "/sim/clear_data")}
            disabled={isLoading("clear")}
            variant="ghost" size="sm"
            className="gap-1 w-full text-xs justify-start text-destructive hover:text-destructive hover:bg-destructive/10"
          >
            {isLoading("clear") ? <Loader2 className="w-3 h-3 animate-spin" /> : <Trash2 className="w-3 h-3" />}
            Clear InfluxDB Data
          </Button>

          <a href="/grafana/" target="_blank" rel="noreferrer" className="w-full">
            <Button variant="ghost" size="sm" className="gap-1 w-full text-xs justify-start text-muted-foreground hover:text-foreground">
              <ExternalLink className="w-3 h-3" /> Open Grafana
            </Button>
          </a>
        </div>
      </aside>

      {/* ── MAIN CANVAS ── */}
      <main className="flex-1 relative">
        {DASHBOARDS.map(d => (
          <iframe
            key={d.id}
            src={d.url}
            className={`w-full h-full border-none absolute inset-0 transition-opacity duration-300 ${
              activeDash === d.id ? "opacity-100 z-10" : "opacity-0 z-0 pointer-events-none"
            }`}
            title={d.label}
            allow="fullscreen"
          />
        ))}
      </main>

    </div>
  )
}
