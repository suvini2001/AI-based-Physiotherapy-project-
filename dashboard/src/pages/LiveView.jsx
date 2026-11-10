import { useEffect, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'
import { api } from '../api'

const MAX_POINTS = 150
const WS_URL = 'ws://localhost:8000/ws/live'

const STATE_COLORS = {
  EXTENDED: '#64748b', FLEXING: '#3b82f6',
  FLEXED: '#8b5cf6', EXTENDING: '#10b981',
  RESTING: '#64748b', RAISING: '#3b82f6',
  RAISED: '#8b5cf6', LOWERING: '#10b981',
}

function StatPill({ label, value, color = 'text-white' }) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl px-5 py-4 text-center min-w-28">
      <p className={`text-2xl font-bold ${color} mb-0.5`}>{value}</p>
      <p className="text-slate-500 text-xs">{label}</p>
    </div>
  )
}

export default function LiveView() {
  const [connected, setConnected] = useState(false)
  const [patientId, setPatientId] = useState('1')
  const [patients, setPatients] = useState([])
  const [data, setData] = useState([])
  const [latest, setLatest] = useState(null)
  const wsRef = useRef(null)

  useEffect(() => {
    api.getPatients().then(r => setPatients(r.data)).catch(() => {})
    return () => wsRef.current?.close()
  }, [])

  function connect() {
    if (wsRef.current) wsRef.current.close()
    const ws = new WebSocket(`${WS_URL}/${patientId}`)
    wsRef.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (e) => {
      const frame = JSON.parse(e.data)
      setLatest(frame)
      setData(prev => {
        const next = [...prev, { t: frame.timestamp?.toFixed(1), angle: frame.joint_angle }]
        return next.length > MAX_POINTS ? next.slice(-MAX_POINTS) : next
      })
    }
  }

  function disconnect() {
    wsRef.current?.close()
    setConnected(false)
    setData([])
    setLatest(null)
  }

  const stateColor = STATE_COLORS[latest?.state] || '#64748b'

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-white mb-1">Live session</h1>
        <p className="text-slate-400 text-sm">Real-time kinematic feed from patient device</p>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 mb-6 flex items-end gap-4 flex-wrap">
        <div>
          <label className="text-slate-400 text-xs mb-1 block">Patient</label>
          <select value={patientId} onChange={e => setPatientId(e.target.value)} disabled={connected}
            className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-blue-500 disabled:opacity-50 min-w-44">
            {patients.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
            {patients.length === 0 && <option value="1">Patient 1</option>}
          </select>
        </div>
        <div className="flex gap-3">
          {!connected ? (
            <button onClick={connect}
              className="bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium px-5 py-2 rounded-lg transition-colors flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-300 animate-pulse"/>
              Connect
            </button>
          ) : (
            <button onClick={disconnect}
              className="bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-medium px-5 py-2 rounded-lg transition-colors flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400"/>
              Disconnect
            </button>
          )}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-slate-600'}`}/>
          <span className={`text-xs ${connected ? 'text-emerald-400' : 'text-slate-500'}`}>
            {connected ? 'Live' : 'Disconnected'}
          </span>
        </div>
      </div>

      <div className="flex gap-4 mb-6 flex-wrap">
        <StatPill label="Joint angle" value={latest?.joint_angle != null ? `${latest.joint_angle.toFixed(0)}°` : '—'} color="text-blue-400" />
        <StatPill label="State" value={latest?.state || '—'} />
        <StatPill label="Reps" value={latest?.rep_count ?? '—'} color="text-emerald-400" />
        <div className="bg-slate-800 border border-slate-700 rounded-xl px-5 py-4 text-center min-w-28">
          <p className="text-sm font-semibold mb-1 mt-1" style={{ color: stateColor }}>
            {latest?.fault || 'None'}
          </p>
          <p className="text-slate-500 text-xs">Last fault</p>
        </div>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-sm font-medium text-white">Joint angle — live</h2>
          {connected && (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/>
              Streaming ~10 Hz
            </span>
          )}
        </div>

        {data.length === 0 ? (
          <div className="flex items-center justify-center h-56 text-slate-600 text-sm flex-col gap-3">
            {connected
              ? <><span className="text-2xl animate-spin">⟳</span><span>Waiting for frames…</span></>
              : <><p>Connect to a live session to see the angle chart.</p>
                  <p className="text-slate-700 text-xs">Make sure demo_cv.py is running on the patient device.</p></>
            }
          </div>
        ) : (
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#64748b' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false}
                  tickFormatter={v => `${v}°`} domain={[0, 180]} />
                <Tooltip formatter={(v) => [`${v?.toFixed(1)}°`, 'Angle']}
                  contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: '#64748b' }} itemStyle={{ color: '#38bdf8' }} />
                <ReferenceLine y={145} stroke="#3b82f6" strokeDasharray="4 4" strokeOpacity={0.5}
                  label={{ value: 'Extended', position: 'insideTopRight', fontSize: 10, fill: '#3b82f6' }} />
                <ReferenceLine y={55} stroke="#8b5cf6" strokeDasharray="4 4" strokeOpacity={0.5}
                  label={{ value: 'Flexed', position: 'insideTopRight', fontSize: 10, fill: '#8b5cf6' }} />
                <Line type="monotone" dataKey="angle" stroke="#38bdf8" strokeWidth={2}
                  dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        <div className="flex gap-4 mt-4 flex-wrap">
          {Object.entries(STATE_COLORS).slice(0, 4).map(([state, color]) => (
            <div key={state} className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: color }}/>
              <span className="text-slate-500 text-xs">{state}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}