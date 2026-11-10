import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, BarChart, Bar
} from 'recharts'
import { api } from '../api'

const COLORS = { good: '#34d399', warn: '#fbbf24', bad: '#f87171' }

function qualityColor(score) {
  if (score == null) return '#64748b'
  return score >= 80 ? COLORS.good : score >= 60 ? COLORS.warn : COLORS.bad
}

function FaultBadge({ fault }) {
  const map = {
    'Incomplete flexion': 'bg-amber-950 text-amber-300 border-amber-800',
    'Too fast': 'bg-blue-950 text-blue-300 border-blue-800',
    'Keep shoulders level': 'bg-red-950 text-red-300 border-red-800',
    'Keep your back straight': 'bg-purple-950 text-purple-300 border-purple-800',
  }
  const style = map[fault] || 'bg-slate-800 text-slate-300 border-slate-700'
  return (
    <span className={`inline-block border rounded-full px-2 py-0.5 text-xs mr-1 mb-1 ${style}`}>{fault}</span>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs">
      <p className="text-slate-400 mb-1">{label}</p>
      {payload.map(p => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(1)}{p.name.includes('ROM') ? '°' : '%'}
        </p>
      ))}
    </div>
  )
}

export default function PatientDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [patient, setPatient] = useState(null)
  const [sessions, setSessions] = useState([])
  const [selectedSession, setSelectedSession] = useState(null)
  const [reps, setReps] = useState([])
  const [presc, setPresc] = useState({ exercise: 'elbow_flexion', target_reps: 10, target_rom: 100 })
  const [prescSaved, setPrescSaved] = useState(false)
  const [tab, setTab] = useState('progress')

  useEffect(() => {
    api.getPatients().then(r => {
      const p = r.data.find(x => String(x.id) === String(id))
      setPatient(p)
      if (p?.prescription) setPresc(p.prescription)
    })
    api.getSessions(id).then(r => {
      setSessions(r.data)
      if (r.data.length > 0) selectSession(r.data[0])
    })
  }, [id])

  async function selectSession(s) {
    setSelectedSession(s)
    const r = await api.getReps(s.id)
    setReps(r.data)
  }

  async function savePresc() {
    await api.setPrescription({ patient_id: Number(id), ...presc })
    setPrescSaved(true)
    setTimeout(() => setPrescSaved(false), 2000)
  }

  const chartData = [...sessions].reverse().map((s, i) => ({
    session: `S${i + 1}`,
    date: new Date(s.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }),
    'Mean ROM': s.mean_rom,
    'Quality': s.mean_quality,
  }))

  if (!patient) return <div className="p-8 text-slate-500 text-sm">Loading patient…</div>

  return (
    <div className="p-8">
      <button onClick={() => navigate('/patients')}
        className="flex items-center gap-2 text-slate-500 hover:text-slate-300 text-sm mb-6 transition-colors">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="15,18 9,12 15,6"/>
        </svg>
        Back to patients
      </button>

      <div className="flex items-start justify-between mb-8 gap-4 flex-wrap">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center text-xl font-bold text-slate-300">
            {patient.name.split(' ').map(w => w[0]).join('').slice(0,2)}
          </div>
          <div>
            <h1 className="text-2xl font-semibold text-white mb-0.5">{patient.name}</h1>
            <p className="text-slate-500 text-sm">{patient.notes || 'No clinical notes'}</p>
          </div>
        </div>
        <div className="flex gap-3 flex-shrink-0">
          <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-center min-w-20">
            <p className="text-xl font-bold text-white">{sessions.length}</p>
            <p className="text-slate-500 text-xs">sessions</p>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-center min-w-20">
            <p className="text-xl font-bold text-emerald-400">
              {sessions.length ? `${sessions[0].mean_rom?.toFixed(0) || '—'}°` : '—'}
            </p>
            <p className="text-slate-500 text-xs">latest ROM</p>
          </div>
          <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-center min-w-20">
            <p className="text-xl font-bold text-blue-400">{presc.target_reps}</p>
            <p className="text-slate-500 text-xs">target reps</p>
          </div>
        </div>
      </div>

      <div className="flex gap-1 mb-6 bg-slate-900 border border-slate-800 rounded-lg p-1 w-fit">
        {['progress', 'reps', 'prescription'].map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors capitalize ${
              tab === t ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'
            }`}>
            {t}
          </button>
        ))}
      </div>

      {tab === 'progress' && (
        <div className="space-y-6">
          {sessions.length === 0 ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-600 text-sm">
              No sessions recorded yet.
            </div>
          ) : (
            <>
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h2 className="text-sm font-medium text-white mb-5">Range of motion over sessions</h2>
                <div style={{ height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false}
                        tickFormatter={v => `${v}°`} domain={['auto', 'auto']} />
                      <Tooltip content={<CustomTooltip />} />
                      {presc.target_rom && (
                        <ReferenceLine y={presc.target_rom} stroke="#3b82f6" strokeDasharray="4 4"
                          label={{ value: 'Target', position: 'insideTopRight', fontSize: 10, fill: '#3b82f6' }} />
                      )}
                      <Line type="monotone" dataKey="Mean ROM" stroke="#34d399" strokeWidth={2}
                        dot={{ fill: '#34d399', r: 3 }} activeDot={{ r: 5 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-800">
                  <h2 className="text-sm font-medium text-white">Session history</h2>
                </div>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-slate-800">
                      {['Date', 'Exercise', 'Reps', 'Mean ROM', 'Quality'].map(h => (
                        <th key={h} className="text-left text-xs text-slate-500 font-medium px-6 py-3">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.map(s => (
                      <tr key={s.id} onClick={() => { selectSession(s); setTab('reps') }}
                        className={`cursor-pointer hover:bg-slate-800 transition-colors border-b border-slate-800 last:border-0 ${selectedSession?.id === s.id ? 'bg-slate-800/50' : ''}`}>
                        <td className="px-6 py-3 text-sm text-slate-300">
                          {new Date(s.date).toLocaleDateString('en-GB', {day:'numeric',month:'short',year:'numeric'})}
                        </td>
                        <td className="px-6 py-3 text-sm text-slate-400 capitalize">{s.exercise?.replace('_',' ')}</td>
                        <td className="px-6 py-3 text-sm text-slate-300">{s.total_reps}</td>
                        <td className="px-6 py-3 text-sm text-slate-300">{s.mean_rom?.toFixed(1)}°</td>
                        <td className="px-6 py-3">
                          {s.mean_quality != null
                            ? <span className="text-sm font-semibold" style={{ color: qualityColor(s.mean_quality) }}>{s.mean_quality.toFixed(0)}%</span>
                            : <span className="text-slate-600 text-xs">—</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {tab === 'reps' && (
        <div className="space-y-5">
          {!selectedSession ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-12 text-center text-slate-600 text-sm">
              Select a session from the Progress tab.
            </div>
          ) : (
            <>
              <div className="flex items-center gap-3">
                <p className="text-slate-400 text-sm">Session:</p>
                <select value={selectedSession.id}
                  onChange={e => selectSession(sessions.find(s => s.id === Number(e.target.value)))}
                  className="bg-slate-900 border border-slate-700 text-white text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-blue-500">
                  {sessions.map(s => (
                    <option key={s.id} value={s.id}>
                      {new Date(s.date).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'})} — {s.total_reps} reps
                    </option>
                  ))}
                </select>
              </div>

              {reps.length > 0 && (
                <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                  <h2 className="text-sm font-medium text-white mb-5">ROM per rep</h2>
                  <div style={{ height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={reps} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="rep_number" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} />
                        <YAxis tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} tickFormatter={v => `${v}°`} />
                        <Tooltip content={<CustomTooltip />} />
                        {presc.target_rom && (
                          <ReferenceLine y={presc.target_rom} stroke="#3b82f6" strokeDasharray="4 4" />
                        )}
                        <Bar dataKey="rom" name="ROM" fill="#34d399" radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-slate-800">
                  <h2 className="text-sm font-medium text-white">Rep breakdown</h2>
                </div>
                {reps.length === 0 ? (
                  <p className="text-slate-600 text-sm text-center py-10">No rep data for this session.</p>
                ) : (
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-slate-800">
                        {['Rep', 'Max angle', 'Min angle', 'ROM', 'Duration', 'Quality', 'Faults'].map(h => (
                          <th key={h} className="text-left text-xs text-slate-500 font-medium px-5 py-3">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {reps.map(r => (
                        <tr key={r.rep_number} className="border-b border-slate-800 last:border-0">
                          <td className="px-5 py-3 text-sm text-slate-400 font-mono">{r.rep_number}</td>
                          <td className="px-5 py-3 text-sm text-slate-300">{r.max_angle?.toFixed(1)}°</td>
                          <td className="px-5 py-3 text-sm text-slate-300">{r.min_angle?.toFixed(1)}°</td>
                          <td className="px-5 py-3 text-sm font-semibold"
                            style={{ color: r.rom >= (presc.target_rom || 80) ? COLORS.good : COLORS.warn }}>
                            {r.rom?.toFixed(1)}°
                          </td>
                          <td className="px-5 py-3 text-sm text-slate-400">{r.duration_s?.toFixed(1)}s</td>
                          <td className="px-5 py-3">
                            {r.quality_score != null
                              ? <span className="text-sm font-semibold" style={{ color: qualityColor(r.quality_score) }}>{r.quality_score.toFixed(0)}%</span>
                              : <span className="text-slate-600 text-xs">—</span>}
                          </td>
                          <td className="px-5 py-3">
                            {r.faults?.length > 0
                              ? r.faults.map(f => <FaultBadge key={f} fault={f} />)
                              : <span className="text-emerald-500 text-xs">Clean</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {tab === 'prescription' && (
        <div className="max-w-md">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-5">
            <h2 className="text-sm font-medium text-white">Exercise prescription</h2>
            <div>
              <label className="text-slate-400 text-xs mb-1 block">Exercise</label>
              <select value={presc.exercise} onChange={e => setPresc({...presc, exercise: e.target.value})}
                className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-blue-500">
                <option value="elbow_flexion">Elbow flexion</option>
                <option value="shoulder_abduction">Shoulder abduction</option>
              </select>
            </div>
            <div>
              <label className="text-slate-400 text-xs mb-1 block">Target reps per session</label>
              <input type="number" min={1} max={50} value={presc.target_reps}
                onChange={e => setPresc({...presc, target_reps: Number(e.target.value)})}
                className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-blue-500" />
            </div>
            <div>
              <label className="text-slate-400 text-xs mb-1 block">Target ROM (degrees)</label>
              <input type="number" min={10} max={180} value={presc.target_rom}
                onChange={e => setPresc({...presc, target_rom: Number(e.target.value)})}
                className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:border-blue-500" />
              <p className="text-slate-600 text-xs mt-1">Shown as a reference line on ROM charts</p>
            </div>
            <button onClick={savePresc}
              className={`w-full py-2.5 rounded-lg text-sm font-medium transition-colors ${
                prescSaved ? 'bg-emerald-700 text-emerald-100' : 'bg-blue-600 hover:bg-blue-500 text-white'
              }`}>
              {prescSaved ? 'Saved' : 'Save prescription'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}