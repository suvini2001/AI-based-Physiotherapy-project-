import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../context/AuthContext'

function StatCard({ label, value, sub, color = 'blue' }) {
  const colors = {
    blue: 'text-blue-400', green: 'text-emerald-400',
    amber: 'text-amber-400', slate: 'text-slate-300',
  }
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
      <p className="text-slate-500 text-xs mb-3">{label}</p>
      <p className={`text-3xl font-bold ${colors[color]} mb-1`}>{value}</p>
      {sub && <p className="text-slate-500 text-xs">{sub}</p>}
    </div>
  )
}

export default function Dashboard() {
  const { physio } = useAuth()
  const [patients, setPatients] = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    api.getPatients()
      .then(r => setPatients(r.data))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const totalSessions = patients.reduce((s, p) => s + (p.total_sessions || 0), 0)
  const withRecent = patients.filter(p => p.last_session).length
  const romsWithData = patients.filter(p => p.last_rom)
  const avgRom = romsWithData.length
    ? (romsWithData.reduce((s, p) => s + p.last_rom, 0) / romsWithData.length).toFixed(1)
    : '—'

  const hour = new Date().getHours()
  const greeting = hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening'

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-semibold text-white mb-1">
          {greeting}, {physio?.name?.split(' ')[0]}
        </h1>
        <p className="text-slate-400 text-sm">{physio?.specialty}</p>
      </div>

      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total patients" value={patients.length} color="blue" />
        <StatCard label="Total sessions" value={totalSessions} color="slate" />
        <StatCard label="Active this week" value={withRecent} sub="had a session recently" color="green" />
        <StatCard label="Mean ROM (last session)" value={avgRom === '—' ? '—' : `${avgRom}°`} color="amber" />
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <h2 className="text-sm font-medium text-white">Patient roster</h2>
          <button onClick={() => navigate('/patients')}
            className="text-xs text-blue-400 hover:text-blue-300 transition-colors">
            View all →
          </button>
        </div>
        {loading ? (
          <div className="px-6 py-12 text-center text-slate-600 text-sm">Loading…</div>
        ) : patients.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <p className="text-slate-500 text-sm mb-4">No patients yet.</p>
            <button onClick={() => api.seed().then(() => window.location.reload())}
              className="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-4 py-2 rounded-lg transition-colors">
              Load demo patients
            </button>
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="border-b border-slate-800">
                {['Name', 'Exercise', 'Last session', 'ROM', 'Sessions'].map(h => (
                  <th key={h} className="text-left text-xs text-slate-500 font-medium px-6 py-3">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id} onClick={() => navigate(`/patients/${p.id}`)}
                  className="cursor-pointer hover:bg-slate-800 transition-colors border-b border-slate-800 last:border-0">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center text-xs text-slate-300 font-medium flex-shrink-0">
                        {p.name.split(' ').map(w => w[0]).join('').slice(0,2)}
                      </div>
                      <span className="text-sm text-white font-medium">{p.name}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-400 capitalize">
                    {p.prescription?.exercise?.replace('_', ' ') || '—'}
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-400">
                    {p.last_session ? new Date(p.last_session).toLocaleDateString('en-GB', {
                      day: 'numeric', month: 'short', year: 'numeric'
                    }) : '—'}
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-300">
                    {p.last_rom ? `${p.last_rom.toFixed(1)}°` : '—'}
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-400">{p.total_sessions}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}