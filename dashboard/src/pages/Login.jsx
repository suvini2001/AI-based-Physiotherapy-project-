import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { api } from '../api'

export default function Login() {
  const [id, setId] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { login } = useAuth()
  const navigate = useNavigate()

  async function handleSubmit(e) {
    e.preventDefault()
    if (!id.trim() || !password) { setError('Enter your ID and password'); return }
    setError('')
    setLoading(true)
    try {
      const res = await api.login(id.trim(), password)
      login(res.data.access_token, res.data.physio)
      navigate('/', { replace: true })
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed — check your ID and password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex">
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-14"
        style={{ background: 'linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)' }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-500 flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <span className="text-white font-semibold text-lg tracking-tight">PhysioTrack</span>
        </div>
        <div>
          <h1 className="text-5xl font-bold text-white leading-tight mb-6">
            Remote physiotherapy<br />
            <span className="text-blue-400">monitoring platform</span>
          </h1>
          <p className="text-slate-400 text-lg leading-relaxed max-w-md">
            Monitor patient exercise sessions in real time, track ROM progress over weeks,
            and review flagged movements — without the patient ever leaving home.
          </p>
          <div className="mt-12 grid grid-cols-3 gap-6">
            {[
              { value: '9.96°', label: 'MAE accuracy' },
              { value: 'Real-time', label: 'Session streaming' },
              { value: 'No video', label: 'Privacy preserved' },
            ].map(stat => (
              <div key={stat.label}>
                <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                <div className="text-slate-500 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
        <p className="text-slate-600 text-sm">University of Peradeniya · E/21/140 · 2026</p>
      </div>

      <div className="flex-1 flex items-center justify-center p-8 bg-slate-950">
        <div className="w-full max-w-md">
          <div className="flex items-center gap-3 mb-10 lg:hidden">
            <div className="w-9 h-9 rounded-lg bg-blue-500 flex items-center justify-center">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            </div>
            <span className="text-white font-semibold text-lg">PhysioTrack</span>
          </div>

          <h2 className="text-2xl font-semibold text-white mb-1">Sign in</h2>
          <p className="text-slate-400 text-sm mb-8">Use your physiotherapist ID and password</p>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm text-slate-400 mb-2">Physiotherapist ID</label>
              <input
                type="text"
                value={id}
                onChange={e => { setId(e.target.value.toUpperCase()); setError('') }}
                placeholder="DR001"
                className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg px-4 py-3 text-sm placeholder-slate-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors font-mono tracking-wider"
                autoComplete="username"
                autoFocus
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-2">Password</label>
              <input
                type="password"
                value={password}
                onChange={e => { setPassword(e.target.value); setError('') }}
                placeholder="••••••••"
                className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg px-4 py-3 text-sm placeholder-slate-600 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-colors"
                autoComplete="current-password"
              />
            </div>
            {error && (
              <div className="bg-red-950 border border-red-800 rounded-lg px-4 py-3 text-red-300 text-sm">
                {error}
              </div>
            )}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 disabled:text-blue-500 text-white font-medium rounded-lg px-4 py-3 text-sm transition-colors mt-2"
            >
              {loading ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <div className="mt-8 p-4 bg-slate-900 rounded-lg border border-slate-800">
            <p className="text-slate-500 text-xs mb-2 font-medium">Demo credentials</p>
            <p className="text-slate-400 text-xs font-mono">ID: DR001 · Password: physio2026</p>
            <p className="text-slate-400 text-xs font-mono">ID: DR002 · Password: physio2026</p>
          </div>
        </div>
      </div>
    </div>
  )
}