import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const BASE = 'http://localhost:8000'

export default function PatientLogin() {
  const [mode, setMode] = useState('login')
  const [name, setName] = useState('')
  const [patientId, setPatientId] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [newId, setNewId] = useState(null)
  const navigate = useNavigate()

  async function handleRegister(e) {
    e.preventDefault()
    if (!name.trim()) { setError('Enter your full name'); return }
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${BASE}/patient/register`, { name: name.trim() })
      setNewId(res.data.patient_id)
      localStorage.setItem('patient', JSON.stringify(res.data))
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed')
    } finally {
      setLoading(false) }
  }

  async function handleLogin(e) {
    e.preventDefault()
    if (!patientId.trim() || !name.trim()) { setError('Enter your Patient ID and name'); return }
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${BASE}/patient/login`, {
        patient_id: patientId.trim(),
        name: name.trim(),
      })
      localStorage.setItem('patient', JSON.stringify(res.data))
      navigate('/patient/session')
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed')
    } finally {
      setLoading(false) }
  }

  if (newId) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
        <div className="w-full max-w-sm text-center">
          <div className="w-16 h-16 rounded-2xl bg-emerald-900 flex items-center justify-center mx-auto mb-6">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round">
              <polyline points="20,6 9,17 4,12"/>
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">You're registered!</h2>
          <p className="text-slate-400 text-sm mb-6">Write down your Patient ID — you'll need it every time you log in.</p>
          <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 mb-6">
            <p className="text-slate-500 text-xs mb-2">Your Patient ID</p>
            <p className="text-5xl font-bold text-emerald-400 tracking-widest font-mono">{newId}</p>
            <p className="text-slate-500 text-xs mt-3">Keep this safe — it cannot be recovered</p>
          </div>
          <button
            onClick={() => navigate('/patient/session')}
            className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium rounded-xl py-3 text-sm transition-colors"
          >
            Start my first session
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-950 flex">
      {/* Left branding */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-14"
        style={{ background: 'linear-gradient(135deg, #0f172a 0%, #0c2340 100%)' }}>
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-500 flex items-center justify-center">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <span className="text-white font-semibold text-lg">PhysioTrack</span>
        </div>
        <div>
          <h1 className="text-5xl font-bold text-white leading-tight mb-6">
            Your home<br />
            <span className="text-blue-400">physio sessions</span>
          </h1>
          <p className="text-slate-400 text-lg leading-relaxed max-w-md">
            Do your prescribed exercises at home. Your physiotherapist monitors your progress remotely — no clinic visit needed.
          </p>
          <div className="mt-10 space-y-4">
            {[
              { icon: '①', text: 'Log in with your Patient ID and name' },
              { icon: '②', text: 'Follow the on-screen exercise guide' },
              { icon: '③', text: 'Your physio reviews your session automatically' },
            ].map(s => (
              <div key={s.icon} className="flex items-center gap-4">
                <span className="text-blue-400 text-xl font-bold">{s.icon}</span>
                <span className="text-slate-300 text-sm">{s.text}</span>
              </div>
            ))}
          </div>
        </div>
        <p className="text-slate-600 text-sm">University of Peradeniya · E/21/140 · 2026</p>
      </div>

      {/* Right form */}
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

          {/* Tab switcher */}
          <div className="flex gap-1 bg-slate-900 border border-slate-800 rounded-xl p-1 mb-8">
            <button onClick={() => { setMode('login'); setError('') }}
              className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-colors ${mode === 'login' ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>
              Returning patient
            </button>
            <button onClick={() => { setMode('register'); setError('') }}
              className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-colors ${mode === 'register' ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}>
              First time
            </button>
          </div>

          {mode === 'login' ? (
            <>
              <h2 className="text-2xl font-semibold text-white mb-1">Welcome back</h2>
              <p className="text-slate-400 text-sm mb-8">Enter your Patient ID and name to continue</p>
              <form onSubmit={handleLogin} className="space-y-5">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Patient ID</label>
                  <input value={patientId}
                    onChange={e => { setPatientId(e.target.value.toUpperCase()); setError('') }}
                    placeholder="PT0001"
                    className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg px-4 py-3 text-sm placeholder-slate-600 focus:outline-none focus:border-blue-500 font-mono tracking-wider transition-colors" />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Full name</label>
                  <input value={name}
                    onChange={e => { setName(e.target.value); setError('') }}
                    placeholder="Your full name"
                    className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg px-4 py-3 text-sm placeholder-slate-600 focus:outline-none focus:border-blue-500 transition-colors" />
                </div>
                {error && <div className="bg-red-950 border border-red-800 rounded-lg px-4 py-3 text-red-300 text-sm">{error}</div>}
                <button type="submit" disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-3 text-sm transition-colors">
                  {loading ? 'Signing in…' : 'Start session'}
                </button>
              </form>
            </>
          ) : (
            <>
              <h2 className="text-2xl font-semibold text-white mb-1">Create your account</h2>
              <p className="text-slate-400 text-sm mb-8">You'll get a Patient ID to use every time you log in</p>
              <form onSubmit={handleRegister} className="space-y-5">
                <div>
                  <label className="block text-sm text-slate-400 mb-2">Full name</label>
                  <input value={name}
                    onChange={e => { setName(e.target.value); setError('') }}
                    placeholder="Your full name"
                    className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg px-4 py-3 text-sm placeholder-slate-600 focus:outline-none focus:border-blue-500 transition-colors" />
                </div>
                {error && <div className="bg-red-950 border border-red-800 rounded-lg px-4 py-3 text-red-300 text-sm">{error}</div>}
                <button type="submit" disabled={loading}
                  className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium rounded-lg px-4 py-3 text-sm transition-colors">
                  {loading ? 'Creating…' : 'Create account'}
                </button>
              </form>
            </>
          )}

          <div className="mt-8 text-center">
            <p className="text-slate-600 text-xs">Are you a physiotherapist?</p>
            <a href="/" className="text-blue-400 hover:text-blue-300 text-xs transition-colors">Go to physio portal →</a>
          </div>
        </div>
      </div>
    </div>
  )
}