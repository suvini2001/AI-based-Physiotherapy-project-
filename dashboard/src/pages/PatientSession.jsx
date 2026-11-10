import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'

export default function PatientSession() {
  const [patient, setPatient] = useState(null)
  const [lastSession, setLastSession] = useState(null)
  const [screen, setScreen] = useState('ready')
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState('')
  const pollRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    const stored = localStorage.getItem('patient')
    if (!stored) { navigate('/patient'); return }
    const p = JSON.parse(stored)
    setPatient(p)
    api.sessionStatus(p.id)
      .then(r => { if (r.data.running) setScreen('active') })
      .catch(() => {})
    fetch(`http://localhost:8000/patient/${JSON.parse(stored).id}/last-session`)
      .then(r => r.json()).then(setLastSession).catch(() => {})
    return () => clearInterval(pollRef.current)
  }, [])

  function startPolling(patientId) {
    clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const r = await api.sessionStatus(patientId)
        if (!r.data.running) {
          clearInterval(pollRef.current)
          setScreen('done')
        }
      } catch {
        clearInterval(pollRef.current)
      }
    }, 2000)
  }

  async function handleStart() {
    setLaunching(true)
    setError('')
    try {
      await api.launchSession(patient.id, patient.exercise)
      setScreen('active')
      startPolling(patient.id)
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not launch session. Make sure the backend can find demo_cv.py.')
    } finally {
      setLaunching(false)
    }
  }

  async function handleStop() {
    clearInterval(pollRef.current)
    await api.stopSession(patient.id).catch(() => {})
    setScreen('done')
  }

  function handleLogout() {
    clearInterval(pollRef.current)
    api.stopSession(patient?.id).catch(() => {})
    localStorage.removeItem('patient')
    navigate('/patient')
  }

  if (!patient) return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center">
      <p className="text-slate-500 text-sm">Loading…</p>
    </div>
  )

  const exerciseLabel = patient.exercise === 'shoulder_abduction'
    ? 'Shoulder abduction' : 'Elbow flexion'

  const steps = patient.exercise === 'shoulder_abduction' ? [
    'Stand straight with your arm at your side.',
    'Slowly raise your arm out to the side until it reaches shoulder height.',
    'Hold for a moment, then slowly lower it back down.',
    'That is one rep. Repeat smoothly.',
  ] : [
    'Sit or stand with your arm relaxed at your side.',
    'Slowly bend your elbow, bringing your hand up toward your shoulder.',
    'Lower your arm back down slowly and fully.',
    'That is one rep. Repeat smoothly.',
  ]

  const TopBar = () => (
    <div className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <span className="font-semibold text-sm text-white">PhysioTrack</span>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right hidden sm:block">
          <p className="text-xs font-medium text-white">{patient.name}</p>
          <p className="text-xs text-slate-500 font-mono">{patient.patient_id}</p>
        </div>
        <button onClick={handleLogout}
          className="text-slate-500 hover:text-slate-300 text-xs border border-slate-700 px-3 py-1.5 rounded-lg transition-colors">
          Sign out
        </button>
      </div>
    </div>
  )

  // ── Ready screen ───────────────────────────────────────────────────
  if (screen === 'ready') return (
    <div className="min-h-screen bg-slate-950 text-white">
      <TopBar />
      <div className="max-w-2xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-3xl font-bold text-white mb-2">
            Hello, {patient.name.split(' ')[0]}
          </h1>
          <p className="text-slate-400">Ready for your exercise session?</p>
        </div>

        {lastSession?.date && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 mb-6 flex items-center gap-5">
            <div className="w-10 h-10 rounded-xl bg-emerald-900 flex items-center justify-center flex-shrink-0">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round">
                <polyline points="20,6 9,17 4,12"/>
              </svg>
            </div>
            <div>
              <p className="text-xs text-slate-500 mb-1">Last session</p>
              <p className="text-white text-sm font-medium">
                {lastSession.total_reps} reps
                {lastSession.mean_rom ? ` · ${lastSession.mean_rom.toFixed(0)}° mean ROM` : ''}
              </p>
              <p className="text-slate-600 text-xs">
                {new Date(lastSession.date).toLocaleDateString('en-GB', {
                  weekday: 'long', day: 'numeric', month: 'long'
                })}
              </p>
            </div>
          </div>
        )}

        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-6">
          <p className="text-xs text-slate-500 mb-4">Today's prescription</p>
          <div className="flex items-center gap-4 mb-5">
            <div className="w-14 h-14 rounded-2xl bg-blue-900 flex items-center justify-center flex-shrink-0">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeLinecap="round">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
              </svg>
            </div>
            <div>
              <p className="text-white font-semibold text-lg">{exerciseLabel}</p>
              <p className="text-slate-400 text-sm">Target: {patient.target_reps ?? '—'} reps</p>
            </div>
          </div>
          <div className="bg-slate-800 rounded-xl p-4 space-y-2">
            <p className="text-slate-400 text-xs font-medium mb-3">How to do this exercise</p>
            {steps.map((s, i) => (
              <div key={i} className="flex gap-3 text-sm text-slate-300">
                <span className="text-blue-400 font-mono flex-shrink-0">{i + 1}.</span>
                <span>{s}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-amber-950 border border-amber-800 rounded-2xl p-5 mb-8">
          <div className="flex gap-3">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round" className="flex-shrink-0 mt-0.5">
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            <div>
              <p className="text-amber-300 text-sm font-medium mb-1">What will happen</p>
              <p className="text-amber-400 text-xs leading-relaxed">
                Clicking the button below will automatically open the camera window on this computer.
                Make sure your webcam can see your full arm clearly.
                Press <span className="font-mono bg-amber-900 px-1 rounded">Q</span> in the camera window when you finish your reps.
                Stop immediately if you feel pain.
              </p>
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-950 border border-red-800 rounded-xl p-4 mb-6 text-red-300 text-sm">
            {error}
          </div>
        )}

        <button
          onClick={handleStart}
          disabled={launching}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold rounded-2xl py-4 text-base transition-colors"
        >
          {launching ? 'Opening camera…' : "I'm ready — start session"}
        </button>
      </div>
    </div>
  )

  // ── Active session screen ──────────────────────────────────────────
  if (screen === 'active') return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col">
      <TopBar />
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-12 max-w-lg mx-auto w-full">
        <div className="relative mb-8">
          <div className="w-24 h-24 rounded-3xl bg-blue-900 flex items-center justify-center">
            <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="1.5" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <span className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
            <span className="w-3 h-3 rounded-full bg-emerald-400 animate-ping absolute"/>
            <span className="w-2 h-2 rounded-full bg-white"/>
          </span>
        </div>

        <h2 className="text-2xl font-bold text-white mb-2 text-center">Session in progress</h2>
        <p className="text-slate-400 text-sm text-center mb-2">
          The camera window is open on your screen.
        </p>
        <p className="text-slate-500 text-xs text-center mb-10">
          Do your {patient.target_reps} reps, then press{' '}
          <span className="font-mono bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">Q</span>{' '}
          in the camera window to finish.
        </p>

        <div className="w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-6">
          <div className="flex justify-between items-center mb-3">
            <span className="text-slate-400 text-sm">Exercise</span>
            <span className="text-white text-sm font-medium">{exerciseLabel}</span>
          </div>
          <div className="border-t border-slate-800 pt-3 flex justify-between items-center mb-3">
            <span className="text-slate-400 text-sm">Target reps</span>
            <span className="text-white font-bold text-lg">{patient.target_reps ?? '—'}</span>
          </div>
          <div className="border-t border-slate-800 pt-3 flex justify-between items-center">
            <span className="text-slate-400 text-sm">Status</span>
            <span className="flex items-center gap-2 text-emerald-400 text-sm">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"/>
              Tracking
            </span>
          </div>
        </div>

        <p className="text-slate-600 text-xs text-center mb-6">
          This page will update automatically when you finish. Or press the button below.
        </p>

        <button
          onClick={handleStop}
          className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-2xl py-4 text-base transition-colors mb-3"
        >
          I've finished my reps
        </button>
        <button
          onClick={() => { handleStop(); }}
          className="w-full border border-slate-700 hover:border-slate-600 text-slate-400 hover:text-slate-300 font-medium rounded-2xl py-3 text-sm transition-colors"
        >
          Cancel session
        </button>
      </div>
    </div>
  )

  // ── Done screen ────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col">
      <TopBar />
      <div className="flex-1 flex flex-col items-center justify-center px-6">
        <div className="max-w-sm w-full text-center">
          <div className="w-20 h-20 rounded-3xl bg-emerald-900 flex items-center justify-center mx-auto mb-8">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2" strokeLinecap="round">
              <polyline points="20,6 9,17 4,12"/>
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-white mb-3">Session complete!</h2>
          <p className="text-slate-400 mb-2">Great work. Your physiotherapist will review your data.</p>
          <p className="text-slate-600 text-sm mb-10">Your session was saved automatically.</p>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 mb-8 text-left">
            <div className="flex items-center justify-between mb-3">
              <span className="text-slate-400 text-sm">Patient ID</span>
              <span className="text-white font-mono font-semibold">{patient.patient_id}</span>
            </div>
            <div className="border-t border-slate-800 pt-3 flex items-center justify-between">
              <span className="text-slate-400 text-sm">Exercise</span>
              <span className="text-white text-sm">{exerciseLabel}</span>
            </div>
          </div>

          <div className="flex gap-3">
            <button onClick={() => setScreen('ready')}
              className="flex-1 bg-slate-800 hover:bg-slate-700 text-white font-medium rounded-xl py-3 text-sm transition-colors">
              Do another session
            </button>
            <button onClick={handleLogout}
              className="flex-1 border border-slate-700 hover:border-slate-600 text-slate-300 font-medium rounded-xl py-3 text-sm transition-colors">
              Sign out
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}