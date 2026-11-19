
import { useEffect, useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'

export default function PatientSession() {
  const navigate = useNavigate()

  // ============================================
  // STATE
  // ============================================

  const [patient, setPatient] = useState(null)
  const [lastSession, setLastSession] = useState(null)
  const [screen, setScreen] = useState('ready')
  const [launching, setLaunching] = useState(false)
  const [error, setError] = useState('')

  const pollRef = useRef(null)

  // ============================================
  // LOAD PATIENT
  // ============================================

  useEffect(() => {
    const stored = localStorage.getItem('patient')
    const token = sessionStorage.getItem('patient_token')

    if (!stored || !token) {
      localStorage.removeItem('patient')
      sessionStorage.removeItem('patient_token')
      navigate('/patient', { replace: true })
      return
    }

    let currentPatient

    try {
      currentPatient = JSON.parse(stored)

      if (!currentPatient?.id || !currentPatient?.patient_id) {
        throw new Error('Invalid patient information')
      }
    } catch {
      localStorage.removeItem('patient')
      sessionStorage.removeItem('patient_token')
      navigate('/patient', { replace: true })
      return
    }

    setPatient(currentPatient)

    // Existing desktop session status
    api.sessionStatus(currentPatient.id)
      .then(response => {
        if (response.data.running) {
          setScreen('active')
        }
      })
      .catch(() => {})

    // Existing last-session API
    fetch(
      `http://localhost:8000/patient/${currentPatient.id}/last-session`
    )
      .then(response => {
        if (!response.ok) {
          throw new Error('Could not load last session')
        }
        return response.json()
      })
      .then(setLastSession)
      .catch(() => {})

    return () => clearInterval(pollRef.current)
  }, [navigate])

  // ============================================
  // SESSION POLLING
  // ============================================

  function startPolling(patientId) {
    clearInterval(pollRef.current)

    pollRef.current = setInterval(async () => {
      try {
        const response = await api.sessionStatus(patientId)

        if (!response.data.running) {
          clearInterval(pollRef.current)
          setScreen('done')
        }
      } catch {
        clearInterval(pollRef.current)
      }
    }, 2000)
  }

  // ============================================
  // DESKTOP AI SESSION
  // ============================================

  async function handleStart() {
    if (!patient) return

    setLaunching(true)
    setError('')

    try {
      await api.launchSession(
        patient.id,
        patient.exercise
      )

      setScreen('active')
      startPolling(patient.id)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Could not launch the desktop AI session.'
      )
    } finally {
      setLaunching(false)
    }
  }

  async function handleStop() {
    clearInterval(pollRef.current)

    if (patient) {
      await api.stopSession(patient.id).catch(() => {})
    }

    setScreen('done')
  }

  // ============================================
  // SECURE LOGIN SESSION CLEANUP
  // ============================================

  function handleLogout() {
    clearInterval(pollRef.current)

    // Stop an active desktop session if necessary.
    if (screen === 'active' && patient) {
      api.stopSession(patient.id).catch(() => {})
    }

    // Remove saved patient information.
    localStorage.removeItem('patient')

    // Remove the secure-login access token.
    sessionStorage.removeItem('patient_token')

    // Return to login.
    navigate('/patient', { replace: true })
  }

  // ============================================
  // BROWSER EXERCISE DEMO
  // ============================================

  function handleOpenBrowserDemo() {
    navigate('/patient/exercise-demo')
  }

  // ============================================
  // LOADING SCREEN
  // ============================================

  if (!patient) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950">
        <p className="text-sm text-slate-400">
          Loading patient information...
        </p>
      </div>
    )
  }

  // ============================================
  // EXERCISE INFORMATION
  // ============================================

  const exerciseLabel =
    patient.exercise === 'shoulder_abduction'
      ? 'Shoulder Abduction'
      : 'Elbow Flexion'

  const steps =
    patient.exercise === 'shoulder_abduction'
      ? [
          'Stand straight with your arm at your side.',
          'Slowly raise your arm out to shoulder height.',
          'Hold for a moment, then lower it slowly.',
          'Repeat smoothly for the prescribed repetitions.',
        ]
      : [
          'Sit or stand with your arm relaxed at your side.',
          'Slowly bend your elbow toward your shoulder.',
          'Lower your arm slowly and fully.',
          'Repeat smoothly for the prescribed repetitions.',
        ]

  // ============================================
  // TOP NAVIGATION
  // ============================================

  const topBar = (
    <header className="flex items-center justify-between border-b border-slate-800 px-6 py-4">

      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 font-bold text-white">
          +
        </div>

        <span className="font-semibold text-white">
          PhysioTrack
        </span>
      </div>

      <div className="flex items-center gap-4">
        <div className="hidden text-right sm:block">
          <p className="text-sm font-medium text-white">
            {patient.name}
          </p>

          <p className="font-mono text-xs text-slate-500">
            {patient.patient_id}
          </p>
        </div>

        <button
          type="button"
          onClick={handleLogout}
          className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 hover:bg-slate-800"
        >
          Sign Out
        </button>
      </div>

    </header>
  )

  // ============================================
  // READY SCREEN
  // ============================================

  if (screen === 'ready') {
    return (
      <div className="min-h-screen bg-slate-950 text-white">
        {topBar}

        <main className="mx-auto max-w-2xl px-6 py-12">

          <div className="mb-8">
            <h1 className="text-3xl font-bold">
              Hello, {patient.name.split(' ')[0]}
            </h1>

            <p className="mt-2 text-slate-400">
              Ready for your exercise session?
            </p>
          </div>

          {/* LAST SESSION */}

          {lastSession?.date && (
            <div className="mb-6 rounded-2xl border border-slate-800 bg-slate-900 p-5">

              <p className="mb-2 text-xs text-slate-500">
                Last Session
              </p>

              <p className="font-medium text-white">
                {lastSession.total_reps} repetitions
              </p>

              {lastSession.mean_rom != null && (
                <p className="mt-1 text-sm text-slate-400">
                  Mean ROM: {Number(lastSession.mean_rom).toFixed(0)}°
                </p>
              )}

              <p className="mt-2 text-xs text-slate-500">
                {new Date(lastSession.date).toLocaleDateString(
                  'en-GB',
                  {
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                  }
                )}
              </p>
            </div>
          )}

          {/* PRESCRIPTION */}

          <section className="mb-6 rounded-2xl border border-slate-800 bg-slate-900 p-6">

            <p className="mb-4 text-xs text-slate-400">
              Today's Prescription
            </p>

            <div className="mb-5 flex items-center gap-4">

              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-900 text-2xl text-blue-300">
                +
              </div>

              <div>
                <h2 className="text-lg font-semibold">
                  {exerciseLabel}
                </h2>

                <p className="text-sm text-slate-400">
                  Target: {patient.target_reps ?? '—'} repetitions
                </p>
              </div>

            </div>

            {/* INSTRUCTIONS */}

            <div className="space-y-3 rounded-xl bg-slate-800 p-4">

              <h3 className="mb-3 text-sm font-medium text-slate-300">
                Exercise Instructions
              </h3>

              {steps.map((step, index) => (
                <div
                  key={index}
                  className="flex gap-3 text-sm text-slate-300"
                >
                  <span className="font-mono text-blue-400">
                    {index + 1}.
                  </span>

                  <p>{step}</p>
                </div>
              ))}

            </div>
          </section>

          {/* DEVELOPMENT NOTICE */}

          <div className="mb-6 rounded-2xl border border-amber-800 bg-amber-950 p-5">

            <h3 className="mb-2 font-medium text-amber-300">
              Development Mode
            </h3>

            <p className="text-sm leading-relaxed text-amber-200">
              The browser exercise demo uses your device's
              camera and simulated AI measurements.
              It does not analyze your movements or save
              clinical results.
            </p>

            <p className="mt-3 text-xs text-amber-400">
              Stop exercising if you experience pain.
            </p>

          </div>

          {/* ERRORS */}

          {error && (
            <div
              role="alert"
              className="mb-5 rounded-xl border border-red-800 bg-red-950 p-4 text-sm text-red-300"
            >
              {error}
            </div>
          )}

          {/* BROWSER EXERCISE BUTTON */}

          <button
            type="button"
            onClick={handleOpenBrowserDemo}
            className="w-full rounded-2xl bg-emerald-600 px-5 py-4 font-semibold text-white transition-colors hover:bg-emerald-500"
          >
            Open Browser Exercise Demo
          </button>

          <p className="mt-2 text-center text-xs text-slate-500">
            Browser camera + simulated AI results
          </p>

          {/* EXISTING DESKTOP AI BUTTON */}

          <div className="mt-8 border-t border-slate-800 pt-6">

            <p className="mb-3 text-sm text-slate-400">
              Legacy desktop AI session
            </p>

            <button
              type="button"
              onClick={handleStart}
              disabled={launching}
              className="w-full rounded-2xl border border-slate-700 bg-slate-800 px-5 py-4 font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
            >
              {launching
                ? 'Launching Desktop AI...'
                : 'Start Desktop AI Session'}
            </button>

            <p className="mt-2 text-xs text-slate-500">
              This existing function launches the Python
              AI application on the backend computer.
              It is separate from the browser demo and
              may not work with the current macOS setup.
            </p>

          </div>

        </main>
      </div>
    )
  }

  // ============================================
  // ACTIVE DESKTOP SESSION
  // ============================================

  if (screen === 'active') {
    return (
      <div className="min-h-screen bg-slate-950 text-white">
        {topBar}

        <main className="mx-auto max-w-xl px-6 py-16 text-center">

          <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-blue-900 text-4xl text-blue-300">
            +
          </div>

          <h1 className="text-3xl font-bold">
            Session in Progress
          </h1>

          <p className="mt-3 text-slate-400">
            The existing Python AI session is running
            on the backend computer.
          </p>

          <div className="mt-8 space-y-4 rounded-2xl border border-slate-800 bg-slate-900 p-6 text-left">

            <div className="flex justify-between">
              <span className="text-slate-400">
                Exercise
              </span>

              <span>{exerciseLabel}</span>
            </div>

            <div className="flex justify-between border-t border-slate-800 pt-4">
              <span className="text-slate-400">
                Target Repetitions
              </span>

              <span className="font-bold">
                {patient.target_reps ?? '—'}
              </span>
            </div>

            <div className="flex justify-between border-t border-slate-800 pt-4">
              <span className="text-slate-400">
                Status
              </span>

              <span className="text-emerald-400">
                Running
              </span>
            </div>

          </div>

          <button
            type="button"
            onClick={handleStop}
            className="mt-8 w-full rounded-xl bg-emerald-600 py-4 font-semibold text-white hover:bg-emerald-500"
          >
            I've Finished My Reps
          </button>

          <button
            type="button"
            onClick={handleStop}
            className="mt-3 w-full rounded-xl border border-slate-700 py-3 text-slate-300 hover:bg-slate-800"
          >
            Cancel Session
          </button>

        </main>
      </div>
    )
  }

  // ============================================
  // FINISHED DESKTOP SESSION
  // ============================================

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      {topBar}

      <main className="mx-auto max-w-md px-6 py-16 text-center">

        <div className="mx-auto mb-6 flex h-20 w-20 items-center justify-center rounded-3xl bg-emerald-900 text-4xl text-emerald-300">
          ✓
        </div>

        <h1 className="text-3xl font-bold">
          Session Finished
        </h1>

        <p className="mt-3 text-slate-400">
          Your desktop session has ended.
        </p>

        <p className="mt-2 text-sm text-slate-500">
          Session saving has not been verified.
          Check your history before assuming results
          were stored.
        </p>

        <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-900 p-6 text-left">

          <div className="flex justify-between">
            <span className="text-slate-400">
              Patient ID
            </span>

            <span className="font-mono">
              {patient.patient_id}
            </span>
          </div>

          <div className="mt-4 flex justify-between border-t border-slate-800 pt-4">
            <span className="text-slate-400">
              Exercise
            </span>

            <span>{exerciseLabel}</span>
          </div>

        </div>

        <button
          type="button"
          onClick={() => setScreen('ready')}
          className="mt-8 w-full rounded-xl bg-blue-600 py-3 font-semibold hover:bg-blue-500"
        >
          Return to Dashboard
        </button>

        <button
          type="button"
          onClick={handleLogout}
          className="mt-3 w-full rounded-xl border border-slate-700 py-3 text-slate-300 hover:bg-slate-800"
        >
          Sign Out
        </button>

      </main>
    </div>
  )
}
