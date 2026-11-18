
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { patientApi, clearPatientSession } from '../api'
import { usePatient } from '../context/PatientContext'

export default function PatientSession() {
  const navigate = useNavigate()

  const patient = usePatient()
  const [lastSession, setLastSession] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    patientApi.lastSession().then(({ data }) => {
      if (active) setLastSession(data)
    }).catch(() => { if (active) setError('Could not load session history.') })
    return () => { active = false }
  }, [])

  function handleLogout() {
    clearPatientSession()
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
    !patient.has_prescription ? 'Awaiting clinician prescription' : patient.exercise === 'shoulder_abduction'
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

              {patient.has_prescription && steps.map((step, index) => (
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
            disabled={!patient.has_prescription}
            className="w-full rounded-2xl bg-emerald-600 px-5 py-4 font-semibold text-white transition-colors hover:bg-emerald-500 disabled:opacity-50"
          >
            Open Browser Exercise Demo
          </button>

          <p className="mt-2 text-center text-xs text-slate-500">
            Browser camera + simulated AI results
          </p>

          <p className="mt-6 text-sm text-slate-400">Real AI and desktop sessions are unavailable pending adapter verification.</p>
        </main>
      </div>
    )
}
