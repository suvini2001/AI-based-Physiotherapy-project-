
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const BASE = 'http://127.0.0.1:8000'

export default function PatientLogin() {
  const navigate = useNavigate()

  const [mode, setMode] = useState('login')
  const [name, setName] = useState('')
  const [patientId, setPatientId] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [newId, setNewId] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function saveSession(data) {
    // Development-only storage.
    // We will replace this with secure server-managed
    // sessions before real patient deployment.
    localStorage.setItem(
      'patient',
      JSON.stringify(data.patient)
    )

    sessionStorage.setItem(
      'patient_token',
      data.access_token
    )
  }

  async function handleLogin(e) {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const response = await axios.post(
        `${BASE}/patient/login-secure`,
        {
          patient_id: patientId.trim().toUpperCase(),
          password,
        }
      )

      saveSession(response.data)
      navigate('/patient/session')
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Login failed. Please try again.'
      )
    } finally {
      setLoading(false)
    }
  }

  async function handleRegister(e) {
    e.preventDefault()
    setError('')

    if (!name.trim()) {
      setError('Please enter your full name.')
      return
    }

    if (password.length < 12) {
      setError('Password must contain at least 12 characters.')
      return
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    setLoading(true)

    try {
      const response = await axios.post(
        `${BASE}/patient/register-secure`,
        {
          name: name.trim(),
          password,
        }
      )

      saveSession(response.data)
      setNewId(response.data.patient.patient_id)
      setPassword('')
      setConfirmPassword('')
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Registration failed.'
      )
    } finally {
      setLoading(false)
    }
  }

  function switchMode(nextMode) {
    setMode(nextMode)
    setError('')
    setPassword('')
    setConfirmPassword('')
  }

  // ==============================================
  // REGISTRATION SUCCESS
  // ==============================================

  if (newId) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 text-white">
        <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 p-8 text-center">

          <div className="mb-5 text-5xl text-emerald-400">
            ✓
          </div>

          <h1 className="text-2xl font-bold">
            Registration Successful!
          </h1>

          <p className="mt-3 text-sm text-slate-400">
            Your PhysioTrack patient account is ready.
            Save your Patient ID securely.
          </p>

          <div className="mt-6 rounded-xl bg-slate-800 p-5">
            <p className="text-xs text-slate-400">
              Your Patient ID
            </p>

            <p className="mt-2 break-all font-mono text-3xl font-bold text-emerald-400">
              {newId}
            </p>
          </div>

          <button
            onClick={() => navigate('/patient/session')}
            className="mt-6 w-full rounded-xl bg-blue-600 px-4 py-3 font-semibold hover:bg-blue-500"
          >
            Continue to Patient Dashboard
          </button>
        </div>
      </div>
    )
  }

  // ==============================================
  // LOGIN / REGISTRATION
  // ==============================================

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 text-white">

      <div className="w-full max-w-md">

        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-blue-600 text-2xl">
            +
          </div>

          <h1 className="text-3xl font-bold">
            PhysioTrack
          </h1>

          <p className="mt-2 text-sm text-slate-400">
            Your physiotherapy exercise portal
          </p>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-7">

          <div className="mb-6 grid grid-cols-2 gap-2 rounded-xl bg-slate-800 p-1">

            <button
              type="button"
              onClick={() => switchMode('login')}
              className={`rounded-lg py-3 text-sm font-medium ${
                mode === 'login'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400'
              }`}
            >
              Login
            </button>

            <button
              type="button"
              onClick={() => switchMode('register')}
              className={`rounded-lg py-3 text-sm font-medium ${
                mode === 'register'
                  ? 'bg-blue-600 text-white'
                  : 'text-slate-400'
              }`}
            >
              Register
            </button>

          </div>

          <h2 className="mb-5 text-xl font-semibold">
            {mode === 'login'
              ? 'Patient Login'
              : 'Create Patient Account'}
          </h2>

          <form
            onSubmit={
              mode === 'login'
                ? handleLogin
                : handleRegister
            }
            className="space-y-5"
          >

            {mode === 'register' && (
              <div>
                <label className="mb-2 block text-sm text-slate-300">
                  Full Name
                </label>

                <input
                  type="text"
                  value={name}
                  onChange={e => setName(e.target.value)}
                  required
                  maxLength={150}
                  autoComplete="name"
                  placeholder="Enter your full name"
                  className="w-full rounded-xl border border-slate-700 bg-slate-800 p-3 text-white outline-none focus:border-blue-500"
                />
              </div>
            )}

            {mode === 'login' && (
              <div>
                <label className="mb-2 block text-sm text-slate-300">
                  Patient ID
                </label>

                <input
                  type="text"
                  value={patientId}
                  onChange={e => setPatientId(e.target.value)}
                  required
                  autoComplete="username"
                  placeholder="Example: PT439699"
                  className="w-full rounded-xl border border-slate-700 bg-slate-800 p-3 text-white outline-none focus:border-blue-500"
                />
              </div>
            )}

            <div>
              <label className="mb-2 block text-sm text-slate-300">
                Password
              </label>

              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                minLength={mode === 'register' ? 12 : undefined}
                maxLength={128}
                autoComplete={
                  mode === 'login'
                    ? 'current-password'
                    : 'new-password'
                }
                placeholder="Enter password"
                className="w-full rounded-xl border border-slate-700 bg-slate-800 p-3 text-white outline-none focus:border-blue-500"
              />
            </div>

            {mode === 'register' && (
              <div>
                <label className="mb-2 block text-sm text-slate-300">
                  Confirm Password
                </label>

                <input
                  type="password"
                  value={confirmPassword}
                  onChange={e => setConfirmPassword(e.target.value)}
                  required
                  minLength={12}
                  maxLength={128}
                  autoComplete="new-password"
                  placeholder="Confirm your password"
                  className="w-full rounded-xl border border-slate-700 bg-slate-800 p-3 text-white outline-none focus:border-blue-500"
                />
              </div>
            )}

            {mode === 'register' && (
              <p className="text-xs text-slate-400">
                Use a unique password with at least
                12 characters.
              </p>
            )}

            {error && (
              <div
                role="alert"
                className="rounded-xl border border-red-800 bg-red-950 p-3 text-sm text-red-300"
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-xl bg-blue-600 py-3 font-semibold text-white hover:bg-blue-500 disabled:opacity-50"
            >
              {loading
                ? 'Please wait...'
                : mode === 'login'
                  ? 'Login Securely'
                  : 'Create Account'}
            </button>

          </form>

        </div>

        <p className="mt-5 text-center text-xs text-slate-500">
          Development environment — fictional patient data only.
        </p>
      </div>
    </div>
  )
}
