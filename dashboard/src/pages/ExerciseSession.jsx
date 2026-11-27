import { useNavigate, Link } from 'react-router-dom'
import { usePatient } from '../context/PatientContext'
import { clearPatientSession } from '../api'

import { useEffect, useRef, useState } from 'react'
import { createMockAIService } from '../services/mockAIService'

export default function ExerciseSession() {
  // ===================================================
  // 1. CAMERA REFERENCES
  // ===================================================

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const requestIdRef = useRef(0)

  // Mock AI service reference.
  const mockServiceRef = useRef(null)

  // ===================================================
  // 2. PATIENT PRESCRIPTION
  // ===================================================

  const patient = usePatient()
  const navigate = useNavigate()

  const prescribedTarget = Number(patient?.target_reps)

  const targetReps =
    Number.isInteger(prescribedTarget) &&
    prescribedTarget > 0
      ? prescribedTarget
      : 10

  const exerciseName =
    !patient?.has_prescription ? 'Awaiting clinician prescription' : patient?.exercise === 'shoulder_abduction'
      ? 'Shoulder Abduction'
      : 'Elbow Flexion'

  // ===================================================
  // 3. REACT STATE
  // ===================================================

  // Camera state
  const [cameraStatus, setCameraStatus] = useState('idle')
  const [cameraError, setCameraError] = useState('')

  // Mock AI state
  const [isSimulating, setIsSimulating] = useState(false)
  const [isCompleted, setIsCompleted] = useState(false)

  const [repCount, setRepCount] = useState(0)
  const [jointAngle, setJointAngle] = useState(null)

  const [movementState, setMovementState] =
    useState('Not tracking')

  const [feedback, setFeedback] = useState(
    'AI analysis is not connected yet.'
  )

  // ===================================================
  // 4. CAMERA FUNCTIONS
  // ===================================================

  function releaseCamera() {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach(track => track.stop())

      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  async function startCamera() {
    if (
      cameraStatus !== 'idle' &&
      cameraStatus !== 'error'
    ) {
      return
    }

    const requestId = ++requestIdRef.current

    setCameraError('')
    setCameraStatus('starting')

    let stream = null

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          'Camera API unavailable. Use localhost or HTTPS.'
        )
      }

      stream =
        await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        })

      // A previous request may have been cancelled.
      if (requestId !== requestIdRef.current) {
        stream.getTracks().forEach(track => track.stop())
        return
      }

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      if (requestId !== requestIdRef.current) {
        // Stop only this request's stream.
        stream.getTracks().forEach(track => track.stop())

        if (streamRef.current === stream) {
          streamRef.current = null
        }

        return
      }

      setCameraStatus('running')
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        stream?.getTracks().forEach(track => track.stop())
        return
      }

      releaseCamera()

      setCameraError(
        err.message || 'Unable to access the camera.'
      )

      setCameraStatus('error')
    }
  }

  function stopCamera() {
    // Invalidate pending camera requests.
    requestIdRef.current += 1

    releaseCamera()
    setCameraError('')
    setCameraStatus('idle')
  }

  // ===================================================
  // 5. SIMULATION FUNCTIONS
  // ===================================================

  function startSimulation() {
    // Prevent multiple simulations running together.
    if (mockServiceRef.current || !patient?.has_prescription) return

    // Reset UI values for a new session.
    setIsCompleted(false)
    setRepCount(0)
    setJointAngle(null)
    setMovementState('Not tracking')

    setFeedback(
      'SIMULATION: No real movement analysis'
    )

    const service = createMockAIService(
      result => {
        setRepCount(result.repCount)
        setJointAngle(result.jointAngle)
        setMovementState(result.movementState)
        setFeedback(result.feedback)

        if (result.completed) {
          setIsCompleted(true)
          setIsSimulating(false)

          // The mock service already stopped its interval.
          mockServiceRef.current = null
        }
      },
      targetReps
    )

    mockServiceRef.current = service
    setIsSimulating(true)

    service.start()
  }

  function stopSimulation() {
    mockServiceRef.current?.stop()
    mockServiceRef.current = null

    setIsSimulating(false)
    setIsCompleted(false)

    setRepCount(0)
    setJointAngle(null)
    setMovementState('Not tracking')

    setFeedback(
      'AI analysis is not connected yet.'
    )
  }

  function restartSimulation() {
    stopSimulation()

    // The next click on Start Simulation
    // begins a fresh simulated session.
  }

  // ===================================================
  // 6. CLEANUP
  // ===================================================

  useEffect(() => {
    return () => {
      // Invalidate any camera request still pending.
      requestIdRef.current += 1

      // Release camera.
      streamRef.current
        ?.getTracks()
        .forEach(track => track.stop())

      streamRef.current = null

      // Stop simulation timer.
      mockServiceRef.current?.stop()
      mockServiceRef.current = null
    }
  }, [])

  // ===================================================
  // 7. DERIVED VALUES
  // ===================================================

  const progress =
    Math.min((repCount / targetReps) * 100, 100)

  const simulationStatus = isCompleted
    ? 'Completed'
    : isSimulating
      ? 'Running'
      : 'Not started'

  // ===================================================
  // 8. USER INTERFACE
  // ===================================================

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-white">

      <div className="mx-auto max-w-4xl space-y-6">

        <nav className="flex justify-between">
          <Link to="/patient/session" onClick={() => { stopCamera(); stopSimulation() }}>← Patient dashboard</Link>
          <button onClick={() => { stopCamera(); stopSimulation(); clearPatientSession(); navigate('/patient', { replace: true }) }}>Sign Out</button>
        </nav>
        {!patient?.has_prescription && <p role="status">No prescription yet. Ask your clinician to prescribe an exercise before starting.</p>}
        {/* HEADER */}
        <header>
          <p className="text-sm text-blue-400">
            PhysioTrack / Exercise Session
          </p>

          <h1 className="mt-2 text-3xl font-bold">
            {exerciseName}
          </h1>

          <p className="mt-2 text-slate-400">
            Browser camera and simulated AI demonstration
          </p>

          {patient && (
            <p className="mt-2 text-sm text-slate-500">
              Patient: {patient.name} ({patient.patient_id})
            </p>
          )}
        </header>

        {/* DEVELOPMENT WARNING */}
        <div className="rounded-xl border border-amber-700 bg-amber-950 p-4">
          <p className="font-semibold text-amber-300">
            Development / Simulation Mode
          </p>

          <p className="mt-1 text-sm text-amber-200">
            These measurements are simulated.
            They are not calculated from your camera.
            No exercise results are uploaded or saved.
          </p>
        </div>

        {/* CAMERA PREVIEW */}
        <div className="overflow-hidden rounded-xl border border-slate-800 bg-black">

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="aspect-video w-full object-contain"
          />

        </div>

        {/* CAMERA STATUS */}
        <div className="text-sm text-slate-400">
          Camera status:{' '}

          <span className="font-medium text-white">
            {cameraStatus}
          </span>
        </div>

        {/* MEASUREMENT CARDS */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">

          {/* REPETITIONS */}
          <div className="rounded-xl bg-slate-900 p-5">

            <p className="text-sm text-slate-400">
              Repetitions
            </p>

            <p className="mt-2 text-3xl font-bold text-blue-400">
              {repCount} / {targetReps}
            </p>

            <p className="mt-2 text-xs text-slate-500">
              Simulated repetition count
            </p>
          </div>

          {/* JOINT ANGLE */}
          <div className="rounded-xl bg-slate-900 p-5">

            <p className="text-sm text-slate-400">
              Joint Angle
            </p>

            <p className="mt-2 text-3xl font-bold text-emerald-400">
              {jointAngle === null
                ? '--'
                : `${jointAngle}°`}
            </p>

            <p className="mt-2 text-xs text-slate-500">
              Simulated angle measurement
            </p>
          </div>

          {/* MOVEMENT STATE */}
          <div className="rounded-xl bg-slate-900 p-5">

            <p className="text-sm text-slate-400">
              Movement State
            </p>

            <p className="mt-2 text-lg font-semibold text-purple-400">
              {movementState}
            </p>

            <p className="mt-2 text-xs text-slate-500">
              Simulated movement state
            </p>
          </div>

        </div>

        {/* PROGRESS */}
        <div className="rounded-xl bg-slate-900 p-5">

          <div className="mb-3 flex items-center justify-between">

            <h2 className="font-semibold">
              Session Progress
            </h2>

            <span className="text-sm text-slate-400">
              {Math.round(progress)}%
            </span>

          </div>

          <div
            className="h-3 overflow-hidden rounded-full bg-slate-800"
            role="progressbar"
            aria-label="Simulated repetition progress"
            aria-valuemin={0}
            aria-valuemax={targetReps}
            aria-valuenow={repCount}
          >
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>

          <p className="mt-3 text-sm text-slate-400">
            Simulation status: {simulationStatus}
          </p>

        </div>

        {/* FEEDBACK */}
        <div className="rounded-xl bg-slate-900 p-5">

          <h2 className="text-lg font-semibold">
            Movement Feedback
          </h2>

          <p className="mt-2 text-sm text-amber-400">
            {feedback}
          </p>

        </div>

        {/* COMPLETION MESSAGE */}
        {isCompleted && (

          <div
            role="status"
            className="rounded-xl border border-emerald-700 bg-emerald-950 p-5"
          >

            <h2 className="text-xl font-bold text-emerald-300">
              Simulation Complete!
            </h2>

            <p className="mt-2 text-sm text-emerald-200">
              The simulated session reached the target
              of {targetReps} repetitions.
            </p>

            <p className="mt-2 text-xs text-emerald-400">
              These are demonstration results only.
              Nothing has been saved to patient records.
            </p>

            <button
              onClick={restartSimulation}
              className="mt-4 rounded-lg bg-emerald-700 px-5 py-3 font-medium text-white hover:bg-emerald-600"
            >
              Reset Simulation
            </button>

          </div>

        )}

        {/* CAMERA ERRORS */}
        {cameraError && (

          <div
            role="alert"
            className="rounded-lg bg-red-950 p-4 text-red-400"
          >
            Camera error: {cameraError}
          </div>

        )}

        {/* CAMERA CONTROLS */}
        <div className="space-y-3">

          <h2 className="font-semibold">
            Camera Controls
          </h2>

          <div className="flex flex-wrap gap-3">

            <button
              onClick={startCamera}
              disabled={
                cameraStatus === 'running' ||
                cameraStatus === 'starting'
              }
              className="rounded-lg bg-blue-600 px-6 py-3 font-medium disabled:opacity-50"
            >
              {cameraStatus === 'starting'
                ? 'Starting...'
                : 'Start Camera'}
            </button>

            <button
              onClick={stopCamera}
              disabled={cameraStatus === 'idle'}
              className="rounded-lg bg-slate-700 px-6 py-3 font-medium disabled:opacity-50"
            >
              Stop Camera
            </button>

          </div>

        </div>

        {/* SIMULATION CONTROLS */}
        <div className="space-y-3">

          <h2 className="font-semibold">
            Mock AI Controls
          </h2>

          <div className="flex flex-wrap gap-3">

            <button
              onClick={startSimulation}
              disabled={isSimulating || isCompleted || !patient?.has_prescription}
              className="rounded-lg bg-emerald-600 px-6 py-3 font-medium disabled:opacity-50"
            >
              Start Simulation
            </button>

            <button
              onClick={stopSimulation}
              disabled={!isSimulating && !isCompleted}
              className="rounded-lg bg-amber-600 px-6 py-3 font-medium disabled:opacity-50"
            >
              {isCompleted
                ? 'Clear Results'
                : 'Stop Simulation'}
            </button>

          </div>

        </div>

        {/* FOOTER */}
        <p className="border-t border-slate-800 pt-4 text-xs text-slate-500">
          Development preview only. Camera frames remain
          inside the browser. Simulated results are
          not suitable for clinical decisions.
        </p>

      </div>

    </div>
  )
}
