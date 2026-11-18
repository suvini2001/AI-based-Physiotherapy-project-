
import { useEffect, useRef, useState } from 'react'
import { createMockAIService } from '../services/mockAIService'

export default function ExerciseSession() {
  // ==================================================
  // 1. CAMERA REFERENCES
  // ==================================================

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const requestIdRef = useRef(0)

  // Keeps track of the simulated AI service.
  const mockServiceRef = useRef(null)

  // ==================================================
  // 2. REACT STATE
  // ==================================================

  // Camera state
  const [cameraStatus, setCameraStatus] = useState('idle')
  const [error, setError] = useState('')

  // Simulated AI state
  const [isSimulating, setIsSimulating] = useState(false)
  const [repCount, setRepCount] = useState(0)
  const [jointAngle, setJointAngle] = useState(null)
  const [movementState, setMovementState] =
    useState('Not tracking')

  const [feedback, setFeedback] =
    useState('AI analysis is not connected yet.')

//   // Temporary target until we integrate prescriptions.
//   const targetReps = 10

  const [patient] = useState(() => {
  try {
    const stored = localStorage.getItem('patient')
    return stored ? JSON.parse(stored) : null
  } catch {
    return null
  }
})

const targetReps = patient?.target_reps ?? 10

const exerciseName =
  patient?.exercise === 'shoulder_abduction'
    ? 'Shoulder Abduction'
    : 'Elbow Flexion'

  // ==================================================
  // 3. CAMERA FUNCTIONS
  // ==================================================

  function releaseCamera() {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop())

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

    setError('')
    setCameraStatus('starting')

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          'Camera API unavailable. Use localhost or HTTPS.'
        )
      }

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        })

      // Stop a camera stream from an outdated request.
      if (requestId !== requestIdRef.current) {
        stream.getTracks().forEach((track) => track.stop())
        return
      }

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream

        await videoRef.current.play()
      }

      if (requestId !== requestIdRef.current) {
        releaseCamera()
        return
      }

      setCameraStatus('running')
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return
      }

      releaseCamera()

      setError(
        err.message || 'Unable to access the camera.'
      )

      setCameraStatus('error')
    }
  }

  function stopCamera() {
    // Invalidate any camera request still in progress.
    requestIdRef.current += 1

    releaseCamera()
    setError('')
    setCameraStatus('idle')
  }

  // ==================================================
  // 4. MOCK AI SIMULATION FUNCTIONS
  // ==================================================

  function startSimulation() {
    // Prevent multiple simulation intervals.
    if (mockServiceRef.current) {
      return
    }

    // Reset previous simulated measurements.
    setRepCount(0)
    setJointAngle(null)
    setMovementState('Not tracking')

    const mockService = createMockAIService((result) => {
      // Update React when new mock data arrives.
      setRepCount(result.repCount)
      setJointAngle(result.jointAngle)
      setMovementState(result.movementState)
      setFeedback(result.feedback)
    })

    mockServiceRef.current = mockService

    mockService.start()

    setIsSimulating(true)
  }

  function stopSimulation() {
    if (mockServiceRef.current) {
      mockServiceRef.current.stop()
      mockServiceRef.current = null
    }

    setIsSimulating(false)

    // Reset the UI values.
    setRepCount(0)
    setJointAngle(null)
    setMovementState('Not tracking')

    setFeedback('AI analysis is not connected yet.')
  }

  // ==================================================
  // 5. CLEANUP
  // ==================================================

  useEffect(() => {
    return () => {
      // Cancel outstanding camera requests.
      requestIdRef.current += 1

      // Release camera hardware.
      streamRef.current
        ?.getTracks()
        .forEach((track) => track.stop())

      streamRef.current = null

      // Stop background mock simulation.
      mockServiceRef.current?.stop()
      mockServiceRef.current = null
    }
  }, [])

  // ==================================================
  // 6. USER INTERFACE
  // ==================================================

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-white">
      <div className="mx-auto max-w-4xl space-y-6">

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
        </header>

        {/* SIMULATION WARNING */}
        <div
          className="rounded-xl border border-amber-700
                     bg-amber-950 p-4"
        >
          <p className="font-semibold text-amber-300">
            Development / Simulation Mode
          </p>

          <p className="mt-1 text-sm text-amber-200">
            Measurements shown during simulation are fake.
            The camera is not analyzed and no results
            are saved as patient assessments.
          </p>
        </div>

        {/* CAMERA PREVIEW */}
        <div
          className="overflow-hidden rounded-xl
                     border border-slate-800 bg-black"
        >
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
              {isSimulating
                ? 'Simulated count'
                : 'Waiting for analysis'}
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
              {isSimulating
                ? 'Simulated angle'
                : 'No measurement'}
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
              {isSimulating
                ? 'Simulated movement'
                : 'Not tracking'}
            </p>
          </div>

        </div>

        {/* FEEDBACK PANEL */}
        <div className="rounded-xl bg-slate-900 p-5">
          <h2 className="text-lg font-semibold">
            Movement Feedback
          </h2>

          <p
            className={`mt-2 text-sm ${
              isSimulating
                ? 'text-amber-400'
                : 'text-slate-400'
            }`}
          >
            {feedback}
          </p>
        </div>

        {/* ERROR DISPLAY */}
        {error && (
          <div
            role="alert"
            className="rounded-lg bg-red-950 p-4 text-red-400"
          >
            Camera error: {error}
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
                cameraStatus === 'starting' ||
                cameraStatus === 'running'
              }
              className="rounded-lg bg-blue-600 px-6 py-3
                         font-medium disabled:opacity-50"
            >
              {cameraStatus === 'starting'
                ? 'Starting...'
                : 'Start Camera'}
            </button>

            <button
              onClick={stopCamera}
              disabled={cameraStatus === 'idle'}
              className="rounded-lg bg-slate-700 px-6 py-3
                         font-medium disabled:opacity-50"
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
              disabled={isSimulating}
              className="rounded-lg bg-emerald-600 px-6 py-3
                         font-medium disabled:opacity-50"
            >
              Start Simulation
            </button>

            <button
              onClick={stopSimulation}
              disabled={!isSimulating}
              className="rounded-lg bg-amber-600 px-6 py-3
                         font-medium disabled:opacity-50"
            >
              Stop Simulation
            </button>

          </div>
        </div>

        {/* FOOTER */}
        <p className="border-t border-slate-800 pt-4 text-xs text-slate-500">
          Development preview only. Camera frames remain
          in the browser. Simulated measurements must not
          be used for medical decisions.
        </p>

      </div>
    </div>
  )
}
