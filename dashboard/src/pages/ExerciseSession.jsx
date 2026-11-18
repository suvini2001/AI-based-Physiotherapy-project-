
import { useEffect, useRef, useState } from 'react'

export default function ExerciseSession() {
  // References do not cause React to re-render.
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const requestIdRef = useRef(0)

  // State controls what appears on the screen.
  const [cameraStatus, setCameraStatus] = useState('idle')
  const [error, setError] = useState('')

  // These are placeholders until we connect the AI service.
  const targetReps = 10
  const repCount = 0
  const jointAngle = null
  const movementState = 'Not tracking'

  function releaseCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  async function startCamera() {
    if (cameraStatus !== 'idle' && cameraStatus !== 'error') {
      return
    }

    const requestId = ++requestIdRef.current

    setError('')
    setCameraStatus('starting')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      })

      // Ignore requests that were cancelled or whose page unmounted.
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
        releaseCamera()
        return
      }

      setCameraStatus('running')
    } catch (err) {
      if (requestId !== requestIdRef.current) return

      releaseCamera()
      setError(err.message || 'Unable to access the camera.')
      setCameraStatus('error')
    }
  }

  function stopCamera() {
    // Invalidates an in-progress camera request.
    requestIdRef.current += 1

    releaseCamera()
    setError('')
    setCameraStatus('idle')
  }

  // Release the camera when this page is removed.
  useEffect(() => {
    return () => {
      requestIdRef.current += 1
      releaseCamera()
    }
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6">
      <div className="mx-auto max-w-4xl space-y-6">

        <header>
          <p className="text-sm text-blue-400">
            PhysioTrack / Exercise Session
          </p>
          <h1 className="mt-2 text-3xl font-bold">
            Elbow Flexion
          </h1>
          <p className="mt-2 text-slate-400">
            Browser camera demonstration
          </p>
        </header>

        {/* Browser camera preview */}
        <div className="overflow-hidden rounded-xl bg-black">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="aspect-video w-full object-contain"
          />
        </div>

        {/* Exercise measurements */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">

          <div className="rounded-xl bg-slate-900 p-5">
            <p className="text-sm text-slate-400">
              Repetitions
            </p>
            <p className="mt-2 text-3xl font-bold">
              {repCount} / {targetReps}
            </p>
          </div>

          <div className="rounded-xl bg-slate-900 p-5">
            <p className="text-sm text-slate-400">
              Joint Angle
            </p>
            <p className="mt-2 text-3xl font-bold">
              {jointAngle === null ? '--' : `${jointAngle}°`}
            </p>
          </div>

          <div className="rounded-xl bg-slate-900 p-5">
            <p className="text-sm text-slate-400">
              Movement State
            </p>
            <p className="mt-2 text-lg font-semibold">
              {movementState}
            </p>
          </div>
        </div>

        {/* Feedback */}
        <div className="rounded-xl bg-slate-900 p-5">
          <h2 className="font-semibold">Movement Feedback</h2>
          <p className="mt-2 text-sm text-slate-400">
            AI analysis is not connected yet.
          </p>
        </div>

        {error && (
          <p role="alert" className="text-red-400">
            Camera error: {error}
          </p>
        )}

        {/* Camera controls */}
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

        <p className="text-xs text-slate-500">
          Development preview only. No video is uploaded,
          analyzed, or saved.
        </p>

      </div>
    </div>
  )
}
