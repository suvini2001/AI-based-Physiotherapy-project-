import { useRef, useState, useEffect } from 'react'

export default function CameraTest() {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [error, setError] = useState('')
  const [isRunning, setIsRunning] = useState(false)

  async function startCamera() {
    setError('')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      })

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }

      setIsRunning(true)
    } catch (err) {
      setError(`Camera access failed: ${err.message}`)
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setIsRunning(false)
  }

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  return (
    <div className="p-6 space-y-4">
      <h2 className="text-xl font-bold">Camera Test</h2>

        <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full max-w-xl rounded-lg bg-black"
        style={{ transform: 'scaleX(-1)' }}
        />

      {error && <p className="text-red-500">{error}</p>}

      <div className="flex gap-3">
        <button
          onClick={startCamera}
          disabled={isRunning}
          className="rounded bg-blue-600 px-4 py-2 text-white"
        >
          Start Camera
        </button>

        <button
          onClick={stopCamera}
          disabled={!isRunning}
          className="rounded bg-red-600 px-4 py-2 text-white"
        >
          Stop Camera
        </button>
      </div>
    </div>
  )
}