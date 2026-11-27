import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { BASE, patientClient, clearPatientSession } from '../api'
import { usePatient } from '../context/PatientContext'

const terminal = new Set(['COMPLETE', 'CALIBRATION_FAILED', 'TRACKING_LOST'])
const errorMessage = error => typeof error.response?.data?.detail === 'string'
  ? error.response.data.detail : error.message || 'The AI connection failed. Stop and restart.'

export default function RealExerciseSession() {
  const patient = usePatient()
  const navigate = useNavigate()
  const video = useRef(null)
  const stream = useRef(null)
  const sessionId = useRef(null)
  const generation = useRef(0)
  const timer = useRef(null)
  const token = useRef(sessionStorage.getItem('patient_token'))
  const [side, setSide] = useState('right')
  const [consent, setConsent] = useState(false)
  const [status, setStatus] = useState('idle')
  const [measurement, setMeasurement] = useState(null)
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(null)
  const [prescription, setPrescription] = useState(null)

  function releaseCamera() {
    stream.current?.getTracks().forEach(track => track.stop())
    stream.current = null
    if (video.current) video.current.srcObject = null
  }

  function discardId(id) {
    // Capture credentials before logout clears browser storage. No image in cleanup requests.
    return fetch(`${BASE}/patient/me/ai-sessions/${id}`, {
      method: 'DELETE', headers: { Authorization: `Bearer ${token.current}` }, keepalive: true,
    }).catch(() => {})
  }

  function cleanup() {
    generation.current += 1
    clearTimeout(timer.current)
    releaseCamera()
    const id = sessionId.current
    sessionId.current = null
    if (id) discardId(id)
  }

  useEffect(() => {
    function leave() { cleanup() }
    window.addEventListener('pagehide', leave)
    return () => { window.removeEventListener('pagehide', leave); cleanup() }
    // Refs hold resources; cleanup must not restart when measurements change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function start() {
    if (status === 'starting' || status === 'running' || !consent) return
    const run = ++generation.current
    setStatus('starting'); setError(''); setSaved(null); setMeasurement(null)
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or HTTPS.')
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      if (run !== generation.current) { media.getTracks().forEach(track => track.stop()); return }
      stream.current = media
      video.current.srcObject = media
      await video.current.play()
      if (run !== generation.current) return
      const { data } = await patientClient.post('/patient/me/ai-sessions', { side }, { timeout: 200000 })
      if (run !== generation.current) { discardId(data.session_id); return }
      sessionId.current = data.session_id
      setPrescription(data.prescription)
      setStatus('running')
      const canvas = document.createElement('canvas')
      canvas.width = 640; canvas.height = 480
      const ctx = canvas.getContext('2d')
      async function nextFrame() {
        const frameStart = performance.now()
        if (run !== generation.current) return
        try {
          const element = video.current
          if (!element?.videoWidth || !stream.current?.active) throw new Error('Camera stopped. Restart the session.')
          // Preserve aspect ratio; do not mirror frames sent to the research pipeline.
          const scale = Math.min(640 / element.videoWidth, 480 / element.videoHeight)
          const w = element.videoWidth * scale, h = element.videoHeight * scale
          ctx.fillStyle = 'black'; ctx.fillRect(0, 0, 640, 480)
          ctx.drawImage(element, (640 - w) / 2, (480 - h) / 2, w, h)
          const jpeg = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.7))
          if (run !== generation.current) return
          if (!jpeg) throw new Error('Could not capture a camera frame.')
          const response = await patientClient.post(`/patient/me/ai-sessions/${data.session_id}/frame`, jpeg, {
            headers: { 'Content-Type': 'image/jpeg' }, timeout: 25000,
          })
          if (run !== generation.current) return
          setMeasurement(response.data)
          if (terminal.has(response.data.phase)) {
            setStatus('finished'); releaseCamera(); return
          }
          // One outstanding frame only: slow inference cannot build a frame queue.
          timer.current = setTimeout(nextFrame, Math.max(10, 60 - (performance.now() - frameStart)))
        } catch (err) {
          if (run !== generation.current) return
          if (err.response?.status === 429) { timer.current = setTimeout(nextFrame, 500); return }
          cleanup(); setStatus('idle'); setError(errorMessage(err))
        }
      }
      nextFrame()
    } catch (err) {
      if (run !== generation.current) return
      cleanup(); setStatus('idle'); setError(errorMessage(err))
    }
  }

  function stop() {
    cleanup(); setStatus('idle'); setMeasurement(null); setError(''); setSaved(null)
  }

  async function save() {
    const id = sessionId.current
    if (!id) return
    const run = ++generation.current
    clearTimeout(timer.current); releaseCamera(); setStatus('saving'); setError('')
    try {
      const { data } = await patientClient.post(`/patient/me/ai-sessions/${id}/finish`, null, { timeout: 30000 })
      if (run !== generation.current) return
      setSaved(data); setStatus('saved')
      sessionId.current = null
      discardId(id)
    } catch (err) {
      if (run !== generation.current) return
      setError(errorMessage(err)); setStatus('finished')
    }
  }

  const busy = status === 'starting' || status === 'running' || status === 'saving'
  const reps = measurement?.rep_count || 0
  const target = prescription?.target_reps || patient?.target_reps
  return (
    <div className="min-h-screen bg-slate-950 p-6 text-white">
      <main className="mx-auto max-w-4xl space-y-5">
        <nav className="flex justify-between">
          <Link to="/patient/session" onClick={cleanup}>← Patient dashboard</Link>
          <button onClick={() => { cleanup(); clearPatientSession(); navigate('/patient', { replace: true }) }}>Sign Out</button>
        </nav>
        <h1 className="text-3xl font-bold">Camera exercise analysis</h1>
        <p>{(prescription?.exercise || patient?.exercise || 'No prescription').replaceAll('_', ' ')} · Target: {target ?? '—'} repetitions</p>
        <div className="rounded-xl border border-amber-700 bg-amber-950 p-4 text-amber-200">
          Real pose-based research measurements, not a simulation. This prototype is not clinically validated.
          Frames are sent to your configured AI server for processing and are not stored as video.
          Saved results include angles, repetitions, faults and model provenance. Use fictional accounts only.
        </div>
        <label className="block">
          <input type="checkbox" checked={consent} disabled={busy} onChange={e => setConsent(e.target.checked)} className="mr-2" />
          I agree to send camera frames to the AI server for this research session.
        </label>
        <label className="block">Arm to analyse{' '}
          <select value={side} disabled={busy || status === 'finished'} onChange={e => setSide(e.target.value)} className="rounded bg-slate-800 p-2">
            <option value="right">Right</option><option value="left">Left</option>
          </select>
        </label>
        <video ref={video} autoPlay playsInline muted className="aspect-video w-full rounded-xl bg-black object-contain" />
        <div className="grid grid-cols-3 gap-3 rounded-xl bg-slate-900 p-5">
          <div>Repetitions<p className="text-2xl">{reps} / {target ?? '—'}</p></div>
          <div>Measured angle<p className="text-2xl">{measurement?.joint_angle == null ? '—' : `${measurement.joint_angle.toFixed(1)}°`}</p></div>
          <div>Phase<p className="text-lg">{measurement?.phase || status}</p></div>
        </div>
        <p role="status">{measurement?.feedback || 'Select your prescribed arm, allow frame processing, then start.'}</p>
        {measurement && <p className="text-sm text-slate-400">Tracking: {measurement.tracking ? 'visible' : 'unavailable'} · Angle source: {measurement.angle_source || 'none'} · Movement: {measurement.state}</p>}
        <p className="text-sm text-slate-400">Calibration uses your observed range. The prescribed ROM is recorded for clinician review; it does not override the research algorithm’s thresholds.</p>
        {error && <p role="alert" className="rounded bg-red-950 p-4 text-red-300">{error}</p>}
        {saved && <p role="status" className="rounded bg-emerald-950 p-4">{saved.saved ? `Saved ${saved.total_reps} repetitions. Your clinician can review this research session.` : saved.reason}</p>}
        <div className="flex flex-wrap gap-3">
          <button onClick={start} disabled={busy || status === 'finished' || !consent || !patient?.has_prescription}
            className="rounded-lg bg-blue-600 px-5 py-3 disabled:opacity-50">{status === 'starting' ? 'Starting AI…' : 'Start real analysis'}</button>
          <button onClick={stop} disabled={status === 'idle' || status === 'saving'} className="rounded-lg bg-slate-700 px-5 py-3 disabled:opacity-50">Stop and discard / Reset</button>
          <button onClick={save} disabled={!sessionId.current || reps < 1 || status === 'saving'}
            className="rounded-lg bg-emerald-700 px-5 py-3 disabled:opacity-50">{status === 'saving' ? 'Saving…' : 'Finish and save results'}</button>
        </div>
        <Link className="block text-blue-300" to="/patient/exercise-demo" onClick={cleanup}>Use the separate simulation instead</Link>
      </main>
    </div>
  )
}
