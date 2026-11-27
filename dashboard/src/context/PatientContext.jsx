import { createContext, useContext, useEffect, useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { patientApi, clearPatientSession } from '../api'

const PatientContext = createContext(null)
export const usePatient = () => useContext(PatientContext)

export function PatientRoute({ children }) {
  const location = useLocation()
  const navigate = useNavigate()
  const [state, setState] = useState({ key: null, patient: null, error: '' })
  const [retry, setRetry] = useState(0)
  const token = sessionStorage.getItem('patient_token')

  useEffect(() => {
    if (!token) return
    let active = true
    let expiryTimer
    patientApi.me().then(({ data }) => {
      if (!active) return
      setState({ key: location.key, patient: data, error: '' })
      // Server verified this token; its expiration also ends an idle exercise page.
      const payload = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')))
      expiryTimer = setTimeout(() => {
        clearPatientSession()
        navigate('/patient', { replace: true })
      }, Math.max(0, payload.exp * 1000 - Date.now()))
    }).catch(error => {
      if (!active) return
      if (error.response?.status === 401) {
        clearPatientSession()
        navigate('/patient', { replace: true })
      } else {
        setState({ key: location.key, patient: null, error: 'Could not verify your session. Check the API connection and try again.' })
      }
    })
    return () => { active = false; clearTimeout(expiryTimer) }
  }, [token, location.key, navigate, retry])

  if (!token) return <Navigate to="/patient" replace />
  if (state.key !== location.key) return <p className="p-8 text-white">Verifying patient session…</p>
  if (state.error) return <div className="p-8 text-white" role="alert">{state.error} <button onClick={() => setRetry(v => v + 1)}>Retry</button> <button onClick={() => { clearPatientSession(); navigate('/patient', { replace: true }) }}>Sign out</button></div>
  return <PatientContext.Provider value={state.patient}>{children}</PatientContext.Provider>
}
