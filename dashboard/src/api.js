import axios from 'axios'

export const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function authHeaders() {
  const token = localStorage.getItem('physio_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export const api = {
  login: (id, password) => {
    const form = new URLSearchParams()
    form.append('username', id)
    form.append('password', password)
    return axios.post(`${BASE}/auth/login`, form, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    })
  },

  getPatients: () =>
    axios.get(`${BASE}/api/patients`, { headers: authHeaders() }),

  createPatient: (data) =>
    axios.post(`${BASE}/api/patients`, data, { headers: authHeaders() }),

  getSessions: (patientId) =>
    axios.get(`${BASE}/api/patients/${patientId}/sessions`, { headers: authHeaders() }),

  getReps: (sessionId) =>
    axios.get(`${BASE}/api/sessions/${sessionId}/reps`, { headers: authHeaders() }),

  setPrescription: (data) =>
    axios.post(`${BASE}/api/prescriptions`, data, { headers: authHeaders() }),


}
export function clearPatientSession() {
  sessionStorage.removeItem('patient_token')
  localStorage.removeItem('patient')
}

export const patientClient = axios.create({ baseURL: BASE, timeout: 10000 })
patientClient.interceptors.request.use(config => {
  const token = sessionStorage.getItem('patient_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})
patientClient.interceptors.response.use(response => response, error => {
  if (error.response?.status === 401 && sessionStorage.getItem('patient_token')) {
    clearPatientSession()
    window.location.replace('/patient')
  }
  return Promise.reject(error)
})
export const patientApi = {
  me: () => patientClient.get('/patient/me'),
  lastSession: () => patientClient.get('/patient/me/last-session'),
}
