import axios from 'axios'

const BASE = 'http://localhost:8000'

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

  seed: () =>
    axios.post(`${BASE}/api/seed`),

  launchSession: (patientId, exercise) =>
    axios.post(`${BASE}/api/launch-session`, {
      patient_id: patientId,
      exercise: exercise,
      side: 'right',
    }),

  stopSession: (patientId) =>
    axios.post(`${BASE}/api/stop-session`, { patient_id: patientId }),

  sessionStatus: (patientId) =>
    axios.get(`${BASE}/api/session-status/${patientId}`),
}