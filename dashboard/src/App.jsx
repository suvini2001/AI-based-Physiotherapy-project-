import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import Login from './pages/Login'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import PatientList from './pages/PatientList'
import PatientDetail from './pages/PatientDetail'
import LiveView from './pages/LiveView'
import PatientLogin from './pages/PatientLogin'
import PatientSession from './pages/PatientSession'

function PrivateRoute({ children }) {
  const { token } = useAuth()
  return token ? children : <Navigate to="/login" replace />
}

function PatientRoute({ children }) {
  const stored = localStorage.getItem('patient')
  return stored ? children : <Navigate to="/patient" replace />
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Physio portal */}
          <Route path="/login" element={<Login />} />
          <Route path="/" element={
            <PrivateRoute><Layout /></PrivateRoute>
          }>
            <Route index element={<Dashboard />} />
            <Route path="patients" element={<PatientList />} />
            <Route path="patients/:id" element={<PatientDetail />} />
            <Route path="live" element={<LiveView />} />
          </Route>

          {/* Patient portal */}
          <Route path="/patient" element={<PatientLogin />} />
          <Route path="/patient/session" element={
            <PatientRoute><PatientSession /></PatientRoute>
          } />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}