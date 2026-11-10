import { createContext, useContext, useState, useCallback } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem('physio_token'))
  const [physio, setPhysio] = useState(() => {
    try { return JSON.parse(localStorage.getItem('physio_user')) } catch { return null }
  })

  const login = useCallback((accessToken, physioData) => {
    localStorage.setItem('physio_token', accessToken)
    localStorage.setItem('physio_user', JSON.stringify(physioData))
    setToken(accessToken)
    setPhysio(physioData)
  }, [])

  const logout = useCallback(() => {
    localStorage.removeItem('physio_token')
    localStorage.removeItem('physio_user')
    setToken(null)
    setPhysio(null)
  }, [])

  return (
    <AuthContext.Provider value={{ token, physio, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)