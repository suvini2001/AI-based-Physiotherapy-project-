import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NAV = [
  { to: '/', label: 'Overview', exact: true, icon: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
      <rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>
    </svg>
  )},
  { to: '/patients', label: 'Patients', icon: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
  )},
  { to: '/live', label: 'Live session', icon: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
    </svg>
  )},
]

export default function Layout() {
  const { physio, logout } = useAuth()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100">
      <aside className="w-60 flex-shrink-0 flex flex-col border-r border-slate-800 bg-slate-950">
        <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
          </div>
          <span className="font-semibold text-white text-sm tracking-tight">PhysioTrack</span>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ to, label, icon, exact }) => (
            <NavLink
              key={to}
              to={to}
              end={exact}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`
              }
            >
              {icon}
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-slate-800 p-3">
          <div className="flex items-center gap-3 px-2 py-2">
            <div className="w-8 h-8 rounded-full bg-blue-900 flex items-center justify-center flex-shrink-0 text-blue-300 text-xs font-bold">
              {physio?.name?.split(' ').map(w => w[0]).join('').slice(0, 2)}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-white truncate">{physio?.name}</p>
              <p className="text-xs text-slate-500 truncate">{physio?.id}</p>
            </div>
            <button onClick={handleLogout} title="Sign out"
              className="text-slate-500 hover:text-slate-300 transition-colors flex-shrink-0">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16,17 21,12 16,7"/>
                <line x1="21" y1="12" x2="9" y2="12"/>
              </svg>
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-auto bg-slate-950">
        <Outlet />
      </main>
    </div>
  )
}