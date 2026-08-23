import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  MicrophoneIcon, SparklesIcon, MusicalNoteIcon,
  LockClosedIcon, Bars3Icon, XMarkIcon,
} from '@heroicons/react/24/outline'
import { useAuthStore } from '../../store/useAuthStore'
import { clsx } from 'clsx'

const navItems = [
  { to: '/library',  label: 'Library',    Icon: MicrophoneIcon },
  { to: '/generate', label: 'Generate',   Icon: SparklesIcon   },
  { to: '/song-cover',label:'Song Cover', Icon: MusicalNoteIcon, locked: true },
]

export function Navbar() {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <>
      {/* Floating glass pill nav */}
      <nav className="nav-pill" role="navigation" aria-label="Main navigation">
        {/* Logo */}
        <NavLink
          to="/library"
          className="flex items-center gap-2.5 mr-5 group"
          aria-label="VoiceLib home"
        >
          <div className="w-9 h-9 rounded-xl bg-primary-500/20 border border-primary-500/30 flex items-center justify-center
                          group-hover:bg-primary-500/30 transition-all duration-300 ease-spring shrink-0">
            <MicrophoneIcon className="w-5 h-5 text-primary-400" />
          </div>
          <span className="text-base sm:text-lg font-bold text-white tracking-tight hidden sm:block">VoiceLib</span>
        </NavLink>

        {/* Desktop nav links */}
        <div className="hidden sm:flex items-center gap-1.5">
          {navItems.map(({ to, label, Icon, locked }) =>
            locked ? (
              <span
                key={to}
                className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold
                           text-zinc-500 cursor-not-allowed select-none"
                title="Coming Soon"
              >
                <Icon className="w-4 h-4" />
                {label}
                <LockClosedIcon className="w-3.5 h-3.5" />
              </span>
            ) : (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold transition-all duration-300 ease-spring',
                    isActive
                      ? 'bg-primary-500/20 text-primary-300 border border-primary-500/30 shadow-glow-sm'
                      : 'text-surface-200 hover:bg-white/[0.08] hover:text-white'
                  )
                }
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            )
          )}
        </div>

        {/* User / logout */}
        {user && (
          <div className="flex items-center gap-3 ml-4 pl-4 border-l border-white/[0.12]">
            <span className="text-sm font-medium text-surface-200 hidden md:block truncate max-w-[150px]">
              {user.email}
            </span>
            <button
              onClick={handleLogout}
              className="text-sm font-semibold text-surface-300 hover:text-red-400 transition-colors"
            >
              Logout
            </button>
          </div>
        )}

        {/* Mobile hamburger */}
        <button
          onClick={() => setMenuOpen(!menuOpen)}
          className="sm:hidden ml-2 w-8 h-8 flex items-center justify-center rounded-full
                     bg-white/[0.05] hover:bg-white/[0.1] transition-colors"
          aria-label="Toggle menu"
          aria-expanded={menuOpen}
        >
          {menuOpen
            ? <XMarkIcon className="w-4 h-4 text-white" />
            : <Bars3Icon className="w-4 h-4 text-surface-200" />
          }
        </button>
      </nav>

      {/* Mobile menu overlay */}
      {menuOpen && (
        <div
          className="fixed inset-0 z-40 flex flex-col items-center justify-center gap-4 sm:hidden animate-fade-in"
          style={{ backdropFilter: 'blur(30px)', background: 'rgba(5,5,5,0.85)' }}
        >
          {navItems.map(({ to, label, Icon, locked }, i) => (
            locked ? (
              <span
                key={to}
                className="flex items-center gap-3 text-2xl font-medium text-zinc-700 cursor-not-allowed"
                style={{ animationDelay: `${i * 80}ms` }}
              >
                <Icon className="w-6 h-6" /> {label}
                <LockClosedIcon className="w-4 h-4" />
              </span>
            ) : (
              <NavLink
                key={to}
                to={to}
                onClick={() => setMenuOpen(false)}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-3 text-2xl font-medium transition-all duration-300 animate-fade-up',
                    isActive ? 'text-primary-400' : 'text-white hover:text-primary-300',
                  )
                }
                style={{ animationDelay: `${i * 80}ms` }}
              >
                <Icon className="w-6 h-6" /> {label}
              </NavLink>
            )
          ))}
          {user && (
            <button
              onClick={() => { handleLogout(); setMenuOpen(false) }}
              className="mt-8 text-sm text-surface-700 hover:text-red-400 transition-colors animate-fade-up delay-300"
            >
              Logout
            </button>
          )}
        </div>
      )}
    </>
  )
}
