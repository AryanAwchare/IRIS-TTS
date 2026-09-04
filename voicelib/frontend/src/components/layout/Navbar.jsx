import { useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import {
  MicrophoneIcon, SparklesIcon, MusicalNoteIcon,
  LockClosedIcon, Bars3Icon, XMarkIcon,
} from '@heroicons/react/24/outline'
import { useAuthStore } from '../../store/useAuthStore'
import { clsx } from 'clsx'

const navItems = [
  { to: '/library',   label: 'Library',    Icon: MicrophoneIcon },
  { to: '/generate',  label: 'Generate',   Icon: SparklesIcon   },
  { to: '/song-cover', label: 'Song Cover', Icon: MusicalNoteIcon, locked: false },
]

/**
 * gpuStatus: { online: boolean, label: string }
 * Passed as a prop from the page that polls engineStatus,
 * or falls back to a default offline state.
 */
export function Navbar({ gpuStatus = { online: false, label: 'OFFLINE' } }) {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const [menuOpen, setMenuOpen] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <>
      {/* Floating tactical glass pill nav */}
      <nav className="nav-pill" role="navigation" aria-label="Main navigation">

        {/* ── Logo — IRIS // TØP-STUDIO ─────────────────────────────── */}
        <NavLink
          to="/library"
          className="flex items-center gap-2.5 mr-5 group"
          aria-label="IRIS home"
        >
          {/* Acid-yellow mic icon box */}
          <div className="w-9 h-9 rounded-xl border flex items-center justify-center shrink-0 transition-all duration-300 ease-spring group-hover:shadow-glow-sm"
            style={{
              background: 'rgba(229,255,0,0.08)',
              borderColor: 'rgba(229,255,0,0.3)',
            }}
          >
            <MicrophoneIcon className="w-5 h-5" style={{ color: '#E5FF00' }} />
          </div>

          {/* Wordmark */}
          <div className="hidden sm:flex items-baseline gap-1.5 leading-none">
            <span className="font-display font-bold text-base tracking-tight" style={{ color: '#E5FF00' }}>
              IRIS
            </span>
            <span className="text-surface-600 text-xs font-mono">//</span>
            <span className="font-display font-medium text-sm tracking-wide text-bone">
              TØP-STUDIO
            </span>
          </div>

          {/* GPU heartbeat indicator */}
          <div className="flex items-center gap-1.5 ml-1 hidden sm:flex" title={`GPU ${gpuStatus.label}`}>
            <span className={gpuStatus.online ? 'gpu-dot-online' : 'gpu-dot-offline'} />
            <span className="font-mono text-[9px] uppercase tracking-widest"
              style={{ color: gpuStatus.online ? '#E5FF00' : '#FF003C', opacity: 0.75 }}
            >
              {gpuStatus.label}
            </span>
          </div>
        </NavLink>

        {/* ── Desktop nav links ─────────────────────────────────────── */}
        <div className="hidden sm:flex items-center gap-1.5">
          {navItems.map(({ to, label, Icon, locked }) =>
            locked ? (
              <span
                key={to}
                className="flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold text-zinc-600 cursor-not-allowed select-none"
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
                      ? 'text-obsidian font-bold'
                      : 'text-surface-300 hover:bg-white/[0.07] hover:text-bone'
                  )
                }
                style={({ isActive }) => isActive ? {
                  background: 'rgba(229,255,0,0.15)',
                  border: '1px solid rgba(229,255,0,0.3)',
                  boxShadow: '0 0 12px rgba(229,255,0,0.2)',
                  color: '#E5FF00',
                } : {}}
              >
                <Icon className="w-4 h-4" />
                {label}
              </NavLink>
            )
          )}
        </div>

        {/* ── User / logout ─────────────────────────────────────────── */}
        {user && (
          <div className="flex items-center gap-3 ml-4 pl-4" style={{ borderLeft: '1px solid rgba(255,255,255,0.08)' }}>
            <span className="text-xs font-mono text-surface-500 hidden md:block truncate max-w-[130px]">
              {user.email}
            </span>
            <button
              onClick={handleLogout}
              className="text-xs font-mono font-semibold text-surface-500 hover:text-crimson transition-colors tracking-wider uppercase"
            >
              LOGOUT
            </button>
          </div>
        )}

        {/* ── Mobile hamburger ──────────────────────────────────────── */}
        <button
          onClick={() => setMenuOpen(!menuOpen)}
          className="sm:hidden ml-2 w-8 h-8 flex items-center justify-center rounded-full bg-white/[0.05] hover:bg-white/[0.1] transition-colors"
          aria-label="Toggle menu"
          aria-expanded={menuOpen}
        >
          {menuOpen
            ? <XMarkIcon className="w-4 h-4 text-bone" />
            : <Bars3Icon className="w-4 h-4 text-surface-300" />
          }
        </button>
      </nav>

      {/* ── Mobile menu overlay ────────────────────────────────────────── */}
      {menuOpen && (
        <div
          className="fixed inset-0 z-40 flex flex-col items-center justify-center gap-5 sm:hidden animate-fade-in"
          style={{ backdropFilter: 'blur(30px)', background: 'rgba(7,8,10,0.92)' }}
        >
          {/* Scanline on mobile overlay */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.08) 3px, rgba(0,0,0,0.08) 4px)',
            }}
          />

          {navItems.map(({ to, label, Icon, locked }, i) => (
            locked ? (
              <span
                key={to}
                className="relative z-10 flex items-center gap-3 text-2xl font-display font-medium text-zinc-700 cursor-not-allowed"
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
                    'relative z-10 flex items-center gap-3 text-2xl font-display font-bold transition-all duration-300 animate-fade-up',
                    isActive ? '' : 'text-bone hover:text-acid'
                  )
                }
                style={({ isActive }) => isActive ? { color: '#E5FF00' } : {}}
                aria-current={undefined}
              >
                <Icon className="w-6 h-6" /> {label}
              </NavLink>
            )
          ))}

          {user && (
            <button
              onClick={() => { handleLogout(); setMenuOpen(false) }}
              className="relative z-10 mt-8 text-xs font-mono uppercase tracking-widest text-surface-600 hover:text-crimson transition-colors animate-fade-up delay-300"
            >
              LOGOUT
            </button>
          )}
        </div>
      )}
    </>
  )
}
