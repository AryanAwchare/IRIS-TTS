import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { authApi } from '../api/auth'
import { useAuthStore } from '../store/useAuthStore'
import { ErrorBanner } from '../components/ui/ErrorBanner'
import { Spinner } from '../components/ui/Spinner'
import { MicrophoneIcon } from '@heroicons/react/24/outline'

export default function Register() {
  const navigate = useNavigate()
  const login    = useAuthStore((s) => s.login)

  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    if (password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    setLoading(true)
    try {
      const data = await authApi.register(email, password)
      login(data.user, data.access_token)
      navigate('/library')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-dvh flex items-center justify-center px-4 py-24">
      <div className="w-full max-w-md animate-fade-up">
        {/* Brand */}
        <div className="flex flex-col items-center mb-10 text-center">
          <div className="w-20 h-20 rounded-4xl bg-primary-500/15 border border-primary-500/30 flex items-center justify-center mb-5 shadow-glow">
            <MicrophoneIcon className="w-10 h-10 text-primary-400" />
          </div>
          <span className="eyebrow mb-3">Free to Start</span>
          <h1 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">Create your account</h1>
          <p className="text-base text-surface-200 mt-2 font-medium">Build and manage your personal voice library</p>
        </div>

        <div className="card-shell">
          <div className="card-inner">
            <form onSubmit={handleSubmit} className="space-y-5">
              <ErrorBanner message={error} onDismiss={() => setError(null)} />

              <div>
                <label htmlFor="reg-email" className="label">Email Address</label>
                <input
                  id="reg-email"
                  type="email"
                  className="input"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                />
              </div>

              <div>
                <label htmlFor="reg-password" className="label">Password</label>
                <input
                  id="reg-password"
                  type="password"
                  className="input"
                  placeholder="Min. 8 characters"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={8}
                  autoComplete="new-password"
                />
              </div>

              <button
                type="submit"
                disabled={loading || !email || password.length < 8}
                className="btn-primary w-full justify-center mt-3 py-4 text-base font-bold shadow-glow"
              >
                {loading ? <><Spinner size="sm" /><span>Creating account...</span></> : 'Create Account'}
              </button>
            </form>
          </div>
        </div>

        <p className="text-center text-base text-surface-200 mt-8 font-medium">
          Already have an account?{' '}
          <Link to="/login" className="text-primary-300 hover:text-white transition-colors font-bold underline underline-offset-4">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  )
}
