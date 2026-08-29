import { useState, useCallback } from 'react'
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { Navbar } from './components/layout/Navbar'
import { ProtectedRoute } from './components/layout/ProtectedRoute'
import Login from './pages/Login'
import Register from './pages/Register'
import VoiceLibrary from './pages/VoiceLibrary'
import Generate from './pages/Generate'
import SongCover from './pages/SongCover'

// Shared GPU status lifted to App level so Navbar can display it
// without prop-drilling through every route. Generate page calls
// setGpuStatus via the onGpuStatus prop.
function AppLayout({ gpuStatus }) {
  return (
    <>
      <Navbar gpuStatus={gpuStatus} />
      <Outlet />
    </>
  )
}

function ProtectedLayout({ gpuStatus }) {
  return (
    <ProtectedRoute>
      <AppLayout gpuStatus={gpuStatus} />
    </ProtectedRoute>
  )
}

export default function App() {
  const [gpuStatus, setGpuStatus] = useState({ online: false, label: 'OFFLINE' })

  const handleGpuStatus = useCallback((statuses) => {
    const neural = statuses?.find(e => e.id === 'gpt-sovits-v3')
    setGpuStatus({
      online: neural?.ready || false,
      label:  neural?.ready ? 'ONLINE' : 'OFFLINE',
    })
  }, [])

  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login"    element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected routes with shared layout */}
        <Route element={<ProtectedLayout gpuStatus={gpuStatus} />}>
          <Route path="/library"    element={<VoiceLibrary />} />
          <Route path="/generate"   element={<Generate onGpuStatus={handleGpuStatus} />} />
          <Route path="/song-cover" element={<SongCover />} />
        </Route>

        {/* Default redirect */}
        <Route path="*" element={<Navigate to="/library" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
