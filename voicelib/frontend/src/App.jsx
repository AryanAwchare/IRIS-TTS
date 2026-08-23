import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom'
import { Navbar } from './components/layout/Navbar'
import { ProtectedRoute } from './components/layout/ProtectedRoute'
import Login from './pages/Login'
import Register from './pages/Register'
import VoiceLibrary from './pages/VoiceLibrary'
import Generate from './pages/Generate'
import SongCover from './pages/SongCover'

function AppLayout() {
  return (
    <>
      <Navbar />
      <Outlet />
    </>
  )
}

function ProtectedLayout() {
  return (
    <ProtectedRoute>
      <AppLayout />
    </ProtectedRoute>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login"    element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected routes with shared layout */}
        <Route element={<ProtectedLayout />}>
          <Route path="/library"    element={<VoiceLibrary />} />
          <Route path="/generate"   element={<Generate />} />
          <Route path="/song-cover" element={<SongCover />} />
        </Route>

        {/* Default redirect */}
        <Route path="*" element={<Navigate to="/library" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
