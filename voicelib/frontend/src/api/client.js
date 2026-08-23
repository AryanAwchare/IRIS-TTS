import axios from 'axios'
import { useAuthStore } from '../store/useAuthStore'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 120_000, // 2 min for TTS generation
})

// Attach JWT to every request
client.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Normalize error responses into plain Error objects
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      // Auto-logout on token expiry
      useAuthStore.getState().logout()
    }
    const detail = err.response?.data?.detail
    const message =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
        ? detail.map((d) => d.msg).join(', ')
        : err.message || 'An unexpected error occurred. Please try again.'
    return Promise.reject(new Error(message))
  }
)

export default client
