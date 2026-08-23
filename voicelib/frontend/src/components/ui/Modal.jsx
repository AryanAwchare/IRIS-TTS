import { useEffect, useRef } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'

export function Modal({ isOpen, onClose, title, children, size = 'md' }) {
  const overlayRef = useRef(null)

  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    if (isOpen) window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, onClose])

  // Prevent body scroll
  useEffect(() => {
    document.body.style.overflow = isOpen ? 'hidden' : ''
    return () => { document.body.style.overflow = '' }
  }, [isOpen])

  if (!isOpen) return null

  const sizes = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
  }

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
      style={{ backdropFilter: 'blur(20px)', background: 'rgba(5,5,5,0.7)' }}
      onClick={(e) => { if (e.target === overlayRef.current) onClose() }}
    >
      <div
        className={`relative w-full ${sizes[size]} animate-fade-up`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        {/* Outer shell (Double-Bezel) */}
        <div className="card-shell" style={{ boxShadow: '0 24px 80px -12px rgba(0,0,0,0.5)' }}>
          <div className="card-inner">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <h2 id="modal-title" className="text-lg font-semibold text-white">
                {title}
              </h2>
              <button
                onClick={onClose}
                className="w-8 h-8 flex items-center justify-center rounded-full
                           bg-white/[0.05] hover:bg-white/[0.1] transition-colors"
                aria-label="Close modal"
              >
                <XMarkIcon className="w-4 h-4 text-surface-200" />
              </button>
            </div>
            {children}
          </div>
        </div>
      </div>
    </div>
  )
}
