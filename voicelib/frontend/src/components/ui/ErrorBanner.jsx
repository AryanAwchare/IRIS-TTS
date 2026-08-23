import { XMarkIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline'

export function ErrorBanner({ message, onDismiss }) {
  if (!message) return null
  return (
    <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/20 rounded-2xl p-4 animate-fade-up">
      <ExclamationTriangleIcon className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
      <p className="text-sm text-red-300 flex-1">{message}</p>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-300 transition-colors"
          aria-label="Dismiss error"
        >
          <XMarkIcon className="w-4 h-4" />
        </button>
      )}
    </div>
  )
}
