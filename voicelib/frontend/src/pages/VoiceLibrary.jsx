import { useEffect, useState } from 'react'
import { PlusIcon, MicrophoneIcon } from '@heroicons/react/24/outline'
import { useVoiceStore } from '../store/useVoiceStore'
import { VoiceCard } from '../components/voices/VoiceCard'
import { AddVoiceModal } from '../components/voices/AddVoiceModal'
import { Spinner } from '../components/ui/Spinner'
import { ErrorBanner } from '../components/ui/ErrorBanner'

/* Skeleton card placeholder */
function SkeletonCard() {
  return (
    <div className="card-shell">
      <div className="card-inner animate-pulse">
        <div className="flex gap-3 mb-4">
          <div className="w-10 h-10 bg-white/[0.05] rounded-2xl shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="h-3 bg-white/[0.05] rounded w-2/3" />
            <div className="h-2 bg-white/[0.03] rounded w-1/3" />
          </div>
        </div>
        <div className="h-8 bg-white/[0.04] rounded-full mt-6" />
      </div>
    </div>
  )
}

function EmptyState({ onAdd }) {
  return (
    <div className="col-span-full flex flex-col items-center justify-center py-24 animate-fade-up">
      <div className="w-24 h-24 rounded-4xl bg-primary-500/15 border border-primary-500/30 flex items-center justify-center mb-6 shadow-glow-sm">
        <MicrophoneIcon className="w-12 h-12 text-primary-400" />
      </div>
      <span className="eyebrow mb-4">Your library is empty</span>
      <h2 className="text-2xl sm:text-3xl font-bold text-white mb-3">No cloned voices yet</h2>
      <p className="text-base text-surface-200 mb-8 text-center max-w-md font-medium leading-relaxed">
        Upload a short audio sample (up to 30s) to clone a voice with high fidelity and add it to your library.
      </p>
      <button onClick={onAdd} className="btn-primary">
        <PlusIcon className="w-5 h-5" />
        Add Your First Voice
      </button>
    </div>
  )
}

export default function VoiceLibrary() {
  const { voices, isLoading, error, fetchVoices, clearError } = useVoiceStore()
  const [showModal, setShowModal] = useState(false)

  useEffect(() => { fetchVoices() }, [])

  return (
    <div className="min-h-dvh pt-32 pb-20 px-4 sm:px-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6 mb-12 animate-fade-up">
        <div>
          <span className="eyebrow mb-3 inline-block">Voice Library</span>
          <h1 className="text-4xl sm:text-5xl font-bold text-white tracking-tight">Your Voices</h1>
          <p className="text-base text-surface-200 mt-2 font-medium">
            {voices.length} {voices.length === 1 ? 'voice' : 'voices'} saved in your personal library
          </p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="btn-primary shrink-0"
          id="add-voice-btn"
        >
          <PlusIcon className="w-5 h-5" />
          <span>Add Voice</span>
        </button>
      </div>

      {/* Error */}
      <ErrorBanner message={error} onDismiss={clearError} />

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {isLoading
          ? Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)
          : voices.length === 0
          ? <EmptyState onAdd={() => setShowModal(true)} />
          : voices.map((v, i) => (
              <div
                key={v.id}
                className="animate-fade-up"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <VoiceCard voice={v} />
              </div>
            ))
        }
      </div>

      <AddVoiceModal isOpen={showModal} onClose={() => setShowModal(false)} />
    </div>
  )
}
