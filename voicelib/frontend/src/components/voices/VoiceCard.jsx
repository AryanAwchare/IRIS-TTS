import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  TrashIcon,
  MicrophoneIcon,
  SparklesIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline'
import { PlayIcon, PauseIcon, ArrowPathIcon } from '@heroicons/react/24/solid'
import { voicesApi } from '../../api/voices'
import { useVoiceStore } from '../../store/useVoiceStore'
import { VoiceTuneModal } from './VoiceTuneModal'
import { Spinner } from '../ui/Spinner'

function formatDate(iso) {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

export function VoiceCard({ voice }) {
  const navigate = useNavigate()
  const removeVoice = useVoiceStore((s) => s.removeVoice)
  const [deleting, setDeleting] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const [playingSample, setPlayingSample] = useState(false)
  const [showTuneModal, setShowTuneModal] = useState(false)
  const audioRef = useRef(null)

  const handleDelete = async () => {
    setDeleting(true)
    try {
      await voicesApi.remove(voice.id)
      removeVoice(voice.id)
    } catch {
      setDeleting(false)
      setConfirm(false)
    }
  }

  const toggleSamplePlay = (e) => {
    e.stopPropagation()
    if (!audioRef.current) return
    if (playingSample) {
      audioRef.current.pause()
      setPlayingSample(false)
    } else {
      audioRef.current
        .play()
        .then(() => setPlayingSample(true))
        .catch(() => setPlayingSample(false))
    }
  }

  const cfgVal = voice.opt_weights?.cfg_weight ?? 0.70

  return (
    /* Double-Bezel card */
    <div className="card-shell group animate-fade-up">
      <div className="card-inner flex flex-col h-full min-h-[200px]">
        {voice.sample_url && (
          <audio
            ref={audioRef}
            src={voice.sample_url}
            onEnded={() => setPlayingSample(false)}
            preload="none"
          />
        )}

        {/* Icon + name */}
        <div className="flex items-start gap-3.5 mb-auto">
          <button
            onClick={toggleSamplePlay}
            disabled={!voice.sample_url}
            className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 transition-all ${
              playingSample
                ? 'bg-primary-500 text-white shadow-glow-sm scale-105'
                : 'bg-primary-500/15 border border-primary-500/30 text-primary-400 hover:bg-primary-500/25'
            }`}
            title={
              voice.sample_url
                ? playingSample
                  ? 'Pause sample'
                  : 'Listen to original voice sample'
                : 'No sample audio'
            }
          >
            {playingSample ? (
              <PauseIcon className="w-6 h-6 text-white" />
            ) : (
              <PlayIcon className="w-5 h-5 ml-0.5" />
            )}
          </button>

          <div className="min-w-0 flex-1">
            <div className="flex items-center justify-between gap-1">
              <h3 className="font-bold text-surface-50 truncate text-base sm:text-lg">
                {voice.name}
              </h3>
              <div className="flex items-center gap-1 shrink-0">
                <span className="text-[10px] text-cyan-300 font-mono bg-cyan-500/10 px-2 py-0.5 rounded-md border border-cyan-500/20" title="Speaker Fidelity Lock">
                  {cfgVal}x
                </span>
                {voice.sample_url && (
                  <span className="text-[10px] text-primary-300 font-mono bg-primary-500/10 px-2 py-0.5 rounded-md border border-primary-500/20">
                    Sample
                  </span>
                )}
              </div>
            </div>
            <p className="text-xs font-medium text-surface-200 mt-1">
              Added {formatDate(voice.created_at)}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="mt-5">
          {confirm ? (
            <div className="flex items-center gap-2.5 animate-fade-in">
              <p className="text-sm font-semibold text-red-400 flex-1">
                Delete this voice?
              </p>
              <button
                onClick={() => setConfirm(false)}
                className="btn-ghost text-sm px-3.5 py-2"
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                className="btn-danger text-sm px-3.5 py-2"
                disabled={deleting}
              >
                {deleting ? <Spinner size="sm" /> : 'Delete'}
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <button
                onClick={() => navigate(`/generate?voice_id=${voice.id}`)}
                className="btn-primary flex-1 justify-center text-xs sm:text-sm py-2 px-3 gap-1.5"
              >
                <SparklesIcon className="w-4 h-4" />
                <span>Use Voice</span>
              </button>

              <button
                onClick={() => setShowTuneModal(true)}
                className="p-2 flex items-center justify-center rounded-xl
                           bg-white/[0.05] border border-white/[0.1] text-surface-200
                           hover:text-violet-400 hover:bg-violet-500/15 hover:border-violet-500/30 transition-all duration-300 shrink-0"
                title="Calibrate pitch, accent lock, and vocal warmth"
              >
                <AdjustmentsHorizontalIcon className="w-5 h-5" />
              </button>

              <button
                onClick={() => setConfirm(true)}
                className="p-2 flex items-center justify-center rounded-xl
                           bg-white/[0.05] border border-white/[0.1]
                           hover:bg-red-500/15 hover:border-red-500/30 transition-all duration-300 shrink-0"
                aria-label="Delete voice"
                title="Delete voice"
              >
                <TrashIcon className="w-5 h-5 text-surface-300 hover:text-red-400 transition-colors" />
              </button>
            </div>
          )}
        </div>
      </div>

      <VoiceTuneModal
        isOpen={showTuneModal}
        onClose={() => setShowTuneModal(false)}
        voice={voice}
      />
    </div>
  )
}
