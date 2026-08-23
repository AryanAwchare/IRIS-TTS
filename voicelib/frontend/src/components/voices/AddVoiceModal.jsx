import { useState } from 'react'
import { Modal } from '../ui/Modal'
import { Dropzone } from '../ui/Dropzone'
import { ErrorBanner } from '../ui/ErrorBanner'
import { Spinner } from '../ui/Spinner'
import { voicesApi } from '../../api/voices'
import { useVoiceStore } from '../../store/useVoiceStore'
import { ShieldCheckIcon } from '@heroicons/react/24/outline'

const MAX_NAME = 100

export function AddVoiceModal({ isOpen, onClose }) {
  const addVoice = useVoiceStore((s) => s.addVoice)

  const [file, setFile]             = useState(null)
  const [name, setName]             = useState('')
  const [promptText, setPromptText] = useState('')
  const [promptLang, setPromptLang] = useState('en')
  const [consent, setConsent]       = useState(false)
  const [state, setState]           = useState('idle') // idle | uploading | success
  const [error, setError]           = useState(null)

  const canSubmit = file && name.trim().length > 0 && consent && state === 'idle'

  const handleClose = () => {
    if (state === 'uploading') return
    setFile(null); setName(''); setPromptText(''); setPromptLang('en'); setConsent(false); setState('idle'); setError(null)
    onClose()
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!canSubmit) return
    setError(null)
    setState('uploading')

    const fd = new FormData()
    fd.append('file', file)
    fd.append('name', name.trim())
    fd.append('prompt_text', promptText.trim())
    fd.append('prompt_lang', promptLang)
    fd.append('consent_confirmed', 'true')

    try {
      const voice = await voicesApi.create(fd)
      addVoice(voice)
      setState('success')
      setTimeout(handleClose, 800)
    } catch (err) {
      setError(err.message)
      setState('idle')
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title="Add a Voice" size="md">
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Error */}
        <ErrorBanner message={error} onDismiss={() => setError(null)} />

        {/* Dropzone */}
        <div>
          <label className="label">Voice Sample</label>
          <Dropzone onFileAccepted={setFile} />
        </div>

        {/* Voice name */}
        <div>
          <label htmlFor="voice-name" className="label">Voice Name</label>
          <input
            id="voice-name"
            type="text"
            className="input"
            placeholder="e.g. Alex, Jarod, Morgan"
            value={name}
            onChange={(e) => setName(e.target.value.slice(0, MAX_NAME))}
            maxLength={MAX_NAME}
            disabled={state === 'uploading'}
          />
        </div>

        {/* Prompt Text / Transcript (GPT-SoVITS Requirement) */}
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="prompt-text" className="label mb-0">Audio Transcript (Prompt Text)</label>
            <span className="text-xs text-primary-300 font-mono font-semibold">GPT-SoVITS v3 Feature</span>
          </div>
          <p className="text-sm font-medium text-surface-200 mb-2.5">
            Exact words spoken in the audio sample for high-precision voice alignment.
          </p>
          <textarea
            id="prompt-text"
            className="input resize-none h-24 text-sm font-mono"
            placeholder="e.g. Hello, this is my sample audio recording for voice cloning."
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            disabled={state === 'uploading'}
          />
        </div>

        {/* Language Selection */}
        <div>
          <label htmlFor="prompt-lang" className="label">Primary Language</label>
          <select
            id="prompt-lang"
            className="input appearance-none cursor-pointer text-base font-medium"
            value={promptLang}
            onChange={(e) => setPromptLang(e.target.value)}
            disabled={state === 'uploading'}
          >
            <option value="en">English (en)</option>
            <option value="zh">Chinese (zh)</option>
            <option value="ja">Japanese (ja)</option>
            <option value="ko">Korean (ko)</option>
            <option value="yue">Cantonese (yue)</option>
          </select>
        </div>

        {/* Consent */}
        <label className="flex items-start gap-3.5 cursor-pointer group p-1">
          <div className="relative mt-1 shrink-0">
            <input
              id="consent"
              type="checkbox"
              className="sr-only"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              disabled={state === 'uploading'}
            />
            <div
              className={`w-6 h-6 rounded-lg border-2 flex items-center justify-center transition-all duration-300 ease-spring
                ${consent
                  ? 'bg-primary-500 border-primary-500 shadow-glow-sm'
                  : 'bg-transparent border-white/30 group-hover:border-white/60'
                }`}
            >
              {consent && (
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
            </div>
          </div>
          <div>
            <p className="text-base font-semibold text-white flex items-center gap-2">
              <ShieldCheckIcon className="w-5 h-5 text-primary-400 shrink-0" />
              I confirm explicit consent
            </p>
            <p className="text-sm font-medium text-surface-200 mt-1 leading-relaxed">
              I confirm that I have this person's explicit permission to clone their voice.
            </p>
          </div>
        </label>

        {/* Submit */}
        <button
          type="submit"
          disabled={!canSubmit}
          className="btn-primary w-full justify-center py-4 text-base font-bold shadow-glow"
        >
          {state === 'uploading' ? (
            <>
              <Spinner size="sm" />
              <span>Cloning Voice with GPT-SoVITS...</span>
            </>
          ) : state === 'success' ? (
            '✓ Voice Cloned Successfully!'
          ) : (
            'Add Voice to Library'
          )}
        </button>
      </form>
    </Modal>
  )
}
