import { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import {
  SparklesIcon, MicrophoneIcon, ClockIcon, AdjustmentsHorizontalIcon,
  AdjustmentsVerticalIcon, FaceSmileIcon, CommandLineIcon, CpuChipIcon, BoltIcon
} from '@heroicons/react/24/outline'
import { PlayIcon, ArrowDownTrayIcon, CheckIcon, ChartBarIcon } from '@heroicons/react/24/solid'
import { useVoiceStore } from '../store/useVoiceStore'
import { generateApi } from '../api/generate'
import { AudioPlayer } from '../components/generate/AudioPlayer'
import { ErrorBanner } from '../components/ui/ErrorBanner'
import { Spinner } from '../components/ui/Spinner'
import { useAuthStore } from '../store/useAuthStore'
import VoiceSimilarityModal from '../components/generate/VoiceSimilarityModal'

const MAX_CHARS = 5000

const PARALINGUISTIC_TAGS = [
  { tag: '[laughter]', label: '😂 Laughter', color: 'bg-amber-500/10 border-amber-500/30 text-amber-300' },
  { tag: '[sigh]', label: '😮‍💨 Sigh', color: 'bg-blue-500/10 border-blue-500/30 text-blue-300' },
  { tag: '[gasp]', label: '😲 Gasp', color: 'bg-purple-500/10 border-purple-500/30 text-purple-300' },
  { tag: '[whisper]', label: '🤫 Whisper', color: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' },
  { tag: '[chuckle]', label: '🤭 Chuckle', color: 'bg-pink-500/10 border-pink-500/30 text-pink-300' },
  { tag: '[clears throat]', label: '🗣️ Throat', color: 'bg-orange-500/10 border-orange-500/30 text-orange-300' },
]

const ENGINES = [
  { id: 'gpt-sovits-v3', name: 'Neural Voice Cloning', desc: 'Zero-shot deep learning cloning with XTTS-v2 & Colab GPU', badge: 'High Fidelity' },
  { id: 'zonos-expressive', name: 'Zonos TTS Expressive', desc: 'Expressive 8D emotion vectors & acoustic conditioning', badge: 'Expressive' },
  { id: 'pocket-tts', name: 'Standard Fast TTS', desc: 'Fast CPU local synthesis fallback', badge: 'Fast' },
]

function GenerationHistoryItem({ gen, isSelected, isLatest, onPlay, onAnalyze }) {
  const [downloadingFmt, setDownloadingFmt] = useState(null)
  const cleanId = gen.id ? gen.id.slice(0, 8) : 'audio'

  const handleDownload = async (e, fmt) => {
    e.stopPropagation()
    if (downloadingFmt) return
    setDownloadingFmt(fmt)
    try {
      await generateApi.downloadAudio(gen.id, fmt, `voicelib-${cleanId}.${fmt}`)
    } catch (err) {
      console.error('History download failed:', err)
      if (gen.audio_url) window.open(gen.audio_url, '_blank')
    } finally {
      setDownloadingFmt(null)
    }
  }

  return (
    <div
      onClick={() => onPlay(gen)}
      className={`flex items-center gap-3 p-3 rounded-2xl border cursor-pointer transition-all ${
        isSelected
          ? 'bg-primary-500/15 border-primary-500/40 shadow-glow-sm ring-1 ring-primary-500/30'
          : 'bg-white/[0.03] border-white/[0.08] hover:bg-white/[0.06]'
      }`}
    >
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onPlay(gen)
        }}
        className={`w-9 h-9 rounded-full flex items-center justify-center text-white shrink-0 shadow-sm transition-transform hover:scale-105 active:scale-95 ${
          isSelected ? 'bg-primary-500 ring-2 ring-primary-400/50' : 'bg-primary-500/80 hover:bg-primary-500'
        }`}
        title="Load and hear this audio"
      >
        <PlayIcon className="w-4 h-4 ml-0.5" />
      </button>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-xs sm:text-sm font-semibold text-surface-50 truncate">{gen.input_text || 'Generated Speech'}</p>
          {isLatest && (
            <span className="text-[9px] font-bold text-emerald-300 bg-emerald-500/20 border border-emerald-500/30 px-1.5 py-0.2 rounded shrink-0">
              Latest
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-[11px] font-medium text-surface-300">
            {gen.created_at ? new Date(gen.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Recent'}
          </span>
          <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded bg-white/[0.06] text-surface-300">
            {gen.engine === 'pocket-tts' ? 'Pocket' : 'Neural'}
          </span>
          {isSelected && (
            <span className="text-[10px] font-semibold text-primary-400 font-mono">
              ● Active
            </span>
          )}
        </div>
      </div>

      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          onAnalyze(gen)
        }}
        className="p-1.5 rounded-lg hover:bg-violet-500/15 text-violet-400 hover:text-violet-300 transition-colors shrink-0"
        title="Analyze voice similarity"
      >
        <ChartBarIcon className="w-4 h-4" />
      </button>

      {/* MP3 Download */}
      <button
        type="button"
        onClick={(e) => handleDownload(e, 'mp3')}
        disabled={downloadingFmt === 'mp3'}
        className="px-2 py-1 rounded-lg hover:bg-primary-500/20 text-primary-300 hover:text-white font-mono text-[11px] font-semibold transition-colors shrink-0 border border-primary-500/30 bg-primary-500/10 disabled:opacity-50 flex items-center gap-1"
        title="Download MP3"
      >
        {downloadingFmt === 'mp3' ? <Spinner size="xs" /> : 'MP3'}
      </button>

      {/* WAV Download */}
      <button
        type="button"
        onClick={(e) => handleDownload(e, 'wav')}
        disabled={downloadingFmt === 'wav'}
        className="px-2 py-1 rounded-lg hover:bg-white/[0.12] text-surface-200 hover:text-white font-mono text-[11px] font-semibold transition-colors shrink-0 border border-white/[0.1] bg-white/[0.04] disabled:opacity-50 flex items-center gap-1"
        title="Download WAV"
      >
        {downloadingFmt === 'wav' ? <Spinner size="xs" /> : 'WAV'}
      </button>
    </div>
  )
}


export default function Generate() {
  const [params] = useSearchParams()
  const { voices, fetchVoices, isLoading: voicesLoading } = useVoiceStore()

  const [selectedVoiceId, setSelectedVoiceId] = useState(params.get('voice_id') || '')
  const [engine, setEngine]                     = useState('gpt-sovits-v3')
  const [text, setText]                         = useState('')
  const [loading, setLoading]                   = useState(false)
  const [error, setError]                       = useState(null)
  const [activeResult, setActiveResult]         = useState(null)
  const [latestResult, setLatestResult]         = useState(null)
  const [history, setHistory]                   = useState([])

  // Colab External GPU Status
  const [colabStatus, setColabStatus]           = useState(null)

  // Voice similarity modal
  const [similarityGen, setSimilarityGen]       = useState(null)

  // Studio Control Tabs: 'text' | 'emotions' | 'hyperparams'
  const [activeTab, setActiveTab]               = useState('text')
  const [selectedEmotion, setSelectedEmotion]   = useState('auto')

  // GPT-SoVITS / XTTS Hyperparameters
  const [rank, setRank]                         = useState(128)
  const [topP, setTopP]                         = useState(0.8)
  const [temperature, setTemperature]           = useState(0.7)
  const [textLang, setTextLang]                 = useState('en')

  // Zonos 8D Emotion Vector Sliders
  const [emotions, setEmotions]                 = useState({
    happiness: 0.4,
    sadness: 0.0,
    disgust: 0.0,
    fear: 0.0,
    surprise: 0.1,
    anger: 0.0,
    neutral: 0.5,
    other: 0.0,
  })

  // Speech Modifiers (Default 1.0 for 100% natural authentic speaker pacing)
  const [speed, setSpeed]                       = useState(1.0)
  const [pitch, setPitch]                       = useState(0)

  useEffect(() => {
    fetchVoices()
    
    // Poll Colab GPU status
    const checkColab = () => {
      generateApi.getColabStatus()
        .then((s) => setColabStatus(s))
        .catch(() => setColabStatus({ online: false }))
    }
    checkColab()
    const interval = setInterval(checkColab, 15000)
    return () => clearInterval(interval)
  }, [fetchVoices])

  // Sync selected voice from URL query params
  useEffect(() => {
    const vId = params.get('voice_id')
    if (vId) setSelectedVoiceId(vId)
  }, [params])

  // Auto-select first voice once loaded
  useEffect(() => {
    if (!selectedVoiceId && voices.length > 0) {
      setSelectedVoiceId(voices[0].id)
    }
  }, [voices, selectedVoiceId])

  // Fetch generation history
  const refreshHistory = async () => {
    try {
      const hist = await generateApi.history({ limit: 12 })
      setHistory(hist)
    } catch (err) {
      console.error('Failed to load history:', err)
    }
  }

  useEffect(() => {
    refreshHistory()
  }, [])

  const selectedVoice = voices.find((v) => v.id === selectedVoiceId)
  const simVoice = voices.find((v) => v.id === similarityGen?.voice_id) || selectedVoice
  const canGenerate = selectedVoiceId && text.trim().length > 0 && !loading

  const handleEmotionSelect = (emoId) => {
    setSelectedEmotion(emoId)
    // Synchronize 8D emotion vector for Zonos engine
    if (emoId === 'happy') setEmotions({ happiness: 0.8, sadness: 0, disgust: 0, fear: 0, surprise: 0.1, anger: 0, neutral: 0.1, other: 0 })
    else if (emoId === 'excited') setEmotions({ happiness: 0.5, sadness: 0, disgust: 0, fear: 0, surprise: 0.5, anger: 0, neutral: 0, other: 0 })
    else if (emoId === 'sad') setEmotions({ happiness: 0, sadness: 0.8, disgust: 0, fear: 0.1, surprise: 0, anger: 0, neutral: 0.1, other: 0 })
    else if (emoId === 'angry') setEmotions({ happiness: 0, sadness: 0, disgust: 0.2, fear: 0, surprise: 0, anger: 0.8, neutral: 0, other: 0 })
    else if (emoId === 'calm') setEmotions({ happiness: 0.1, sadness: 0, disgust: 0, fear: 0, surprise: 0, anger: 0, neutral: 0.9, other: 0 })
    else if (emoId === 'neutral') setEmotions({ happiness: 0.1, sadness: 0, disgust: 0, fear: 0, surprise: 0, anger: 0, neutral: 0.9, other: 0 })
  }

  const insertTag = (tag) => {
    setText((prev) => prev ? `${prev} ${tag} ` : `${tag} `)
    // Auto-select corresponding emotion mode for convenience
    if (tag === '[laughter]' || tag === '[chuckle]') handleEmotionSelect('happy')
    else if (tag === '[sigh]') handleEmotionSelect('sad')
    else if (tag === '[gasp]') handleEmotionSelect('excited')
    else if (tag === '[whisper]') handleEmotionSelect('calm')
  }

  const handleEnhancePrompt = () => {
    if (!text.trim()) return
    let s = text.trim()
    const conjunctions = ['but', 'however', 'although', 'because', 'which', 'while', 'otherwise']
    conjunctions.forEach((conj) => {
      const regex = new RegExp(`(?<=[a-zA-Z0-9])\\s+(${conj}\\b)`, 'gi')
      s = s.replace(regex, ', $1')
    })
    s = s.replace(/,\s*,+/g, ',')
    s = s.replace(/\.\s*\.+/g, '...')
    s = s.replace(/\?\s*\?+/g, '?')
    s = s.replace(/!\s*!+/g, '!')
    if (s && !'.!?]"\''.includes(s.slice(-1))) {
      s += '.'
    }
    s = s.replace(/\s+/g, ' ').trim()
    setText(s)
  }

  const handleGenerate = async () => {
    if (!canGenerate) return
    setError(null)
    setLoading(true)
    try {
      const gen = await generateApi.generate(selectedVoiceId, text, {
        engine,
        emotion: selectedEmotion,
        rank,
        top_p: topP,
        temperature,
        text_lang: textLang,
        emotions,
        speed,
        pitch
      })
      setLatestResult(gen)
      setActiveResult(gen)
      refreshHistory()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }


  return (
    <div className="min-h-dvh pt-28 pb-16 px-4 sm:px-8 max-w-6xl mx-auto">
      {/* Studio Header */}
      <div className="mb-8 animate-fade-up flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <span className="eyebrow mb-3 inline-block">AI Voice Studio • Neural Voice Cloning</span>
          <h1 className="text-4xl sm:text-5xl font-bold text-white tracking-tight">Voice Cloning & Speech Synthesis</h1>
          <p className="text-base text-surface-200 mt-2 font-medium">
            Fine-tune vocal characteristics, emotions, and perform zero-shot neural voice cloning.
          </p>
        </div>

        {/* Engine Badge Selector */}
        <div className="flex items-center gap-2 bg-white/[0.04] border border-white/[0.1] p-2 rounded-2xl shrink-0">
          {ENGINES.map((e) => (
            <button
              key={e.id}
              onClick={() => setEngine(e.id)}
              className={`px-4 py-2.5 rounded-xl text-sm font-semibold transition-all ${
                engine === e.id
                  ? 'bg-primary-500 text-white shadow-glow-sm'
                  : 'text-surface-200 hover:bg-white/[0.08] hover:text-white'
              }`}
            >
              {e.name}
            </button>
          ))}
        </div>
      </div>

      {/* Colab External GPU Status Pill */}
      <div className="mb-6 animate-fade-up">
        {colabStatus?.online ? (
          <div className="flex items-center justify-between px-4 py-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-sm">
            <div className="flex items-center gap-3">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
              </span>
              <span className="font-semibold text-white">External GPU Active</span>
              <span className="text-emerald-400/80 text-xs hidden sm:inline">
                Connected to {colabStatus.gpu || 'CUDA GPU'} via Colab (XTTS-v2 Neural Engine)
              </span>
            </div>
            <span className="text-xs font-mono bg-emerald-500/20 px-2.5 py-1 rounded-lg text-emerald-200 border border-emerald-500/30">
              ⚡ High-Speed Neural Inference
            </span>
          </div>
        ) : (
          <div className="flex items-center justify-between px-4 py-3 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-sm">
            <div className="flex items-center gap-3">
              <span className="h-3 w-3 rounded-full bg-amber-500 shrink-0"></span>
              <div>
                <span className="font-semibold text-white">Colab External GPU Offline</span>
                <span className="text-amber-200/80 text-xs block sm:inline sm:ml-2">
                  (Run Step 7 in Colab notebook to activate Tesla T4 GPU neural cloning)
                </span>
              </div>
            </div>
            <span className="text-xs font-mono bg-amber-500/20 px-2.5 py-1 rounded-lg text-amber-200 border border-amber-500/30 shrink-0">
              Local Standby
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* LEFT: Controls Studio (3/5) */}
        <div className="lg:col-span-3 space-y-5 animate-fade-up delay-100">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />

          {/* Voice Selector */}
          <div className="card-shell">
            <div className="card-inner">
              <label htmlFor="voice-select" className="label">Cloned Voice Profile</label>

              {voicesLoading ? (
                <div className="flex items-center gap-2 text-sm text-surface-700 py-2">
                  <Spinner size="sm" /> Loading voice models...
                </div>
              ) : voices.length === 0 ? (
                <p className="text-sm text-surface-700">
                  No cloned voices in library.{' '}
                  <Link to="/library" className="text-primary-400 hover:text-primary-300 font-medium">
                    Upload voice sample →
                  </Link>
                </p>
              ) : (
                <select
                  id="voice-select"
                  value={selectedVoiceId}
                  onChange={(e) => setSelectedVoiceId(e.target.value)}
                  className="input appearance-none cursor-pointer"
                >
                  <option value="">Choose a voice profile…</option>
                  {voices.map((v) => (
                    <option key={v.id} value={v.id}>{v.name}</option>
                  ))}
                </select>
              )}

              {selectedVoice && (
                <div className="flex items-center justify-between mt-3 text-xs text-surface-200 bg-white/[0.02] border border-white/[0.05] p-2.5 rounded-xl">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded-lg bg-primary-500/10 border border-primary-500/20 flex items-center justify-center">
                      <MicrophoneIcon className="w-3.5 h-3.5 text-primary-400" />
                    </div>
                    <span>{selectedVoice.name}</span>
                  </div>
                  <span className="text-[10px] text-emerald-400 font-mono">
                    {colabStatus?.online ? '🟢 Neural GPU Ready' : '🟡 Acoustic Ready'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Studio Tabbed Controls */}
          <div className="card-shell">
            <div className="card-inner space-y-4">
              {/* Tab Header */}
              <div className="flex items-center border-b border-white/[0.1] pb-3 gap-6 overflow-x-auto">
                <button
                  onClick={() => setActiveTab('text')}
                  className={`flex items-center gap-2 text-sm sm:text-base font-semibold pb-2 border-b-2 transition-all whitespace-nowrap ${
                    activeTab === 'text'
                      ? 'border-primary-500 text-white'
                      : 'border-transparent text-surface-200 hover:text-white'
                  }`}
                >
                  <CommandLineIcon className="w-5 h-5" /> Script & Paralinguistics
                </button>
                <button
                  onClick={() => setActiveTab('emotions')}
                  className={`flex items-center gap-2 text-sm sm:text-base font-semibold pb-2 border-b-2 transition-all whitespace-nowrap ${
                    activeTab === 'emotions'
                      ? 'border-primary-500 text-white'
                      : 'border-transparent text-surface-200 hover:text-white'
                  }`}
                >
                  <FaceSmileIcon className="w-5 h-5" /> Zonos 8D Emotions
                </button>
                <button
                  onClick={() => setActiveTab('hyperparams')}
                  className={`flex items-center gap-2 text-sm sm:text-base font-semibold pb-2 border-b-2 transition-all whitespace-nowrap ${
                    activeTab === 'hyperparams'
                      ? 'border-primary-500 text-white'
                      : 'border-transparent text-surface-200 hover:text-white'
                  }`}
                >
                  <AdjustmentsHorizontalIcon className="w-5 h-5" /> GPT-SoVITS v3 Rank & Tuning
                </button>
              </div>

              {/* TAB 1: Script & Paralinguistics */}
              {activeTab === 'text' && (
                <div className="space-y-5 animate-fade-in">
                  {/* Emotion Mode Preset Selector */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="label mb-0">Emotion & Delivery Mode</label>
                      <span className="text-xs font-semibold text-primary-300 font-mono">
                        Active: {selectedEmotion.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {[
                        { id: 'auto', label: '⚡ Auto (NLP Sentiment)', color: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40' },
                        { id: 'happy', label: '😊 Happy', color: 'bg-amber-500/20 text-amber-300 border-amber-500/40' },
                        { id: 'excited', label: '🤩 Excited', color: 'bg-purple-500/20 text-purple-300 border-purple-500/40' },
                        { id: 'calm', label: '😌 Calm', color: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40' },
                        { id: 'sad', label: '😢 Sad', color: 'bg-blue-500/20 text-blue-300 border-blue-500/40' },
                        { id: 'angry', label: '😠 Angry', color: 'bg-rose-500/20 text-rose-300 border-rose-500/40' },
                        { id: 'neutral', label: '😐 Neutral', color: 'bg-zinc-800 text-zinc-300 border-zinc-700' },
                      ].map((emo) => {
                        const isCurrent = selectedEmotion === emo.id
                        return (
                          <button
                            key={emo.id}
                            type="button"
                            onClick={() => handleEmotionSelect(emo.id)}
                            className={`text-xs sm:text-sm px-3.5 py-1.5 rounded-xl border font-semibold transition-all ${
                              isCurrent
                                ? `${emo.color} ring-2 ring-primary-400 shadow-glow-sm scale-[1.03]`
                                : 'bg-white/[0.04] border-white/[0.1] text-surface-200 hover:bg-white/[0.08] hover:text-white'
                            }`}
                          >
                            {emo.label}
                          </button>
                        )
                      })}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="label mb-0">Expressiveness Tags (Click to Insert & Set Emotion)</label>
                      <span className="text-xs font-medium text-surface-300">Click to insert tag</span>
                    </div>
                    <div className="flex flex-wrap gap-2.5">
                      {PARALINGUISTIC_TAGS.map((t) => (
                        <button
                          key={t.tag}
                          type="button"
                          onClick={() => insertTag(t.tag)}
                          className={`text-sm px-3.5 py-1.5 rounded-xl border font-medium ${t.color} hover:scale-105 active:scale-95 transition-all`}
                        >
                          {t.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label htmlFor="tts-text" className="label mb-0">Target Speech Text</label>
                      <button
                        type="button"
                        onClick={handleEnhancePrompt}
                        disabled={!text.trim()}
                        className="text-xs font-semibold px-3 py-1.5 rounded-xl border border-primary-500/30 bg-primary-500/15 text-primary-300 hover:text-white hover:bg-primary-500/25 transition-all flex items-center gap-1.5 shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
                        title="Auto-format text with natural human pauses and punctuation"
                      >
                        <SparklesIcon className="w-4 h-4 text-primary-400" />
                        ✨ Enhance Prompt for Human Voice
                      </button>
                    </div>
                    <textarea
                      id="tts-text"
                      className="input resize-none h-44 text-base sm:text-lg leading-relaxed"
                      placeholder="Type your script here... Use [laughter] or [sigh] for expressive delivery."
                      value={text}
                      onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
                      disabled={loading}
                    />
                    <div className="flex items-center justify-between mt-2.5 text-sm text-surface-200 font-medium">
                      <div className="flex items-center gap-2">
                        <span>Language:</span>
                        <select
                          value={textLang}
                          onChange={(e) => setTextLang(e.target.value)}
                          className="bg-white/[0.06] border border-white/[0.12] rounded-xl px-3 py-1 text-sm font-semibold text-white cursor-pointer"
                        >
                          <option value="en">English (en)</option>
                          <option value="zh">Chinese (zh)</option>
                          <option value="ja">Japanese (ja)</option>
                          <option value="ko">Korean (ko)</option>
                        </select>
                      </div>
                      <span className="tabular-nums font-mono">
                        {text.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: Zonos 8D Emotion Sliders */}
              {activeTab === 'emotions' && (
                <div className="space-y-5 animate-fade-in">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-semibold text-white">8-Dimensional Emotion Vector</p>
                      <p className="text-xs font-medium text-surface-200 mt-0.5">Fine-tune vocal emotion intensities</p>
                    </div>
                    <button
                      type="button"
                      onClick={normalizeEmotions}
                      className="text-xs font-semibold text-primary-300 hover:text-white bg-primary-500/15 border border-primary-500/30 px-3 py-1.5 rounded-xl"
                    >
                      Auto-Normalize (1.0 Total)
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                    {Object.keys(emotions).map((emo) => (
                      <div key={emo} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-medium">
                          <span className="capitalize text-surface-200">{emo}</span>
                          <span className="text-primary-300 font-mono font-semibold">{emotions[emo].toFixed(2)}</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={emotions[emo]}
                          onChange={(e) => handleEmotionChange(emo, e.target.value)}
                          className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-primary-500"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 3: GPT-SoVITS v3 Rank & Tuning */}
              {activeTab === 'hyperparams' && (
                <div className="space-y-5 animate-fade-in">
                  <div className="bg-primary-500/15 border border-primary-500/30 rounded-2xl p-4 text-sm text-surface-200 flex items-start gap-3">
                    <SparklesIcon className="w-6 h-6 text-primary-400 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold text-white">Jarod's Benchmark Tip:</span>
                      <p className="text-xs text-surface-200 mt-1 leading-relaxed">
                        In benchmark testing, <strong>Rank 128</strong> produces optimal voice fidelity without overfitting. Excessive epochs are unnecessary for clean prompt samples.
                      </p>
                    </div>
                  </div>

                  <div>
                    <label className="label">LoRA / Model Rank Setting</label>
                    <div className="grid grid-cols-4 gap-2.5">
                      {[32, 64, 128, 256].map((r) => (
                        <button
                          key={r}
                          type="button"
                          onClick={() => setRank(r)}
                          className={`py-2.5 text-sm rounded-xl border transition-all ${
                            rank === r
                              ? 'bg-primary-500 text-white border-primary-500 font-bold shadow-glow-sm'
                              : 'bg-white/[0.05] border-white/[0.1] text-surface-200 hover:bg-white/[0.1] font-semibold'
                          }`}
                        >
                          Rank {r} {r === 128 ? '⭐' : ''}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <div className="flex justify-between text-xs font-semibold mb-1.5">
                        <span className="text-surface-200">Sampling Temp</span>
                        <span className="text-primary-300 font-mono">{temperature}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="1.5"
                        step="0.05"
                        value={temperature}
                        onChange={(e) => setTemperature(parseFloat(e.target.value))}
                        className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-primary-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-xs font-semibold mb-1.5">
                        <span className="text-surface-200">Top-P Nucleus</span>
                        <span className="text-primary-300 font-mono">{topP}</span>
                      </div>
                      <input
                        type="range"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                        value={topP}
                        onChange={(e) => setTopP(parseFloat(e.target.value))}
                        className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-primary-500"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-5 pt-2 border-t border-white/[0.08]">
                    <div>
                      <div className="flex justify-between text-xs font-semibold mb-1.5">
                        <span className="text-surface-200">Speaking Pace / Speed</span>
                        <span className="text-primary-300 font-mono">{speed.toFixed(2)}x {speed === 1.0 ? '(Native)' : ''}</span>
                      </div>
                      <input
                        type="range"
                        min="0.5"
                        max="2.0"
                        step="0.05"
                        value={speed}
                        onChange={(e) => setSpeed(parseFloat(e.target.value))}
                        className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-primary-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-xs font-semibold mb-1.5">
                        <span className="text-surface-200">Pitch Offset (Semitones)</span>
                        <span className="text-primary-300 font-mono">{pitch > 0 ? `+${pitch}` : pitch} st</span>
                      </div>
                      <input
                        type="range"
                        min="-6"
                        max="6"
                        step="0.5"
                        value={pitch}
                        onChange={(e) => setPitch(parseFloat(e.target.value))}
                        className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-primary-500"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Generate Button */}
          <button
            id="generate-btn"
            onClick={handleGenerate}
            disabled={!canGenerate}
            className="btn-primary w-full justify-center py-4 text-base sm:text-lg font-bold shadow-glow"
          >
            {loading ? (
              <>
                <Spinner size="sm" />
                <span>
                  {colabStatus?.online
                    ? '⚡ Synthesizing via Colab External GPU (XTTS-v2)...'
                    : 'Synthesizing Audio...'}
                </span>
              </>
            ) : (
              <>
                <SparklesIcon className="w-6 h-6" />
                <span>
                  {colabStatus?.online
                    ? 'Generate Cloned Speech (External GPU)'
                    : 'Generate Cloned Speech (Local)'}
                </span>
              </>
            )}
          </button>
        </div>

        {/* RIGHT: Audio Output & History (2/5) */}
        <div className="lg:col-span-2 space-y-5 animate-fade-up delay-200">
          {/* Audio Output Player or Live Synthesis Loader */}
          {loading ? (
            <div className="card-shell animate-fade-in border-primary-500/40 shadow-glow-sm">
              <div className="card-inner flex flex-col items-center justify-center py-12 text-center space-y-4">
                <div className="flex items-center gap-1.5 h-12">
                  {Array.from({ length: 14 }).map((_, i) => (
                    <div
                      key={i}
                      className="waveform-bar"
                      style={{
                        height: `${[35, 65, 90, 45, 75, 100, 85, 55, 95, 70, 40, 80, 60, 90][i]}%`,
                        animationPlayState: 'running',
                        animationDelay: `${i * 75}ms`,
                      }}
                    />
                  ))}
                </div>
                <div>
                  <h4 className="text-base font-bold text-white flex items-center justify-center gap-2">
                    <Spinner size="sm" />
                    {colabStatus?.online
                      ? '⚡ Synthesizing via Colab GPU (XTTS-v2)...'
                      : 'Synthesizing Audio Output...'}
                  </h4>
                  <p className="text-xs text-surface-200 mt-1">
                    Generating zero-shot acoustic waveform &amp; neural embeddings
                  </p>
                </div>
              </div>
            </div>
          ) : activeResult ? (
            <div className="animate-fade-in">
              {/* History vs Latest Badge & Quick Switch */}
              {latestResult && activeResult.id !== latestResult.id ? (
                <div className="flex items-center justify-between bg-amber-500/10 border border-amber-500/25 px-3 py-2 rounded-xl mb-3">
                  <span className="text-xs font-semibold text-amber-300 flex items-center gap-1.5">
                    📜 Viewing Studio History Item
                  </span>
                  <button
                    type="button"
                    onClick={() => setActiveResult(latestResult)}
                    className="text-xs font-bold text-primary-300 hover:text-white bg-primary-500/20 hover:bg-primary-500/30 px-2.5 py-1 rounded-lg border border-primary-500/30 transition-all flex items-center gap-1"
                  >
                    <BoltIcon className="w-3.5 h-3.5" />
                    Switch to Latest Output
                  </button>
                </div>
              ) : (
                <div className="flex items-center justify-between mb-3">
                  <p className="label mb-0">Speech Output</p>
                  {latestResult && activeResult.id === latestResult.id && (
                    <span className="text-[11px] font-semibold text-emerald-400 font-mono bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/20">
                      ⚡ Latest Synthesis
                    </span>
                  )}
                </div>
              )}

              <AudioPlayer
                key={activeResult.id || activeResult.audio_url}
                url={activeResult.audio_url}
                voiceName={activeVoice?.name}
                generationId={activeResult.id}
                autoPlay={true}
              />

              {/* Analyze Similarity button */}
              <button
                type="button"
                onClick={() => setSimilarityGen(activeResult)}
                className="mt-3 w-full flex items-center justify-center gap-2.5 py-3 rounded-2xl border border-violet-500/30 bg-violet-500/10 text-violet-300 hover:bg-violet-500/20 hover:text-white font-semibold text-sm transition-all shadow-sm active:scale-98"
              >
                <ChartBarIcon className="w-5 h-5" />
                Analyse Voice Similarity
              </button>
            </div>
          ) : (
            <div className="card-shell">
              <div className="card-inner flex flex-col items-center justify-center py-14 text-center">
                <div className="flex items-center gap-1 mb-4 h-12">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <div
                      key={i}
                      className="waveform-bar opacity-20"
                      style={{
                        height: `${[40, 70, 90, 50, 80, 100, 60, 40, 80, 50][i]}%`,
                        animationPlayState: 'paused',
                      }}
                    />
                  ))}
                </div>
                <p className="text-xs text-surface-200 font-medium">
                  Your synthesized audio waveform will display here
                </p>
              </div>
            </div>
          )}

          {/* Generation History */}
          {history.length > 0 && (
            <div className="card-shell animate-fade-up delay-300">
              <div className="card-inner space-y-3">
                <div className="flex items-center justify-between">
                  <p className="label mb-0">Studio History</p>
                  <span className="text-[11px] text-surface-300 font-mono">{history.length} recordings</span>
                </div>
                <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
                  {history.map((g) => (
                    <GenerationHistoryItem
                      key={g.id}
                      gen={g}
                      isSelected={activeResult?.id === g.id}
                      isLatest={latestResult?.id === g.id}
                      onPlay={(item) => setActiveResult(item)}
                      onAnalyze={(item) => setSimilarityGen(item)}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Voice Similarity Modal */}
      {similarityGen && (
        <VoiceSimilarityModal
          generation={similarityGen}
          voiceName={simVoice?.name || ''}
          referenceSampleUrl={simVoice?.sample_url}
          onClose={() => setSimilarityGen(null)}
        />
      )}
    </div>
  )
}

