import { useEffect, useState, useRef, useCallback } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import {
  SparklesIcon, MicrophoneIcon, AdjustmentsHorizontalIcon,
  FaceSmileIcon, CommandLineIcon, CpuChipIcon, BoltIcon
} from '@heroicons/react/24/outline'
import { PlayIcon, ChartBarIcon } from '@heroicons/react/24/solid'
import { useVoiceStore } from '../store/useVoiceStore'
import { generateApi } from '../api/generate'
import { AudioPlayer } from '../components/generate/AudioPlayer'
import { ErrorBanner } from '../components/ui/ErrorBanner'
import { Spinner } from '../components/ui/Spinner'
import VoiceSimilarityModal from '../components/generate/VoiceSimilarityModal'
import InteractiveHero from '../components/generate/InteractiveHero'

const MAX_CHARS = 5000

const PARALINGUISTIC_TAGS = [
  { tag: '[laughter]', label: '😂 Laughter', color: 'bg-amber-500/10 border-amber-500/30 text-amber-300' },
  { tag: '[sigh]', label: '😮‍💨 Sigh', color: 'bg-blue-500/10 border-blue-500/30 text-blue-300' },
  { tag: '[gasp]', label: '😲 Gasp', color: 'bg-purple-500/10 border-purple-500/30 text-purple-300' },
  { tag: '[whisper]', label: '🤫 Whisper', color: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' },
  { tag: '[chuckle]', label: '🤭 Chuckle', color: 'bg-pink-500/10 border-pink-500/30 text-pink-300' },
  { tag: '[clears throat]', label: '🗣️ Throat', color: 'bg-orange-500/10 border-orange-500/30 text-orange-300' },
]

const CARRIER_VOICES = [
  { id: 'auto', label: 'Auto (Best Match)' },
  { id: 'jean', label: 'Jean (Male, Deep)' },
  { id: 'marius', label: 'Marius (Male, Mid)' },
  { id: 'françois', label: 'François (Male, Warm)' },
  { id: 'alba', label: 'Alba (Female, Alto)' },
  { id: 'laura', label: 'Laura (Female, Mid)' },
  { id: 'anna', label: 'Anna (Female, Bright)' },
]

const POCKET_PRESETS = {
  natural_conversational: { label: '🎙️ Natural', carrier_voice: null, morph_strength: 0.85, warmth_gain_db: 0, brightness_gain_db: 0, speed: 1.0 },
  studio_broadcast: { label: '📻 Broadcast', carrier_voice: null, morph_strength: 0.90, warmth_gain_db: 1.5, brightness_gain_db: 1.0, speed: 0.95 },
  crisp_narration: { label: '📖 Narration', carrier_voice: null, morph_strength: 0.80, warmth_gain_db: -0.5, brightness_gain_db: 2.0, speed: 0.90 },
  deep_warmth: { label: '🔥 Deep Warmth', carrier_voice: null, morph_strength: 0.90, warmth_gain_db: 3.5, brightness_gain_db: -1.0, speed: 0.92 },
}

const ENGINE_ICON = {
  'pocket-tts': '🧠',
  'gpt-sovits-v3': '⚡',
  'zonos-expressive': '🎭',
}

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
          {/* Objective Multi-Metric Composite Grade */}
          {gen.composite_grade && (
            <span
              className={`text-[10px] font-bold font-mono px-1.5 py-0.2 rounded border shrink-0 transition-transform ${
                gen.composite_grade === 'A'
                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                  : gen.composite_grade === 'B'
                  ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                  : gen.composite_grade === 'C'
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                  : 'bg-rose-500/20 text-rose-300 border-rose-500/30'
              }`}
              title={`Objective Eval: Grade ${gen.composite_grade} | Sim: ${gen.speaker_similarity != null ? (gen.speaker_similarity * 100).toFixed(0) + '%' : 'N/A'} | WER: ${gen.word_error_rate != null ? (gen.word_error_rate * 100).toFixed(0) + '%' : 'N/A'} | F0 σ: ${gen.prosody_f0_std != null ? gen.prosody_f0_std.toFixed(0) + 'Hz' : 'N/A'}`}
            >
              Grade {gen.composite_grade}
            </span>
          )}
          {gen.eval_status === 'pending' && (
            <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-primary-500/20 text-primary-300 border border-primary-500/30 animate-pulse shrink-0" title="Computing ECAPA-TDNN & Whisper metrics in background...">
              ⚡ Eval...
            </span>
          )}
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


export default function Generate({ onGpuStatus }) {
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

  // Colab External GPU Status (legacy) + Engine Status (new)
  const [colabStatus, setColabStatus]           = useState(null)
  const [engineStatuses, setEngineStatuses]     = useState([])

  // Pocket TTS Fine-Tuning State
  const [carrierVoice, setCarrierVoice]         = useState('auto')
  const [morphStrength, setMorphStrength]       = useState(0.85)
  const [warmthGainDb, setWarmthGainDb]         = useState(0.0)
  const [brightnessGainDb, setBrightnessGainDb] = useState(0.0)
  const [activePreset, setActivePreset]         = useState('natural_conversational')

  // Voice similarity modal
  const [similarityGen, setSimilarityGen]       = useState(null)

  // Studio Control Tabs: 'text' | 'emotions' | 'hyperparams' | 'pocket'
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
  const [userIntensity, setUserIntensity]       = useState(0.50)

  // Restore Pocket TTS settings from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('iris_pocket_tts_settings')
      if (saved) {
        const s = JSON.parse(saved)
        if (s.carrierVoice) setCarrierVoice(s.carrierVoice)
        if (s.morphStrength != null) setMorphStrength(s.morphStrength)
        if (s.warmthGainDb != null) setWarmthGainDb(s.warmthGainDb)
        if (s.brightnessGainDb != null) setBrightnessGainDb(s.brightnessGainDb)
        if (s.activePreset) setActivePreset(s.activePreset)
      }
    } catch { /* ignore */ }
  }, [])

  // Persist Pocket TTS settings to localStorage
  useEffect(() => {
    localStorage.setItem('iris_pocket_tts_settings', JSON.stringify({
      carrierVoice, morphStrength, warmthGainDb, brightnessGainDb, activePreset,
    }))
  }, [carrierVoice, morphStrength, warmthGainDb, brightnessGainDb, activePreset])

  useEffect(() => {
    fetchVoices()

    // Poll engine statuses (replaces old Colab-only poll)
    const checkEngines = () => {
      generateApi.engineStatus()
        .then((statuses) => {
          setEngineStatuses(statuses)
          // Derive legacy colabStatus for backwards compat
          const neural = statuses.find(e => e.id === 'gpt-sovits-v3')
          setColabStatus({ online: neural?.ready || false, gpu: neural?.description })
          // Notify App-level GPU status for Navbar indicator
          if (typeof onGpuStatus === 'function') onGpuStatus(statuses)
        })
        .catch(() => {
          // Fallback to legacy Colab status endpoint
          generateApi.getColabStatus()
            .then((s) => setColabStatus(s))
            .catch(() => setColabStatus({ online: false }))
        })
    }
    checkEngines()
    const interval = setInterval(checkEngines, 15000)
    return () => clearInterval(interval)
  }, [fetchVoices, onGpuStatus])

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

  // Auto-poll evaluation status for pending generations
  useEffect(() => {
    if (!history || history.length === 0) return
    const hasPending = history.some((g) => g.eval_status === 'pending')
    if (!hasPending) return

    const pollInterval = setInterval(() => {
      refreshHistory()
    }, 3000)

    const timer = setTimeout(() => {
      clearInterval(pollInterval)
    }, 25000)

    return () => {
      clearInterval(pollInterval)
      clearTimeout(timer)
    }
  }, [history])

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
    else if (emoId === 'fearful') setEmotions({ happiness: 0, sadness: 0.2, disgust: 0, fear: 0.8, surprise: 0.2, anger: 0, neutral: 0, other: 0 })
    else if (emoId === 'disgusted') setEmotions({ happiness: 0, sadness: 0.1, disgust: 0.8, fear: 0, surprise: 0, anger: 0.3, neutral: 0, other: 0 })
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

  const applyPreset = (presetKey) => {
    const p = POCKET_PRESETS[presetKey]
    if (!p) return
    setActivePreset(presetKey)
    if (p.carrier_voice != null) setCarrierVoice(p.carrier_voice || 'auto')
    setMorphStrength(p.morph_strength)
    setWarmthGainDb(p.warmth_gain_db)
    setBrightnessGainDb(p.brightness_gain_db)
    if (p.speed) setSpeed(p.speed)
  }

  const resetPocketDefaults = () => {
    setCarrierVoice('auto')
    setMorphStrength(0.85)
    setWarmthGainDb(0.0)
    setBrightnessGainDb(0.0)
    setActivePreset('natural_conversational')
    setSpeed(1.0)
  }

  // ── Zonos 8D Emotion Vector handlers ─────────────────────────────────────
  const handleEmotionChange = (emo, value) => {
    setEmotions((prev) => ({
      ...prev,
      [emo]: Math.min(1, Math.max(0, parseFloat(value) || 0)),
    }))
    setSelectedEmotion('custom')
  }

  const normalizeEmotions = () => {
    setEmotions((prev) => {
      const total = Object.values(prev).reduce((s, v) => s + v, 0)
      if (total < 0.001) {
        // All zero — distribute evenly
        const keys = Object.keys(prev)
        const even = parseFloat((1.0 / keys.length).toFixed(3))
        return Object.fromEntries(keys.map((k) => [k, even]))
      }
      return Object.fromEntries(
        Object.entries(prev).map(([k, v]) => [k, parseFloat((v / total).toFixed(3))])
      )
    })
  }
  // ─────────────────────────────────────────────────────────────────────────

  const getEngineStatus = (engineId) => {
    return engineStatuses.find(e => e.id === engineId) || { status: 'unknown', ready: false }
  }

  const handleGenerate = async () => {
    if (!canGenerate) return
    setError(null)
    setLoading(true)
    try {
      const opts = {
        engine,
        emotion: selectedEmotion,
        user_intensity: userIntensity,
        rank,
        top_p: topP,
        temperature,
        text_lang: textLang,
        emotions,
        speed,
        pitch,
      }
      // Include Pocket TTS params only when that engine is selected
      if (engine === 'pocket-tts') {
        opts.carrier_voice = carrierVoice === 'auto' ? null : carrierVoice
        opts.morph_strength = morphStrength
        opts.warmth_gain_db = warmthGainDb
        opts.brightness_gain_db = brightnessGainDb
      }
      const gen = await generateApi.generate(selectedVoiceId, text, opts)
      setLatestResult(gen)
      setActiveResult(gen)
      refreshHistory()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // Scroll-to-studio ref for hero CTA button
  const studioRef = useRef(null)
  const scrollToStudio = useCallback(() => {
    studioRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [])

  // Engine pill styling per engine
  const engineStyle = (eid, isActive) => {
    if (eid === 'pocket-tts') return isActive
      ? { background: 'rgba(255,145,0,0.18)', border: '1px solid rgba(255,145,0,0.4)', color: '#FF9100', boxShadow: '0 0 10px rgba(255,145,0,0.25)' }
      : { border: '1px solid rgba(255,145,0,0.15)', color: 'rgba(255,145,0,0.6)' }
    if (eid === 'gpt-sovits-v3') return isActive
      ? { background: 'rgba(229,255,0,0.15)', border: '1px solid rgba(229,255,0,0.4)', color: '#E5FF00', boxShadow: '0 0 12px rgba(229,255,0,0.25)' }
      : { border: '1px solid rgba(229,255,0,0.15)', color: 'rgba(229,255,0,0.6)' }
    if (eid === 'zonos-expressive') return isActive
      ? { background: 'rgba(255,0,60,0.15)', border: '1px solid rgba(255,0,60,0.4)', color: '#FF003C', boxShadow: '0 0 12px rgba(255,0,60,0.25)' }
      : { border: '1px solid rgba(255,0,60,0.15)', color: 'rgba(255,0,60,0.6)' }
    return {}
  }

  return (
    <div className="min-h-dvh pb-16">
      {/* ── Interactive Hero ──────────────────────────────────────── */}
      <InteractiveHero
        engineStatuses={engineStatuses}
        isPlaying={!!activeResult}
        onScrollDown={scrollToStudio}
      />

      {/* ── Studio Body ──────────────────────────────────────────── */}
      <div ref={studioRef} className="px-4 sm:px-8 max-w-6xl mx-auto pt-10">

        {/* Engine Switcher — Tactical Pills */}
        <div className="mb-6 animate-fade-up flex flex-wrap items-center gap-2">
          <span className="font-mono text-[10px] uppercase tracking-widest text-surface-500 mr-2">// Engine</span>
          {engineStatuses.length > 0 ? engineStatuses.map((e) => {
            const isActive = engine === e.id
            const isReady = e.ready || e.status === 'not_loaded'
            return (
              <button
                key={e.id}
                onClick={() => isReady && setEngine(e.id)}
                disabled={!isReady}
                title={isReady ? e.description : `${e.name} — ${e.status}`}
                className={`px-4 py-2 rounded-xl text-sm font-display font-semibold transition-all flex items-center gap-2 ${
                  !isReady ? 'opacity-40 cursor-not-allowed' : 'hover:opacity-100 cursor-pointer'
                }`}
                style={engineStyle(e.id, isActive)}
              >
                <span>{ENGINE_ICON[e.id] || '🔊'}</span>
                <span>{e.name}</span>
                <span className={`w-1.5 h-1.5 rounded-full ${
                  e.status === 'ready' ? 'bg-acid animate-pulse'
                    : e.status === 'not_loaded' ? 'bg-amber-signal'
                    : 'bg-crimson'
                }`} />
              </button>
            )
          }) : (
            ['pocket-tts', 'gpt-sovits-v3', 'zonos-expressive'].map((eid) => (
              <button
                key={eid}
                onClick={() => setEngine(eid)}
                className="px-4 py-2 rounded-xl text-sm font-display font-semibold transition-all"
                style={engineStyle(eid, engine === eid)}
              >
                {ENGINE_ICON[eid]} {eid === 'pocket-tts' ? 'Pocket CPU' : eid === 'gpt-sovits-v3' ? 'Neural GPU' : 'Zonos 8D'}
              </button>
            ))
          )}
        </div>

      {/* GPU Status Bar — Tactical style */}
      <div className="mb-6 animate-fade-up">
        {colabStatus?.online ? (
          <div className="flex items-center justify-between px-4 py-2.5 rounded-xl font-mono text-xs"
            style={{ background: 'rgba(229,255,0,0.07)', border: '1px solid rgba(229,255,0,0.2)' }}>
            <div className="flex items-center gap-3">
              <span className="gpu-dot-online" />
              <span className="font-semibold text-acid tracking-wider uppercase">Neural GPU Active</span>
              <span className="text-acid/50 hidden sm:inline">
                // {colabStatus.gpu || 'CUDA GPU'} — Chatterbox TTS — Zero-Shot Mode
              </span>
            </div>
            <span className="badge-telemetry-acid">⚡ GPU INFERENCE</span>
          </div>
        ) : (
          <div className="flex items-center justify-between px-4 py-2.5 rounded-xl font-mono text-xs"
            style={{ background: 'rgba(255,145,0,0.06)', border: '1px solid rgba(255,145,0,0.18)' }}>
            <div className="flex items-center gap-3">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ background: '#FF9100' }} />
              <span className="font-semibold tracking-wider uppercase" style={{ color: '#FF9100' }}>Neural GPU Offline</span>
              <span className="text-surface-600 hidden sm:inline">
                // Run Colab notebook to activate Tesla T4 GPU
              </span>
            </div>
            <span className="badge-telemetry-amber">// LOCAL STANDBY</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* LEFT: Controls Studio (3/5) */}
        <div className="lg:col-span-3 space-y-5 animate-fade-up delay-100">
          <ErrorBanner message={error} onDismiss={() => setError(null)} />

          {/* Voice Selector */}
          <div className="card-tactical">
            <div className="card-tactical-inner">
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
          <div className="card-tactical card-tactical-amber">
            <div className="card-tactical-inner space-y-4">
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
                {engine === 'pocket-tts' && (
                  <button
                    onClick={() => setActiveTab('pocket')}
                    className={`flex items-center gap-2 text-sm sm:text-base font-semibold pb-2 border-b-2 transition-all whitespace-nowrap ${
                      activeTab === 'pocket'
                        ? 'border-primary-500 text-white'
                        : 'border-transparent text-surface-200 hover:text-white'
                    }`}
                  >
                    <CpuChipIcon className="w-5 h-5" /> 🧠 Pocket TTS Studio
                  </button>
                )}
                {(engine === 'zonos-expressive' || engine === 'gpt-sovits-v3') && (
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
                )}
                {engine !== 'pocket-tts' && (
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
                )}
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
                        { id: 'fearful', label: '😨 Fearful', color: 'bg-amber-700/20 text-amber-300 border-amber-600/40' },
                        { id: 'disgusted', label: '🤢 Disgusted', color: 'bg-lime-500/20 text-lime-300 border-lime-500/40' },
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

                    {/* Manual Emotion Delivery Intensity Slider */}
                    <div className="mt-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.08]">
                      <div className="flex items-center justify-between mb-1.5">
                        <label className="text-xs font-mono text-surface-300 flex items-center gap-1.5">
                          <span className="text-cyber-yellow">◈</span> Delivery Intensity Override: {(userIntensity * 100).toFixed(0)}%
                        </label>
                        <span className="text-[10px] font-mono text-cyber-cyan font-bold tracking-wider">
                          {userIntensity <= 0.35 ? 'SUBTLE' : userIntensity <= 0.65 ? 'NATURAL' : userIntensity <= 0.85 ? 'EXPRESSIVE' : 'MAXIMUM INTENSITY'}
                        </span>
                      </div>
                      <input
                        type="range"
                        min="0.05"
                        max="1.0"
                        step="0.05"
                        value={userIntensity}
                        onChange={(e) => setUserIntensity(parseFloat(e.target.value))}
                        className="w-full accent-cyber-yellow h-1.5 bg-zinc-800 rounded-lg cursor-pointer"
                      />
                      <div className="flex justify-between mt-1.5 text-[10px] font-mono text-surface-400">
                        <button type="button" onClick={() => setUserIntensity(0.25)} className="hover:text-cyber-yellow transition-colors">Subtle (25%)</button>
                        <button type="button" onClick={() => setUserIntensity(0.50)} className="hover:text-cyber-yellow transition-colors">Natural (50%)</button>
                        <button type="button" onClick={() => setUserIntensity(0.75)} className="hover:text-cyber-yellow transition-colors">Strong (75%)</button>
                        <button type="button" onClick={() => setUserIntensity(1.00)} className="hover:text-cyber-yellow transition-colors">Max (100%)</button>
                      </div>
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

              {/* POCKET TTS STUDIO TAB */}
              {activeTab === 'pocket' && engine === 'pocket-tts' && (
                <div className="space-y-5 animate-fade-in">
                  {/* Presets Row */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="label mb-0">Quick Presets</label>
                      <button
                        type="button"
                        onClick={resetPocketDefaults}
                        className="text-xs font-semibold text-surface-300 hover:text-white bg-white/[0.06] border border-white/[0.1] px-3 py-1 rounded-xl transition-all"
                      >
                        ↺ Reset Defaults
                      </button>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                      {Object.entries(POCKET_PRESETS).map(([key, preset]) => (
                        <button
                          key={key}
                          type="button"
                          onClick={() => applyPreset(key)}
                          className={`py-2.5 px-3 text-sm rounded-xl border transition-all text-center ${
                            activePreset === key
                              ? 'bg-primary-500/20 text-white border-primary-500/40 font-bold shadow-glow-sm'
                              : 'bg-white/[0.04] border-white/[0.1] text-surface-200 hover:bg-white/[0.08] font-medium'
                          }`}
                        >
                          {preset.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Carrier Voice Selector */}
                  <div>
                    <div className="flex items-center gap-2 mb-2">
                      <label className="label mb-0">Carrier Voice</label>
                      <span className="text-[10px] text-surface-300 bg-white/[0.06] px-2 py-0.5 rounded-md" title="The base voice Pocket TTS uses before acoustic morphing is applied">
                        ℹ️ Base voice for morphing
                      </span>
                    </div>
                    <select
                      value={carrierVoice}
                      onChange={(e) => { setCarrierVoice(e.target.value); setActivePreset(null) }}
                      className="input appearance-none cursor-pointer"
                    >
                      {CARRIER_VOICES.map((cv) => (
                        <option key={cv.id} value={cv.id}>{cv.label}</option>
                      ))}
                    </select>
                  </div>

                  {/* Morph Strength Slider */}
                  <div>
                    <div className="flex justify-between text-xs font-semibold mb-1.5">
                      <div className="flex items-center gap-2">
                        <span className="text-surface-200">Morph Strength</span>
                        <span className="text-[10px] text-surface-300 bg-white/[0.06] px-2 py-0.5 rounded-md" title="How strongly the output is morphed toward your reference voice">
                          ℹ️ Voice similarity intensity
                        </span>
                      </div>
                      <span className="text-primary-300 font-mono">{morphStrength.toFixed(2)}</span>
                    </div>
                    <input
                      type="range" min="0" max="1" step="0.05"
                      value={morphStrength}
                      onChange={(e) => { setMorphStrength(parseFloat(e.target.value)); setActivePreset(null) }}
                      className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-primary-500"
                    />
                    <div className="flex justify-between text-[10px] text-surface-400 mt-1">
                      <span>0 — Original carrier</span>
                      <span>1 — Full morph</span>
                    </div>
                  </div>

                  {/* Warmth & Brightness Side-by-Side */}
                  <div className="grid grid-cols-2 gap-5">
                    <div>
                      <div className="flex justify-between text-xs font-semibold mb-1.5">
                        <div className="flex items-center gap-1">
                          <span className="text-surface-200">Warmth</span>
                          <span className="text-[10px] text-surface-400" title="Boost/cut low-mid frequencies (220Hz)">
                            🔥
                          </span>
                        </div>
                        <span className="text-primary-300 font-mono">{warmthGainDb > 0 ? `+${warmthGainDb.toFixed(1)}` : warmthGainDb.toFixed(1)} dB</span>
                      </div>
                      <input
                        type="range" min="-6" max="6" step="0.5"
                        value={warmthGainDb}
                        onChange={(e) => { setWarmthGainDb(parseFloat(e.target.value)); setActivePreset(null) }}
                        className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-amber-500"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between text-xs font-semibold mb-1.5">
                        <div className="flex items-center gap-1">
                          <span className="text-surface-200">Presence</span>
                          <span className="text-[10px] text-surface-400" title="Boost/cut high-frequency clarity (4kHz)">
                            ✨
                          </span>
                        </div>
                        <span className="text-primary-300 font-mono">{brightnessGainDb > 0 ? `+${brightnessGainDb.toFixed(1)}` : brightnessGainDb.toFixed(1)} dB</span>
                      </div>
                      <input
                        type="range" min="-6" max="6" step="0.5"
                        value={brightnessGainDb}
                        onChange={(e) => { setBrightnessGainDb(parseFloat(e.target.value)); setActivePreset(null) }}
                        className="w-full h-2 bg-white/[0.1] rounded-lg appearance-none cursor-pointer accent-cyan-500"
                      />
                    </div>
                  </div>

                  {/* Active Settings Summary */}
                  <div className="bg-white/[0.03] border border-white/[0.08] rounded-xl p-3 text-xs text-surface-300 font-mono">
                    <span className="text-white font-semibold">Active Config:</span>{' '}
                    carrier={carrierVoice}, morph={morphStrength}, warmth={warmthGainDb > 0 ? '+' : ''}{warmthGainDb}dB, presence={brightnessGainDb > 0 ? '+' : ''}{brightnessGainDb}dB
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
                  {engine === 'pocket-tts'
                    ? '🧠 Synthesizing locally via Pocket TTS...'
                    : colabStatus?.online
                      ? '⚡ Synthesizing via Colab External GPU (XTTS-v2)...'
                      : 'Synthesizing Audio...'}
                </span>
              </>
            ) : (
              <>
                <SparklesIcon className="w-6 h-6" />
                <span>
                  {engine === 'pocket-tts'
                    ? '🧠 Generate with Pocket TTS (Local CPU)'
                    : colabStatus?.online
                      ? '⚡ Generate Cloned Speech (External GPU)'
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
                voiceName={(voices.find((v) => v.id === activeResult?.voice_id) || selectedVoice)?.name || 'Cloned Voice'}
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

