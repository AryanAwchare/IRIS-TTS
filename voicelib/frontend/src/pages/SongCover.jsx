import { useEffect, useState, useRef } from 'react'
import {
  MusicalNoteIcon, ArrowUpTrayIcon, MagnifyingGlassIcon,
  FolderIcon, PlayIcon, ArrowDownTrayIcon, SparklesIcon,
  CheckCircleIcon, ExclamationTriangleIcon, BoltIcon, AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline'
import { useVoiceStore } from '../store/useVoiceStore'
import { songCoverApi } from '../api/songCover'
import { AudioPlayer } from '../components/generate/AudioPlayer'
import { Spinner } from '../components/ui/Spinner'
import { ErrorBanner } from '../components/ui/ErrorBanner'

const INPUT_TABS = [
  { id: 'UPLOAD', label: '📁 Direct Upload', desc: 'WAV / MP3 / FLAC file up to 5 min' },
  { id: 'SEARCH', label: '🌐 Search & Fetch URL', desc: 'YouTube, SoundCloud or audio URL' },
  { id: 'LIBRARY', label: '💾 Library & Curated Demos', desc: 'Instant 1-click test with pre-separated stems' },
]

export default function SongCover() {
  const { voices, fetchVoices } = useVoiceStore()
  const [activeTab, setActiveTab] = useState('UPLOAD')

  // Form State
  const [selectedVoiceId, setSelectedVoiceId] = useState('')
  const selectedVoice = voices.find(v => v.id === selectedVoiceId)
  const [songFile, setSongFile] = useState(null)
  const [sourceUrl, setSourceUrl] = useState('')
  const [selectedLibraryHash, setSelectedLibraryHash] = useState('')
  const [title, setTitle] = useState('')
  const [pitchShift, setPitchShift] = useState(0)
  const [indexRate, setIndexRate] = useState(0.75)
  const [protectVoiceless, setProtectVoiceless] = useState(0.33)
  const [previewOnly, setPreviewOnly] = useState(false)
  const [tosConfirmed, setTosConfirmed] = useState(true)

  // Catalogs & History
  const [curatedSongs, setCuratedSongs] = useState([])
  const [personalLibrary, setPersonalLibrary] = useState([])
  const [coverHistory, setCoverHistory] = useState([])

  // Audio Preview & Stem Player State
  const [previewTrack, setPreviewTrack] = useState(null)
  const [activeStem, setActiveStem] = useState('mix') // 'mix' | 'vocals' | 'instrumental' | 'original'

  // Pipeline Execution State
  const [activeJob, setActiveJob] = useState(null)
  const [activeCoverResult, setActiveCoverResult] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)
  const pollTimerRef = useRef(null)

  // Update preview when song file changes
  useEffect(() => {
    if (songFile) {
      const blobUrl = URL.createObjectURL(songFile)
      setPreviewTrack({
        title: songFile.name,
        subtitle: `Local Upload (${(songFile.size / (1024 * 1024)).toFixed(2)} MB)`,
        url: blobUrl,
        type: 'upload',
      })
      return () => URL.revokeObjectURL(blobUrl)
    }
  }, [songFile])

  // Update preview when activeCoverResult is ready
  useEffect(() => {
    if (activeCoverResult) {
      setPreviewTrack({
        title: activeCoverResult.title || 'Song Cover Master',
        subtitle: `Target Voice: ${selectedVoice?.name || 'Cloned Voice'}`,
        url: activeCoverResult.audio_url || activeCoverResult.preview_url,
        vocals_url: activeCoverResult.vocals_url || activeCoverResult.converted_vocals_url,
        instrumental_url: activeCoverResult.instrumental_url,
        original_url: activeCoverResult.original_audio_url,
        type: 'result',
      })
      setActiveStem('mix')
    }
  }, [activeCoverResult, selectedVoice])

  const handleSelectCurated = (c) => {
    setSelectedLibraryHash(c.song_hash)
    setTitle(c.title)
    if (c.preview_audio_url || c.audio_url) {
      setPreviewTrack({
        title: c.title,
        subtitle: `Demo by ${c.artist} (${c.duration}s)`,
        url: c.preview_audio_url || c.audio_url,
        type: 'curated',
      })
    }
  }

  const handleSelectHistory = (item) => {
    setActiveCoverResult(item)
    setPreviewTrack({
      title: item.title,
      subtitle: `${item.status.toUpperCase()} ${item.is_preview ? '· Preview' : '· Master'}`,
      url: item.audio_url || item.preview_url,
      vocals_url: item.vocals_url,
      instrumental_url: item.instrumental_url,
      type: 'history',
    })
    setActiveStem('mix')
  }

  const getCurrentAudioUrl = () => {
    if (!previewTrack) return null
    if (activeStem === 'vocals' && previewTrack.vocals_url) return previewTrack.vocals_url
    if (activeStem === 'instrumental' && previewTrack.instrumental_url) return previewTrack.instrumental_url
    if (activeStem === 'original' && previewTrack.original_url) return previewTrack.original_url
    return previewTrack.url
  }

  // Load voices and catalogs on mount
  useEffect(() => {
    fetchVoices()
    loadCatalogs()
    loadHistory()
  }, [])

  useEffect(() => {
    if (voices.length > 0 && !selectedVoiceId) {
      setSelectedVoiceId(voices[0].id)
    }
  }, [voices])

  const loadCatalogs = async () => {
    try {
      const curated = await songCoverApi.getCuratedCatalog()
      setCuratedSongs(curated || [])
    } catch (err) {
      console.debug('Could not load curated catalog:', err)
    }
    try {
      const lib = await songCoverApi.getLibrarySongs()
      setPersonalLibrary(lib || [])
    } catch (err) {
      console.debug('Could not load personal library:', err)
    }
  }

  const loadHistory = async () => {
    try {
      const list = await songCoverApi.list({ limit: 15 })
      setCoverHistory(list || [])
    } catch (err) {
      console.debug('Could not load history:', err)
    }
  }

  // Polling for active job progress
  useEffect(() => {
    if (!activeJob || activeJob.status === 'completed' || activeJob.status === 'failed') {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      return
    }

    pollTimerRef.current = setInterval(async () => {
      try {
        const stat = await songCoverApi.getStatus(activeJob.id)
        setActiveJob(stat)
        if (stat.status === 'completed') {
          clearInterval(pollTimerRef.current)
          const fullDetail = await songCoverApi.getDetail(activeJob.id)
          setActiveCoverResult(fullDetail)
          loadHistory()
          loadCatalogs()
        } else if (stat.status === 'failed') {
          clearInterval(pollTimerRef.current)
          setError(stat.error_message || 'Conversion pipeline encountered an error.')
        }
      } catch (pollErr) {
        console.debug('Polling notice:', pollErr)
      }
    }, 1500)

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
    }
  }, [activeJob?.id, activeJob?.status])

  const handleStartConversion = async (e) => {
    e.preventDefault()
    setError(null)
    if (!selectedVoiceId) {
      setError('Please select a target voice from your library.')
      return
    }

    const formData = new FormData()
    formData.append('voice_id', selectedVoiceId)
    formData.append('source_type', activeTab)
    formData.append('pitch_shift', pitchShift)
    formData.append('index_rate', indexRate)
    formData.append('protect_voiceless', protectVoiceless)
    formData.append('preview_only', previewOnly)
    formData.append('tos_confirmed', tosConfirmed)

    if (title.trim()) formData.append('title', title.trim())

    if (activeTab === 'UPLOAD') {
      if (!songFile) {
        setError('Please drop or select an audio file (max 5 minutes).')
        return
      }
      formData.append('file', songFile)
    } else if (activeTab === 'SEARCH') {
      if (!sourceUrl.trim()) {
        setError('Please enter a valid song streaming URL.')
        return
      }
      formData.append('source_url', sourceUrl.trim())
    } else if (activeTab === 'LIBRARY') {
      if (!selectedLibraryHash) {
        setError('Please select a demo or personal song from the library.')
        return
      }
      formData.append('library_song_hash', selectedLibraryHash)
    }

    setSubmitting(true)
    try {
      const job = await songCoverApi.create(formData)
      setActiveJob(job)
      setActiveCoverResult(null)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to submit song cover request.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-dvh pt-24 pb-16 px-4 sm:px-6 max-w-7xl mx-auto">
      
      {/* Cyberpunk Top Header */}
      <div className="mb-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyber-yellow/10 border-l-2 border-cyber-yellow text-cyber-yellow font-mono text-xs font-bold tracking-widest uppercase mb-2">
          <span>// SVC-RVC V2 NEURAL ENGINE</span>
          <span className="text-white/40">|</span>
          <span className="text-surface-300">楽曲音声クローンスタジオ</span>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
          <h1 className="font-display font-black text-3xl sm:text-4xl text-white tracking-tight">
            SONG VOICE <span className="text-cyber-yellow neon-text-yellow">CONVERSION</span>
          </h1>
          <span className="font-mono text-xs text-cyber-cyan tracking-wider">
            DEMUCS V4 · RMVPE F0 · 5-MIN MAX CAP · ZERO BLEED
          </span>
        </div>
      </div>

      {error && <div className="mb-6"><ErrorBanner message={error} onDismiss={() => setError(null)} /></div>}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

        {/* ── Left Column: Controls & Ingestion (7 cols) ── */}
        <div className="lg:col-span-7 flex flex-col gap-6">

          {/* Ingestion Source Tabs */}
          <div className="bg-cyber-panel border border-cyber-cyan/30 rounded-2xl p-1 shadow-glow-cyan/10">
            <div className="grid grid-cols-3 gap-1">
              {INPUT_TABS.map((tab) => {
                const isActive = activeTab === tab.id
                return (
                  <button
                    key={tab.id}
                    type="button"
                    onClick={() => { setActiveTab(tab.id); setError(null) }}
                    className={`py-3 px-2 rounded-xl text-center transition-all ${
                      isActive
                        ? 'bg-cyber-yellow text-black font-bold shadow-glow-yellow/40'
                        : 'text-surface-300 hover:text-white hover:bg-white/[0.04]'
                    }`}
                  >
                    <div className="text-xs sm:text-sm font-display font-semibold">{tab.label}</div>
                    <div className={`text-[10px] truncate font-mono mt-0.5 ${isActive ? 'text-black/70' : 'text-surface-400'}`}>
                      {tab.desc}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Tab Content Panels */}
          <div className="bg-cyber-panel border border-white/[0.1] rounded-2xl p-6 relative cyber-clip">
            <div className="h-1 w-full cyber-hazard-tape absolute top-0 left-0 right-0" />

            {/* TAB 1: UPLOAD */}
            {activeTab === 'UPLOAD' && (
              <div className="space-y-4">
                <label className="text-xs font-mono font-bold text-cyber-yellow uppercase tracking-wider block">
                  1. Select Song File (Max 5 Minutes)
                </label>
                <div
                  className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
                    songFile
                      ? 'border-cyber-cyan bg-cyber-cyan/10'
                      : 'border-white/20 hover:border-cyber-yellow bg-black/40'
                  }`}
                  onClick={() => document.getElementById('song-file-input').click()}
                >
                  <input
                    id="song-file-input"
                    type="file"
                    accept="audio/*,.mp3,.wav,.flac,.m4a,.ogg"
                    className="hidden"
                    onChange={(e) => {
                      if (e.target.files?.[0]) setSongFile(e.target.files[0])
                    }}
                  />
                  <ArrowUpTrayIcon className="w-10 h-10 mx-auto text-cyber-cyan mb-2" />
                  {songFile ? (
                    <div>
                      <p className="font-mono font-bold text-sm text-white">{songFile.name}</p>
                      <p className="font-mono text-xs text-cyber-yellow mt-1">
                        {(songFile.size / (1024 * 1024)).toFixed(2)} MB · Click to change
                      </p>
                    </div>
                  ) : (
                    <div>
                      <p className="font-mono text-sm text-surface-200">Drag & drop or click to upload</p>
                      <p className="font-mono text-[11px] text-surface-400 mt-1">MP3, WAV, FLAC, M4A up to 100MB</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TAB 2: SEARCH */}
            {activeTab === 'SEARCH' && (
              <div className="space-y-4">
                <label className="text-xs font-mono font-bold text-cyber-yellow uppercase tracking-wider block">
                  1. Paste Remote Audio / Video URL
                </label>
                <div className="relative">
                  <MagnifyingGlassIcon className="w-5 h-5 absolute left-3.5 top-3.5 text-cyber-cyan" />
                  <input
                    type="url"
                    value={sourceUrl}
                    onChange={(e) => setSourceUrl(e.target.value)}
                    placeholder="https://www.youtube.com/watch?v=... or audio URL"
                    className="w-full bg-black/60 border border-cyber-cyan/30 rounded-xl pl-11 pr-4 py-3 text-sm font-mono text-white placeholder-surface-500 focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan"
                  />
                </div>
                <div className="p-3 bg-cyber-cyan/10 border border-cyber-cyan/20 rounded-xl text-[11px] font-mono text-surface-300">
                  <span className="text-cyber-cyan font-bold">⚡ AUTO PRE-CHECK:</span> Remote audio will be validated for the strict 5-minute cap before separation begins.
                </div>
                <label className="flex items-center gap-2 cursor-pointer pt-1">
                  <input
                    type="checkbox"
                    checked={tosConfirmed}
                    onChange={(e) => setTosConfirmed(e.target.checked)}
                    className="accent-cyber-yellow w-4 h-4 rounded"
                  />
                  <span className="text-xs font-mono text-surface-300">
                    I affirm this track is for personal, non-commercial evaluation.
                  </span>
                </label>
              </div>
            )}

            {/* TAB 3: LIBRARY */}
            {activeTab === 'LIBRARY' && (
              <div className="space-y-5">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-mono font-bold text-cyber-yellow uppercase tracking-wider block">
                      ⚡ Curated Demo Catalog (0ms Separation Wait)
                    </label>
                    <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/20 px-2 py-0.5 rounded">
                      PRE-SEPARATED STEMS ✓
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                    {curatedSongs.map((c) => {
                      const isSel = selectedLibraryHash === c.song_hash
                      return (
                        <div
                          key={c.song_hash}
                          onClick={() => handleSelectCurated(c)}
                          className={`p-3 rounded-xl border cursor-pointer transition-all relative group ${
                            isSel
                              ? 'bg-cyber-yellow/20 border-cyber-yellow shadow-glow-yellow/30'
                              : 'bg-black/40 border-white/10 hover:border-cyber-cyan/50'
                          }`}
                        >
                          <div className="flex items-start justify-between gap-1">
                            <div className="min-w-0 flex-1">
                              <div className="font-display font-bold text-xs text-white truncate">{c.title}</div>
                              <div className="font-mono text-[10px] text-cyber-cyan mt-0.5 truncate">{c.artist}</div>
                            </div>
                            <span className="w-6 h-6 rounded-full bg-cyber-yellow/20 group-hover:bg-cyber-yellow text-cyber-yellow group-hover:text-black flex items-center justify-center shrink-0 transition-colors">
                              <PlayIcon className="w-3 h-3 ml-0.5" />
                            </span>
                          </div>
                          <div className="flex justify-between font-mono text-[10px] text-surface-400 mt-2">
                            <span>{c.genre}</span>
                            <span>{c.duration}s · Click to play</span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {personalLibrary.length > 0 && (
                  <div>
                    <label className="text-xs font-mono font-bold text-cyber-cyan uppercase tracking-wider block mb-2">
                      📁 Your Previously Separated Songs (Instant Re-conversion)
                    </label>
                    <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                      {personalLibrary.map((item) => {
                        const isSel = selectedLibraryHash === item.song_hash
                        return (
                          <div
                            key={item.id}
                            onClick={() => {
                              setSelectedLibraryHash(item.song_hash)
                              setTitle(item.title)
                              if (item.audio_url || item.vocals_url) {
                                setPreviewTrack({
                                  title: item.title,
                                  subtitle: 'Separated Stems Available',
                                  url: item.audio_url || item.final_mix_url,
                                  vocals_url: item.vocals_url,
                                  instrumental_url: item.instrumental_url,
                                  type: 'library',
                                })
                              }
                            }}
                            className={`p-2.5 rounded-lg border text-xs font-mono flex items-center justify-between cursor-pointer ${
                              isSel ? 'bg-cyber-cyan/20 border-cyber-cyan text-white' : 'bg-black/30 border-white/10 text-surface-300 hover:bg-white/[0.04]'
                            }`}
                          >
                            <span className="truncate">{item.title}</span>
                            <div className="flex items-center gap-2 shrink-0">
                              <span className="text-[10px] text-cyber-yellow">STOCKED ✓</span>
                              <PlayIcon className="w-3.5 h-3.5 text-cyber-cyan" />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Custom Cover Title Input */}
            <div className="mt-5 pt-4 border-t border-white/[0.08]">
              <label className="text-xs font-mono text-surface-300 block mb-1.5">
                Song Title (Optional)
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. My Favorite Song"
                className="w-full bg-black/60 border border-white/20 rounded-xl px-4 py-2 text-sm font-mono text-white focus:outline-none focus:border-cyber-yellow"
              />
            </div>
          </div>

          {/* Performance & SVC Fine-Tuning Matrix */}
          <div className="bg-cyber-panel border border-white/[0.1] rounded-2xl p-6 space-y-5">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
              <span className="font-mono text-xs font-bold uppercase tracking-wider text-cyber-yellow flex items-center gap-2">
                <AdjustmentsHorizontalIcon className="w-4 h-4" /> 2. Conversion & Transposition Matrix
              </span>
              <span className="text-[10px] font-mono text-surface-400">ALGORITHM: RVC-V2</span>
            </div>

            {/* Voice Picker */}
            <div>
              <label className="text-xs font-mono text-surface-300 block mb-1.5">
                Target Cloned Voice
              </label>
              <select
                value={selectedVoiceId}
                onChange={(e) => setSelectedVoiceId(e.target.value)}
                className="w-full bg-black/80 border border-white/20 rounded-xl px-4 py-2.5 text-sm font-mono text-white focus:outline-none focus:border-cyber-yellow"
              >
                {voices.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.name} {v.singing_capable ? '★ [SINGING CAPABLE]' : '[SPEECH BASE]'}
                  </option>
                ))}
              </select>
            </div>

            {/* Pitch Shift (Semitones) */}
            <div>
              <div className="flex justify-between items-center text-xs font-mono mb-1.5">
                <span className="text-surface-300">Pitch Shift (Semitones): {pitchShift > 0 ? `+${pitchShift}` : pitchShift} st</span>
                <span className="text-[10px] text-cyber-cyan">
                  {pitchShift === 0 ? 'Original Pitch' : pitchShift === 12 ? '+1 Octave (Male->Female)' : pitchShift === -12 ? '-1 Octave (Female->Male)' : 'Transposed'}
                </span>
              </div>
              <input
                type="range"
                min="-24"
                max="24"
                step="1"
                value={pitchShift}
                onChange={(e) => setPitchShift(parseInt(e.target.value, 10))}
                className="w-full accent-cyber-yellow h-1.5 bg-zinc-800 rounded cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-surface-400 mt-1">
                <button type="button" onClick={() => setPitchShift(-12)} className="hover:text-cyber-yellow">-12 (Male)</button>
                <button type="button" onClick={() => setPitchShift(0)} className="hover:text-cyber-yellow">0 (Neutral)</button>
                <button type="button" onClick={() => setPitchShift(12)} className="hover:text-cyber-yellow">+12 (Female)</button>
              </div>
            </div>

            {/* RVC Index & Protection Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-surface-300">Index Rate: {indexRate}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={indexRate}
                  onChange={(e) => setIndexRate(parseFloat(e.target.value))}
                  className="w-full accent-cyber-cyan h-1.5 bg-zinc-800 rounded cursor-pointer"
                />
                <span className="text-[9px] font-mono text-surface-400">Timbre index search weight</span>
              </div>

              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-surface-300">Voiceless Protect: {protectVoiceless}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.5"
                  step="0.01"
                  value={protectVoiceless}
                  onChange={(e) => setProtectVoiceless(parseFloat(e.target.value))}
                  className="w-full accent-cyber-cyan h-1.5 bg-zinc-800 rounded cursor-pointer"
                />
                <span className="text-[9px] font-mono text-surface-400">Protects breath & consonants</span>
              </div>
            </div>

            {/* Fast 20s Preview Toggle */}
            <div className="p-3 rounded-xl bg-cyber-yellow/10 border border-cyber-yellow/30 flex items-center justify-between">
              <div>
                <span className="font-mono text-xs font-bold text-cyber-yellow block">
                  ⚡ Render 20-30s Fast Preview Only
                </span>
                <span className="font-mono text-[10px] text-surface-300 block">
                  Test the vocals in seconds before rendering the full song.
                </span>
              </div>
              <input
                type="checkbox"
                checked={previewOnly}
                onChange={(e) => setPreviewOnly(e.target.checked)}
                className="accent-cyber-yellow w-5 h-5 rounded cursor-pointer"
              />
            </div>

            {/* Submit Button */}
            <button
              type="button"
              onClick={handleStartConversion}
              disabled={submitting || (activeJob && activeJob.status !== 'completed' && activeJob.status !== 'failed')}
              className="w-full py-3.5 bg-cyber-yellow hover:bg-white text-black font-display font-black text-sm tracking-widest uppercase transition-all shadow-glow-yellow/50 cyber-clip-btn disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {submitting ? (
                <>
                  <Spinner size="sm" />
                  <span>INITIALIZING PIPELINE...</span>
                </>
              ) : (
                <>
                  <BoltIcon className="w-4 h-4" />
                  <span>{previewOnly ? 'RENDER 20S PREVIEW' : 'START FULL SONG CONVERSION'}</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── Right Column: Pipeline Monitor & Audio Output (5 cols) ── */}
        <div className="lg:col-span-5 flex flex-col gap-6">

          {/* Active Job Progress Console */}
          <div className="bg-cyber-panel border-2 border-cyber-cyan/30 rounded-2xl p-6 shadow-glow-cyan/20 cyber-clip relative">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3 mb-4">
              <span className="font-mono text-xs font-bold text-cyber-cyan tracking-wider flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyber-cyan animate-pulse" />
                NEURAL CONVERSION HUD
              </span>
              <span className="font-mono text-[10px] uppercase text-surface-400">
                STATUS: {activeJob?.status || 'IDLE'}
              </span>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between font-mono text-xs">
                <span className="text-surface-300 truncate">
                  {activeJob?.stage_message || 'Waiting for song submission...'}
                </span>
                <span className="text-cyber-yellow font-bold">
                  {activeJob ? `${activeJob.progress.toFixed(0)}%` : '0%'}
                </span>
              </div>
              <div className="w-full h-2.5 bg-black/60 rounded-full overflow-hidden border border-white/10">
                <div
                  className="h-full bg-gradient-to-r from-cyber-cyan via-cyber-yellow to-cyber-neon transition-all duration-500 rounded-full"
                  style={{ width: `${activeJob ? activeJob.progress : 0}%` }}
                />
              </div>
            </div>

            {/* Stage Steps Indicator */}
            <div className="mt-5 space-y-2 font-mono text-[11px]">
              {[
                { stage: 'separating', label: '[01] Stem Separation (Demucs v4)' },
                { stage: 'analyzing', label: '[02] Vocal F0 & Register Profiling' },
                { stage: 'converting', label: '[03] Chunked RVC Timbre Conversion' },
                { stage: 'mixing', label: '[04] Vocal Mastering & Ducking (-1.5dB)' },
                { stage: 'completed', label: '[05] Render Complete' },
              ].map((st) => {
                const isCurrent = activeJob?.status === st.stage
                const isPassed = activeJob?.status === 'completed' || (
                  (st.stage === 'separating' && ['analyzing', 'converting', 'mixing'].includes(activeJob?.status)) ||
                  (st.stage === 'analyzing' && ['converting', 'mixing'].includes(activeJob?.status)) ||
                  (st.stage === 'converting' && ['mixing'].includes(activeJob?.status))
                )
                return (
                  <div
                    key={st.stage}
                    className={`flex items-center justify-between p-2 rounded border ${
                      isCurrent
                        ? 'bg-cyber-yellow/15 border-cyber-yellow text-cyber-yellow'
                        : isPassed
                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                        : 'bg-black/30 border-white/[0.05] text-surface-500'
                    }`}
                  >
                    <span>{st.label}</span>
                    <span>{isCurrent ? 'PROCESSING...' : isPassed ? 'DONE ✓' : 'QUEUED'}</span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Dedicated Studio Audio Monitor & Stem Player Card */}
          {previewTrack && (
            <div className="bg-cyber-panel border-2 border-cyber-yellow/40 rounded-2xl p-6 shadow-glow-yellow/20 space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-cyber-yellow uppercase tracking-wider flex items-center gap-2">
                  <MusicalNoteIcon className="w-5 h-5 text-cyber-cyan animate-pulse" />
                  STUDIO AUDIO MONITOR
                </span>
                <span className="font-mono text-[10px] text-surface-400 uppercase">
                  {previewTrack.type}
                </span>
              </div>

              <div>
                <h3 className="font-display font-bold text-lg text-white truncate">
                  {previewTrack.title}
                </h3>
                <p className="font-mono text-xs text-cyber-cyan">
                  {previewTrack.subtitle}
                </p>
              </div>

              {/* Stem Switching Pills (When stems are available) */}
              {(previewTrack.vocals_url || previewTrack.instrumental_url || previewTrack.original_url) && (
                <div className="flex flex-wrap gap-1.5 p-1 bg-black/50 border border-white/10 rounded-xl">
                  <button
                    type="button"
                    onClick={() => setActiveStem('mix')}
                    className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-mono font-bold transition-all ${
                      activeStem === 'mix' ? 'bg-cyber-yellow text-black' : 'text-surface-300 hover:text-white'
                    }`}
                  >
                    🎧 Master Mix
                  </button>
                  {previewTrack.vocals_url && (
                    <button
                      type="button"
                      onClick={() => setActiveStem('vocals')}
                      className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-mono font-bold transition-all ${
                        activeStem === 'vocals' ? 'bg-cyber-cyan text-black' : 'text-surface-300 hover:text-white'
                      }`}
                    >
                      🎤 Vocals Only
                    </button>
                  )}
                  {previewTrack.instrumental_url && (
                    <button
                      type="button"
                      onClick={() => setActiveStem('instrumental')}
                      className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-mono font-bold transition-all ${
                        activeStem === 'instrumental' ? 'bg-cyber-neon text-white' : 'text-surface-300 hover:text-white'
                      }`}
                    >
                      🎹 Instrumental
                    </button>
                  )}
                  {previewTrack.original_url && (
                    <button
                      type="button"
                      onClick={() => setActiveStem('original')}
                      className={`flex-1 py-1.5 px-2 rounded-lg text-xs font-mono font-bold transition-all ${
                        activeStem === 'original' ? 'bg-amber-400 text-black' : 'text-surface-300 hover:text-white'
                      }`}
                    >
                      📻 Original
                    </button>
                  )}
                </div>
              )}

              {/* Master / Preview Audio Player */}
              {getCurrentAudioUrl() && (
                <AudioPlayer
                  url={getCurrentAudioUrl()}
                  voiceName={selectedVoice?.name || 'Cloned Voice'}
                  autoPlay={true}
                />
              )}

              {/* Download Master Button */}
              {previewTrack.url && previewTrack.type === 'result' && (
                <a
                  href={previewTrack.url}
                  download="IRIS_Song_Master.wav"
                  target="_blank"
                  rel="noreferrer"
                  className="w-full py-2.5 bg-cyber-panel border border-cyber-cyan text-cyber-cyan hover:bg-cyber-cyan hover:text-black font-mono text-xs font-bold uppercase transition-all rounded-xl flex items-center justify-center gap-2"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" /> Download Master WAV
                </a>
              )}
            </div>
          )}

          {/* Past Song Covers History List */}
          <div className="bg-cyber-panel border border-white/[0.08] rounded-2xl p-5 space-y-3">
            <h3 className="font-mono text-xs font-bold text-surface-300 uppercase tracking-wider flex items-center justify-between">
              <span>Past Song Covers</span>
              <span className="text-surface-400">{coverHistory.length}</span>
            </h3>

            {coverHistory.length === 0 ? (
              <p className="font-mono text-xs text-surface-500 py-4 text-center">
                No previous song covers generated yet.
              </p>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                {coverHistory.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => handleSelectHistory(item)}
                    className="p-3 bg-black/40 border border-white/10 hover:border-cyber-yellow/40 rounded-xl flex items-center justify-between cursor-pointer transition-colors"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="text-xs font-display font-bold text-white truncate">{item.title}</div>
                      <div className="text-[10px] font-mono text-surface-400 mt-0.5">
                        {item.status.toUpperCase()} {item.is_preview && '· [PREVIEW]'}
                      </div>
                    </div>
                    {item.audio_url && (
                      <button
                        type="button"
                        className="w-8 h-8 rounded-full bg-cyber-yellow/20 hover:bg-cyber-yellow text-cyber-yellow hover:text-black flex items-center justify-center transition-colors ml-2 shrink-0"
                        onClick={(e) => { e.stopPropagation(); handleSelectHistory(item) }}
                      >
                        <PlayIcon className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
