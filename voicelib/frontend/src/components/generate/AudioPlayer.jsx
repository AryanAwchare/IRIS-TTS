import { useRef, useState, useEffect } from 'react'
import {
  PlayIcon, PauseIcon, ArrowDownTrayIcon, ArrowPathIcon
} from '@heroicons/react/24/solid'
import { Spinner } from '../ui/Spinner'
import { AudioCanvasVisualizer } from '../ui/AudioCanvasVisualizer'
import { generateApi } from '../../api/generate'

/** Visualizer mode cycle */
const MODES = ['bars', 'oscilloscope', 'particles']
const MODE_LABELS = { bars: '≋', oscilloscope: '∿', particles: '✦' }

export function AudioPlayer({ url, voiceName = 'audio', generationId = null, autoPlay = true }) {
  const audioRef = useRef(null)
  const [playing, setPlaying]         = useState(false)
  const [progress, setProgress]       = useState(0)
  const [duration, setDuration]       = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [audioError, setAudioError]   = useState(null)
  const [downloadingFmt, setDownloadingFmt] = useState(null)
  const [vizMode, setVizMode]         = useState(0) // index into MODES

  useEffect(() => {
    setPlaying(false)
    setProgress(0)
    setCurrentTime(0)
    setAudioError(null)

    if (url && audioRef.current) {
      audioRef.current.load()
      if (autoPlay) {
        const timer = setTimeout(() => {
          if (audioRef.current) {
            audioRef.current.play()
              .then(() => setPlaying(true))
              .catch((err) => {
                console.debug('Autoplay waiting for user interaction:', err)
                setPlaying(false)
              })
          }
        }, 150)
        return () => clearTimeout(timer)
      }
    }
  }, [url, autoPlay, generationId])

  const togglePlay = () => {
    if (!audioRef.current || audioError) return
    if (audioRef.current.muted) audioRef.current.muted = false
    audioRef.current.volume = 1.0
    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
    } else {
      const playPromise = audioRef.current.play()
      if (playPromise !== undefined) {
        playPromise
          .then(() => setPlaying(true))
          .catch((err) => {
            console.warn('Audio playback error:', err)
            setPlaying(false)
            setAudioError('Playback failed. Please click play or try generating again.')
          })
      } else {
        setPlaying(true)
      }
    }
  }

  const handleReplay = () => {
    if (!audioRef.current) return
    audioRef.current.currentTime = 0
    audioRef.current.play()
      .then(() => setPlaying(true))
      .catch((err) => console.warn('Replay error:', err))
  }

  const handleTimeUpdate = () => {
    if (!audioRef.current) return
    const p = (audioRef.current.currentTime / (audioRef.current.duration || 1)) * 100
    setProgress(isNaN(p) ? 0 : p)
    setCurrentTime(audioRef.current.currentTime || 0)
  }

  const handleLoadedMetadata = () => {
    setDuration(audioRef.current?.duration || 0)
    setAudioError(null)
  }

  const handleEnded = () => setPlaying(false)

  const handleAudioError = () => {
    setAudioError('Could not load audio. Please try generating again.')
    setPlaying(false)
  }

  const handleSeek = (e) => {
    if (!audioRef.current || audioError) return
    const rect = e.currentTarget.getBoundingClientRect()
    const pct  = (e.clientX - rect.left) / rect.width
    audioRef.current.currentTime = pct * (audioRef.current.duration || 0)
  }

  const fmt = (s) => {
    if (!s || isNaN(s)) return '0:00'
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  const cleanName = (voiceName || 'voice').toLowerCase().replace(/[^a-z0-9]/g, '-')

  const handleDownload = async (format) => {
    if (downloadingFmt) return
    setDownloadingFmt(format)
    try {
      if (generationId) {
        await generateApi.downloadAudio(generationId, format, `${cleanName}-${format}.wav`.replace('.wav', `.${format}`))
      } else if (url) {
        const link = document.createElement('a')
        link.href = url
        link.download = `${cleanName}-speech.${format}`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }
    } catch (err) {
      console.error('Download error:', err)
      window.open(url, '_blank')
    } finally {
      setDownloadingFmt(null)
    }
  }

  const cycleMode = () => setVizMode((m) => (m + 1) % MODES.length)
  const currentMode = MODES[vizMode]

  return (
    <div
      className="card-tactical"
      key={generationId || url}
      style={playing ? { boxShadow: '0 0 20px rgba(229,255,0,0.12)', borderColor: 'rgba(229,255,0,0.2)' } : {}}
    >
      <div className="card-tactical-inner space-y-4">
        <audio
          ref={audioRef}
          src={url}
          crossOrigin="anonymous"
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
          onError={handleAudioError}
          preload="auto"
        />

        {/* Error message */}
        {audioError && (
          <div className="text-crimson text-sm text-center py-2 rounded-xl font-mono"
            style={{ background: 'rgba(255,0,60,0.08)', border: '1px solid rgba(255,0,60,0.2)' }}>
            ⚠ {audioError}
          </div>
        )}

        {/* ── Canvas Visualizer ────────────────────────────────────── */}
        <div className="relative rounded-xl overflow-hidden" style={{ background: 'rgba(0,0,0,0.3)' }}>
          <AudioCanvasVisualizer
            audioRef={audioRef}
            mode={currentMode}
            height={72}
            isPlaying={playing}
          />
          {/* Mode toggle pill */}
          <button
            onClick={cycleMode}
            type="button"
            title={`Visualizer: ${currentMode}`}
            className="absolute top-2 right-2 w-7 h-7 rounded-full flex items-center justify-center font-mono text-[11px] font-bold transition-all hover:scale-110 active:scale-95"
            style={{
              background: 'rgba(229,255,0,0.12)',
              border: '1px solid rgba(229,255,0,0.25)',
              color: '#E5FF00',
            }}
            aria-label={`Switch visualizer mode, current: ${currentMode}`}
          >
            {MODE_LABELS[currentMode]}
          </button>
        </div>

        {/* ── Progress bar — acid yellow ───────────────────────────── */}
        <div
          className="h-1 rounded-full cursor-pointer overflow-hidden relative"
          style={{ background: 'rgba(255,255,255,0.06)' }}
          onClick={handleSeek}
          role="slider"
          aria-label="Audio progress"
          aria-valuenow={progress}
        >
          <div
            className="h-full rounded-full transition-all duration-100 ease-linear"
            style={{
              width: `${progress}%`,
              background: playing
                ? 'linear-gradient(90deg, #E5FF00, #FF003C)'
                : '#E5FF00',
              boxShadow: playing ? '0 0 8px rgba(229,255,0,0.5)' : 'none',
            }}
          />
        </div>

        {/* ── Controls ─────────────────────────────────────────────── */}
        <div className="flex items-center justify-between pt-1 gap-2 flex-wrap sm:flex-nowrap">
          {/* Time */}
          <span className="text-xs font-mono tabular-nums min-w-[75px]"
            style={{ color: 'rgba(226,226,223,0.6)' }}>
            {fmt(currentTime)} / {fmt(duration)}
          </span>

          {/* Play controls */}
          <div className="flex items-center gap-3">
            {/* Replay */}
            <button
              onClick={handleReplay}
              type="button"
              className="w-9 h-9 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 active:scale-95"
              style={{
                background: 'rgba(229,255,0,0.07)',
                border: '1px solid rgba(229,255,0,0.2)',
                color: 'rgba(229,255,0,0.7)',
              }}
              title="Replay from start"
              aria-label="Replay from start"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>

            {/* Main Play / Pause */}
            <button
              onClick={togglePlay}
              type="button"
              className="w-12 h-12 rounded-full flex items-center justify-center transition-all duration-300 ease-spring hover:scale-105 active:scale-95 shrink-0"
              style={{
                background: playing ? '#FF003C' : '#E5FF00',
                boxShadow: playing
                  ? '0 0 20px rgba(255,0,60,0.5)'
                  : '0 0 16px rgba(229,255,0,0.4)',
              }}
              aria-label={playing ? 'Pause' : 'Play'}
            >
              {playing
                ? <PauseIcon className="w-6 h-6 text-white" />
                : <PlayIcon  className="w-6 h-6 text-obsidian ml-0.5" />
              }
            </button>
          </div>

          {/* Download buttons */}
          <div className="inline-flex rounded-xl p-0.5 gap-0.5"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
            <button
              type="button"
              onClick={() => handleDownload('mp3')}
              disabled={downloadingFmt === 'mp3'}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-colors disabled:opacity-50"
              style={{ color: '#E5FF00' }}
              title="Download as MP3"
            >
              {downloadingFmt === 'mp3' ? <Spinner size="xs" /> : <ArrowDownTrayIcon className="w-3.5 h-3.5" />}
              MP3
            </button>
            <button
              type="button"
              onClick={() => handleDownload('wav')}
              disabled={downloadingFmt === 'wav'}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-colors disabled:opacity-50"
              style={{ color: 'rgba(226,226,223,0.7)' }}
              title="Download as WAV"
            >
              {downloadingFmt === 'wav' ? <Spinner size="xs" /> : <ArrowDownTrayIcon className="w-3.5 h-3.5" />}
              WAV
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
