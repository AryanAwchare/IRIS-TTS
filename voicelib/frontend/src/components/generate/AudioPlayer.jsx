import { useRef, useState, useEffect } from 'react'
import {
  PlayIcon, PauseIcon, ArrowDownTrayIcon, ArrowPathIcon
} from '@heroicons/react/24/solid'
import { Spinner } from '../ui/Spinner'
import { generateApi } from '../../api/generate'

function WaveformBars({ playing }) {
  const heights = [40, 70, 55, 90, 45, 80, 60, 100, 50, 75, 40, 65, 85]
  return (
    <div className="flex items-center gap-0.5 h-8">
      {heights.map((h, i) => (
        <div
          key={i}
          className="waveform-bar transition-all duration-300"
          style={{
            height: playing ? `${h}%` : '30%',
            animationPlayState: playing ? 'running' : 'paused',
            animationDelay: `${i * 60}ms`,
          }}
        />
      ))}
    </div>
  )
}

export function AudioPlayer({ url, voiceName = 'audio', generationId = null, autoPlay = true }) {
  const audioRef = useRef(null)
  const [playing, setPlaying]         = useState(false)
  const [progress, setProgress]       = useState(0)
  const [duration, setDuration]       = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [audioError, setAudioError]   = useState(null)
  const [downloadingFmt, setDownloadingFmt] = useState(null)

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
            setAudioError('Playback failed. The audio file may still be finalizing.')
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
      // Fallback direct link
      window.open(url, '_blank')
    } finally {
      setDownloadingFmt(null)
    }
  }

  return (
    <div className="card-shell" key={generationId || url}>
      <div className="card-inner space-y-4">
        <audio
          ref={audioRef}
          src={url}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
          onError={handleAudioError}
          preload="auto"
        />

        {/* Error message */}
        {audioError && (
          <div className="text-red-400 text-sm text-center py-2 bg-red-500/10 border border-red-500/20 rounded-xl">
            ⚠️ {audioError}
          </div>
        )}

        {/* Waveform */}
        <div className="flex items-center justify-center py-2">
          <WaveformBars playing={playing} />
        </div>

        {/* Progress bar */}
        <div
          className="h-1.5 bg-white/[0.08] rounded-full cursor-pointer overflow-hidden relative"
          onClick={handleSeek}
          role="slider"
          aria-label="Audio progress"
          aria-valuenow={progress}
        >
          <div
            className="h-full bg-primary-500 rounded-full transition-all duration-100 ease-linear shadow-glow-sm"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between pt-1 gap-2 flex-wrap sm:flex-nowrap">
          <span className="text-sm font-semibold text-surface-200 font-mono tabular-nums min-w-[75px]">
            {fmt(currentTime)} / {fmt(duration)}
          </span>

          <div className="flex items-center gap-3">
            {/* Replay Button */}
            <button
              onClick={handleReplay}
              type="button"
              className="w-9 h-9 rounded-full bg-white/[0.06] hover:bg-white/[0.12] border border-white/[0.1]
                         flex items-center justify-center text-surface-200 hover:text-white
                         transition-all duration-200 active:scale-95 shrink-0"
              title="Hear again from start"
              aria-label="Hear again from start"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>

            {/* Main Play / Pause Button */}
            <button
              onClick={togglePlay}
              type="button"
              className="w-12 h-12 rounded-full bg-primary-500 hover:bg-primary-600
                         flex items-center justify-center shadow-glow
                         transition-all duration-300 ease-spring hover:scale-105 active:scale-95 shrink-0"
              aria-label={playing ? 'Pause' : 'Play'}
            >
              {playing
                ? <PauseIcon className="w-6 h-6 text-white" />
                : <PlayIcon  className="w-6 h-6 text-white ml-0.5" />
              }
            </button>
          </div>

          {/* Download Buttons (MP3 + WAV) */}
          <div className="relative">
            <div className="inline-flex rounded-xl shadow-sm border border-white/[0.1] bg-white/[0.04] p-0.5">
              <button
                type="button"
                onClick={() => handleDownload('mp3')}
                disabled={downloadingFmt === 'mp3'}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs sm:text-sm font-semibold text-primary-300 hover:text-white hover:bg-primary-500/20 rounded-lg transition-colors disabled:opacity-50"
                title="Download as high-compatibility MP3"
              >
                {downloadingFmt === 'mp3' ? (
                  <Spinner size="xs" />
                ) : (
                  <ArrowDownTrayIcon className="w-4 h-4" />
                )}
                <span>MP3</span>
              </button>
              <span className="w-px bg-white/[0.1] my-1"></span>
              <button
                type="button"
                onClick={() => handleDownload('wav')}
                disabled={downloadingFmt === 'wav'}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs sm:text-sm font-semibold text-surface-200 hover:text-white hover:bg-white/[0.08] rounded-lg transition-colors disabled:opacity-50"
                title="Download as lossless WAV"
              >
                {downloadingFmt === 'wav' ? (
                  <Spinner size="xs" />
                ) : (
                  <ArrowDownTrayIcon className="w-4 h-4" />
                )}
                <span>WAV</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

