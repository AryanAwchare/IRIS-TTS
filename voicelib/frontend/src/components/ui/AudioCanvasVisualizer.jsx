import { useRef, useEffect, useCallback } from 'react'

/**
 * AudioCanvasVisualizer — Web Audio API frequency visualizer.
 *
 * Props:
 *   audioRef  {React.RefObject<HTMLAudioElement>}  — the <audio> element to analyze
 *   mode      {'bars' | 'oscilloscope' | 'particles'}
 *   height    {number}   — canvas height in px (default 80)
 *   isPlaying {boolean}  — whether audio is currently playing
 */
export function AudioCanvasVisualizer({ audioRef, mode = 'bars', height = 80, isPlaying = false }) {
  const canvasRef     = useRef(null)
  const ctxRef        = useRef(null)   // AudioContext
  const analyserRef   = useRef(null)
  const sourceRef     = useRef(null)
  const rafRef        = useRef(null)
  const particlesRef  = useRef([])

  // Lazy-init AudioContext on first use (browser autoplay policy)
  const initAudio = useCallback(() => {
    if (ctxRef.current || !audioRef?.current) return
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      if (!AudioCtx) return
      const ctx      = new AudioCtx()
      const analyser = ctx.createAnalyser()
      analyser.fftSize           = 128
      analyser.smoothingTimeConstant = 0.82
      const source   = ctx.createMediaElementSource(audioRef.current)
      source.connect(analyser)
      analyser.connect(ctx.destination)
      ctxRef.current    = ctx
      analyserRef.current = analyser
      sourceRef.current   = source
    } catch (err) {
      // Cross-origin or already connected — fall back to idle animation
      console.debug('AudioCanvasVisualizer: could not connect analyser', err)
    }
  }, [audioRef])

  // Init particles for particles mode
  const initParticles = useCallback((w, h) => {
    particlesRef.current = Array.from({ length: 80 }, (_, i) => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      size: Math.random() * 2 + 0.5,
      phase: Math.random() * Math.PI * 2,
      alpha: Math.random() * 0.6 + 0.2,
    }))
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx2d = canvas.getContext('2d')
    let w = canvas.offsetWidth
    let h = canvas.offsetHeight
    canvas.width  = w
    canvas.height = h

    if (mode === 'particles') initParticles(w, h)

    let time = 0

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      time += 0.016

      // Resize check
      if (canvas.offsetWidth !== w) {
        w = canvas.offsetWidth
        h = canvas.offsetHeight
        canvas.width  = w
        canvas.height = h
        if (mode === 'particles') initParticles(w, h)
      }

      ctx2d.clearRect(0, 0, w, h)

      const analyser = analyserRef.current
      const acidColor   = isPlaying ? '#FF003C' : '#E5FF00'
      const acidColorDim = isPlaying ? 'rgba(255,0,60,' : 'rgba(229,255,0,'

      if (mode === 'bars') {
        // ── Frequency bar chart ───────────────────────────────────────
        const bufLen = analyser ? analyser.frequencyBinCount : 32
        const data   = new Uint8Array(bufLen)
        if (analyser) analyser.getByteFrequencyData(data)

        const barW = (w / bufLen) * 1.2
        let x = 0

        for (let i = 0; i < bufLen; i++) {
          const val    = analyser ? data[i] : (Math.sin(time * 2 + i * 0.5) * 0.5 + 0.5) * 60
          const barH   = (val / 255) * h
          const ratio  = val / 255
          // Gradient: acid-yellow → crimson by intensity
          const r = Math.round(229 + (255 - 229) * ratio * (isPlaying ? 1 : 0))
          const g = Math.round(255 * (1 - ratio * 0.8))
          const b = Math.round(isPlaying ? 60 * ratio : 0)
          ctx2d.fillStyle = `rgba(${r},${g},${b},0.85)`
          const rx = 3
          ctx2d.beginPath()
          ctx2d.roundRect(x, h - barH, barW - 2, barH, [rx, rx, 0, 0])
          ctx2d.fill()
          x += barW
        }

      } else if (mode === 'oscilloscope') {
        // ── Phosphor waveform line ────────────────────────────────────
        const bufLen = analyser ? analyser.fftSize : 128
        const data   = new Uint8Array(bufLen)
        if (analyser) analyser.getByteTimeDomainData(data)

        ctx2d.strokeStyle = isPlaying ? '#FF003C' : '#00FF8C'
        ctx2d.lineWidth   = 1.5
        ctx2d.shadowColor = isPlaying ? '#FF003C' : '#00FF8C'
        ctx2d.shadowBlur  = 6
        ctx2d.beginPath()
        const sliceW = w / bufLen
        let xPos = 0
        for (let i = 0; i < bufLen; i++) {
          const v = analyser ? (data[i] / 128.0 - 1) : Math.sin(time * 3 + i * 0.15) * 0.25
          const y = (v * h * 0.35) + h / 2
          i === 0 ? ctx2d.moveTo(xPos, y) : ctx2d.lineTo(xPos, y)
          xPos += sliceW
        }
        ctx2d.stroke()
        ctx2d.shadowBlur = 0

      } else if (mode === 'particles') {
        // ── Radial particle field ─────────────────────────────────────
        let avgFreq = 0
        if (analyser) {
          const data = new Uint8Array(analyser.frequencyBinCount)
          analyser.getByteFrequencyData(data)
          avgFreq = data.reduce((a, b) => a + b, 0) / data.length / 255
        } else {
          avgFreq = (Math.sin(time) * 0.5 + 0.5) * 0.3
        }

        particlesRef.current.forEach((p) => {
          p.x += p.vx * (1 + avgFreq * 3)
          p.y += p.vy * (1 + avgFreq * 3)
          p.alpha = 0.2 + avgFreq * 0.8 + Math.sin(time + p.phase) * 0.2

          if (p.x < 0) p.x = w
          if (p.x > w) p.x = 0
          if (p.y < 0) p.y = h
          if (p.y > h) p.y = 0

          ctx2d.beginPath()
          ctx2d.arc(p.x, p.y, p.size * (1 + avgFreq), 0, Math.PI * 2)
          ctx2d.fillStyle = `${acidColorDim}${Math.min(p.alpha, 1)})`
          ctx2d.fill()
        })
      }
    }

    draw()
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [mode, isPlaying, initParticles])

  // Init audio context when audio element is ready
  useEffect(() => {
    if (!audioRef?.current) return
    const el = audioRef.current
    const onPlay = () => {
      initAudio()
      if (ctxRef.current?.state === 'suspended') ctxRef.current.resume()
    }
    el.addEventListener('play', onPlay)
    return () => el.removeEventListener('play', onPlay)
  }, [audioRef, initAudio])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      if (sourceRef.current) { try { sourceRef.current.disconnect() } catch {} }
      if (ctxRef.current) { try { ctxRef.current.close() } catch {} }
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: `${height}px`, display: 'block' }}
      aria-hidden="true"
    />
  )
}
