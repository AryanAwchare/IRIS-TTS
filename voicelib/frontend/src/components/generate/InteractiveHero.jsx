import { useRef, useEffect, useState, useCallback } from 'react'

/**
 * InteractiveHero — Twenty One Pilots Cyber-Studio Hero Section
 *
 * Props:
 *   engineStatuses  {Array}   — list of { id, ready, description } from engineStatus API
 *   isPlaying       {boolean} — whether audio is currently generating / playing
 *   onScrollDown    {() => void} — optional callback to scroll to studio section
 */
export default function InteractiveHero({ engineStatuses = [], isPlaying = false, onScrollDown }) {
  const canvasRef = useRef(null)
  const rafRef    = useRef(null)
  const mouseRef  = useRef({ x: 0, y: 0 })
  const particles = useRef([])
  const [gpuStatus, setGpuStatus] = useState({ online: false, label: 'OFFLINE', latency: '—' })

  // Derive GPU status from engineStatuses prop
  useEffect(() => {
    const neural = engineStatuses.find(e => e.id === 'gpt-sovits-v3')
    if (neural) {
      setGpuStatus({
        online: neural.ready,
        label: neural.ready ? 'ONLINE' : 'OFFLINE',
        latency: neural.ready ? '~120ms' : '—',
      })
    }
  }, [engineStatuses])

  // ── Particle canvas setup ──────────────────────────────────────────
  const initParticles = useCallback((w, h) => {
    particles.current = Array.from({ length: 120 }, () => ({
      x:     Math.random() * w,
      y:     Math.random() * h,
      vx:    (Math.random() - 0.5) * 0.5,
      vy:    (Math.random() - 0.5) * 0.5,
      size:  Math.random() * 1.8 + 0.4,
      phase: Math.random() * Math.PI * 2,
      alpha: Math.random() * 0.5 + 0.15,
    }))
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    let w = canvas.offsetWidth
    let h = canvas.offsetHeight
    canvas.width  = w
    canvas.height = h
    initParticles(w, h)
    let time = 0

    const handleMouse = (e) => {
      const rect = canvas.getBoundingClientRect()
      mouseRef.current = { x: e.clientX - rect.left, y: e.clientY - rect.top }
    }
    canvas.addEventListener('mousemove', handleMouse)

    const draw = () => {
      rafRef.current = requestAnimationFrame(draw)
      time += 0.012

      if (canvas.offsetWidth !== w) {
        w = canvas.offsetWidth
        h = canvas.offsetHeight
        canvas.width  = w
        canvas.height = h
        initParticles(w, h)
      }

      ctx.clearRect(0, 0, w, h)

      const acidR = isPlaying ? 255 : 229
      const acidG = isPlaying ? 0   : 255
      const acidB = isPlaying ? 60  : 0

      particles.current.forEach((p) => {
        // Sine-wave drift
        p.x += p.vx + Math.sin(time + p.phase) * 0.3
        p.y += p.vy + Math.cos(time * 0.7 + p.phase) * 0.2

        // Mouse gravity — particles drift toward mouse
        const dx = mouseRef.current.x - p.x
        const dy = mouseRef.current.y - p.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 120) {
          p.x += (dx / dist) * 0.25
          p.y += (dy / dist) * 0.25
        }

        // Wrap edges
        if (p.x < 0) p.x = w
        if (p.x > w) p.x = 0
        if (p.y < 0) p.y = h
        if (p.y > h) p.y = 0

        const alpha = p.alpha + Math.sin(time * 1.5 + p.phase) * 0.15
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(${acidR},${acidG},${acidB},${Math.max(0, Math.min(1, alpha))})`
        ctx.fill()
      })

      // Subtle sine wave across the bottom
      ctx.beginPath()
      ctx.strokeStyle = isPlaying
        ? `rgba(255,0,60,0.3)`
        : `rgba(229,255,0,0.2)`
      ctx.lineWidth = 1
      for (let x = 0; x <= w; x += 2) {
        const y = h * 0.8 + Math.sin(x * 0.02 + time * 2) * 12
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()
    }

    draw()
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      canvas.removeEventListener('mousemove', handleMouse)
    }
  }, [isPlaying, initParticles])

  const neuralEngine  = engineStatuses.find(e => e.id === 'gpt-sovits-v3')
  const pocketEngine  = engineStatuses.find(e => e.id === 'pocket-tts')
  const zonosEngine   = engineStatuses.find(e => e.id === 'zonos-expressive')

  return (
    <section
      className="relative w-full overflow-hidden"
      style={{ minHeight: '380px', background: 'linear-gradient(180deg, #07080A 0%, #0D0E10 100%)' }}
      aria-label="IRIS Studio Hero"
    >
      {/* Scanline overlay */}
      <div
        className="absolute inset-0 pointer-events-none z-0"
        style={{
          background: 'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.06) 3px, rgba(0,0,0,0.06) 4px)',
        }}
      />

      {/* Particle canvas — covers full hero */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-auto z-0"
        style={{ mixBlendMode: 'screen' }}
        aria-hidden="true"
      />

      {/* Content grid */}
      <div className="relative z-10 max-w-6xl mx-auto px-6 pt-28 pb-12 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">

        {/* ── Left — Dithered Singer Art ─────────────────────────────── */}
        <div className="flex justify-center lg:justify-start animate-slide-in-left">
          <div className="relative">
            {/* Glow ring behind image */}
            <div
              className="absolute inset-0 rounded-2xl blur-3xl opacity-20"
              style={{ background: isPlaying ? '#FF003C' : '#E5FF00', transform: 'scale(0.9)' }}
            />
            {/* Dithered singer image */}
            <img
              src="/singer_dither.jpg"
              alt="IRIS — Cyber-Rock Voice Studio"
              className="relative rounded-2xl animate-hero-float scanline"
              style={{
                width: '100%',
                maxWidth: '380px',
                filter: 'contrast(1.1) brightness(0.92)',
                boxShadow: isPlaying
                  ? '0 0 40px rgba(255,0,60,0.35), 0 0 80px rgba(255,0,60,0.12)'
                  : '0 0 40px rgba(229,255,0,0.2), 0 0 80px rgba(229,255,0,0.06)',
                transition: 'box-shadow 0.8s ease',
              }}
              draggable={false}
            />
            {/* TAPE label overlay */}
            <div className="absolute bottom-4 left-4 badge-telemetry-acid">
              // TAPE: CLANCY_v3
            </div>
          </div>
        </div>

        {/* ── Right — HUD Panel ──────────────────────────────────────── */}
        <div className="flex flex-col gap-6 animate-slide-in-right">

          {/* Eyebrow */}
          <div className="eyebrow w-fit">
            // NEURAL VOICE ENGINE
          </div>

          {/* Hero Title */}
          <div>
            <h1
              className="font-display font-bold leading-none tracking-tight text-bone mb-1"
              style={{ fontSize: 'clamp(2rem, 5vw, 3.5rem)' }}
            >
              ZERO-SHOT
              <br />
              <span style={{ color: '#E5FF00', WebkitTextStroke: '1px rgba(229,255,0,0.3)' }}>
                VOICE CLONING
              </span>
            </h1>
            <p className="text-surface-400 text-sm font-mono mt-2 tracking-wide">
              // Powered by Chatterbox · Pocket TTS · Zonos 8D
            </p>
          </div>

          {/* Engine Status Badges */}
          <div className="flex flex-wrap gap-2">
            <span className={`badge-telemetry ${pocketEngine?.ready ? 'badge-telemetry-amber' : ''}`}>
              🧠 Pocket CPU — {pocketEngine?.ready ? 'READY' : 'LOCAL'}
            </span>
            <span className={`badge-telemetry ${neuralEngine?.ready ? 'badge-telemetry-acid' : ''}`}>
              ⚡ Neural GPU — {gpuStatus.label}
            </span>
            <span className={`badge-telemetry ${zonosEngine?.ready ? 'badge-telemetry-crimson' : ''}`}>
              🎭 Zonos 8D — {zonosEngine?.ready ? 'READY' : 'COLD'}
            </span>
          </div>

          {/* Live Telemetry Tags */}
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <span className={gpuStatus.online ? 'gpu-dot-online' : 'gpu-dot-offline'} />
              <span className="font-mono text-[11px] text-surface-400 tracking-wider uppercase">
                GPU {gpuStatus.label} — latency {gpuStatus.latency}
              </span>
            </div>
            <div className="font-mono text-[10px] text-surface-500 tracking-widest uppercase">
              // FREQ: 32KHZ / 16-BIT &nbsp;·&nbsp; FORMANT LOCK: 99.4% &nbsp;·&nbsp; ZERO-SHOT ✓
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-wrap gap-3 mt-2">
            <button
              type="button"
              onClick={onScrollDown}
              className="btn-acid"
              aria-label="Open voice studio"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm-1-11a1 1 0 112 0v3.586l1.707 1.707a1 1 0 01-1.414 1.414l-2-2A1 1 0 019 11V7z" />
              </svg>
              Open Studio
            </button>
            <button
              type="button"
              onClick={onScrollDown}
              className="btn-crimson-ghost"
              aria-label="Configure 8D emotions"
            >
              Configure 8D Emotions →
            </button>
          </div>
        </div>
      </div>

      {/* Bottom fade to page */}
      <div
        className="absolute bottom-0 left-0 right-0 h-16 pointer-events-none"
        style={{ background: 'linear-gradient(to bottom, transparent, #07080A)' }}
      />
    </section>
  )
}
