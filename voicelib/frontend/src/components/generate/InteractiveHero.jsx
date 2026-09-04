import { useRef, useEffect, useState, useCallback } from 'react'

/**
 * InteractiveHero — Cyberpunk 2077 Anime-Tech Neural Audio Console
 *
 * Props:
 *   engineStatuses  {Array}   — list of { id, ready, description } from engineStatus API
 *   isPlaying       {boolean} — whether audio is currently generating / playing
 *   onScrollDown    {() => void} — optional callback to scroll to studio section
 */
export default function InteractiveHero({ engineStatuses = [], isPlaying = false, onScrollDown }) {
  const canvasRef = useRef(null)
  const hudCanvasRef = useRef(null)
  const rafRef = useRef(null)
  const hudRafRef = useRef(null)
  const mouseRef = useRef({ x: 0, y: 0 })
  const particles = useRef([])
  const [pulseActive, setPulseActive] = useState(false)
  const [gpuStatus, setGpuStatus] = useState({ online: false, label: 'OFFLINE', latency: '—' })

  // Derive GPU status from engineStatuses prop
  useEffect(() => {
    const neural = engineStatuses.find(e => e.id === 'gpt-sovits-v3')
    if (neural) {
      setGpuStatus({
        online: neural.ready,
        label: neural.ready ? 'ONLINE' : 'OFFLINE',
        latency: neural.ready ? '~38ms' : '—',
      })
    }
  }, [engineStatuses])

  // ── Background Cyber Grid & Particles ──────────────────────────────
  const initParticles = useCallback((w, h) => {
    particles.current = Array.from({ length: 90 }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      size: Math.random() * 2.0 + 0.5,
      phase: Math.random() * Math.PI * 2,
      color: Math.random() > 0.6 ? '#FCEE0A' : '#00F0FF',
    }))
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    let w = canvas.offsetWidth
    let h = canvas.offsetHeight
    canvas.width = w
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
      time += 0.015

      if (canvas.offsetWidth !== w) {
        w = canvas.offsetWidth
        h = canvas.offsetHeight
        canvas.width = w
        canvas.height = h
        initParticles(w, h)
      }

      ctx.clearRect(0, 0, w, h)

      // Cyberpunk scanning laser grid
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.04)'
      ctx.lineWidth = 1
      const gridSize = 40
      for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, h)
        ctx.stroke()
      }
      for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      // Particles
      particles.current.forEach((p) => {
        p.x += p.vx
        p.y += p.vy + Math.sin(time + p.phase) * 0.2

        if (p.x < 0) p.x = w
        if (p.x > w) p.x = 0
        if (p.y < 0) p.y = h
        if (p.y > h) p.y = 0

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.globalAlpha = 0.35 + Math.sin(time * 2 + p.phase) * 0.2
        ctx.fill()
        ctx.globalAlpha = 1.0
      })

      // Dynamic reactive neon audio beam
      ctx.beginPath()
      ctx.strokeStyle = isPlaying ? 'rgba(255, 0, 60, 0.5)' : 'rgba(0, 240, 255, 0.3)'
      ctx.lineWidth = 1.5
      for (let x = 0; x <= w; x += 3) {
        const amp = isPlaying ? 24 : 8
        const y = h * 0.88 + Math.sin(x * 0.02 + time * 3) * amp
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

  // ── HUD Waveform & Equalizer Simulator ──────────────────────────────
  useEffect(() => {
    const canvas = hudCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let time = 0

    const drawHud = () => {
      hudRafRef.current = requestAnimationFrame(drawHud)
      time += 0.035

      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      // Cyberpunk Equalizer Columns
      const numBars = 32
      const barWidth = w / numBars - 2
      for (let i = 0; i < numBars; i++) {
        const offset = i * 0.25
        const baseH = (Math.sin(time * 2.5 + offset) * 0.5 + 0.5) * (h * 0.65) + 6
        const extra = pulseActive ? Math.random() * (h * 0.3) : 0
        const barH = Math.min(h - 4, baseH + extra)

        const x = i * (barWidth + 2) + 2
        const y = h - barH - 2

        // Color gradient: Yellow -> Cyan -> Magenta
        const grad = ctx.createLinearGradient(x, y, x, h)
        if (i < 10) {
          grad.addColorStop(0, '#00F0FF')
          grad.addColorStop(1, 'rgba(0, 240, 255, 0.15)')
        } else if (i < 24) {
          grad.addColorStop(0, '#FCEE0A')
          grad.addColorStop(1, 'rgba(252, 238, 10, 0.15)')
        } else {
          grad.addColorStop(0, '#FF003C')
          grad.addColorStop(1, 'rgba(255, 0, 60, 0.15)')
        }

        ctx.fillStyle = grad
        ctx.fillRect(x, y, barWidth, barH)

        // Peak cap
        ctx.fillStyle = '#FFFFFF'
        ctx.fillRect(x, y - 2, barWidth, 2)
      }

      // F0 Pitch Oscilloscope Trace
      ctx.beginPath()
      ctx.strokeStyle = '#00F0FF'
      ctx.lineWidth = 2
      ctx.shadowColor = '#00F0FF'
      ctx.shadowBlur = 8
      for (let x = 0; x < w; x += 4) {
        const freq = isPlaying ? 0.05 : 0.025
        const y = h * 0.45 + Math.sin(x * freq + time * 3) * (h * 0.25) * Math.cos(time * 0.5)
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()
      ctx.shadowBlur = 0
    }

    drawHud()
    return () => {
      if (hudRafRef.current) cancelAnimationFrame(hudRafRef.current)
    }
  }, [isPlaying, pulseActive])

  const triggerNeuralPing = () => {
    setPulseActive(true)
    setTimeout(() => setPulseActive(false), 1200)
  }

  const neuralEngine = engineStatuses.find(e => e.id === 'gpt-sovits-v3')
  const pocketEngine = engineStatuses.find(e => e.id === 'pocket-tts')
  const zonosEngine  = engineStatuses.find(e => e.id === 'zonos-expressive')

  return (
    <section
      className="relative w-full overflow-hidden bg-cyber-dark text-bone border-b border-cyber-border/40"
      style={{ minHeight: '440px' }}
      aria-label="IRIS Neural Studio Hero"
    >
      {/* Scanline CRT overlay */}
      <div
        className="absolute inset-0 pointer-events-none z-0"
        style={{
          background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.12) 2px, rgba(0,0,0,0.12) 4px)',
        }}
      />

      {/* Background canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-auto z-0"
        style={{ mixBlendMode: 'screen' }}
        aria-hidden="true"
      />

      {/* Hero Content Grid */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 pt-24 pb-14 grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 items-center">

        {/* ── Left Column — Cyberpunk Cyberdeck Neural Console (Replaces Floating Singer) ── */}
        <div className="lg:col-span-6 flex justify-center lg:justify-start">
          <div className="w-full max-w-md bg-cyber-panel border-2 border-cyber-cyan/40 cyber-clip shadow-glow-cyan/20 relative p-1">
            
            {/* Top Bar / Warning Stripe Accent */}
            <div className="h-1.5 w-full cyber-hazard-tape mb-1" />

            {/* Cyberdeck Header */}
            <div className="bg-cyber-raised px-4 py-2.5 flex items-center justify-between border-b border-white/[0.08]">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-sm bg-cyber-yellow animate-ping" />
                <span className="font-mono text-xs font-bold tracking-wider text-cyber-yellow">
                  ARASAKA // NEURAL HUD [V2.0.77]
                </span>
              </div>
              <span className="font-mono text-[10px] text-cyber-cyan tracking-widest uppercase">
                SYNAPSE: ACTIVE
              </span>
            </div>

            {/* Main Cyber Visualizer Screen */}
            <div className="relative bg-black/90 p-4 border border-cyber-cyan/20">
              
              {/* Corner tech brackets */}
              <span className="absolute top-1 left-1 text-[10px] font-mono text-cyber-cyan/50">┌ NODE_01</span>
              <span className="absolute top-1 right-1 text-[10px] font-mono text-cyber-cyan/50">NODE_02 ┐</span>
              <span className="absolute bottom-1 left-1 text-[10px] font-mono text-cyber-cyan/50">└ LAT: 38MS</span>
              <span className="absolute bottom-1 right-1 text-[10px] font-mono text-cyber-yellow/60">44.1KHZ ┘</span>

              {/* Japanese Cyberpunk Kanji Tag */}
              <div className="flex justify-between items-center mb-2">
                <div className="text-[11px] font-mono text-surface-400">
                  <span className="text-cyber-cyan font-semibold">// 音声合成プロトコル</span> · CORE_MATRIX
                </div>
                <button
                  type="button"
                  onClick={triggerNeuralPing}
                  className="px-2 py-0.5 text-[9px] font-mono uppercase bg-cyber-yellow/10 border border-cyber-yellow/40 text-cyber-yellow hover:bg-cyber-yellow hover:text-black transition-colors rounded"
                  title="Test neural blip"
                >
                  ⚡ PING SYNAPSE
                </button>
              </div>

              {/* Live Canvas Spectrum Analyzer */}
              <div className="relative h-32 w-full rounded border border-white/[0.06] bg-[#07090D] overflow-hidden">
                <canvas
                  ref={hudCanvasRef}
                  width={380}
                  height={128}
                  className="w-full h-full block"
                />
                <div className="absolute top-1 left-2 text-[9px] font-mono text-cyber-cyan/70 tracking-widest">
                  F0 HARMONIC TRACKER
                </div>
              </div>

              {/* Telemetry Matrix Grid */}
              <div className="grid grid-cols-3 gap-2 mt-3 pt-2 border-t border-white/[0.08] text-center font-mono">
                <div className="bg-white/[0.03] p-1.5 rounded border border-white/[0.05]">
                  <div className="text-[9px] text-surface-400 uppercase">Formant Lock</div>
                  <div className="text-xs font-bold text-cyber-cyan">99.8%</div>
                </div>
                <div className="bg-white/[0.03] p-1.5 rounded border border-white/[0.05]">
                  <div className="text-[9px] text-surface-400 uppercase">Register</div>
                  <div className="text-xs font-bold text-cyber-yellow">C2 — G5</div>
                </div>
                <div className="bg-white/[0.03] p-1.5 rounded border border-white/[0.05]">
                  <div className="text-[9px] text-surface-400 uppercase">Dual-Engine</div>
                  <div className="text-xs font-bold text-cyber-neon">SVC + TTS</div>
                </div>
              </div>

              {/* Cyber Alert Footer Banner */}
              <div className="mt-3 py-1 px-2.5 bg-cyber-cyan/10 border-l-2 border-cyber-cyan flex items-center justify-between text-[10px] font-mono">
                <span className="text-cyber-cyan font-bold tracking-wider">
                  // RVC-V2 VOCAL CONVERSION: ARMED
                </span>
                <span className="text-cyber-yellow font-mono">
                  ZERO-LEAKAGE ✓
                </span>
              </div>
            </div>

            {/* Bottom Cyber Edge Cut */}
            <div className="h-1 bg-cyber-cyan/40 w-full" />
          </div>
        </div>

        {/* ── Right Column — Cyberpunk Typography & CTA HUD ── */}
        <div className="lg:col-span-6 flex flex-col gap-5">

          {/* Cyberpunk Eyebrow with Kanji Accent */}
          <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyber-yellow/10 border-l-2 border-cyber-yellow text-cyber-yellow font-mono text-xs font-bold tracking-widest uppercase w-fit">
            <span>// NEURAL VOICE & SONG ENGINE</span>
            <span className="text-white/40">|</span>
            <span className="text-surface-300">次世代音声</span>
          </div>

          {/* Cyberpunk Main Title */}
          <div>
            <h1 className="font-display font-black leading-none tracking-tight text-white text-3xl sm:text-5xl lg:text-6xl">
              CYBERNETIC
              <br />
              <span className="text-cyber-yellow neon-text-yellow">
                VOICE SYNTHESIS
              </span>
            </h1>
            <p className="text-surface-300 text-sm sm:text-base font-mono mt-3 leading-relaxed">
              Studio-grade zero-shot voice cloning and song voice conversion. 
              Powered by <span className="text-cyber-cyan font-semibold">RVC v2</span>,{' '}
              <span className="text-cyber-yellow font-semibold">Demucs v4</span>, and{' '}
              <span className="text-cyber-pink font-semibold">Chatterbox Neural Core</span>.
            </p>
          </div>

          {/* Live Engine Diagnostic Chips */}
          <div className="flex flex-wrap gap-2.5">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-cyber-panel border border-cyber-cyan/30 text-cyber-cyan font-mono text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-cyber-cyan animate-pulse" />
              RVC v2 Conversion: READY
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-cyber-panel border border-cyber-yellow/30 text-cyber-yellow font-mono text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-cyber-yellow" />
              Demucs Stem Isolation: ACTIVE
            </span>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded bg-cyber-panel border border-white/15 text-surface-200 font-mono text-xs font-semibold">
              <span className={gpuStatus.online ? 'w-2 h-2 rounded-full bg-emerald-400' : 'w-2 h-2 rounded-full bg-rose-500'} />
              GPU Tunnel: {gpuStatus.label}
            </span>
          </div>

          {/* Action CTAs */}
          <div className="flex flex-wrap gap-3.5 mt-2">
            <button
              type="button"
              onClick={onScrollDown}
              className="cyber-clip-btn px-6 py-3 bg-cyber-yellow hover:bg-white text-black font-display font-bold text-sm tracking-wider uppercase transition-all shadow-glow-yellow/50 flex items-center gap-2 active:scale-95"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm-1-11a1 1 0 112 0v3.586l1.707 1.707a1 1 0 01-1.414 1.414l-2-2A1 1 0 019 11V7z" />
              </svg>
              Open Neural Studio
            </button>
            <a
              href="/song-cover"
              className="cyber-clip-btn px-6 py-3 bg-cyber-panel border-2 border-cyber-cyan hover:bg-cyber-cyan hover:text-black text-cyber-cyan font-display font-bold text-sm tracking-wider uppercase transition-all shadow-glow-cyan/30 flex items-center gap-2 active:scale-95"
            >
              Song Covers (SVC v2) →
            </a>
          </div>

          {/* Bottom Telemetry Readout */}
          <div className="font-mono text-[11px] text-surface-400 tracking-wider flex items-center gap-3">
            <span>// ZERO-SHOT TIMBRE EXTRACTION</span>
            <span>·</span>
            <span className="text-cyber-yellow">EQUAL-POWER OLA STITCHING</span>
            <span>·</span>
            <span className="text-cyber-cyan">5-MIN CAP</span>
          </div>
        </div>
      </div>
    </section>
  )
}
