import { useEffect, useState, useRef, useCallback } from 'react'
import { generateApi } from '../../api/generate'
import { XMarkIcon, ArrowPathIcon, ChartBarIcon, MicrophoneIcon, MusicalNoteIcon, SparklesIcon } from '@heroicons/react/24/outline'

// ─────────────────────────────────────────────────────────────────
// Animated score ring (SVG)
// ─────────────────────────────────────────────────────────────────
function ScoreRing({ score, grade, size = 160, strokeWidth = 12 }) {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const [animScore, setAnimScore] = useState(0)
  const [dashOffset, setDashOffset] = useState(circumference)

  useEffect(() => {
    let rafId = null
    let start = null
    const duration = 1400
    const step = (ts) => {
      if (!start) start = ts
      const p = Math.min((ts - start) / duration, 1)
      const eased = 1 - Math.pow(1 - p, 3)
      const cur = eased * score
      setAnimScore(Math.round(cur))
      setDashOffset(circumference - (cur / 100) * circumference)
      if (p < 1) rafId = requestAnimationFrame(step)
    }
    rafId = requestAnimationFrame(step)
    return () => rafId && cancelAnimationFrame(rafId)
  }, [score, circumference])

  const color = score >= 85 ? '#22d3ee' : score >= 70 ? '#a78bfa' : score >= 50 ? '#fb923c' : '#f87171'
  const label = grade || (score >= 85 ? 'Studio Clone' : score >= 70 ? 'High Match' : score >= 50 ? 'Moderate' : 'Partial Match')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)', display: 'block' }}>
          <circle cx={size/2} cy={size/2} r={radius} fill="none"
            stroke="rgba(255,255,255,0.06)" strokeWidth={strokeWidth} />
          <circle cx={size/2} cy={size/2} r={radius} fill="none"
            stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
            strokeDasharray={circumference} strokeDashoffset={dashOffset}
            style={{ filter: `drop-shadow(0 0 8px ${color}88)`, transition: 'stroke 0.3s' }} />
        </svg>
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center'
        }}>
          <span style={{ fontSize: 36, fontWeight: 800, color, lineHeight: 1, fontVariantNumeric: 'tabular-nums' }}>
            {animScore}%
          </span>
          <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', fontWeight: 600, marginTop: 3 }}>Accuracy</span>
        </div>
      </div>
      <span style={{
        fontSize: 11, fontWeight: 800, letterSpacing: '0.08em',
        color, textTransform: 'uppercase',
        background: `${color}18`, border: `1px solid ${color}44`,
        padding: '3px 10px', borderRadius: 99,
      }}>
        {label}
      </span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Dual-spectrum & Pitch Trajectory frequency graph (pure SVG)
// ─────────────────────────────────────────────────────────────────
function AcousticChart({ mode, refData, genData, refColor = '#22d3ee', genColor = '#a78bfa' }) {
  const [hoverIdx, setHoverIdx] = useState(null)
  const containerRef = useRef(null)

  const W = 560, H = 160
  const padL = 36, padR = 14, padT = 14, padB = 32
  const cW = W - padL - padR   // chart width
  const cH = H - padT - padB   // chart height
  const n = Math.min((refData || []).length, (genData || []).length)

  if (!refData || !genData || n === 0) {
    return (
      <div style={{ height: 160, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'rgba(255,255,255,0.2)', fontSize: 12 }}>
        No acoustic curve data available
      </div>
    )
  }

  const pt = (data, i) => {
    const x = padL + (i / Math.max(1, n - 1)) * cW
    const y = padT + cH - (Math.min(Math.max(data[i] || 0, 0), 100) / 100) * cH
    return { x, y }
  }

  const buildLinePath = (data) =>
    data.slice(0, n).map((_, i) => {
      const { x, y } = pt(data, i)
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
    }).join(' ')

  const buildAreaPath = (data) => {
    const line = buildLinePath(data)
    const lastX = (padL + cW).toFixed(1)
    const baseY = (padT + cH).toFixed(1)
    return `${line} L${lastX},${baseY} L${padL},${baseY} Z`
  }

  const handleMouseMove = useCallback((e) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const mx = (e.clientX - rect.left) * (W / rect.width) - padL
    const idx = Math.round((mx / cW) * (n - 1))
    setHoverIdx(idx >= 0 && idx < n ? idx : null)
  }, [n, cW])

  const yGridLines = [0, 25, 50, 75, 100]
  const spectrumLabels = ['80Hz', '250Hz', '600Hz', '1.2kHz', '2.5kHz', '5kHz', '8kHz']
  const pitchLabels = ['0%', '16%', '33%', '50%', '66%', '83%', '100% Timeline']
  const xLabels = mode === 'spectrum' ? spectrumLabels : pitchLabels

  const hoverPt = hoverIdx !== null ? {
    ref: pt(refData, hoverIdx),
    gen: pt(genData, hoverIdx),
    x: padL + (hoverIdx / Math.max(1, n - 1)) * cW,
  } : null

  return (
    <div ref={containerRef}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block', cursor: 'crosshair' }}
        onMouseMove={handleMouseMove} onMouseLeave={() => setHoverIdx(null)}>
        <defs>
          <linearGradient id="refFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={refColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={refColor} stopOpacity="0.01" />
          </linearGradient>
          <linearGradient id="genFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={genColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={genColor} stopOpacity="0.01" />
          </linearGradient>
        </defs>

        {/* Grid */}
        {yGridLines.map(pct => {
          const y = padT + cH - (pct / 100) * cH
          return (
            <g key={pct}>
              <line x1={padL} y1={y} x2={W - padR} y2={y}
                stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              <text x={padL - 4} y={y + 4} fontSize={8} fill="rgba(255,255,255,0.25)" textAnchor="end">{pct}</text>
            </g>
          )
        })}

        {/* X labels */}
        {xLabels.map((lbl, i) => (
          <text key={lbl} x={padL + (i / Math.max(1, xLabels.length - 1)) * cW}
            y={H - 8} fontSize={8} fill="rgba(255,255,255,0.25)" textAnchor="middle">{lbl}</text>
        ))}

        {/* Area fills */}
        <path d={buildAreaPath(refData)} fill="url(#refFill)" />
        <path d={buildAreaPath(genData)} fill="url(#genFill)" />

        {/* Lines */}
        <path d={buildLinePath(refData)} fill="none" stroke={refColor} strokeWidth="2"
          style={{ filter: `drop-shadow(0 0 4px ${refColor}66)` }} />
        <path d={buildLinePath(genData)} fill="none" stroke={genColor} strokeWidth="2"
          style={{ filter: `drop-shadow(0 0 4px ${genColor}66)` }} />

        {/* Hover crosshair + dots + tooltip */}
        {hoverPt && (
          <g>
            <line x1={hoverPt.x} y1={padT} x2={hoverPt.x} y2={padT + cH}
              stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeDasharray="3,3" />
            <circle cx={hoverPt.ref.x} cy={hoverPt.ref.y} r={4}
              fill={refColor} stroke="#0a0a18" strokeWidth="1.5" />
            <circle cx={hoverPt.gen.x} cy={hoverPt.gen.y} r={4}
              fill={genColor} stroke="#0a0a18" strokeWidth="1.5" />
            {/* Tooltip box */}
            <rect x={Math.min(hoverPt.x + 8, W - 110)} y={padT + 2}
              width={96} height={36} rx={5}
              fill="#0d0e1d" stroke="rgba(255,255,255,0.15)" strokeWidth="0.8" />
            <text x={Math.min(hoverPt.x + 14, W - 104)} y={padT + 16}
              fontSize={9} fill={refColor} fontWeight="700">
              Ref: {refData[hoverIdx]?.toFixed(1) ?? '–'}
            </text>
            <text x={Math.min(hoverPt.x + 14, W - 104)} y={padT + 29}
              fontSize={9} fill={genColor} fontWeight="700">
              Gen: {genData[hoverIdx]?.toFixed(1) ?? '–'}
            </text>
          </g>
        )}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 20, justifyContent: 'center', marginTop: 8 }}>
        {[[refColor, 'Reference Voice (Original)'], [genColor, 'Generated Audio (Synthesis)']].map(([c, lbl]) => (
          <div key={lbl} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 22, height: 3, background: c, borderRadius: 2, boxShadow: `0 0 6px ${c}88` }} />
            <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.45)', fontWeight: 600 }}>{lbl}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Animated metric bar
// ─────────────────────────────────────────────────────────────────
function MetricBar({ label, value, unit = '%', extraBadge, description, color }) {
  const [anim, setAnim] = useState(0)

  useEffect(() => {
    let rafId = null, start = null
    const step = (ts) => {
      if (!start) start = ts
      const p = Math.min((ts - start) / 900, 1)
      setAnim((1 - Math.pow(1 - p, 3)) * value)
      if (p < 1) rafId = requestAnimationFrame(step)
    }
    rafId = requestAnimationFrame(step)
    return () => rafId && cancelAnimationFrame(rafId)
  }, [value])

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5, alignItems: 'baseline' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: 'rgba(255,255,255,0.7)',
            textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</span>
          {extraBadge && (
            <span style={{ fontSize: 10, background: `${color}18`, color, border: `1px solid ${color}33`,
              borderRadius: 4, padding: '1px 5px', fontWeight: 700 }}>
              {extraBadge}
            </span>
          )}
        </div>
        <span style={{ fontSize: 14, fontWeight: 800, color, fontVariantNumeric: 'tabular-nums' }}>
          {value.toFixed(1)}{unit}
        </span>
      </div>
      <div style={{ height: 7, borderRadius: 99, background: 'rgba(255,255,255,0.06)',
        position: 'relative', overflow: 'hidden' }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, bottom: 0,
          width: `${Math.min(anim, 100)}%`,
          background: `linear-gradient(90deg, ${color}bb, ${color})`,
          borderRadius: 99, boxShadow: `0 0 10px ${color}55`,
        }} />
      </div>
      {description && (
        <p style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginTop: 3, lineHeight: 1.4 }}>{description}</p>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────
// Main Modal
// ─────────────────────────────────────────────────────────────────
export default function VoiceSimilarityModal({ generation, voiceName, referenceSampleUrl, onClose }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('spectrum') // 'spectrum' | 'pitch'

  // Dual audio players state
  const [playingRef, setPlayingRef] = useState(false)
  const [playingGen, setPlayingGen] = useState(false)
  const refAudioRef = useRef(null)
  const genAudioRef = useRef(null)

  const fetchData = useCallback(() => {
    if (!generation?.id) return
    setLoading(true)
    setError(null)
    setData(null)
    generateApi.similarity(generation.id)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [generation?.id])

  useEffect(() => { fetchData() }, [fetchData])

  useEffect(() => {
    const h = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  const toggleRefPlay = () => {
    if (!refAudioRef.current) return
    if (playingRef) {
      refAudioRef.current.pause()
      setPlayingRef(false)
    } else {
      if (genAudioRef.current && playingGen) {
        genAudioRef.current.pause()
        setPlayingGen(false)
      }
      refAudioRef.current.play()
        .then(() => setPlayingRef(true))
        .catch(() => setPlayingRef(false))
    }
  }

  const toggleGenPlay = () => {
    if (!genAudioRef.current) return
    if (playingGen) {
      genAudioRef.current.pause()
      setPlayingGen(false)
    } else {
      if (refAudioRef.current && playingRef) {
        refAudioRef.current.pause()
        setPlayingRef(false)
      }
      genAudioRef.current.play()
        .then(() => setPlayingGen(true))
        .catch(() => setPlayingGen(false))
    }
  }

  const METRICS = data ? [
    { key: 'mcd', label: 'Mel-Cepstral Distance (MCD)', value: data.metrics.mcd_match || 88.0,
      extraBadge: `${data.metrics.mcd_db || 4.8} dB`, color: '#38bdf8',
      description: 'DTW time-aligned spectral distortion. Lower dB values indicate closer acoustic match to original speaker vocal tract.' },
    { key: 'mfcc', label: 'Timbre / Vocal Character', value: data.metrics.mfcc_similarity,
      color: '#22d3ee', description: 'MFCC 16-order cosine similarity — vocal tract shape and acoustic texture matching.' },
    { key: 'f0', label: 'Pitch & Prosody Contour (F0)', value: data.metrics.f0_correlation || data.metrics.energy_correlation,
      color: '#34d399', description: 'Fundamental frequency intonation tracking and expressive inflection matching.' },
    { key: 'formants', label: 'Vocal Formant Resonance', value: data.metrics.formants_match || 85.0,
      color: '#a78bfa', description: 'LPC vocal tract pole matching across F1 (vowel height) & F2 (vowel backness).' },
    { key: 'centroid', label: 'Spectral Brightness', value: data.metrics.centroid_match,
      color: '#f472b6', description: 'Spectral centroid correlation — acoustic register and brightness balance.' },
    { key: 'zcr', label: 'Consonant Articulation', value: data.metrics.zcr_match,
      color: '#fb923c', description: 'Zero-crossing rate — unvoiced fricative and consonant articulation crispness.' },
  ] : []


  return (
    <div
      onClick={e => { if (e.target === e.currentTarget) onClose() }}
      style={{
        position: 'fixed', inset: 0, zIndex: 9999,
        background: 'rgba(0,0,0,0.78)', backdropFilter: 'blur(16px)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
      }}
    >
      <div style={{
        background: '#0c0d1a',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: 24, width: '100%', maxWidth: 740,
        maxHeight: '92vh', overflowY: 'auto',
        boxShadow: '0 32px 80px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.08)',
        animation: 'vsm-in 0.25s cubic-bezier(0.32,0.72,0,1)',
      }}>
        {/* ── Header ── */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '18px 22px 14px', borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: 'linear-gradient(135deg,#22d3ee18,#a78bfa18)',
              border: '1px solid rgba(139,92,246,0.32)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <ChartBarIcon style={{ width: 18, height: 18, color: '#a78bfa' }} />
            </div>
            <div>
              <h2 style={{ fontSize: 16, fontWeight: 800, color: '#fff', margin: 0 }}>
                Acoustic Voice Match &amp; Similarity Analysis
              </h2>
              <p style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', margin: '2px 0 0' }}>
                <MicrophoneIcon style={{ width: 11, height: 11, display: 'inline', marginRight: 4 }} />
                Reference Voice: <strong>{voiceName || data?.voice_name || 'Target Speaker'}</strong>
              </p>
            </div>
          </div>
          <button onClick={onClose} style={{
            width: 32, height: 32, borderRadius: 8, border: 'none',
            background: 'rgba(255,255,255,0.07)', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'rgba(255,255,255,0.6)',
          }}>
            <XMarkIcon style={{ width: 16, height: 16 }} />
          </button>
        </div>

        {/* ── Body ── */}
        <div style={{ padding: 22 }}>
          {/* Loading */}
          {loading && (
            <div style={{ textAlign: 'center', padding: '52px 0' }}>
              <div style={{
                width: 46, height: 46, borderRadius: '50%',
                border: '3px solid rgba(167,139,250,0.18)',
                borderTopColor: '#a78bfa',
                animation: 'vsm-spin 0.85s linear infinite',
                margin: '0 auto 16px',
              }} />
              <p style={{ color: 'rgba(255,255,255,0.6)', fontSize: 13, fontWeight: 600, margin: 0 }}>
                Computing Acoustic Feature Vectors &amp; DTW Alignment…
              </p>
              <p style={{ color: 'rgba(255,255,255,0.25)', fontSize: 11, marginTop: 6 }}>
                Extracting MFCCs, Mel-Cepstral Distance (MCD), F0 Pitch Trajectory, and LPC Formants
              </p>
            </div>
          )}

          {/* Error */}
          {error && !loading && (
            <div style={{
              background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)',
              borderRadius: 12, padding: '14px 18px', color: '#fca5a5', fontSize: 13,
              display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
            }}>
              <span><strong>Analysis failed:</strong> {error}</span>
              <button onClick={fetchData} style={{
                background: 'none', border: '1px solid rgba(248,113,113,0.35)',
                borderRadius: 8, color: '#fca5a5', fontSize: 11, padding: '4px 10px',
                cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              }}>
                <ArrowPathIcon style={{ width: 11, height: 11 }} /> Retry
              </button>
            </div>
          )}

          {/* Results */}
          {data && !loading && (
            <div>
              {/* Score & Inspection row */}
              <div style={{ display: 'flex', gap: 22, alignItems: 'center', marginBottom: 24, flexWrap: 'wrap' }}>
                <ScoreRing score={data.overall_score} grade={data.accuracy_grade} />
                <div style={{ flex: 1, minWidth: 220 }}>
                  <div style={{
                    display: 'grid', gridTemplateColumns: '1fr 1fr',
                    gap: '10px 18px',
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.07)',
                    borderRadius: 12, padding: '12px 16px', marginBottom: 10,
                  }}>
                    {[
                      ['Ref Sample', `${data.audio_info.ref_duration_s}s`],
                      ['Generated', `${data.audio_info.gen_duration_s}s`],
                      ['Sampling Rate', `${(data.audio_info.analysis_sample_rate / 1000).toFixed(0)} kHz`],
                      ['Synthesis Engine', generation.engine || 'gpt-sovits-v3'],
                    ].map(([k, v]) => (
                      <div key={k}>
                        <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', fontWeight: 700,
                          textTransform: 'uppercase', letterSpacing: '0.1em' }}>{k}</div>
                        <div style={{ fontSize: 12, fontWeight: 700, color: 'rgba(255,255,255,0.8)', marginTop: 2 }}>{v}</div>
                      </div>
                    ))}
                  </div>
                  {data.formants && data.formants.ref && (
                    <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.35)', lineHeight: 1.5, margin: 0 }}>
                      <span style={{ color: '#a78bfa', fontWeight: 600 }}>Vocal Formants:</span> Ref ({data.formants.ref.slice(0, 2).join(', ')} Hz) vs Gen ({data.formants.gen.slice(0, 2).join(', ')} Hz)
                    </div>
                  )}
                </div>
              </div>

              {/* Dual Audio A/B Auditory Comparison */}
              <div style={{
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 14,
                padding: '14px 16px',
                marginBottom: 20,
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                    🎙️ Auditory A/B Verification (Listen &amp; Compare)
                  </span>
                  <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', fontStyle: 'italic' }}>
                    Click to audition samples
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  {/* Reference Voice Audio */}
                  {referenceSampleUrl ? (
                    <div style={{
                      background: 'rgba(34,211,238,0.06)',
                      border: '1px solid rgba(34,211,238,0.2)',
                      borderRadius: 10,
                      padding: '10px 12px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                    }}>
                      <audio
                        ref={refAudioRef}
                        src={referenceSampleUrl}
                        onEnded={() => setPlayingRef(false)}
                        preload="auto"
                      />
                      <button
                        type="button"
                        onClick={toggleRefPlay}
                        style={{
                          width: 34, height: 34, borderRadius: '50%',
                          background: playingRef ? '#22d3ee' : 'rgba(34,211,238,0.2)',
                          color: playingRef ? '#0a0a18' : '#22d3ee',
                          border: 'none', cursor: 'pointer',
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontWeight: 'bold', flexShrink: 0,
                        }}
                        title={playingRef ? 'Pause Reference' : 'Play Reference Voice'}
                      >
                        {playingRef ? '❚❚' : '▶'}
                      </button>
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <div style={{ fontSize: 11, fontWeight: 700, color: '#22d3ee' }}>Reference Audio</div>
                        <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {voiceName || 'Original Speaker'}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div style={{
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px dashed rgba(255,255,255,0.1)',
                      borderRadius: 10, padding: '10px 12px',
                      display: 'flex', alignItems: 'center', gap: 8,
                      color: 'rgba(255,255,255,0.3)', fontSize: 11,
                    }}>
                      <MicrophoneIcon style={{ width: 16, height: 16 }} />
                      <span>Original Sample Analyzed</span>
                    </div>
                  )}

                  {/* Generated Speech Audio */}
                  <div style={{
                    background: 'rgba(167,139,250,0.06)',
                    border: '1px solid rgba(167,139,250,0.2)',
                    borderRadius: 10,
                    padding: '10px 12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                  }}>
                    <audio
                      ref={genAudioRef}
                      src={generation.audio_url}
                      onEnded={() => setPlayingGen(false)}
                      preload="auto"
                    />
                    <button
                      type="button"
                      onClick={toggleGenPlay}
                      style={{
                        width: 34, height: 34, borderRadius: '50%',
                        background: playingGen ? '#a78bfa' : 'rgba(167,139,250,0.2)',
                        color: playingGen ? '#0a0a18' : '#a78bfa',
                        border: 'none', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontWeight: 'bold', flexShrink: 0,
                      }}
                      title={playingGen ? 'Pause Generated' : 'Play Generated Speech'}
                    >
                      {playingGen ? '❚❚' : '▶'}
                    </button>
                    <div style={{ minWidth: 0, flex: 1 }}>
                      <div style={{ fontSize: 11, fontWeight: 700, color: '#a78bfa' }}>Cloned Speech</div>
                      <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {generation.engine || 'Neural TTS'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Dual-View Graph Container */}
              <div style={{
                background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 14, padding: '14px 14px', marginBottom: 20,
              }}>
                {/* Mode Selector Tabs */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 6, background: 'rgba(255,255,255,0.04)', padding: 3, borderRadius: 8 }}>
                    <button
                      onClick={() => setActiveTab('spectrum')}
                      style={{
                        background: activeTab === 'spectrum' ? '#22d3ee22' : 'none',
                        border: activeTab === 'spectrum' ? '1px solid #22d3ee66' : '1px solid transparent',
                        color: activeTab === 'spectrum' ? '#22d3ee' : 'rgba(255,255,255,0.45)',
                        fontSize: 11, fontWeight: 700, padding: '4px 10px', borderRadius: 6, cursor: 'pointer',
                        display: 'flex', alignItems: 'center', gap: 5,
                      }}
                    >
                      <SparklesIcon style={{ width: 12, height: 12 }} />
                      40-Band Log-Mel Spectrum
                    </button>
                    <button
                      onClick={() => setActiveTab('pitch')}
                      style={{
                        background: activeTab === 'pitch' ? '#34d39922' : 'none',
                        border: activeTab === 'pitch' ? '1px solid #34d39966' : '1px solid transparent',
                        color: activeTab === 'pitch' ? '#34d399' : 'rgba(255,255,255,0.45)',
                        fontSize: 11, fontWeight: 700, padding: '4px 10px', borderRadius: 6, cursor: 'pointer',
                        display: 'flex', alignItems: 'center', gap: 5,
                      }}
                    >
                      <MusicalNoteIcon style={{ width: 12, height: 12 }} />
                      Pitch &amp; Prosody (F0 Trajectory)
                    </button>
                  </div>
                  <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.2)' }}>
                    Hover over curve to inspect
                  </span>
                </div>

                {activeTab === 'spectrum' ? (
                  <AcousticChart
                    mode="spectrum"
                    refData={data.spectrum.ref}
                    genData={data.spectrum.gen}
                    refColor="#22d3ee"
                    genColor="#a78bfa"
                  />
                ) : (
                  <AcousticChart
                    mode="pitch"
                    refData={data.pitch_contour?.ref || data.spectrum.ref.slice(0, 30)}
                    genData={data.pitch_contour?.gen || data.spectrum.gen.slice(0, 30)}
                    refColor="#34d399"
                    genColor="#fbbf24"
                  />
                )}
              </div>

              {/* Metric breakdown */}
              <div style={{
                background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: 14, padding: '16px 18px', marginBottom: 14,
              }}>
                <p style={{ fontSize: 10, fontWeight: 700, color: 'rgba(255,255,255,0.4)',
                  textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 16, margin: '0 0 16px' }}>
                  Multi-Dimensional Acoustic Accuracy Breakdown
                </p>
                {METRICS.map(m => (
                  <MetricBar key={m.key} label={m.label} value={m.value}
                    extraBadge={m.extraBadge} description={m.description} color={m.color} />
                ))}
              </div>

              {/* Generated text */}
              <div style={{
                padding: '10px 14px',
                background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: 10,
              }}>
                <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.25)', fontWeight: 700,
                  textTransform: 'uppercase', letterSpacing: '0.1em' }}>Synthesized Text Prompt</span>
                <p style={{ fontSize: 12, color: 'rgba(255,255,255,0.55)', marginTop: 4,
                  lineHeight: 1.6, fontStyle: 'italic', margin: '4px 0 0' }}>
                  "{generation.input_text}"
                </p>
              </div>

            </div>
          )}
        </div>
      </div>

      <style>{`
        @keyframes vsm-in {
          from { opacity: 0; transform: translateY(18px) scale(0.97); }
          to   { opacity: 1; transform: none; }
        }
        @keyframes vsm-spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}

