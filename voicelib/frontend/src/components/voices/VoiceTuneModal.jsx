import { useState, useEffect } from 'react'
import {
  AdjustmentsHorizontalIcon,
  XMarkIcon,
  SparklesIcon,
  CheckIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline'
import { voicesApi } from '../../api/voices'
import { useVoiceStore } from '../../store/useVoiceStore'
import { Spinner } from '../ui/Spinner'

export function VoiceTuneModal({ isOpen, onClose, voice }) {
  const updateVoice = useVoiceStore((s) => s.updateVoice)
  const [saving, setSaving] = useState(false)
  const [success, setSuccess] = useState(false)
  const [error, setError] = useState(null)

  // Default parameters
  const initialWeights = voice?.opt_weights || {}
  const [cfgWeight, setCfgWeight] = useState(0.70)
  const [pitchBias, setPitchBias] = useState(0.0)
  const [speedScale, setSpeedScale] = useState(1.0)
  const [temperature, setTemperature] = useState(0.70)
  const [topP, setTopP] = useState(0.82)
  const [warmthGainDb, setWarmthGainDb] = useState(2.0)
  const [exaggeration, setExaggeration] = useState(0.0)
  const [deRobotize, setDeRobotize] = useState(true)

  useEffect(() => {
    if (voice && voice.opt_weights) {
      const w = voice.opt_weights
      setCfgWeight(w.cfg_weight ?? 0.70)
      setPitchBias(w.pitch_bias ?? 0.0)
      setSpeedScale(w.speed_scale ?? 1.0)
      setTemperature(w.temperature ?? 0.70)
      setTopP(w.top_p ?? 0.82)
      setWarmthGainDb(w.warmth_gain_db ?? 2.0)
      setExaggeration(w.exaggeration ?? 0.0)
      setDeRobotize(w.de_robotize !== false)
    } else {
      setCfgWeight(0.70)
      setPitchBias(0.0)
      setSpeedScale(1.0)
      setTemperature(0.70)
      setTopP(0.82)
      setWarmthGainDb(2.0)
      setExaggeration(0.0)
      setDeRobotize(true)
    }
    setSuccess(false)
    setError(null)
  }, [voice, isOpen])

  if (!isOpen || !voice) return null

  const handleResetDefaults = () => {
    setCfgWeight(0.72)
    setPitchBias(0.0)
    setSpeedScale(1.0)
    setTemperature(0.68)
    setTopP(0.82)
    setWarmthGainDb(2.5)
    setExaggeration(0.0)
    setDeRobotize(true)
  }

  const handleSave = async (e) => {
    e.preventDefault()
    setSaving(true)
    setError(null)
    setSuccess(false)

    try {
      const payload = {
        cfg_weight: parseFloat(cfgWeight),
        pitch_bias: parseFloat(pitchBias),
        speed_scale: parseFloat(speedScale),
        temperature: parseFloat(temperature),
        top_p: parseFloat(topP),
        warmth_gain_db: parseFloat(warmthGainDb),
        exaggeration: parseFloat(exaggeration),
        de_robotize: Boolean(deRobotize),
      }

      const updated = await voicesApi.updateSettings(voice.id, payload)
      updateVoice(updated)
      setSuccess(true)
      setTimeout(() => {
        setSuccess(false)
        onClose()
      }, 900)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to save voice settings')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md animate-fade-in">
      <div
        className="w-full max-w-xl bg-gray-900/90 border border-white/10 rounded-3xl p-6 sm:p-8 shadow-[0_20px_60px_rgba(0,0,0,0.7)] backdrop-blur-2xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between pb-4 mb-6 border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-violet-500/20 border border-violet-500/30 flex items-center justify-center text-violet-400">
              <AdjustmentsHorizontalIcon className="w-6 h-6" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-tight">Tune & Calibrate Voice</h2>
              <p className="text-xs text-surface-200">
                Custom isolated profile for <span className="text-violet-300 font-semibold">{voice.name}</span>
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-surface-300 hover:text-white rounded-full hover:bg-white/5 transition-colors"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="mb-4 p-3 bg-rose-500/15 border border-rose-500/30 rounded-xl text-rose-300 text-xs">
            {error}
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-6">
          {/* 1. Speaker & Native Accent Lock (CFG) */}
          <div className="space-y-2 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            <div className="flex justify-between items-center">
              <label className="text-sm font-semibold text-white flex items-center gap-2">
                <span>Native Accent & Speaker Lock</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 font-mono">
                  {cfgWeight}x
                </span>
              </label>
              <span className="text-[11px] text-surface-300">
                {cfgWeight >= 0.70 ? '🎯 High Fidelity (Zero Accent Bleed)' : 'Standard'}
              </span>
            </div>
            <p className="text-xs text-surface-200 leading-relaxed">
              Forces the neural model to strictly clone the authentic native pronunciation and cadence, eliminating generic American/British accent drift.
            </p>
            <input
              type="range"
              min="0.30"
              max="0.85"
              step="0.05"
              value={cfgWeight}
              onChange={(e) => setCfgWeight(e.target.value)}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
            />
            <div className="flex justify-between text-[10px] text-surface-400 font-mono">
              <span>0.30 (Relaxed)</span>
              <span className="text-violet-400 font-bold">0.72 (Recommended for Indian/Native voices)</span>
              <span>0.85 (Maximum Lock)</span>
            </div>
          </div>

          {/* 2. Vocal Warmth & Body */}
          <div className="space-y-2 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            <div className="flex justify-between items-center">
              <label className="text-sm font-semibold text-white flex items-center gap-2">
                <span>Vocal Warmth & Harmonic Body</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono">
                  {warmthGainDb > 0 ? `+${warmthGainDb}` : warmthGainDb} dB
                </span>
              </label>
            </div>
            <p className="text-xs text-surface-200 leading-relaxed">
              Boosts warm low-mid vocal resonances (220Hz) to eliminate thin, metallic, or robotic timbre.
            </p>
            <input
              type="range"
              min="-2.0"
              max="6.0"
              step="0.5"
              value={warmthGainDb}
              onChange={(e) => setWarmthGainDb(e.target.value)}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>

          {/* 3. Pitch Offset */}
          <div className="space-y-2 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            <div className="flex justify-between items-center">
              <label className="text-sm font-semibold text-white flex items-center gap-2">
                <span>Pitch / Vocal Register Calibration</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/30 text-violet-300 font-mono">
                  {pitchBias > 0 ? `+${pitchBias}` : pitchBias} st
                </span>
              </label>
            </div>
            <p className="text-xs text-surface-200 leading-relaxed">
              Calibrates the natural vocal pitch (negative = deeper baritone/bass, positive = lighter tenor/soprano).
            </p>
            <input
              type="range"
              min="-6.0"
              max="6.0"
              step="0.5"
              value={pitchBias}
              onChange={(e) => setPitchBias(e.target.value)}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-violet-500"
            />
          </div>

          {/* 4. Speed / Cadence Scale */}
          <div className="space-y-2 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            <div className="flex justify-between items-center">
              <label className="text-sm font-semibold text-white flex items-center gap-2">
                <span>Natural Speaking Pace</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono">
                  {speedScale}x
                </span>
              </label>
            </div>
            <input
              type="range"
              min="0.80"
              max="1.25"
              step="0.05"
              value={speedScale}
              onChange={(e) => setSpeedScale(e.target.value)}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>

          {/* 5. Anti-Robotic Smoothing Toggle */}
          <div className="flex items-center justify-between p-4 rounded-2xl bg-white/[0.03] border border-white/[0.06]">
            <div>
              <span className="text-sm font-semibold text-white block">Anti-Robotic Harmonic Smoothing</span>
              <span className="text-xs text-surface-200">Applies digital phase-coherent smoothing and anti-comb filter.</span>
            </div>
            <input
              type="checkbox"
              checked={deRobotize}
              onChange={(e) => setDeRobotize(e.target.checked)}
              className="w-5 h-5 rounded bg-gray-800 border-gray-600 text-violet-600 focus:ring-violet-500 cursor-pointer"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-4 border-t border-white/10 gap-3">
            <button
              type="button"
              onClick={handleResetDefaults}
              className="btn-ghost text-xs py-2 px-3 flex items-center gap-1.5"
            >
              <ArrowPathIcon className="w-4 h-4" />
              Reset Recommended
            </button>

            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={onClose}
                className="btn-ghost text-sm py-2 px-4"
                disabled={saving}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={saving}
                className="btn-primary text-sm py-2.5 px-6 flex items-center gap-2"
              >
                {saving ? (
                  <>
                    <Spinner size="sm" />
                    <span>Saving...</span>
                  </>
                ) : success ? (
                  <>
                    <CheckIcon className="w-5 h-5 text-emerald-400" />
                    <span>Saved!</span>
                  </>
                ) : (
                  <>
                    <SparklesIcon className="w-4 h-4" />
                    <span>Save Calibration</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}
