import { LockClosedIcon, MusicalNoteIcon } from '@heroicons/react/24/outline'

export default function SongCover() {
  return (
    <div className="min-h-dvh pt-28 pb-16 px-4 flex items-center justify-center">
      <div className="text-center max-w-md w-full animate-fade-up">
        {/* Outer shell */}
        <div className="card-shell">
          <div className="card-inner flex flex-col items-center py-14 px-8">
            {/* Icon cluster */}
            <div className="relative mb-8">
              <div className="w-20 h-20 rounded-4xl bg-surface-800 border border-white/[0.07] flex items-center justify-center">
                <MusicalNoteIcon className="w-10 h-10 text-surface-700" />
              </div>
              {/* Lock badge */}
              <div className="absolute -bottom-2 -right-2 w-8 h-8 rounded-full bg-surface-900 border border-white/[0.1] flex items-center justify-center">
                <LockClosedIcon className="w-4 h-4 text-surface-700" />
              </div>
            </div>

            <span className="eyebrow mb-4">Coming in v2</span>
            <h1 className="text-2xl font-semibold text-white mb-3">Song Covers</h1>
            <p className="text-sm text-surface-700 leading-relaxed mb-8">
              Upload any song and generate a full cover in your cloned voice —
              complete with vocal conversion, instrumental preservation, and an
              optional genre twist.
            </p>

            {/* Feature preview pills */}
            <div className="flex flex-wrap gap-2 justify-center mb-8">
              {[
                'Vocal Separation',
                'Voice Conversion',
                'Genre Twist',
                'Full Song Output',
              ].map((f) => (
                <span
                  key={f}
                  className="text-[11px] px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.07] text-surface-700"
                >
                  {f}
                </span>
              ))}
            </div>

            <div className="w-full h-px bg-white/[0.06] mb-6" />
            <p className="text-xs text-surface-700">
              Build your voice library now — it'll work seamlessly with Song Covers when v2 launches.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
