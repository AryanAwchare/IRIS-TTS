/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // ── Cyberpunk 2077 Anime-Tech Palette ─────────────────────────────
        cyber: {
          yellow: '#FCEE0A',
          cyan:   '#00F0FF',
          neon:   '#FF003C',
          pink:   '#FF2A6D',
          purple: '#9B51E0',
          dark:   '#08090C',
          panel:  '#101218',
          raised: '#181A22',
          border: 'rgba(0, 240, 255, 0.25)',
        },
        // ── Clancy Signal palette ─────────────────────────────
        acid: {
          DEFAULT: '#E5FF00',
          muted:   '#B8CC00',
          dim:     'rgba(229,255,0,0.12)',
        },
        crimson: {
          DEFAULT: '#FF003C',
          dim:     'rgba(255,0,60,0.12)',
        },
        amber: {
          signal:  '#FF9100',
          dim:     'rgba(255,145,0,0.12)',
        },
        obsidian: '#07080A',
        charcoal: {
          DEFAULT: '#111214',
          raised:  '#1A1C1F',
          border:  '#222428',
        },
        bone: '#E2E2DF',
        // ── Legacy primary (blue) — kept for backwards compat ─
        primary: {
          50:  '#f0f4ff',
          100: '#dce6fe',
          200: '#bdd2fd',
          300: '#93b4fc',
          400: '#6b96fb',
          500: '#4f7ef8',
          600: '#3d6ef0',
          700: '#2a5ee8',
          900: '#1a3bba',
        },
        surface: {
          0:   '#ffffff',
          50:  '#fafafa',
          100: '#f4f4f5',
          200: '#e4e4e7',
          300: '#d4d4d8',
          400: '#a1a1aa',
          700: '#a1a1aa',
          800: '#18181b',
          900: '#09090b',
          950: '#050505',
        },
      },
      fontFamily: {
        sans:    ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', '"Plus Jakarta Sans"', 'sans-serif'],
        mono:    ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        cyber:   ['"Orbitron"', '"Space Grotesk"', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      boxShadow: {
        // Cyberpunk 2077 glows
        'glow-cyan':    '0 0 20px rgba(0,240,255,0.4), 0 0 60px rgba(0,240,255,0.15)',
        'glow-yellow':  '0 0 20px rgba(252,238,10,0.4), 0 0 60px rgba(252,238,10,0.15)',
        // Clancy Signal glows
        'glow-acid':    '0 0 20px rgba(229,255,0,0.35), 0 0 60px rgba(229,255,0,0.12)',
        'glow-crimson': '0 0 16px rgba(255,0,60,0.40), 0 0 48px rgba(255,0,60,0.12)',
        'glow-amber':   '0 0 14px rgba(255,145,0,0.35)',
        'glow-sm':      '0 0 12px rgba(229,255,0,0.25)',
        // Legacy
        'card':         '0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.06)',
        'card-hover':   '0 8px 32px -4px rgb(0 0 0 / 0.12)',
        'glow':         '0 0 40px -8px rgb(79 126 248 / 0.4)',
        'glow-sm-blue': '0 0 16px rgba(79,126,248,0.3)',
        'modal':        '0 24px 80px -12px rgb(0 0 0 / 0.3)',
        'inner-light':  'inset 0 1px 1px rgba(255,255,255,0.15)',
      },
      transitionTimingFunction: {
        'spring': 'cubic-bezier(0.32, 0.72, 0, 1)',
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      keyframes: {
        // Existing
        'fade-up':  { from: { opacity: '0', transform: 'translateY(16px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        'fade-in':  { from: { opacity: '0' }, to: { opacity: '1' } },
        'pulse-ring': { '0%': { transform: 'scale(1)', opacity: '0.8' }, '100%': { transform: 'scale(1.6)', opacity: '0' } },
        'waveform': { '0%, 100%': { transform: 'scaleY(0.4)' }, '50%': { transform: 'scaleY(1)' } },
        'spin-slow': { from: { transform: 'rotate(0deg)' }, to: { transform: 'rotate(360deg)' } },
        // Clancy Signal
        'pulse-acid': {
          '0%, 100%': { boxShadow: '0 0 8px rgba(229,255,0,0.3)', opacity: '1' },
          '50%':      { boxShadow: '0 0 24px rgba(229,255,0,0.7)', opacity: '0.85' },
        },
        'signal-flicker': {
          '0%, 100%': { opacity: '1' },
          '10%':      { opacity: '0.6' },
          '20%':      { opacity: '1' },
          '80%':      { opacity: '0.8' },
        },
        'hero-float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-10px)' },
        },
        'slide-in-left': {
          from: { opacity: '0', transform: 'translateX(-24px)' },
          to:   { opacity: '1', transform: 'translateX(0)' },
        },
        'slide-in-right': {
          from: { opacity: '0', transform: 'translateX(24px)' },
          to:   { opacity: '1', transform: 'translateX(0)' },
        },
        'typing-cursor': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      animation: {
        'fade-up':        'fade-up 0.6s cubic-bezier(0.32,0.72,0,1) both',
        'fade-in':        'fade-in 0.4s ease both',
        'pulse-ring':     'pulse-ring 1.5s ease-out infinite',
        'waveform':       'waveform 1s ease-in-out infinite',
        'spin-slow':      'spin-slow 2s linear infinite',
        // Clancy Signal
        'pulse-acid':     'pulse-acid 2.5s ease-in-out infinite',
        'signal-flicker': 'signal-flicker 4s ease-in-out infinite',
        'hero-float':     'hero-float 6s ease-in-out infinite',
        'slide-in-left':  'slide-in-left 0.7s cubic-bezier(0.32,0.72,0,1) both',
        'slide-in-right': 'slide-in-right 0.7s cubic-bezier(0.32,0.72,0,1) both',
        'typing-cursor':  'typing-cursor 1s step-end infinite',
      },
    },
  },
  plugins: [],
}
