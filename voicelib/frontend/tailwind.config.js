/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
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
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        display: ['"Clash Display"', '"Plus Jakarta Sans"', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.06)',
        'card-hover': '0 8px 32px -4px rgb(0 0 0 / 0.12)',
        'glow': '0 0 40px -8px rgb(79 126 248 / 0.4)',
        'modal': '0 24px 80px -12px rgb(0 0 0 / 0.3)',
        'inner-light': 'inset 0 1px 1px rgba(255,255,255,0.15)',
      },
      transitionTimingFunction: {
        'spring': 'cubic-bezier(0.32, 0.72, 0, 1)',
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        'pulse-ring': {
          '0%':   { transform: 'scale(1)',    opacity: '0.8' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
        'waveform': {
          '0%, 100%': { transform: 'scaleY(0.4)' },
          '50%':      { transform: 'scaleY(1)' },
        },
        'spin-slow': {
          from: { transform: 'rotate(0deg)' },
          to:   { transform: 'rotate(360deg)' },
        },
      },
      animation: {
        'fade-up':     'fade-up 0.6s cubic-bezier(0.32,0.72,0,1) both',
        'fade-in':     'fade-in 0.4s ease both',
        'pulse-ring':  'pulse-ring 1.5s ease-out infinite',
        'waveform':    'waveform 1s ease-in-out infinite',
        'spin-slow':   'spin-slow 2s linear infinite',
      },
    },
  },
  plugins: [],
}
