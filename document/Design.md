# Design System & UI Specifications — IRIS / VoiceLib

> **Project Name:** IRIS (VoiceLib)  
> **Aesthetic Archetype:** Ethereal Glass / Obsidian Dark Glassmorphism  
> **Document Version:** 1.0.0  
> **Design Tokens Source:** `frontend/src/index.css` & `frontend/tailwind.config.js`  

---

## 1. Aesthetic Vision & Design Philosophy

IRIS features a state-of-the-art **Obsidian Glassmorphism** design aesthetic inspired by high-end digital audio workstations (DAWs) and premium AI interfaces. The visual language balances deep dark space with glowing violet/cyan accents, multi-layered glass cards, crisp typography, and fluid micro-animations.

### Core Principles
1. **Depth & Translucency:** Semi-transparent dark panels layered over subtle gradient radial glows (`backdrop-blur-xl`).
2. **Precision Typography:** Clean sans-serif UI typography coupled with monospaced accents for technical audio specs.
3. **Tactile Micro-Interactions:** Smooth scale hover states (`hover:scale-[1.02]`), active press dynamics, and neon focus rings.
4. **Vibrant Glow Accents:** High-contrast violet, cyan, and emerald glowing elements that immediately capture user focus.

---

## 2. Color Palette & Design Tokens

### 2.1 Color Tokens

| Token Name | Hex Code | HSL / RGBA | Usage |
|---|---|---|---|
| **Background Primary** | `#0B0F17` | `hsl(220, 35%, 7%)` | Deep canvas background |
| **Glass Card Surface** | `#111827` | `rgba(17, 24, 39, 0.65)` | Glass panel fill with `backdrop-blur-md` |
| **Glass Card Border** | `#1F2937` | `rgba(255, 255, 255, 0.08)` | Subtle 1px glass border |
| **Accent Violet Glow** | `#8B5CF6` | `hsl(263, 90%, 66%)` | Primary action buttons, active states, active tab glow |
| **Accent Cyan Cyber** | `#06B6D4` | `hsl(189, 94%, 43%)` | Audio waveform tracks, badge highlights |
| **Success Emerald** | `#10B981` | `hsl(160, 84%, 39%)` | Status indicators, successful upload state |
| **Error Rose** | `#F43F5E` | `hsl(351, 89%, 60%)` | Error banners, delete confirmation dialogs |
| **Text Primary** | `#F9FAFB` | `hsl(210, 40%, 98%)` | Headers, body text, primary button text |
| **Text Muted** | `#94A3B8` | `hsl(215, 20%, 65%)` | Subtitles, timestamps, placeholders |

---

## 3. Typography & Hierarchy

### 3.1 Font Families
* **Primary UI Font:** `Inter`, `-apple-system`, `BlinkMacSystemFont`, `sans-serif`
* **Heading / Display Font:** `Outfit`, `Inter`, `sans-serif`
* **Audio Specs & Monospace:** `JetBrains Mono`, `Fira Code`, `monospace`

### 3.2 Typography Scale

```css
/* Typography Scale */
h1 { font-size: 2.25rem; line-height: 2.5rem;  font-weight: 700; letter-spacing: -0.025em; } /* 36px */
h2 { font-size: 1.5rem;   line-height: 2.0rem;  font-weight: 600; letter-spacing: -0.02em;  } /* 24px */
h3 { font-size: 1.125rem; line-height: 1.75rem; font-weight: 600; }                           /* 18px */
body { font-size: 0.875rem; line-height: 1.25rem; font-weight: 400; color: #94A3B8; }        /* 14px */
mono { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; }                       /* 12px */
```

---

## 4. Component Styling Specifications

### 4.1 Floating Glass Navbar (`Navbar.jsx`)
* **Layout:** Centered floating pill wrapper with `max-w-6xl mx-auto my-4`.
* **Background:** `bg-gray-900/60 backdrop-blur-xl border border-white/10 rounded-full px-6 py-3`.
* **Shadow:** `shadow-[0_8px_32px_0_rgba(0,0,0,0.37)]`.
* **Active Item:** Glowing text gradient with subtle violet indicator dot.

### 4.2 Voice Sample Card (`VoiceCard.jsx`)
* **Layout:** Double-bezel rounded glass container (`rounded-2xl p-5`).
* **Hover State:** `transition-all duration-300 hover:-translate-y-1 hover:border-violet-500/50 hover:shadow-[0_0_25px_rgba(139,92,246,0.2)]`.
* **Elements:** Voice avatar circle, voice name title, duration pill, inline sample audio trigger, delete confirmation drop-down.

### 4.3 Interactive Audio Player (`AudioPlayer.jsx`)
* **Waveform Visualizer:** Dynamic canvas bars visualizing frequency amplitudes in Cyan/Violet neon gradients (`#06B6D4` to `#8B5CF6`).
* **Scrubbing Track:** Custom range input styled with glowing thumb indicator.
* **Controls:** Circular play/pause toggle (`bg-gradient-to-r from-violet-600 to-cyan-500 hover:scale-105 transition-transform`).

### 4.4 Drag & Drop File Zone (`Dropzone.jsx`)
* **Border:** `border-2 border-dashed border-gray-700 hover:border-violet-500 transition-colors rounded-xl p-8`.
* **Drag-Over Effect:** `bg-violet-500/10 border-violet-400 scale-[1.01]`.
* **State Machines:** Idle → Dragging → File Selected → Uploading (Spinner) → Success Checkmark.

---

## 5. Animations & Micro-Interactions

```css
/* Custom Utility Animations */
@keyframes pulse-glow {
  0%, 100% { opacity: 0.4; transform: scale(1); }
  50% { opacity: 0.8; transform: scale(1.05); }
}

.animate-pulse-glow {
  animation: pulse-glow 3s infinite ease-in-out;
}

/* Glassmorphism Card Utility */
.glass-panel {
  background: rgba(17, 24, 39, 0.65);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.08);
}
```
