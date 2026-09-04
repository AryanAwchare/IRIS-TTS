# IRIS VoiceLib: A Modular Framework for Voice Cloning, Deep Acoustic DNA Profiling, and Multi-Metric Speech Quality Evaluation

**Aryan Awchare**  
Department of Computer Science / Artificial Intelligence Engineering  
*[Institution Name]*  
*[City, Country]*  
*[email@institution.edu]*  

---

**Abstract** — Modern neural text-to-speech (TTS) synthesis and zero-shot voice cloning can reproduce a speaker's vocal characteristics from short reference recordings. However, practical deployment of such systems introduces compound challenges extending far beyond the synthesis model: reference audio quality is rarely controlled, expressive nuance is difficult to parameterize, speaker identity must be preserved across variable input conditions, and evaluation is typically limited to subjective listening. This paper presents **IRIS VoiceLib**, a modular, full-stack framework that integrates voice library management, reference-audio preprocessing, deep acoustic DNA profiling, configurable neural TTS backends, expressive synthesis conditioning with tag-emotion synchronization, anti-harshness vocal mastering, voice-state caching, and automated multi-metric objective evaluation into a single coherent system. The framework employs Silero-VAD for voice activity detection and signal-to-noise ratio (SNR)-guided segment selection (supporting reference audio up to 5 minutes / 300 seconds and 100 MB upload payload), a Deep Acoustic DNA Profiler extracting fundamental frequency ($F_0$) via pYIN, formant resonances ($F_1$–$F_4$) via Linear Predictive Coding (LPC), Harmonics-to-Noise Ratio (HNR), and perturbation metrics (jitter/shimmer) to auto-tune synthesis parameters. The framework features a pluggable TTS engine layer with implementations for GPT-SoVITS v3 (GPU/Colab microservice) and Kyutai Pocket-TTS (CPU fallback), six parameterized emotion presets with dynamic inline tag-emotion sync (`[happy]`, `[excited]`, `[whisper]`, `[angry]`, `[sad]`, `[calm]`), an anti-harshness DSP mastering stage (sibilance de-essing, chest warmth boost, peak brickwall limiting, TPDF dithering), and an embedded evaluation pipeline comprising ECAPA-TDNN cosine speaker-similarity scoring, faster-Whisper automatic speech recognition (ASR) with JiWER word error rate (WER), and fundamental-frequency ($F_0$) variance-based prosodic analysis. Speaker identity is assessed using 192-dimensional ECAPA-TDNN embeddings, content accuracy through ASR-derived WER, and prosodic character through $F_0$ standard-deviation tracking. The current work establishes the implemented architecture and evaluation pipeline; comprehensive controlled quantitative benchmarking across multiple speakers, synthesis engines, expressive conditions, and hardware configurations remains an important direction for future work.

**Keywords** — voice cloning, zero-shot TTS, acoustic profiling, ECAPA-TDNN, speaker similarity, word error rate, expressive speech synthesis, DSP vocal mastering, voice activity detection, FastAPI, neural text-to-speech

---

## I. Introduction

Advances in neural text-to-speech synthesis have made zero-shot voice cloning practically feasible: given a short reference recording of a target speaker, modern models can reproduce that speaker's vocal identity in synthesized speech with increasing fidelity [1]–[4]. Despite this technical progress, deploying voice cloning in a real application surface involves a set of engineering and evaluation challenges that receive comparatively less attention in academic literature, which tends to focus on novel model architectures rather than on end-to-end system design.

**Challenge 1 — Reference Audio Quality & Acoustic DNA Profiling.** A reference audio sample uploaded by a user may range from a brief 3-second utterance to a 5-minute (300-second) recording, containing prolonged silence, background noise, reverberation, low-energy non-speech regions, or codec artefacts. Forwarding such a sample directly to a voice cloning model degrades the extracted speaker representation. Reference-audio preprocessing—including voice activity detection (VAD), SNR-based segment selection, and deep acoustic DNA profiling ($F_0$ distribution, LPC formants, HNR, jitter, shimmer)—is therefore an essential practical step to derive accurate speaker embeddings and auto-tune inference parameters.

**Challenge 2 — Speaker Identity Preservation.** A cloned voice must preserve the target speaker's characteristics, including timbre, accent, prosodic style, and fine-grained vocal quality. Objective speaker similarity metrics derived from speaker-verification embeddings provide an automated proxy for this property and enable systematic quality monitoring without requiring subjective listening tests for every generation.

**Challenge 3 — Linguistic Accuracy.** A cloned voice that sounds perceptually convincing may still mispronounce, substitute, or omit words from the input text. In production systems, linguistic accuracy must be monitored alongside speaker similarity, since a generation that sounds like the target speaker but fails to reproduce the correct words constitutes a synthesis failure.

**Challenge 4 — Expressiveness & Paralinguistic Control.** Practical TTS applications require modulating the affective character of synthesized speech—for example, producing neutral narration, excited promotional content, or calm assistive output—while preserving speaker identity. This requires both parameterized emotion conditioning and automatic inline tag-emotion synchronization (e.g., parsing `[excited]`, `[whisper]`, `[gasp]`).

**Challenge 5 — Vocal Harshness & Audio Artifacts.** Raw neural vocoder outputs frequently exhibit digital sibilance, high-frequency harshness, or quantization noise at lower bitrates. Production-ready voice cloning demands an integrated digital signal processing (DSP) vocal mastering pipeline (high-pass filtering, de-essing, chest warmth resonance, peak limiting, TPDF dithering).

**Challenge 6 — Deployment and Evaluation Infrastructure.** Research models are typically evaluated in controlled offline settings. Production deployment requires an authenticated API surface, high-throughput asynchronous database connection pooling, persistent storage for voice profiles and generation histories, a visual UI with real-time waveform visualization, and embedded evaluation that runs automatically per generation without requiring a dedicated evaluation batch process.

These six challenges jointly motivate the design of IRIS VoiceLib, a modular framework that addresses all six within a single deployable system.

### A. Paper Contributions

This paper makes the following contributions:

1. **System Architecture**: A description of a modular, full-stack voice cloning framework integrating a React 18 SPA frontend (Ethereal Glassmorphism UI with interactive canvas visualizers), FastAPI backend with AsyncPG connection pooling, PostgreSQL metadata store, S3-compatible object storage, pluggable TTS engine layer, and automated pipeline.

2. **Reference-Audio Preprocessing & Deep Acoustic DNA Profiler v2**: Implementation of a Silero-VAD reference audio selector combined with a Deep Acoustic DNA Profiler using Linear Predictive Coding (LPC) for formant extraction ($F_1$–$F_4$), pYIN for pitch ($F_0$) distribution, Harmonics-to-Noise Ratio (HNR), and pitch/amplitude perturbation (jitter and shimmer) to auto-tune generation parameters (`cfg_weight`, `exaggeration`, `temp`, `top_p`, `speed_scale`).

3. **Pluggable TTS Engine Abstraction**: An abstract base class (`BaseTTSEngine`) with concrete implementations for GPT-SoVITS v3 (GPU/Colab microservice with automated patch shims) and Kyutai Pocket-TTS (CPU-compatible fallback), enabling backend substitution without changes to the application layer.

4. **Expressive Conditioning & Tag-Emotion Sync**: Six emotion presets (neutral, calm, happy, excited, sad, angry) parameterized by exaggeration coefficient and CFG weight, paired with an inline tag-emotion synchronization engine.

5. **Anti-Harshness DSP Vocal Mastering Pipeline**: Integrated post-synthesis DSP module featuring a 6–8 kHz sibilance de-esser, 150–300 Hz chest warmth resonance booster, anti-aliasing high-pass filter, soft-knee brickwall peak limiter, and TPDF dithering.

6. **Automated Multi-Metric Evaluation Pipeline**: Per-request computation of ECAPA-TDNN speaker similarity, faster-Whisper WER, and $F_0$ standard deviation—returned as HTTP response headers (`X-Speaker-Similarity`, `X-Word-Error-Rate`, `X-Prosody-Variance`) for downstream monitoring.

7. **Proposed Experimental Protocol**: A controlled experimental design for future quantitative validation of the framework's components.

---

## II. Literature Review

### A. Neural Text-to-Speech Synthesis

**[1] Arik et al. (2017) — Deep Voice: Real-time Neural Text-to-Speech.**
Problem: End-to-end trainable TTS systems faced significant inference latency. Method: WaveNet-based acoustic model with a dedicated neural phoneme duration predictor. Finding: Deep Voice achieved competitive naturalness with real-time generation capability. Relevance to IRIS: Established the feasibility of deploying neural TTS at production time scales, a core requirement IRIS inherits.

**[2] Wang et al. (2017) — Tacotron: Towards End-to-End Speech Synthesis.**
Problem: Traditional TTS pipelines relied on hand-engineered linguistic features and separate vocoder stages. Method: Sequence-to-sequence model with attention consuming characters and producing spectrograms. Finding: End-to-end training simplified the pipeline while maintaining naturalness. Relevance to IRIS: Demonstrated that the sequence-to-sequence paradigm produces high-quality speech, establishing the foundation on which later zero-shot cloning models build.

**[3] Shen et al. (2018) — Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions (Tacotron 2).**
Problem: Tacotron produced spectrograms requiring improved vocoders. Method: Tacotron 2 paired with a WaveNet vocoder conditioned on mel spectrograms. Finding: State-of-the-art naturalness at the time, approaching human-level MOS on held-out speakers. Relevance to IRIS: The mel spectrogram conditioning paradigm is a predecessor to the reference-audio conditioning approach used in zero-shot cloning engines that IRIS wraps.

**[4] Ren et al. (2019) — FastSpeech: Fast, Robust and Controllable Text to Speech.**
Problem: Autoregressive TTS models were slow and produced alignment errors. Method: Non-autoregressive model with explicit duration prediction for parallel generation. Finding: Ten to thirty-eight times faster inference than Tacotron 2 with comparable naturalness. Relevance to IRIS: FastSpeech's non-autoregressive approach influenced the design of lightweight CPU-deployable TTS models, including the Pocket-TTS backend used by IRIS.

### B. Zero-Shot Voice Cloning

**[5] Arik et al. (2018) — Neural Voice Cloning with a Few Samples.**
Problem: Cloning a speaker's voice typically required extensive fine-tuning on many speaker-specific samples. Method: Speaker encoding via speaker-conditioned modules or few-sample fine-tuning. Finding: Acceptable cloning quality from as few as five to ten utterances. Relevance to IRIS: Established the few-shot and zero-shot paradigm that IRIS's cloning pipeline relies upon.

**[6] Jia et al. (2018) — Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis (SV2TTS).**
Problem: Multi-speaker TTS models required large speaker-specific corpora. Method: Speaker embedding from a speaker-verification model (d-vector) used to condition a Tacotron 2 synthesis network. Finding: Zero-shot cloning with perceptually similar speaker identity from short reference recordings. Relevance to IRIS: Directly established the reference-audio-conditioned synthesis paradigm that IRIS implements.

**[7] Casanova et al. (2022) — YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion.**
Problem: Zero-shot speaker adaptation remained challenging for multi-lingual voices. Method: Extending VITS with speaker conditioning using d-vectors. Finding: Effective zero-shot voice conversion and multi-speaker TTS across languages. Relevance to IRIS: Demonstrates the effectiveness of reference-conditioned neural vocoders that IRIS's engine layer integrates.

**[8] Wang et al. (2023) — Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E).**
Problem: Zero-shot TTS quality was limited by existing acoustic model designs. Method: Treating neural codec codes as a language modeling task, conditioned on speaker codec tokens. Finding: Strong speaker similarity from only three seconds of reference audio. Relevance to IRIS: VALL-E represents a state-of-the-art zero-shot approach that contextualizes the capabilities of the engines IRIS wraps.

### C. Speaker Representation and Verification

**[9] Desplanques et al. (2020) — ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.**
Problem: Prior TDNN-based speaker embeddings did not sufficiently capture multi-scale temporal information. Method: ECAPA-TDNN with channel-attention and residual multi-scale connections trained on VoxCeleb. Finding: State-of-the-art equal error rate on VoxCeleb1 and VoxCeleb2 benchmarks. Relevance to IRIS: IRIS's speaker similarity module uses the `speechbrain/spkrec-ecapa-voxceleb` model to compute 192-dimensional cosine-comparable embeddings, directly leveraging this contribution.

**[10] Nagrani et al. (2017) — VoxCeleb: A Large-Scale Speaker Identification Dataset.**
Problem: Publicly available speaker datasets were small and not representative of real-world conditions. Method: Automated collection of audio-visual speaker-labeled data from YouTube. Finding: Large-scale challenging benchmark for speaker verification in unconstrained environments. Relevance to IRIS: VoxCeleb serves as the training corpus for the ECAPA-TDNN model used in IRIS's evaluation module.

### D. Automatic Speech Recognition for TTS Evaluation

**[11] Radford et al. (2023) — Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).**
Problem: ASR systems struggled with robustness across accents, domains, and audio conditions. Method: Large-scale weakly-supervised training across 680,000 hours of multilingual web audio using an encoder-decoder transformer. Finding: Whisper generalizes robustly across diverse conditions without domain-specific fine-tuning. Relevance to IRIS: IRIS's content accuracy module uses the `faster-whisper` implementation of Whisper (`small.en`) on CPU for per-request transcription and WER computation.

### E. Expressive and Controllable TTS

**[12] Wang et al. (2018) — Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis (GST-Tacotron).**
Problem: TTS systems produced monotone prosody without control over speaking style. Method: Global Style Tokens (GST) as a conditioning mechanism for style transfer in Tacotron 2. Finding: Controllable prosodic variation and style transfer without explicit style labels. Relevance to IRIS: GST established the principle of latent conditioning for expressive control, which the CFG-weight parameterization in IRIS's emotion presets extends.

**[13] Li et al. (2022) — StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis.**
Problem: Existing TTS models produced limited prosodic diversity. Method: Style-based generative architecture separating content and style embeddings. Finding: High-quality, diverse prosodic synthesis with improved naturalness. Relevance to IRIS: Reinforces the feasibility of disentangling speaker identity from expressive style, a design goal of IRIS's conditioning parameters.

### F. Voice Activity Detection and Audio Preprocessing

**[14] Silero Team (2021) — Silero VAD: pre-trained enterprise-grade Voice Activity Detector.**
Problem: Reliable, lightweight VAD was needed for production environments without heavy dependencies. Method: Compact ONNX-exportable model trained for binary speech/non-speech classification. Finding: Competitive VAD accuracy with low latency and minimal resource usage. Relevance to IRIS: IRIS's reference-audio preprocessing module (`reference_selector.py`) uses Silero VAD to detect speech regions before SNR-ranked window selection.

### G. Research Gap

Despite substantial individual progress in neural TTS synthesis, zero-shot voice cloning, speaker verification embeddings, and automatic speech recognition, there exists a comparative lack of published, deployment-oriented frameworks that integrate all of the following within a single system:

1. Reference-audio quality preprocessing with VAD, SNR-based segment selection, and deep acoustic DNA profiling.
2. A pluggable, abstracted TTS engine layer supporting multiple synthesis backends (GPU cloud & CPU local).
3. Parameterized expressive conditioning presets with inline tag-emotion synchronization.
4. An integrated DSP vocal mastering pipeline for sibilance de-essing, warmth enhancement, and peak limiting.
5. Persistent authenticated voice library management backed by async database connection pooling.
6. Automated per-request multi-metric objective evaluation (speaker similarity, WER, prosody).

IRIS VoiceLib specifically addresses this integration gap. The contribution is not a new neural TTS model but a system-level framework that makes voice cloning, acoustic profiling, expressive control, DSP mastering, voice management, and objective evaluation work together in a production-deployable application.

---

## III. Methodology

### A. System Architecture

IRIS VoiceLib is structured as a three-tier, multi-service architecture:

```
   +-------------------------------------------------------+
   |        React 18 SPA (Vite + Tailwind CSS)            |
   | Ethereal UI * Interactive Canvas Audio Visualizer     |
   |      Zustand State  *  Axios JWT Bearer               |
   +--------------------------+----------------------------+
                              |  REST / HTTPS + JWT
   +--------------------------v----------------------------+
   |            FastAPI 0.111 (Python 3.11+)               |
   |      Async Routers * Auth * Controllers               |
   +-----------+--------------+-------------------+--------+
               |              |                   |
   +-----------v----+   +-----v-------------------+---+
   | PostgreSQL 15  |   |           TTS Router        |
   | (AsyncPG Pool/ |   |    (Engine Selection Logic) |
   |  SQLAlchemy 2) |   +----+--------------------+---+
   +----------------+        |                    |
                      +------v-------+    +-------v-------+
   +----------------+ | GPT-SoVITS   |    | Pocket-TTS    |
   | S3 / MinIO     | | Engine (GPU) |    | Engine (CPU)  |
   | Object Store   | +--------------+    +---------------+
   +----------------+        |
              +--------------v--------------------------------+
              |     DSP Vocal Mastering Pipeline              |
              | (De-esser * Warmth EQ * Limiter * Dither)     |
              +--------------+--------------------------------+
                             |
              +--------------v--------------------------------+
              |          Evaluation Pipeline                  |
              |     ECAPA * WER * F0 Variance                 |
              +-----------------------------------------------+
```

**Frontend.** The React 18 SPA is built with Vite 5 and styled with Tailwind CSS v3 using an Ethereal Glassmorphism design language (OLED dark palette, radial mesh gradients, backdrop-blur card surfaces, Plus Jakarta Sans typography). It includes an interactive HTML5 Canvas waveform visualizer (`AudioCanvasVisualizer`) with dynamic dither overlays. Zustand provides global state management with JWT persistence. Axios handles all HTTP communication with automatic JWT injection via request interceptors.

**Backend.** The FastAPI application (v0.111, Python 3.11+) operates in an async-first mode using an AsyncPG connection pool (`max_overflow=10`, `pool_recycle=3600`) and SQLAlchemy 2.0 async sessions. Authentication uses HS256 JWTs via `python-jose` with bcrypt password hashing. The application startup lifespan event initializes database pools, verifies object storage buckets, and preloads the TTS model into memory—eliminating cold-start latency.

**Database & Object Storage.** PostgreSQL 15 stores `User`, `Voice`, and `Generation` entities. Voice records track consent confirmation (`consent_confirmed`) for compliance. Raw reference audio and generated WAV files are stored in an S3-compatible object store (MinIO for local development; AWS S3 or Cloudflare R2 for production).

---

### B. Reference-Audio Preprocessing & Deep Acoustic DNA Profiling

The reference processing pipeline (`reference_selector.py` & `voice_profiler.py`) executes a multi-stage procedure on uploaded reference audio:

**Stage 1 — Input Validation.** Uploaded audio files are validated for MIME type, file size (maximum 100 MB), and duration (3 to 300 seconds / 5 minutes).

**Stage 2 — Voice Activity Detection (VAD).** Silero VAD [14] (ONNX variant loaded via PyTorch Hub) resamples audio to 16 kHz and identifies speech-active regions, trimming silence and non-speech artifacts.

**Stage 3 — SNR-Based Window Selection.** For recordings longer than 12 seconds, a sliding window calculates estimated Signal-to-Noise Ratio (SNR):

$$\text{SNR}_{\text{est}} = 10 \cdot \log_{10} \left( \frac{P_{\text{signal}} + \epsilon}{P_{\text{noise\_10th}} + \epsilon} \right)$$

The window with the highest SNR is selected as the primary prompt for TTS voice state derivation.

**Stage 4 — Deep Acoustic DNA Profiling (v2).** The profile extractor (`voice_profiler.py`) extracts a detailed acoustic fingerprint:
- **Pitch Distribution ($F_0$)**: Extracted via pYIN algorithm (65 Hz to 500 Hz search range) to determine mean $F_0$, median $F_0$, standard deviation, and register classification (`male` vs `female` vocal range).
- **Formant Resonances ($F_1$–$F_4$)**: Center frequencies calculated via Linear Predictive Coding (LPC order $\approx 2 + \frac{f_s}{1000}$) on a pre-emphasized signal ($1 - 0.97z^{-1}$).
- **Harmonics-to-Noise Ratio (HNR)**: Harmonic-percussive source separation (HPSS) yields vocal breathiness vs. tonality ratio in dB.
- **Perturbation Metrics**: Local pitch jitter (%) and amplitude shimmer (%) derived from consecutive $F_0$ period differences.
- **Spectral Features**: 13 Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, spectral tilt, and spectral rolloff.
- **Auto-Tuning Engine**: The extracted acoustic DNA automatically optimizes synthesis hyper-parameters (`cfg_weight`, `exaggeration`, `temperature`, `top_p`, `speed_scale`) tailored to the speaker's vocal characteristics.

---

### C. TTS Engine Layer & Dual Routing

IRIS provides a unified abstract base class (`BaseTTSEngine` in `tts_engines/base.py`):

```python
class BaseTTSEngine(ABC):
    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def derive_voice_state(self, audio_source, voice_id) -> Any: ...

    @abstractmethod
    def generate_audio(self, voice_state, text, **kwargs) -> bytes: ...

    @abstractmethod
    def invalidate_cache(self, voice_id) -> None: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...
```

1. **GPT-SoVITS Engine (`gptsovits_engine.py`)**: Microservice backend routing synthesis to a Chatterbox / GPT-SoVITS v3 model running on a Google Colab Tesla T4 GPU via a secure Ngrok tunnel (`COLAB_GPU_API_URL`). Generates 32 kHz zero-shot audio with high expressiveness.
2. **Pocket-TTS Engine (`pocket_engine.py`)**: Kyutai Labs' lightweight CPU model generating 24 kHz audio for low-latency offline fallback.
3. **Mock TTS Engine**: Development mode fallback generating test tones when `VOICELIB_USE_MOCK_TTS=true`.

---

### D. Anti-Harshness Vocal Mastering & Timbre Morpher

Synthesized audio is routed through an automated DSP vocal mastering pipeline (`timbre_morpher.py`) before delivery:

1. **High-Pass Filtering**: Second-order Butterworth filter ($f_c = 80\text{ Hz}$) removes sub-bass rumble and DC offset.
2. **Sibilance De-Esser**: Band-pass notch filter targeting $6\text{ kHz}$–$8\text{ kHz}$ attenuates harsh high-frequency sibilants.
3. **Chest Warmth Resonance Boost**: Low-mid parametric bell filter ($150\text{ Hz}$–$300\text{ Hz}$, $+1.5\text{ dB}$ to $+3.0\text{ dB}$) restores natural vocal body and warmth.
4. **Soft-Knee Peak Limiter**: Brickwall peak limiter with a threshold at $-0.5\text{ dBFS}$ prevents digital clipping and distortion.
5. **TPDF Dithering**: Triangular Probability Density Function (TPDF) dithering eliminates quantization distortion when exporting to low-bitrate audio formats.

---

### E. Expressive Conditioning & Tag-Emotion Sync

IRIS features six emotion presets parameterized by exaggeration and Classifier-Free Guidance (CFG) weight, complemented by dynamic inline tag parsing (`[happy]`, `[excited]`, `[whisper]`, `[angry]`, `[sad]`, `[calm]`, `[laughter]`, `[sigh]`, `[gasp]`, `[chuckle]`):

**Table I: IRIS Emotion Conditioning & Tag-Sync Presets**

| Preset     | Exaggeration | CFG Weight | Pitch Shift | Intended Prosodic Character                     |
|:-----------|:------------:|:----------:|:-----------:|:------------------------------------------------|
| `neutral`  | 0.05         | 0.70       | 0.0 semitones | Balanced delivery, tight speaker identity preserve |
| `calm`     | 0.00         | 0.75       | -0.5 semitones | Soft cadence, maximum speaker identity lock    |
| `happy`    | 0.25         | 0.55       | +1.0 semitones | Upward pitch inflections, energetic brightness |
| `excited`  | 0.40         | 0.45       | +2.0 semitones | Dynamic range expansion, fast pacing            |
| `sad`      | 0.15         | 0.65       | -1.5 semitones | Lowered pitch contour, extended pauses          |
| `angry`    | 0.35         | 0.50       | +0.5 semitones | Sharp attack, compressed dynamic range          |

---

### F. Speaker Similarity Evaluation

The speaker similarity module (`speaker_similarity.py`) uses SpeechBrain's `speechbrain/spkrec-ecapa-voxceleb` model [9] to extract 192-dimensional ECAPA-TDNN embeddings $E_{\text{ref}}$ and $E_{\text{gen}}$.

$$\text{Cosine Similarity } S = \frac{E_{\text{ref}} \cdot E_{\text{gen}}}{\|E_{\text{ref}}\| \|E_{\text{gen}}\|}$$

The resulting score $S \in [0.0, 1.0]$ is returned in the `X-Speaker-Similarity` HTTP response header.

---

### G. Content Accuracy Evaluation (WER)

The content accuracy module (`content_accuracy.py`) transcribes generated audio using `faster-whisper` (`small.en`, CPU int8 execution) and computes Word Error Rate (WER) via JiWER:

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is total reference words. Returned in the `X-Word-Error-Rate` HTTP header.

---

### H. Prosodic Analysis

Fundamental frequency ($F_0$) standard deviation across voiced frames evaluates pitch dynamics:

$$\sigma_{F_0} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (f0_i - \bar{f0})^2}$$

Returned in the `X-Prosody-Variance` HTTP header.

---

### I. REST API Design

**Table II: IRIS VoiceLib REST API Endpoints**

| Method   | Endpoint              | Description                                                          |
|:---------|:----------------------|:---------------------------------------------------------------------|
| `POST`   | `/auth/register`      | Create new user account                                              |
| `POST`   | `/auth/login`         | Authenticate; receive JWT                                            |
| `GET`    | `/auth/me`            | Retrieve authenticated user profile                                  |
| `POST`   | `/voices`             | Upload reference audio (up to 100MB / 300s) with consent confirmation|
| `GET`    | `/voices`             | List user's cloned voice profiles & acoustic DNA fingerprints       |
| `DELETE` | `/voices/{id}`        | Delete voice profile and cascade generation records                  |
| `POST`   | `/generate`           | Generate TTS audio with emotion presets & DSP mastering              |
| `GET`    | `/generations`        | List generation history with metrics & download links                |
| `GET`    | `/health`             | Service health & database pool status check                          |
| `GET`    | `/model-info`         | Engine metadata, sample rates, & active capabilities                 |
| `GET`    | `/colab-status`       | Check GPU microservice connectivity & latency                        |

---

## IV. Experimental Evaluation

### A. Functional System Validation

**Table III: Functional Validation Summary**

| Capability                                              | Status                    |
|:--------------------------------------------------------|:--------------------------|
| User registration / JWT-authenticated login             | Implemented               |
| Voice reference upload (up to 100 MB / 300s duration)   | Implemented               |
| Voice activity detection & SNR window selection         | Implemented               |
| Deep Acoustic DNA Profiler v2 (F0, LPC Formants, HNR)   | Implemented               |
| Automatic generation hyper-parameter auto-tuning        | Implemented               |
| Dual-engine TTS routing (GPT-SoVITS GPU / Pocket-TTS CPU)| Implemented               |
| Anti-harshness DSP mastering (De-esser, Warmth, Limiter)| Implemented               |
| Tag-emotion sync & inline paralinguistic tag parsing     | Implemented               |
| LRU voice-state cache (50 entries)                      | Implemented               |
| Per-request ECAPA-TDNN speaker similarity evaluation    | Implemented               |
| Per-request Whisper WER evaluation                      | Implemented               |
| Per-request $F_0$ prosody variance diagnostic           | Implemented               |
| Interactive HTML5 Canvas audio visualizers & Glassmorphism UI| Implemented               |
| AsyncPG database connection pooling                     | Implemented               |

---

### B. Proposed Experimental Protocol

1. **Experiment 1 — Speaker Similarity Assessment**: Evaluate 192-dim ECAPA-TDNN cosine similarity across 10 diverse speakers (5M/5F) under 3 reference duration bins (5s, 30s, 300s).
2. **Experiment 2 — Content Accuracy (WER)**: Benchmark WER across 6 emotion presets to quantify the trade-off between prosodic exaggeration and ASR transcription accuracy.
3. **Experiment 3 — Deep Acoustic DNA Auto-Tuning Impact**: Compare default fixed synthesis parameters against acoustic-DNA auto-tuned parameters for speaker identity retention.
4. **Experiment 4 — Reference Preprocessing & DSP Mastering Ablation**: Conduct paired ablation testing (Raw Audio vs. VAD + SNR + DSP Mastered) to measure perceptual quality and objective score improvements.

---

## V. Limitations

1. **Benchmark Scale**: Quantitative validation across large multi-speaker corpora (e.g., LibriTTS, VoxCeleb) remains to be completed.
2. **Human MOS Study**: Objective metrics (ECAPA similarity, WER) serve as automated proxies; formal human listener Mean Opinion Score (MOS) evaluations are planned.
3. **English WER Focus**: Faster-Whisper ASR is currently configured for English (`small.en`); non-English WER evaluation requires multilingual model loading.

---

## VI. Future Work

1. **Execution of Proposed Benchmark Protocol**: Complete controlled quantitative evaluations across all 10 test speakers and 600 synthesis cells.
2. **Multilingual ASR Integration**: Expand WER scoring to support multilingual synthesis via Whisper large-v3.
3. **v2 AI Song Cover Pipeline**: Implement full vocal-instrumental separation via Demucs, zero-shot voice conversion, and automated mixing.

---

## VII. Conclusion

IRIS VoiceLib presents a production-ready, full-stack framework for zero-shot voice cloning, deep acoustic DNA profiling, expressive speech synthesis, anti-harshness vocal mastering, and per-request objective evaluation. By integrating reference preprocessing, dual GPU/CPU synthesis engines, dynamic emotion-tag synchronization, studio DSP mastering, and automated quality metrics (`X-Speaker-Similarity`, `X-Word-Error-Rate`, `X-Prosody-Variance`), IRIS bridges the gap between research-grade zero-shot TTS models and robust, real-world application deployments.

---

## References

[1] S. O. Arik et al., "Deep Voice: Real-time Neural Text-to-Speech," in *Proc. ICML*, vol. 70, pp. 195–204, 2017.  
[2] Y. Wang et al., "Tacotron: Towards End-to-End Speech Synthesis," in *Proc. Interspeech*, pp. 4006–4010, 2017.  
[3] J. Shen et al., "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions," in *Proc. IEEE ICASSP*, pp. 4779–4783, 2018.  
[4] Y. Ren et al., "FastSpeech: Fast, Robust and Controllable Text to Speech," in *NeurIPS*, vol. 32, 2019.  
[5] S. O. Arik et al., "Neural Voice Cloning with a Few Samples," in *NeurIPS*, vol. 31, 2018.  
[6] Y. Jia et al., "Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis," in *NeurIPS*, vol. 31, 2018.  
[7] E. Casanova et al., "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion," in *Proc. ICML*, pp. 2709–2720, 2022.  
[8] C. Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," *arXiv:2301.02111*, 2023.  
[9] B. Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," in *Proc. Interspeech*, pp. 3830–3834, 2020.  
[10] A. Nagrani et al., "VoxCeleb: A Large-Scale Speaker Identification Dataset," in *Proc. Interspeech*, pp. 2616–2620, 2017.  
[11] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. ICML*, pp. 28492–28518, 2023.  
[12] Y. Wang et al., "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis," in *Proc. ICML*, pp. 5180–5189, 2018.  
[13] Y. Li et al., "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis," *arXiv:2205.15439*, 2022.  
[14] Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector," GitHub Repository, 2021.  
[15] T. Virtanen, "Monaural Sound Source Separation by Nonnegative Matrix Factorization," *IEEE TASLP*, vol. 15, no. 3, pp. 1066–1074, 2007.  

---
*Licensed under the MIT License.*
