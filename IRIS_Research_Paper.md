# IRIS VoiceLib: A Modular Framework for Voice Cloning and Multi-Metric Speech Quality Evaluation

**Aryan Awchare**
Department of Computer Science / Artificial Intelligence Engineering
*[Institution Name]*
*[City, Country]*
*[email@institution.edu]*

---

**Abstract** — Modern neural text-to-speech (TTS) synthesis and zero-shot voice cloning can reproduce a speaker's vocal characteristics from short reference recordings. However, practical deployment of such systems introduces compound challenges extending far beyond the synthesis model: reference audio quality is rarely controlled, expressive nuance is difficult to parameterize, speaker identity must be preserved across variable input conditions, and evaluation is typically limited to subjective listening. This paper presents **IRIS VoiceLib**, a modular, full-stack framework that integrates voice library management, reference-audio preprocessing, configurable neural TTS backends, expressive synthesis conditioning, voice-state caching, and automated multi-metric objective evaluation into a single coherent system. The framework employs Silero-VAD for voice activity detection and signal-to-noise ratio (SNR)-guided segment selection, a pluggable TTS engine layer with implementations for GPT-SoVITS v3 and Kyutai Pocket-TTS, six parameterized emotion presets with configurable exaggeration and classifier-free guidance (CFG) weights, and an embedded evaluation pipeline comprising ECAPA-TDNN cosine speaker-similarity scoring, faster-Whisper automatic speech recognition (ASR) with JiWER word error rate (WER), and fundamental-frequency F0 variance-based prosodic analysis. Speaker identity is assessed using 192-dimensional ECAPA-TDNN embeddings, content accuracy through ASR-derived WER, and prosodic character through F0 standard-deviation tracking. The current work establishes the implemented architecture and evaluation pipeline; comprehensive controlled quantitative benchmarking across multiple speakers, synthesis engines, expressive conditions, and hardware configurations remains an important direction for future work.

**Keywords** — voice cloning, zero-shot TTS, ECAPA-TDNN, speaker similarity, word error rate, expressive speech synthesis, voice activity detection, FastAPI, neural text-to-speech

---

## I. Introduction

Advances in neural text-to-speech synthesis have made zero-shot voice cloning practically feasible: given a short reference recording of a target speaker, modern models can reproduce that speaker's vocal identity in synthesized speech with increasing fidelity [1]–[4]. Despite this technical progress, deploying voice cloning in a real application surface involves a set of engineering and evaluation challenges that receive comparatively less attention in academic literature, which tends to focus on novel model architectures rather than on end-to-end system design.

**Challenge 1 — Reference Audio Quality.** A short audio sample uploaded by a user may contain prolonged silence, background noise, reverberation, low-energy non-speech regions, or codec artefacts. Forwarding such a sample directly to a voice cloning model can degrade the quality of the extracted speaker representation. Reference-audio preprocessing—including voice activity detection (VAD), SNR-based segment selection, and audio normalization—is therefore an important practical step in any deployed system.

**Challenge 2 — Speaker Identity Preservation.** A cloned voice must preserve the target speaker's characteristics, including timbre, accent, prosodic style, and fine-grained vocal quality. Objective speaker similarity metrics derived from speaker-verification embeddings provide an automated proxy for this property and can enable systematic quality monitoring without requiring subjective listening tests for every generation.

**Challenge 3 — Linguistic Accuracy.** A cloned voice that sounds perceptually convincing may still mispronounce, substitute, or omit words from the input text. In production systems, linguistic accuracy must be monitored alongside speaker similarity, since a generation that sounds like the target speaker but fails to reproduce the correct words constitutes a synthesis failure.

**Challenge 4 — Expressiveness.** Practical TTS applications require the ability to modulate the affective character of synthesized speech—for example, to produce neutral narration, excited promotional content, or calm assistive output—while preserving speaker identity. This requires a mechanism for parameterized expressive conditioning.

**Challenge 5 — Deployment and Evaluation Infrastructure.** Research models are typically evaluated in controlled offline settings. Production deployment requires an authenticated API surface, persistent storage for voice profiles and generation histories, a user-facing interface, and embedded evaluation that runs automatically per generation without requiring a dedicated evaluation batch process.

These five challenges jointly motivate the design of IRIS VoiceLib, a modular framework that addresses all five within a single deployable system.

### A. Paper Contributions

This paper makes the following contributions:

1. **System Architecture**: A description of a modular, full-stack voice cloning framework integrating a React SPA frontend, FastAPI backend, PostgreSQL metadata store, S3-compatible object storage, pluggable TTS engine layer, and reference-audio preprocessing pipeline.

2. **Reference-Audio Preprocessing Module**: Implementation of a Silero-VAD-based reference audio selector that identifies speech-active regions and selects the highest-SNR window for use as the voice conditioning prompt.

3. **Pluggable TTS Engine Abstraction**: An abstract base class (BaseTTSEngine) with concrete implementations for GPT-SoVITS v3 (GPU-backed) and Kyutai Pocket-TTS (CPU-compatible), enabling backend substitution without changes to the application layer.

4. **Expressive Conditioning System**: Six emotion presets (neutral, calm, happy, excited, sad, angry) parameterized by exaggeration coefficient and CFG weight, providing a systematic interface for controlling prosodic and stylistic variation.

5. **Automated Multi-Metric Evaluation Pipeline**: Per-request computation of ECAPA-TDNN speaker similarity, faster-Whisper WER, and F0 standard deviation—returned as HTTP response headers (X-Speaker-Similarity, X-Word-Error-Rate, X-Prosody-Variance) for downstream monitoring.

6. **Proposed Experimental Protocol**: A controlled experimental design for future quantitative validation of the framework's components.

The paper is structured as follows. Section II reviews relevant literature and identifies the research gap. Section III details the system methodology. Section IV describes the experimental evaluation framework and functional validation. Section V discusses limitations. Section VI proposes future work. Section VII concludes.

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
Problem: Prior TDNN-based speaker embeddings did not sufficiently capture multi-scale temporal information. Method: ECAPA-TDNN with channel-attention and residual multi-scale connections trained on VoxCeleb. Finding: State-of-the-art equal error rate on VoxCeleb1 and VoxCeleb2 benchmarks. Relevance to IRIS: IRIS's speaker similarity module uses the speechbrain/spkrec-ecapa-voxceleb model to compute 192-dimensional cosine-comparable embeddings, directly leveraging this contribution.

**[10] Nagrani et al. (2017) — VoxCeleb: A Large-Scale Speaker Identification Dataset.**
Problem: Publicly available speaker datasets were small and not representative of real-world conditions. Method: Automated collection of audio-visual speaker-labeled data from YouTube. Finding: Large-scale challenging benchmark for speaker verification in unconstrained environments. Relevance to IRIS: VoxCeleb serves as the training corpus for the ECAPA-TDNN model used in IRIS's evaluation module.

### D. Automatic Speech Recognition for TTS Evaluation

**[11] Radford et al. (2023) — Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).**
Problem: ASR systems struggled with robustness across accents, domains, and audio conditions. Method: Large-scale weakly-supervised training across 680,000 hours of multilingual web audio using an encoder-decoder transformer. Finding: Whisper generalizes robustly across diverse conditions without domain-specific fine-tuning. Relevance to IRIS: IRIS's content accuracy module uses the faster-whisper implementation of Whisper (small.en) on CPU for per-request transcription and WER computation.

### E. Expressive and Controllable TTS

**[12] Wang et al. (2018) — Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis (GST-Tacotron).**
Problem: TTS systems produced monotone prosody without control over speaking style. Method: Global Style Tokens (GST) as a conditioning mechanism for style transfer in Tacotron 2. Finding: Controllable prosodic variation and style transfer without explicit style labels. Relevance to IRIS: GST established the principle of latent conditioning for expressive control, which the CFG-weight parameterization in IRIS's emotion presets extends.

**[13] Li et al. (2022) — StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis.**
Problem: Existing TTS models produced limited prosodic diversity. Method: Style-based generative architecture separating content and style embeddings. Finding: High-quality, diverse prosodic synthesis with improved naturalness. Relevance to IRIS: Reinforces the feasibility of disentangling speaker identity from expressive style, a design goal of IRIS's conditioning parameters.

### F. Voice Activity Detection and Audio Preprocessing

**[14] Silero Team (2021) — Silero VAD: pre-trained enterprise-grade Voice Activity Detector.**
Problem: Reliable, lightweight VAD was needed for production environments without heavy dependencies. Method: Compact ONNX-exportable model trained for binary speech/non-speech classification. Finding: Competitive VAD accuracy with low latency and minimal resource usage. Relevance to IRIS: IRIS's reference-audio preprocessing module (reference_selector.py) uses Silero VAD to detect speech regions before SNR-ranked window selection.

**[15] Virtanen (2007) — Monaural Sound Source Separation by Nonnegative Matrix Factorization with Temporal Continuity and Sparseness Criteria.**
Problem: Sound source separation required principled decomposition of monaural signals. Method: NMF-based separation with temporal and sparseness constraints. Finding: Effective separation of overlapping sound sources in monaural recordings. Relevance to IRIS: Informs signal-separation principles underlying IRIS's preprocessing design and is relevant to the planned v2 song-cover pipeline using Demucs-based vocal separation.

### G. Research Gap

Despite substantial individual progress in neural TTS synthesis, zero-shot voice cloning, speaker verification embeddings, and automatic speech recognition, there exists a comparative lack of published, deployment-oriented frameworks that integrate all of the following within a single system:

1. Reference-audio quality preprocessing with VAD and SNR-based segment selection.
2. A pluggable, abstracted TTS engine layer supporting multiple synthesis backends.
3. Parameterized expressive conditioning presets with documented parameter semantics.
4. Persistent authenticated voice library management.
5. Automated per-request multi-metric objective evaluation (speaker similarity, WER, prosody).
6. Full-stack deployment infrastructure (REST API, SPA frontend, containerized services).

IRIS VoiceLib specifically addresses this integration gap. The contribution is not a new neural TTS model but a system-level framework that makes voice cloning, expressive control, reference processing, voice management, and objective evaluation work together in a production-deployable application.

---

## III. Methodology

### A. System Architecture

IRIS VoiceLib is structured as a three-tier, multi-service system:

```
   +--------------------------------------+
   |   React 18 SPA (Vite + Tailwind)    |
   |  Zustand State  *  Axios JWT Bearer  |
   +------------------+-------------------+
                      |  REST / HTTPS + JWT
   +------------------v-------------------+
   |    FastAPI 0.111 (Python 3.11+)     |
   |  Async Routers * Auth * Controllers  |
   +--------+-----------+-----------------+
            |           |
   +--------v------+ +--v------------------------------+
   |  PostgreSQL   | |         TTS Router               |
   |  (AsyncPG /   | |   (Engine Selection Logic)       |
   |  SQLAlchemy)  | +----------+------------+----------+
   +---------------+            |            |
                         +------v---+  +-----v---------+
   +---------------+     |GPT-SoVITS|  |  Pocket-TTS  |
   |  S3 / MinIO   |     |  Engine  |  |   Engine     |
   |  Object Store |     |(GPU/Colab|  | (CPU-local)  |
   +---------------+     +----------+  +--------------+
                                    |
                      +-------------v-----------------+
                      |     Evaluation Pipeline       |
                      |  ECAPA * WER * F0 Variance    |
                      +-------------------------------+
```

**Frontend.** The React 18 SPA is built with Vite 5 and styled with Tailwind CSS v3 using an Ethereal Glassmorphism design language (OLED black at #050505, radial mesh gradient orbs, backdrop-blur card surfaces, Plus Jakarta Sans variable font). Zustand provides global state management with JWT persistence to localStorage. Axios handles all HTTP communication with automatic JWT injection via request interceptors and standardized error normalization.

**Backend.** The FastAPI application (v0.111, Python 3.11+) operates in an async-first mode using asyncpg and SQLAlchemy 2.0 async sessions. Authentication uses HS256 JWTs via python-jose with bcrypt password hashing (work factor >= 12). The application startup lifespan event creates database tables, ensures the object storage bucket exists, and loads the TTS model into memory—eliminating per-request cold-start latency.

**Database.** PostgreSQL 15 stores three primary entity types: User, Voice, and Generation. Voice records include a consent_confirmed boolean field, providing an audit trail confirming that users have acknowledged voice ownership rights prior to cloning. Alembic is the recommended migration tool for production schema management.

**Object Storage.** Raw reference audio samples and generated WAV files are stored in an S3-compatible object store (MinIO for local development; AWS S3 or Cloudflare R2 for production) via a boto3-based wrapper that abstracts upload, download, presigned URL generation, and deletion operations.

**Docker Compose.** Local development is orchestrated via Docker Compose running four services: postgres, minio, backend, and frontend.

### B. Reference-Audio Preprocessing

The reference-audio preprocessing module (voicelib/backend/app/preprocessing/reference_selector.py) implements a multi-stage pipeline designed to provide the TTS synthesis engine with the cleanest, most informative speech segment extracted from a user-uploaded reference recording.

**Stage 1 — Validation.** Uploaded audio files are validated for MIME type, file size (maximum 20 MB), and duration (3–30 seconds) using mutagen before storage or processing.

**Stage 2 — Voice Activity Detection.** Silero VAD [14] (ONNX variant, loaded via torch.hub) is applied to detect speech-active regions in the reference signal, resampled to 16 kHz for VAD processing. If Silero VAD is unavailable at runtime, an energy-based fallback using librosa.effects.trim is applied automatically.

**Stage 3 — Speech Region Concatenation.** Detected speech timestamps are used to concatenate only speech-active portions of the reference audio, suppressing silence and non-speech segments from the conditioning input.

**Stage 4 — SNR-Based Window Selection.** For reference recordings longer than 12 seconds, a sliding-window SNR scoring procedure selects the highest-quality 8-second window. The SNR estimate compares signal power to the 10th-percentile frame energy (approximating the noise floor):

    SNR_est = 10 * log10( (P_signal + epsilon) / (P_noise_10th + epsilon) )

The window with the highest estimated SNR is selected as the conditioning prompt for the TTS engine.

**Stage 5 — Normalization.** The selected segment is exported as 16-bit PCM WAV. Downstream audio mastering applies peak normalization and a high-pass filter to reduce low-frequency noise artefacts.

> NOTE [PROPOSED — NOT YET EXPERIMENTALLY VERIFIED]: The preprocessing stage is designed to provide a cleaner and more informative reference signal to the TTS conditioning module. A controlled ablation study comparing preprocessing-enabled vs. preprocessing-disabled synthesis quality (measured via ECAPA-TDNN speaker similarity and WER) is a planned future experiment to quantify the contribution of this module. No quality improvement claim is made without that experiment.

### C. TTS Engine Layer

IRIS implements a pluggable TTS engine architecture via an abstract base class defined in tts_engines/base.py:

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

All TTS backends implement this interface, enabling the application layer to interact with any engine through a single, consistent interface. The current repository contains the following engine implementations:

**1. GPT-SoVITS Engine (gptsovits_engine.py).** This engine routes synthesis requests to a Chatterbox / GPT-SoVITS v3 service running on a Google Colab Tesla T4 GPU, accessed over an Ngrok HTTPS tunnel configured via the COLAB_GPU_API_URL environment variable. The engine produces 32 kHz audio with zero-shot speaker conditioning via acoustic prompt embeddings. It also implements paralinguistic tag parsing ([laughter], [sigh], [whisper], [gasp], [chuckle]) and applies emotion modulation factors (pitch shift in semitones, speed scale, energy scale) prior to synthesis. Eight emotion modulation configurations are defined in EMOTION_PRESETS, including fearful and surprised in addition to the six primary user-facing presets.

**2. Pocket-TTS Engine (pocket_engine.py).** This engine uses Kyutai Labs' pocket-tts library, which is CPU-compatible and does not require a GPU. It generates 24 kHz audio and is used when the Colab GPU service is unavailable or when the local CPU path is explicitly selected. The engine maintains an internal LRU cache of 50 voice states.

**3. Mock TTS Engine.** A MockTTSModel fallback can be activated via the VOICELIB_USE_MOCK_TTS=true environment variable, generating clean 1-second reference tones for development and testing without requiring PyTorch or TTS model installation.

The tts_engines/__init__.py routes requests to the appropriate engine based on configuration and Colab availability, which is checked via GET /colab-status.

> IMPORTANT: The framework is designed to support multiple inference backends, including GPU-oriented and CPU-compatible synthesis engines. This is an architectural capability, not an experimentally validated hardware-performance result. No controlled CPU-vs-GPU benchmark has been performed for this study.

### D. Voice-State Management and Caching

Voice state management is centralized in voicelib/backend/app/tts.py. Key design decisions documented in the repository include:

**Startup Preloading.** The TTS model is loaded into memory during the FastAPI lifespan startup event, blocking until the model is ready. This design eliminates a 30–60 second cold-start penalty that would otherwise occur on the first synthesis request after server initialization.

**LRU Voice-State Cache.** A thread-safe LRU cache with a maximum capacity of 50 voice entries holds derived voice representations (acoustic prompt embeddings or internal model states) in memory. When a generation request arrives for a voice whose state is not cached, the system downloads the reference audio from S3/MinIO, re-derives the voice state, and populates the cache. This design ensures that server restarts do not permanently lose voice states—they are re-derived from persistent storage on first use—while bounding memory consumption.

**Thread Safety.** Cache operations use Python threading locks, making the cache safe for concurrent use in Uvicorn's threaded worker context.

### E. Expressive Conditioning

IRIS implements six emotion conditioning presets, each defined by two synthesis parameters:

- **Exaggeration**: Controls the degree of prosodic deviation from the speaker's neutral baseline. Higher values increase pitch excursion, energy variation, and rate dynamics.
- **CFG Weight (Classifier-Free Guidance Weight)**: Controls how closely the synthesis adheres to the speaker's conditioning signal. Higher CFG weights produce tighter speaker identity preservation; lower values allow more stylistic variation.

**Table I: IRIS Emotion Conditioning Presets**

| Emotion    | Exaggeration | CFG Weight | Intended Prosodic Character              |
|:-----------|:------------:|:----------:|:-----------------------------------------|
| `neutral`  | 0.05         | 0.70       | Balanced delivery, tight speaker identity |
| `calm`     | 0.00         | 0.75       | Maximum speaker identity preservation     |
| `happy`    | 0.25         | 0.55       | Lively pitch excursion, positive inflection |
| `excited`  | 0.40         | 0.45       | High dynamic range, energetic cadence     |
| `sad`      | 0.15         | 0.65       | Slower pacing, downward pitch contour     |
| `angry`    | 0.35         | 0.50       | Sharp attack, higher intensity            |

*These values are system configuration parameters representing the current implementation design. Whether and to what degree they modulate the intended prosodic dimensions remains to be quantified in a controlled experiment (see Section IV-C, Experiment 3).*

The GPT-SoVITS engine additionally applies pitch-shift (in semitones), speed scale, and energy scale modulation factors per emotion preset, providing a second complementary modulation pathway beyond the exaggeration and CFG parameters.

> NOTE: The framework provides parameterized expressive conditioning presets intended to alter prosodic and stylistic characteristics. No claim is made that these presets have been experimentally verified to quantitatively reproduce or classify human emotional states. Human perceptual evaluation would be required to validate the subjective expressiveness of each preset.

### F. Speaker Similarity Evaluation

The speaker similarity module (voicelib/backend/app/evaluation/speaker_similarity.py) uses SpeechBrain's speechbrain/spkrec-ecapa-voxceleb model [9]—a VoxCeleb-pretrained ECAPA-TDNN speaker encoder—to compute 192-dimensional speaker embeddings.

Given a reference audio file and a generated audio file, the module:
1. Loads each audio signal via SpeechBrain's load_audio method (automatic resampling to 16 kHz).
2. Computes embeddings E_ref and E_gen using encode_batch with no-gradient inference.
3. Calculates cosine similarity:

    S = (E_ref · E_gen) / (||E_ref|| * ||E_gen||)

4. Clamps the result to [0.0, 1.0].
5. Returns the score as the X-Speaker-Similarity HTTP response header.

A score of 1.0 indicates maximum embedding-space similarity. A score near 0.0 indicates low speaker resemblance at the embedding level.

> IMPORTANT: ECAPA-TDNN cosine similarity is used as an objective proxy for speaker identity preservation and should not be interpreted as equivalent to human perceptual speaker similarity. Human listeners may perceive speaker similarity differently from what the embedding geometry captures, particularly for short utterances or when prosodic style varies substantially from the reference.

### G. Content Accuracy Evaluation

The content accuracy module (voicelib/backend/app/evaluation/content_accuracy.py) uses the faster-whisper library with the small.en model (quantized to int8, CPU inference) to transcribe generated audio and compute WER against the intended input text using JiWER.

**Transcription Pipeline:**
1. The faster-whisper model (WhisperModel("small.en", device="cpu", compute_type="int8")) is loaded once and cached thread-safely at first use.
2. The generated WAV file is transcribed with beam_size=5 and English language forcing.
3. Text normalization is applied to both reference and transcription (lowercase, remove punctuation, strip extra spaces).
4. WER is computed using jiwer.wer:

    WER = (S + D + I) / N

   where S = substitutions, D = deletions, I = insertions, and N = total reference words. Lower WER indicates higher linguistic fidelity.

5. The score is returned as the X-Word-Error-Rate HTTP response header.

WER is an important complementary dimension to speaker similarity: a generation may achieve high speaker similarity while failing to reproduce the intended words (high WER), or it may reproduce words faithfully while exhibiting poor speaker identity (low similarity). The two metrics together provide a more complete diagnostic picture of synthesis quality.

### H. Prosodic Analysis

The prosody evaluation component tracks fundamental frequency (F0) standard deviation across voiced frames of the synthesized audio, returned as the X-Prosody-Variance response header.

    sigma_F0 = sqrt( (1/N) * sum_i (f0_i - mean_f0)^2 )

where f0_i is the fundamental frequency of the i-th voiced frame and mean_f0 is the mean over N voiced frames.

F0 standard deviation is a descriptive measure of pitch dynamics. It can be compared across emotion conditions (e.g., a higher sigma_F0 in excited vs. neutral would confirm that the preset modulates pitch dynamics as intended), but it should not be interpreted as a universal quality score—appropriate F0 variance depends on speaking style, text content, and speaker identity.

### I. REST API Design

IRIS exposes a RESTful API. Authentication endpoints are public; all other endpoints require JWT Bearer token authorization.

**Table II: IRIS VoiceLib REST API Endpoints**

| Method   | Endpoint              | Description                                  |
|:---------|:----------------------|:---------------------------------------------|
| `POST`   | `/auth/register`      | Create new user account                      |
| `POST`   | `/auth/login`         | Authenticate; receive JWT                    |
| `GET`    | `/auth/me`            | Retrieve authenticated user info             |
| `POST`   | `/voices`             | Upload reference audio with consent          |
| `GET`    | `/voices`             | List all voices for authenticated user       |
| `DELETE` | `/voices/{id}`        | Delete voice and cascade to generations      |
| `POST`   | `/generate`           | Generate TTS audio                           |
| `GET`    | `/generations`        | List generation history (paginated)          |
| `GET`    | `/health`             | Service health check                         |
| `GET`    | `/model-info`         | TTS engine metadata and capabilities         |
| `GET`    | `/colab-status`       | Check Colab GPU service availability         |

Generation responses include three evaluation metric headers—X-Speaker-Similarity, X-Word-Error-Rate, and X-Prosody-Variance—making quality metrics immediately available to the consuming application without a separate evaluation request.

---

## IV. Experimental Evaluation

### A. Functional System Validation

The following functional capabilities have been implemented and verified through development and integration testing:

**Table III: Functional Validation Summary**

| Capability                                              | Status                    |
|:--------------------------------------------------------|:--------------------------|
| User registration / JWT-authenticated login             | Implemented               |
| Voice reference upload with consent confirmation        | Implemented               |
| Server-side audio validation (MIME, size, duration)     | Implemented               |
| Voice storage in PostgreSQL + S3/MinIO                  | Implemented               |
| Voice library listing and deletion                      | Implemented               |
| TTS generation via Pocket-TTS engine (CPU)              | Implemented               |
| TTS generation via GPT-SoVITS engine (GPU/Colab)        | Implemented (Colab-dependent) |
| LRU voice-state cache (50 voices) with S3 miss recovery | Implemented               |
| Per-request ECAPA-TDNN speaker similarity evaluation    | Implemented               |
| Per-request Whisper WER evaluation                      | Implemented               |
| Per-request F0 prosody variance evaluation              | Implemented               |
| Evaluation results returned as HTTP response headers    | Implemented               |
| Generation history with waveform playback and download  | Implemented               |
| Emotion conditioning presets (6 presets)                | Implemented               |
| Silero VAD reference preprocessing                      | Implemented               |
| Colab GPU status check (/colab-status)                  | Implemented               |
| Mock TTS fallback for development mode                  | Implemented               |
| Docker Compose local deployment                         | Implemented               |

> NOTE: "Implemented" indicates that the corresponding code is present in the repository and integrated into the system. These entries do not represent results of a controlled quantitative benchmark study.

### B. Dataset

No dedicated benchmark dataset has been assembled for this study. The system has been verified against developer-provided reference recordings during integration testing. A formal benchmark dataset is specified in the proposed experimental protocol in Section IV-C.

### C. Evaluation Metrics

**Table IV: IRIS Evaluation Metric Summary**

| Dimension         | Metric                      | Tool                      | Return Path               |
|:------------------|:----------------------------|:--------------------------|:--------------------------|
| Speaker identity  | ECAPA-TDNN cosine similarity| SpeechBrain               | X-Speaker-Similarity header |
| Linguistic fidelity | Word Error Rate (WER)     | faster-Whisper + JiWER    | X-Word-Error-Rate header  |
| Prosodic dynamics | F0 standard deviation       | Pitch frame analysis      | X-Prosody-Variance header |
| System latency    | End-to-end generation time  | Per-request timing        | —                         |
| Naturalness       | Mean Opinion Score (MOS)    | Human evaluation          | — (planned)               |

### D. Quantitative Results

> DISCLOSURE: Comprehensive controlled experiments across multiple speakers, synthesis engines, expressive conditions, and hardware configurations have not yet been conducted. No numerical results are fabricated or extrapolated in this paper.

What can be stated based on development-level verification:

- The evaluation pipeline (speaker similarity, WER, prosody) is embedded in the synthesis request path and returns non-null scores for valid synthesis outputs.
- The preprocessing module produces valid segmented audio from both short (< 12 s) and long (>= 12 s) reference recordings.
- Both the Pocket-TTS engine (CPU) and the GPT-SoVITS engine (GPU/Colab) produce intelligible synthesized audio from reference voice inputs.

Comprehensive quantitative results—including mean speaker similarity, mean WER, F0 statistics per emotion, and latency distributions—are designated as future work contingent on the experimental protocol below.

### E. Proposed Experimental Protocol

The following experimental design is proposed for future controlled quantitative validation. This protocol has not yet been executed; it is presented as a concrete research plan.

**Dataset Construction:**
- 10 consenting speakers (5 male, 5 female), balanced for accent diversity.
- 3 reference recordings per speaker (5 s, 15 s, and 30 s) to evaluate sensitivity to reference duration.
- 5 text prompts per speaker (10 to 50 words, covering declarative, interrogative, and expressive sentence types).
- 4 expressive conditions: neutral, happy, sad, excited.

This produces 10 x 5 x 4 = 200 condition cells, with 3 reference-duration variants yielding up to 600 generated samples for Experiments 1–3, plus preprocessing ablation samples for Experiment 4.

---

**Experiment 1 — Speaker Similarity Assessment**

Research Question: What degree of speaker embedding-space similarity does IRIS achieve between reference and synthesized audio?

Protocol:
1. For each generated sample, extract E_ref and E_gen using ECAPA-TDNN.
2. Compute cosine similarity S.
3. Report mean +/- standard deviation per engine, per emotion condition, and per reference duration.

Expected Output: Speaker similarity distribution tables stratified by engine, emotion, and reference length.

---

**Experiment 2 — Content Accuracy (WER)**

Research Question: Does expressive conditioning measurably affect linguistic fidelity?

Protocol:
1. For each generated sample, transcribe with faster-Whisper.
2. Compute WER against the input text prompt.
3. Report mean +/- SD per engine and per emotion condition.

Hypothesis: High-exaggeration presets (excited, angry) may show slightly elevated WER relative to neutral due to prosodic distortions that affect ASR transcription accuracy.

---

**Experiment 3 — Expressive Conditioning Characterization**

Research Question: Do the emotion presets systematically modify prosodic properties (F0 mean, sigma_F0, duration, energy) while preserving speaker identity?

Protocol:
1. For each speaker, synthesize the same sentence under all six emotion conditions.
2. Extract: F0 mean, F0 standard deviation, utterance duration, RMS energy.
3. Report per-emotion statistics across speakers.
4. Compute ECAPA speaker similarity between each emotional variant and the reference to assess identity preservation under style change.

Expected Output: A table of prosodic measurements per condition that can confirm or refute whether exaggeration and CFG parameter variations produce the intended prosodic modifications.

---

**Experiment 4 — Reference-Audio Preprocessing Ablation**

Research Question: Does VAD + SNR-based preprocessing improve downstream speaker similarity and content accuracy?

Protocol:
1. Condition A: Raw reference audio passed directly to the TTS engine.
2. Condition B: Silero-VAD preprocessed audio passed to the TTS engine.
3. For each reference recording and each condition, generate speech with the neutral preset.
4. Measure ECAPA speaker similarity and WER for both conditions.
5. Compare means with paired statistical tests.

This experiment would provide direct evidence for or against the utility of the preprocessing module—a result currently unavailable.

---

**Experiment 5 — LRU Cache Effect on Latency**

Research Question: What is the latency reduction achieved by the voice-state LRU cache?

Protocol:
1. Run synthesis requests with cache disabled (voice state re-derived from storage each time).
2. Run identical requests with cache enabled.
3. Measure: first-generation latency, repeated-generation latency, cache hit rate.
4. Report speedup ratio +/- SD.

NOTE: No speedup claim is made until this measurement is performed.

---

**Experiment 6 — Hardware Configuration Comparison (Optional)**

Research Question: What are the synthesis quality and latency trade-offs between the CPU Pocket-TTS backend and the GPU GPT-SoVITS backend?

Protocol: Only to be executed if a controlled benchmark is actually run. Metrics: speaker similarity, WER, end-to-end latency, real-time factor (RTF = synthesis duration / audio duration), memory consumption.

> NOTE: Hardware comparison currently belongs in Future Work. The framework is designed to support multiple inference backends, including GPU-oriented and CPU-compatible synthesis engines. No controlled CPU-vs-GPU performance comparison has been performed for this study.

### F. Discussion

IRIS VoiceLib demonstrates that an integrated, modular voice cloning and evaluation system can be practically implemented with a focused codebase. Several design decisions reflect deliberate engineering trade-offs:

**Reference preprocessing vs. passthrough.** The preprocessing pipeline adds a processing step between audio upload and synthesis conditioning. For short, clean reference recordings this step has minimal effect. For longer or noisier recordings, the SNR-based window selection may provide a meaningfully better conditioning signal. Whether this translates to measurable quality improvements in downstream synthesis is the subject of the planned Experiment 4 ablation.

**LRU caching trade-off.** The 50-voice LRU cache prevents unbounded memory growth while ensuring that frequently used voices do not incur the storage access latency of re-deriving their voice state on every request. The capacity of 50 was chosen as a practical bound; in multi-user production deployments this may require tuning based on observed active-voice distributions.

**Dual engine architecture.** The framework does not present the CPU/GPU dual-engine design as a validated performance advantage. Rather, it is an architectural feature enabling deployment in environments where GPU access varies—a developer running locally on CPU vs. a GPU-backed Colab session for higher-quality synthesis. The comparative performance of these two paths remains to be experimentally characterized.

**Evaluation header design.** Returning evaluation metrics as HTTP headers (X-Speaker-Similarity, X-Word-Error-Rate, X-Prosody-Variance) is a pragmatic design choice that makes evaluation results immediately consumable by the frontend and any downstream monitoring system without requiring separate API calls or post-processing pipelines.

---

## V. Limitations

The current implementation has the following acknowledged limitations:

1. **No Large-Scale Benchmark.** The system has been verified through development-level integration testing but has not been evaluated on a large-scale benchmark dataset with diverse speakers, accents, and text content. Quantitative conclusions about synthesis quality cannot yet be drawn.

2. **No Controlled CPU/GPU Comparative Experiment.** The dual-engine architecture provides backend flexibility, but no controlled experiment comparing Pocket-TTS (CPU) and GPT-SoVITS (GPU) on quality, latency, or real-time factor has been performed.

3. **No Mean Opinion Score (MOS) Study.** Speaker similarity and WER are objective proxies for synthesis quality. Human perceptual evaluation—including MOS for naturalness, comparative MOS for speaker similarity, and expressiveness ratings—has not been conducted.

4. **No Reference Preprocessing Ablation.** The contribution of VAD + SNR preprocessing to downstream synthesis quality has not been quantified in a controlled comparison.

5. **No Cache Latency Measurement.** The latency benefit of the LRU voice-state cache has not been measured with a controlled benchmark.

6. **Colab Dependency.** The GPT-SoVITS engine depends on an externally hosted Colab GPU microservice with an Ngrok tunnel. This is a development-convenience deployment pattern, not a production-grade configuration. The endpoint is ephemeral and requires manual session management.

7. **English Only.** The current evaluation pipeline uses faster-whisper with small.en, limiting content accuracy evaluation to English-language synthesis. The preprocessing and speaker similarity modules are language-agnostic, but the WER evaluation component is not.

8. **Emotion Preset Calibration.** The emotion preset parameter values (exaggeration, CFG weight) are system configuration choices based on design intent, not experimentally optimized values. Whether each preset reliably produces the intended prosodic modification across different speakers and engines has not been validated.

9. **Song Cover Pipeline.** The v2 song cover feature (Demucs vocal separation, voice conversion, genre transformation, mixing) is currently a stub—song_cover.py router contains no active routes, and SongCover.jsx displays a "Coming Soon" placeholder. It is future work.

---

## VI. Future Work

**Controlled Multi-Speaker Benchmark.** Execute the proposed experimental protocol (Section IV-E) with at least 10 consenting speakers, producing quantitative speaker similarity, WER, and prosodic statistics across emotion conditions and reference durations.

**Reference-Audio Preprocessing Ablation.** Run the paired ablation (Experiment 4) to determine whether VAD + SNR preprocessing measurably improves speaker similarity or WER relative to a raw-reference baseline.

**Human MOS Evaluation.** Conduct a listener study with human raters scoring naturalness (MOS), speaker similarity (comparative MOS), and expressiveness for a representative set of generated samples.

**CPU/GPU Hardware Comparison.** Execute a controlled comparison (Experiment 6) of the Pocket-TTS (CPU) and GPT-SoVITS (GPU) backends on matched inputs, measuring latency, real-time factor, memory consumption, speaker similarity, and WER.

**LRU Cache Latency Measurement.** Measure first-generation vs. cache-hit latency to quantify the operational benefit of the voice-state cache.

**Memory Profiling.** Profile server memory consumption under variable-load conditions to determine optimal LRU cache capacity for production deployment.

**Multilingual Voice Cloning.** Extend the preprocessing and evaluation pipeline to support non-English reference audio and synthesis, using multilingual Whisper model variants for WER evaluation.

**Improved Emotion Disentanglement.** Investigate whether speaker identity and expressive style can be more fully disentangled, enabling emotion variation without measurable degradation in speaker similarity.

**Improved Emotion Evaluation.** Develop objective and subjective metrics for evaluating emotion accuracy and naturalness under different conditioning presets, beyond F0 variance alone.

**AI Song Cover Pipeline (v2).** Implement the planned v2 feature: Demucs-based vocal/instrumental separation, voice conversion using cloned voice profiles, optional genre transformation, and final audio mixing and export.

**Production GPU Deployment.** Replace the ephemeral Colab/Ngrok GPU path with a production-grade GPU inference server (containerized GPU instance with CUDA) to enable stable high-quality synthesis in production environments.

---

## VII. Conclusion

IRIS VoiceLib presents a modular, full-stack voice cloning and speech synthesis framework that integrates reference-audio preprocessing, a pluggable TTS engine layer, parameterized expressive conditioning, persistent voice library management, authenticated API access, and automated per-request multi-metric speech quality evaluation. The framework's implementation addresses five practical challenges in deploying voice cloning systems: reference quality, speaker identity preservation, linguistic accuracy, expressiveness, and evaluation infrastructure.

The current contribution is primarily architectural and implementation-oriented. IRIS does not introduce a novel neural TTS model; rather, it provides the integration layer that makes voice cloning, expressive control, reference-audio optimization, voice management, and objective evaluation function together in a production-deployable application. The evaluation pipeline—comprising ECAPA-TDNN cosine speaker similarity, faster-Whisper word error rate, and F0 prosody variance—is embedded directly in the synthesis request path and provides automated, per-generation quality signals without requiring separate offline evaluation runs.

Quantitative validation across multiple speakers, synthesis engines, expressive conditions, and hardware configurations remains an important direction for future work. A concrete experimental protocol has been proposed to guide this work. When executed, this protocol will produce the first systematic quantitative characterization of IRIS's synthesis quality and evaluation pipeline accuracy.

---

## References

[1] S. O. Arik, M. Chrzanowski, A. Coates, G. Diamos, A. Gibiansky, Y. Kang, X. Li, J. Miller, A. Ng, J. Raiman, S. Sengupta, and M. Shoeybi, "Deep Voice: Real-time Neural Text-to-Speech," in *Proc. 34th International Conference on Machine Learning (ICML)*, Sydney, Australia, vol. 70, pp. 195–204, 2017.

[2] Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio, Q. Le, Y. Agiomyrgiannakis, R. Clark, and R. A. Saurous, "Tacotron: Towards End-to-End Speech Synthesis," in *Proc. Interspeech*, Stockholm, Sweden, pp. 4006–4010, 2017.

[3] J. Shen, R. Pang, R. J. Weiss, M. Schuster, N. Jaitly, Z. Yang, Z. Chen, Y. Zhang, Y. Wang, R. Skerry-Ryan, R. A. Saurous, Y. Agiomyrgiannakis, and Y. Wu, "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Calgary, Canada, pp. 4779–4783, 2018.

[4] Y. Ren, Y. Ruan, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, "FastSpeech: Fast, Robust and Controllable Text to Speech," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.

[5] S. O. Arik, J. Chen, K. Peng, W. Ping, and Y. Zhou, "Neural Voice Cloning with a Few Samples," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[6] Y. Jia, Y. Zhang, R. J. Weiss, Q. Wang, J. Shen, F. Ren, P. Nguyen, R. Pang, I. L. Moreno, Y. Wu, and R. A. Saurous, "Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

[7] E. Casanova, J. Weber, C. D. Shulby, A. C. Junior, E. Golge, and M. A. Ponti, "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone," in *Proc. 39th International Conference on Machine Learning (ICML)*, PMLR vol. 162, pp. 2709–2720, 2022.

[8] C. Wang, S. Chen, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei, "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," *arXiv preprint arXiv:2301.02111*, 2023.

[9] B. Desplanques, J. Thienpondt, and K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," in *Proc. Interspeech*, Shanghai, China, pp. 3830–3834, 2020.

[10] A. Nagrani, J. S. Chung, and A. Zisserman, "VoxCeleb: A Large-Scale Speaker Identification Dataset," in *Proc. Interspeech*, Stockholm, Sweden, pp. 2616–2620, 2017.

[11] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust Speech Recognition via Large-Scale Weak Supervision," in *Proc. 40th International Conference on Machine Learning (ICML)*, PMLR vol. 202, pp. 28492–28518, 2023.

[12] Y. Wang, D. Stanton, Y. Zhang, R. J. Skerry-Ryan, E. Battenberg, J. Shor, Y. Xiao, Y. Jia, F. Ren, and R. A. Saurous, "Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis," in *Proc. 35th International Conference on Machine Learning (ICML)*, Stockholm, Sweden, vol. 80, pp. 5180–5189, 2018.

[13] Y. Li, Z. Han, C. Xu, and J. Guo, "StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis," *arXiv preprint arXiv:2205.15439*, 2022.

[14] I. Silero Team, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector," GitHub Repository, https://github.com/snakers4/silero-vad, 2021.

[15] T. Virtanen, "Monaural Sound Source Separation by Nonnegative Matrix Factorization with Temporal Continuity and Sparseness Criteria," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 15, no. 3, pp. 1066–1074, 2007.

[16] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio," *arXiv preprint arXiv:1609.03499*, 2016.

[17] J. S. Chung, A. Nagrani, and A. Zisserman, "VoxCeleb2: Deep Speaker Recognition," in *Proc. Interspeech*, Hyderabad, India, pp. 1086–1090, 2018.

[18] Z. Kong, J. Ping, J. Huang, K. Zhao, and B. Catanzaro, "DiffWave: A Versatile Diffusion Model for Audio Synthesis," in *Proc. 9th International Conference on Learning Representations (ICLR)*, 2021.

[19] J. Kim, J. Kong, and J. Son, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech," in *Proc. 38th International Conference on Machine Learning (ICML)*, PMLR vol. 139, pp. 5530–5540, 2021.

[20] M. Binski, J. Donahue, S. Dieleman, A. Clark, E. Elsen, N. Casagrande, L. C. Cobo, and K. Simonyan, "High Fidelity Speech Synthesis with Adversarial Networks," in *Proc. 8th International Conference on Learning Representations (ICLR)*, 2020.

[21] R. Skerry-Ryan, E. Battenberg, Y. Xiao, Y. Wang, D. Stanton, J. Shor, R. J. Weiss, R. Clark, and R. A. Saurous, "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron," in *Proc. 35th International Conference on Machine Learning (ICML)*, Stockholm, Sweden, vol. 80, pp. 4700–4709, 2018.

[22] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 12449–12460, 2020.

[23] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour, "AudioLM: a Language Modeling Approach to Audio Generation," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 31, pp. 2523–2533, 2023.

[24] A. Nagrani, J. S. Chung, W. Xie, and A. Zisserman, "Voxceleb: Large-scale speaker verification in the wild," *Computer Speech and Language*, vol. 60, p. 101027, 2020.

[25] B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, "librosa: Audio and Music Signal Analysis in Python," in *Proc. 14th Python in Science Conference (SciPy)*, Austin, TX, pp. 18–24, 2015.

---

*This paper describes the IRIS VoiceLib framework at its current implementation stage. All architectural claims are verified against the repository source code. All experimental results are classified as either functional validation (implemented and integration-tested) or proposed (not yet executed). No fabricated numerical benchmarks are presented.*

*Licensed under the MIT License.*
