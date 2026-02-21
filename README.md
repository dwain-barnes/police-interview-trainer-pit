# PIT — Police Interview Trainer

A self-hosted AI training environment for practising PEACE-model police interviews. 523 scenarios across 6 training modes, with optional voice interaction via speech-to-text and text-to-speech with voice cloning.

> **3 containers** &middot; **523 scenarios** &middot; **6 training modes** &middot; **Voice I/O**

---

## Disclaimer

**All names, characters, suspects, witnesses, and cases depicted in this application are entirely fictional and have been created solely for training purposes. Any resemblance to real persons, living or dead, or actual events is purely coincidental.**

This software is provided for **educational and training purposes only**. The author accepts no liability for any loss, damage, or consequence arising from the use or misuse of this application. It is the user's responsibility to ensure that their use complies with all applicable laws, regulations, and organisational policies. This tool is not a substitute for accredited police training programmes.

---

<!-- ![PIT Screenshot](screenshot.png) -->

## What It Does

PIT presents trainee officers with realistic interview scenarios based on UK criminal law. A fine-tuned LLM plays the role of suspects and witnesses, responding dynamically to questioning. After an interview, the model can assess performance against PEACE framework criteria.

The system runs entirely on local hardware — no data leaves your network.

## Features

### Training Modes

| Mode | Description |
|------|-------------|
| **Suspect Roleplay** | Interview an AI suspect who responds in-character based on offence, behaviour type, and evidence |
| **Witness Roleplay** | Take a statement from an AI witness with varying recall and emotional states |
| **PEACE Knowledge** | Flashcard-based Q&A covering PEACE framework theory (10 topics) |
| **Assessment** | Paste an interview transcript and receive structured feedback against PEACE criteria |
| **Scenario Presentation** | Review offence briefings with evidence, points to prove, and suspect background |
| **Special Procedures** | Flashcards on vulnerable suspects, appropriate adults, interpreters, and other procedures |

### Voice

- **Speech-to-Text** — Toggle-mic input using Moonshine STT with Silero VAD (server-side voice activity detection)
- **Text-to-Speech** — AI responses read aloud via [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox), featuring over **30 unique cloned voices** (male and female) so each suspect and witness sounds distinct
- **Voice Cloning** — Per-suspect voices generated from short reference audio clips (~5-10s WAV files)

### Scenarios

523 pre-generated scenarios covering 14 offence types (theft, assault, burglary, fraud, drugs, criminal damage, etc.) with 8 suspect behaviour profiles (cooperative, hostile, no-comment, vulnerable, and more). Scenarios include offence details, statute references, evidence lists, points to prove, and suspect background.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser (:3002)                    │
│              index.html + scenarios.json             │
└──────────────┬──────────────────┬────────────────────┘
               │                  │
          /v1/ │           /api/  │  /api/ws/stt
               │                  │
┌──────────────▼──────┐  ┌───────▼──────────────────┐
│     pit-model       │  │       pit-voice           │
│     (internal)      │  │       (:8001 → :3002)     │
│                     │  │                           │
│  llama-server :8080 │  │  Moonshine STT  (CPU)    │
│  Harmony proxy :8000│◄─│  Silero VAD     (CPU)    │
│                     │  │  Chatterbox TTS (GPU)    │
│  GGUF model (Q8_0)  │  │  Static files + proxy    │
│  All GPUs           │  │                           │
└─────────────────────┘  └───────────────────────────┘
```

| Container | Role | Port |
|-----------|------|------|
| `pit-model` | LLM inference — llama.cpp server with Harmony proxy for stop-token handling | 8000 (internal) |
| `pit-voice` | STT (WebSocket), TTS (REST), reverse proxy to model, static file serving | 8001 → exposed on `PIT_PORT` |

The model GGUF is downloaded automatically from Hugging Face on first start and cached in a Docker volume.

## Prerequisites

- **Docker** with Docker Compose v2
- **NVIDIA GPU** (24 GB+ VRAM recommended — tested on 2× RTX 3090)
- **NVIDIA drivers** (550+) and **nvidia-container-toolkit**
- ~20 GB disk for the GGUF model download (cached after first run)

## Quick Start

```bash
git clone https://github.com/dwain-barnes/police-interview-trainer-pit.git
cd police-interview-trainer-pit

cp .env.example .env        # review and edit if needed
docker compose up -d         # first run downloads the model (~20 GB)
```

Open **http://localhost:3002** once both containers report healthy (model startup can take a few minutes on first launch).

Check progress:

```bash
docker compose logs -f pit-model    # watch GGUF download + llama-server startup
docker compose logs -f pit-voice    # watch STT/TTS model loading
```

## Configuration

All configuration is via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PIT_PORT` | `3002` | Host port for the web UI |
| `MODEL_REPO` | `EryriLabs/PIT-GGUF` | Hugging Face repo containing the GGUF model |
| `GGUF_FILE` | `pit_q8_0.gguf` | Filename of the GGUF to download |
| `N_CTX` | `16384` | Context window size (tokens) |
| `N_GPU_LAYERS` | `-1` | Number of layers to offload to GPU (`-1` = all) |
| `TTS_DEVICE` | `cuda` | Device for Chatterbox TTS (`cuda` or `cpu`) |
| `HF_TOKEN` | _(empty)_ | Hugging Face token (only needed for gated/private models) |

### Voice Cloning

To add custom suspect voices, place WAV reference clips (~5-10 seconds) in `./chatterbox_voices/`:

```
chatterbox_voices/
  male_ryan.wav
  female_sarah.wav
```

Available voices appear in the UI voice dropdown and are listed at `/api/voices`.

## Usage

1. **Select a mode** — choose from the six training mode tabs
2. **Pick a scenario** — filter by difficulty (beginner / intermediate / advanced), browse the card grid
3. **Interview** — type or speak your questions; the AI responds in character
4. **Review** — use the Assessment mode to get structured PEACE-framework feedback on your interview
5. **Download** — export the transcript as a text file for review

### Keyboard Shortcuts (Flashcard Modes)

| Key | Action |
|-----|--------|
| Space | Flip card |
| ← / → | Previous / Next |
| Esc | Close overlay |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Inference | [llama.cpp](https://github.com/ggerganov/llama.cpp) (built from source with CUDA) |
| Fine-tuned Model | [EryriLabs/PIT-GGUF](https://huggingface.co/EryriLabs/PIT-GGUF) (Q8_0 quantised) |
| Base Model | [gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b) (21B MoE, 3.6B active params) |
| Speech-to-Text | [Moonshine](https://github.com/usefulsensors/moonshine) via fastrtc |
| Voice Activity Detection | [Silero VAD](https://github.com/snakers4/silero-vad) |
| Text-to-Speech | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) (with voice cloning) |
| API Layer | FastAPI + uvicorn |
| Frontend | Vanilla JS/CSS single-page app (no build step) |
| Containers | Docker Compose with NVIDIA GPU passthrough |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/scenarios.json` | GET | All 523 training scenarios |
| `/v1/chat/completions` | POST | OpenAI-compatible chat completions (proxied to llama.cpp) |
| `/v1/models` | GET | List loaded models |
| `/api/health` | GET | Service health status |
| `/api/voices` | GET | Available voice IDs for TTS cloning |
| `/api/tts` | POST | Text-to-speech synthesis (`{text, voice_id?}`) |
| `/api/ws/stt` | WebSocket | Speech-to-text stream (16 kHz PCM binary frames) |

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

You are free to share the material for non-commercial purposes with appropriate credit, but you may not distribute modified versions. See [LICENSE](LICENSE) for the full text.
