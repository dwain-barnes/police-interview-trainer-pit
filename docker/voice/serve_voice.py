"""PIT Voice Service — STT, TTS, static files, and model reverse proxy."""

import asyncio
import io
import os

import httpx
import numpy as np
import scipy.io.wavfile as wavfile
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# ── Model server reverse proxy target ────────────────────────────────────────

MODEL_BASE = os.environ.get("MODEL_BASE", "http://pit-model:8000")
VOICE_DIR = os.environ.get("VOICE_DIR", "/voices")

# ── Globals ──────────────────────────────────────────────────────────────────

stt_model = None
tts_model = None
vad_model = None
tts_sr = None
http_client: httpx.AsyncClient = None
voice_files: dict = {"male": [], "female": []}

SAMPLE_RATE = 16000

# Serialise TTS calls (single GPU)
tts_semaphore = asyncio.Semaphore(1)


# ── Lifespan — load models at startup ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_model, tts_model, vad_model, tts_sr, http_client

    # 0. HTTP client for model server reverse proxy
    http_client = httpx.AsyncClient(base_url=MODEL_BASE, timeout=httpx.Timeout(300.0))

    # 1. Moonshine STT (CPU, ~100MB)
    print("[Voice] Loading Moonshine STT model...")
    from fastrtc import get_stt_model
    stt_model = get_stt_model()
    print("[Voice] STT model loaded.")

    # 2. Silero VAD (CPU, ~2MB)
    print("[Voice] Loading Silero VAD model...")
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    print("[Voice] VAD model loaded.")

    # 3. Chatterbox-Turbo TTS (GPU, ~4.5GB VRAM)
    tts_device = os.environ.get("TTS_DEVICE", "cuda")
    if tts_device == "cuda" and not torch.cuda.is_available():
        tts_device = "cpu"
        print("[Voice] CUDA not available, falling back to CPU for TTS.")

    # Chatterbox calls snapshot_download(token=True) which hard-requires
    # a stored HF login.  If HF_TOKEN is provided we log in properly.
    # If not, we monkey-patch snapshot_download to drop the forced token
    # so it falls back to anonymous access for the public Chatterbox repo.
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("[Voice] Logged in to Hugging Face.")
    else:
        import huggingface_hub._snapshot_download as _snap
        _original_snapshot_download = _snap.snapshot_download
        def _patched_snapshot_download(*args, **kwargs):
            kwargs.pop("token", None)
            return _original_snapshot_download(*args, **kwargs)
        _snap.snapshot_download = _patched_snapshot_download
        import huggingface_hub
        huggingface_hub.snapshot_download = _patched_snapshot_download
        print("[Voice] No HF_TOKEN set — patched for anonymous access.")

    print(f"[Voice] Loading Chatterbox-Turbo TTS on {tts_device}...")
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts_model = ChatterboxTurboTTS.from_pretrained(device=tts_device)
    tts_sr = tts_model.sr
    print(f"[Voice] TTS model loaded (sample_rate={tts_sr}).")

    # 4. Scan voice reference clips for voice cloning
    if os.path.isdir(VOICE_DIR):
        for f in sorted(os.listdir(VOICE_DIR)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(VOICE_DIR, f)
            name = f.replace(".wav", "")
            if "female" in f:
                voice_files["female"].append({"id": name, "path": path})
            elif "male" in f:
                voice_files["male"].append({"id": name, "path": path})
        print(f"[Voice] Loaded {len(voice_files['male'])} male + {len(voice_files['female'])} female voice refs from {VOICE_DIR}")
    else:
        print(f"[Voice] Voice directory {VOICE_DIR} not found — voice cloning disabled")

    yield

    await http_client.aclose()
    del stt_model, tts_model, vad_model


app = FastAPI(title="PIT Voice Service", lifespan=lifespan)


# ── Health endpoint ──────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "stt_loaded": stt_model is not None,
        "tts_loaded": tts_model is not None,
    }


# ── Voice listing endpoint ──────────────────────────────────────────────────

@app.get("/api/voices")
async def list_voices():
    return {
        "male": [v["id"] for v in voice_files["male"]],
        "female": [v["id"] for v in voice_files["female"]],
    }


# ── TTS endpoint (REST) ─────────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice_id: str | None = None


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    if not tts_model:
        return Response(status_code=503, content="TTS model not loaded")

    if not req.text.strip():
        return Response(status_code=400, content="Empty text")

    # Resolve voice reference clip if voice_id provided
    audio_prompt = None
    if req.voice_id:
        for gender_list in voice_files.values():
            for v in gender_list:
                if v["id"] == req.voice_id:
                    audio_prompt = v["path"]
                    break
            if audio_prompt:
                break

    async with tts_semaphore:
        loop = asyncio.get_event_loop()
        if audio_prompt:
            wav_tensor = await loop.run_in_executor(
                None, lambda: tts_model.generate(req.text.strip(), audio_prompt_path=audio_prompt)
            )
        else:
            wav_tensor = await loop.run_in_executor(
                None, lambda: tts_model.generate(req.text.strip())
            )

    # Convert torch tensor → int16 WAV bytes
    audio_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    audio_int16 = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, tts_sr, audio_int16)
    buf.seek(0)

    return Response(content=buf.read(), media_type="audio/wav")


# ── STT WebSocket (with Silero VAD for automatic pause detection) ────────────

@app.websocket("/api/ws/stt")
async def stt_websocket(ws: WebSocket):
    await ws.accept()

    # VAD configuration
    FRAME_SIZE = 512          # Silero VAD frame size at 16kHz (32ms)
    SPEECH_THRESHOLD = 0.5    # VAD probability threshold for speech
    PAUSE_FRAMES = 24         # Consecutive silent frames to trigger transcription (~768ms)
    MIN_SPEECH_FRAMES = 3     # Minimum speech frames before we'll bother transcribing (~96ms)

    # Per-connection state
    pcm_buffer = np.array([], dtype=np.int16)
    speech_buffer = np.array([], dtype=np.int16)
    speaking = False
    silence_count = 0
    speech_count = 0

    # Reset VAD LSTM states for this connection
    vad_model.reset_states()

    try:
        while True:
            data = await ws.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)
            pcm_buffer = np.concatenate([pcm_buffer, chunk])

            # Process complete VAD frames
            while len(pcm_buffer) >= FRAME_SIZE:
                frame = pcm_buffer[:FRAME_SIZE]
                pcm_buffer = pcm_buffer[FRAME_SIZE:]

                # Run Silero VAD (expects float32 normalised to [-1, 1])
                frame_f32 = frame.astype(np.float32) / 32768.0
                tensor = torch.from_numpy(frame_f32)
                prob = vad_model(tensor, SAMPLE_RATE).item()

                if prob >= SPEECH_THRESHOLD:
                    # Speech detected
                    speaking = True
                    silence_count = 0
                    speech_count += 1
                    speech_buffer = np.concatenate([speech_buffer, frame])
                elif speaking:
                    # Silence during speech — accumulate and count
                    speech_buffer = np.concatenate([speech_buffer, frame])
                    silence_count += 1

                    if silence_count >= PAUSE_FRAMES and speech_count >= MIN_SPEECH_FRAMES:
                        # Pause detected — transcribe the buffered speech
                        audio_for_stt = speech_buffer.copy()
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(
                            None,
                            lambda buf=audio_for_stt: stt_model.stt((SAMPLE_RATE, buf)),
                        )

                        if text and text.strip():
                            await ws.send_json({
                                "type": "transcript",
                                "text": text.strip(),
                            })

                        # Reset for next utterance
                        speech_buffer = np.array([], dtype=np.int16)
                        speaking = False
                        silence_count = 0
                        speech_count = 0
                        vad_model.reset_states()

    except WebSocketDisconnect:
        # Transcribe any remaining speech on disconnect
        if len(speech_buffer) > FRAME_SIZE * MIN_SPEECH_FRAMES:
            try:
                audio_for_stt = speech_buffer.copy()
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    lambda buf=audio_for_stt: stt_model.stt((SAMPLE_RATE, buf)),
                )
                if text and text.strip():
                    await ws.send_json({
                        "type": "transcript",
                        "text": text.strip(),
                    })
            except Exception:
                pass
    except Exception as e:
        print(f"[STT WS] Error: {e}")
        try:
            await ws.close(code=1011)
        except Exception:
            pass


# ── Model server reverse proxy (/v1/) ────────────────────────────────────────

@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def model_proxy(request: Request, path: str):
    url = f"/v1/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "transfer-encoding", "content-length")}

    resp = await http_client.request(
        method=request.method,
        url=url,
        content=body,
        headers=headers,
    )

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


# ── Static file serving (index.html + scenarios.json) ────────────────────────

INDEX_PATH = os.environ.get("INDEX_PATH", "/app/static/index.html")
SCENARIOS_PATH = os.environ.get("SCENARIOS_PATH", "/app/static/scenarios.json")


@app.get("/")
async def serve_index():
    return FileResponse(
        INDEX_PATH,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/scenarios.json")
async def serve_scenarios():
    return FileResponse(
        SCENARIOS_PATH,
        media_type="application/json",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
