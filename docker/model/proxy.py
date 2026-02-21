"""Proxy for llama-server that injects Harmony stop sequences.

The gpt-oss model uses Harmony format where <|return|> (token 200002) is the
true end-of-generation marker (and IS an EOG token). However, the model
sometimes emits <|end|> (token 200007) instead, which is NOT an EOG token,
causing generation to continue past the intended stop point.

This proxy injects text-based stop sequences so that <|end|> and <|start|>
also halt generation.
"""

import json
import os

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

BACKEND = "http://127.0.0.1:8080"

# Harmony special tokens that should stop generation
HARMONY_STOP_TOKENS = [
    "<|end|>",
    "<|start|>",
]

app = FastAPI(title="PIT Harmony Proxy")


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{BACKEND}/v1/models")
    return Response(
        content=resp.content,
        media_type="application/json",
        status_code=resp.status_code,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.body()
    payload = json.loads(body)

    # --- Inject Harmony stop sequences ---
    existing_stop = payload.get("stop", [])
    if isinstance(existing_stop, str):
        existing_stop = [existing_stop]
    for tok in HARMONY_STOP_TOKENS:
        if tok not in existing_stop:
            existing_stop.append(tok)
    payload["stop"] = existing_stop

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{BACKEND}/v1/chat/completions",
            content=json.dumps(payload),
            headers={"content-type": "application/json"},
        )

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    port = int(os.environ.get("PROXY_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
