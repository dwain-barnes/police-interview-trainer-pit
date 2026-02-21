"""OpenAI-compatible API server with Harmony channel parsing.

Uses openai_harmony's official encoding.render_conversation_for_completion()
and encoding.parse_messages_from_completion_tokens() to properly handle
Harmony's multi-channel output (analysis/commentary/final).
"""

import json
import os
import time
import uuid

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/pit")
model = None
tokenizer = None
harmony_encoding = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, harmony_encoding
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from openai_harmony import load_harmony_encoding, HarmonyEncodingName

    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading Harmony encoding...")
    harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Build max_memory map from env vars (allows reserving VRAM for TTS on GPU 1)
    max_memory = {}
    gpu0_mem = os.environ.get("MAX_MEMORY_GPU0")
    gpu1_mem = os.environ.get("MAX_MEMORY_GPU1")
    if gpu0_mem:
        max_memory[0] = gpu0_mem
    if gpu1_mem:
        max_memory[1] = gpu1_mem

    print(f"Loading model from {MODEL_PATH}...")
    if max_memory:
        print(f"  max_memory: {max_memory}")

    offload_folder = "/tmp/offload"
    os.makedirs(offload_folder, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory if max_memory else None,
        offload_folder=offload_folder,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Device map: {model.hf_device_map}")
    yield
    del model, tokenizer


app = FastAPI(title="PIT Model Server", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "pit-sft"
    messages: list[Message]
    max_tokens: int = Field(default=512, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: list[str] | str | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "pit-sft", "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    from openai_harmony import Conversation, Message as HMessage, Role

    # Build Harmony conversation from the request messages
    harmony_msgs = []
    for m in request.messages:
        role = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM
                }.get(m.role, Role.USER)
        harmony_msgs.append(HMessage.from_role_and_content(role, m.content))
    convo = Conversation.from_messages(harmony_msgs)

    # Render prompt using Harmony encoding (official method)
    prefill_ids = harmony_encoding.render_conversation_for_completion(
        convo, Role.ASSISTANT
    )
    stop_token_ids = harmony_encoding.stop_tokens_for_assistant_actions()

    prompt_len = len(prefill_ids)
    input_ids = torch.tensor([prefill_ids], device=model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature if request.temperature > 0 else None,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            repetition_penalty=1.1,
            eos_token_id=stop_token_ids,
        )

    completion_ids = outputs[0][prompt_len:].tolist()
    completion_len = len(completion_ids)

    # Parse completion tokens using Harmony encoding (official method)
    content = ""
    try:
        entries = harmony_encoding.parse_messages_from_completion_tokens(
            completion_ids, Role.ASSISTANT
        )

        # Debug: dump all parsed messages
        for i, msg in enumerate(entries):
            d = msg.to_dict()
            c = d.get("content", "")
            text_preview = ""
            if isinstance(c, list):
                text_preview = "".join(
                    p.get("text", "")[:80] if isinstance(p, dict) else str(p)[:80]
                    for p in c
                )
            else:
                text_preview = str(c)[:80]
            print(f"[Harmony] msg[{i}]: channel={d.get('channel')!r} "
                  f"role={d.get('role')!r} text={text_preview!r}")

        def extract_text(d):
            c = d.get("content", "")
            if isinstance(c, list):
                return "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in c
                )
            return str(c)

        # Prefer "final" channel
        for msg in entries:
            d = msg.to_dict()
            if d.get("channel") == "final":
                content = extract_text(d)
                break

        # Fallback: use first assistant-role message with no channel (SFT output)
        if not content:
            for msg in entries:
                d = msg.to_dict()
                text = extract_text(d)
                if text:
                    content = text
                    break

        if content:
            print(f"[Harmony] Selected: {content[:200]!r}")
        else:
            print(f"[Harmony] No content parsed from {len(entries)} entries")
    except Exception as e:
        print(f"[Harmony] Parse error: {e}")

    # Last-resort fallback: decode all tokens with skip_special_tokens
    if not content:
        content = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print(f"[Harmony] Fallback decode: {content[:200]!r}")

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model="pit-sft",
        choices=[
            ChatChoice(
                message=Message(role="assistant", content=content),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_len,
            completion_tokens=completion_len,
            total_tokens=prompt_len + completion_len,
        ),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
