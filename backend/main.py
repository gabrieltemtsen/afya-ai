"""
AfyaAI — Voice-first AI health triage for Africa
Backend: FastAPI + Gemini Live (real-time bidirectional audio + vision)
"""
import asyncio
import base64
import json
import os
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from google import genai
from google.genai import types

from stable_mode import generate_triage_text, generate_tts_audio_b64

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("afya")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL_LIVE = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
MODEL_TEXT = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
MODEL_TTS = os.environ.get("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")

SYSTEM_PROMPT = """You are AfyaAI — a warm, compassionate voice-first AI health assistant for Africa.

YOUR ROLE:
- Listen to patients describe their symptoms
- Ask 1-2 simple follow-up questions at a time  
- Give a clear triage: MILD (home care) / MODERATE (clinic soon) / SEVERE (hospital NOW)

COMMUNICATION:
- Speak in the SAME language as the patient: English, Nigerian Pidgin, Yoruba, Hausa, Igbo, Swahili, French, etc.
- Be warm and calm. Simple words. Short sentences.
- If the patient sends a photo, describe what you see and factor it into your assessment.

RULES:
- Always say you are an AI, not a real doctor
- EMERGENCY = go to hospital immediately: chest pain, can't breathe, unconscious, heavy bleeding, stroke signs, severe burns, poisoning
- Never give specific drug dosages
- Never diagnose definitively — say "this sounds like it could be..."

Start by warmly greeting the patient and asking what brings them in today."""

app = FastAPI(title="AfyaAI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "key_set": bool(GEMINI_API_KEY),
        "models": {"live": MODEL_LIVE, "text": MODEL_TEXT, "tts": MODEL_TTS},
    }


@app.post("/api/chat")
async def chat_api(payload: dict = Body(...)):
    """Stable chat endpoint.

    Input: { text: string, langHint?: string }
    Output: { text: string, audio_b64?: string, mime_type?: string }
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}

    text = (payload.get("text") or "").strip()
    if not text:
        return {"error": "Missing text"}

    client = genai.Client(api_key=GEMINI_API_KEY)
    reply_text = generate_triage_text(client, MODEL_TEXT, SYSTEM_PROMPT, text)

    # TTS is optional — if it fails, we still return text.
    tts = generate_tts_audio_b64(client, MODEL_TTS, reply_text)
    return {"text": reply_text, **tts}


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    log.info("New voice session")

    if not GEMINI_API_KEY:
        await websocket.send_text(json.dumps({"type": "error", "message": "Server: GEMINI_API_KEY not set"}))
        await websocket.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Keep config MINIMAL to avoid "invalid argument" from preview endpoints.
    # We can add voice selection etc later once the session is stable.
    config = {
        "response_modalities": ["AUDIO", "TEXT"],
        "system_instruction": SYSTEM_PROMPT,
    }

    try:
        async with client.aio.live.connect(model=MODEL_LIVE, config=config) as session:
            log.info(f"Gemini Live session open (model={MODEL_LIVE})")
            await websocket.send_text(json.dumps({"type": "ready"}))

            async def from_browser():
                try:
                    while True:
                        raw = await websocket.receive_text()
                        msg = json.loads(raw)
                        kind = msg.get("type")

                        if kind == "audio":
                            audio_bytes = base64.b64decode(msg["data"])
                            rate = int(msg.get("rate") or 24000)
                            mime = f"audio/pcm;rate={rate}"
                            log.info(f"[ws] audio chunk bytes={len(audio_bytes)} rate={rate}")
                            await session.send_realtime_input(
                                audio=types.Blob(data=audio_bytes, mime_type=mime)
                            )

                        elif kind == "image":
                            image_bytes = base64.b64decode(msg["data"])
                            mime = msg.get("mime_type", "image/jpeg")
                            # SDK naming varies; try image= first, fall back to video=
                            try:
                                await session.send_realtime_input(
                                    image=types.Blob(data=image_bytes, mime_type=mime)
                                )
                            except TypeError:
                                await session.send_realtime_input(
                                    video=types.Blob(data=image_bytes, mime_type=mime)
                                )

                        elif kind == "text":
                            await session.send(input=msg.get("text", ""), end_of_turn=True)

                        elif kind == "end_of_turn":
                            await session.send(input=" ", end_of_turn=True)

                except WebSocketDisconnect:
                    log.info("Browser disconnected")
                except Exception as e:
                    log.warning(f"from_browser: {e}")

            async def from_gemini():
                try:
                    async for response in session.receive():
                        # Some SDKs expose direct fields
                        if hasattr(response, "data") and response.data:
                            await websocket.send_text(json.dumps({
                                "type": "audio",
                                "data": base64.b64encode(response.data).decode(),
                            }))

                        if hasattr(response, "text") and response.text:
                            await websocket.send_text(json.dumps({
                                "type": "transcript",
                                "text": response.text,
                            }))

                        # Most SDKs expose server_content.model_turn.parts
                        sc = getattr(response, "server_content", None)
                        if sc:
                            mt = getattr(sc, "model_turn", None)
                            if mt:
                                for part in (mt.parts or []):
                                    if getattr(part, "text", None):
                                        await websocket.send_text(json.dumps({
                                            "type": "transcript",
                                            "text": part.text,
                                        }))
                                    inline = getattr(part, "inline_data", None)
                                    if inline and getattr(inline, "data", None):
                                        await websocket.send_text(json.dumps({
                                            "type": "audio",
                                            "data": base64.b64encode(inline.data).decode(),
                                        }))
                            if getattr(sc, "turn_complete", False):
                                await websocket.send_text(json.dumps({"type": "turn_complete"}))

                except Exception as e:
                    log.exception(f"from_gemini: {e}")

            await asyncio.gather(from_browser(), from_gemini())

    except Exception as e:
        msg = str(e)
        log.error(f"Session error: {msg}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": msg}))
        except Exception:
            pass


# Serve frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
