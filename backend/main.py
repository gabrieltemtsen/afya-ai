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
        "enable_live": ENABLE_LIVE,
        "models": {"live": MODEL_LIVE, "text": MODEL_TEXT, "tts": MODEL_TTS},
    }


@app.post("/api/chat")
async def chat_api(payload: dict = Body(...)):
    """Stable chat endpoint (kept for debugging)."""
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}

    text = (payload.get("text") or "").strip()
    if not text:
        return {"error": "Missing text"}

    client = genai.Client(api_key=GEMINI_API_KEY)
    reply_text = generate_triage_text(client, MODEL_TEXT, SYSTEM_PROMPT, text)
    tts = generate_tts_audio_b64(client, MODEL_TTS, reply_text)
    return {"text": reply_text, **tts}


@app.post("/api/voice")
async def voice_api(payload: dict = Body(...)):
    """Voice-to-voice endpoint.

    Input:
      {
        audio_b64: string,
        mime_type: string,
        lang_hint?: string,
        history?: Array<{ role: "user"|"assistant", text: string }>
      }

    Output:
      {
        detected_language?: string,
        reply_text: string,
        reply_audio_b64?: string,
        reply_mime_type?: string
      }
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set"}

    audio_b64 = payload.get("audio_b64")
    mime_type = payload.get("mime_type") or "audio/webm"
    lang_hint = (payload.get("lang_hint") or "").strip()
    history = payload.get("history") or []

    if not audio_b64:
        return {"error": "Missing audio_b64"}

    audio_bytes = base64.b64decode(audio_b64)

    client = genai.Client(api_key=GEMINI_API_KEY)

    # Build lightweight context from last turns
    context_lines: list[str] = []
    for h in history[-8:]:
        role = h.get("role")
        text = (h.get("text") or "").strip()
        if not role or not text:
            continue
        prefix = "User" if role == "user" else "AfyaAI"
        context_lines.append(f"{prefix}: {text}")
    context = "\n".join(context_lines)

    # 1) Transcribe first (gives us real memory + better language detection)
    transcribe_prompt = (
        "Transcribe the user's AUDIO to plain text. "
        "If you can infer the spoken language/dialect, prepend it like: [LANG=...] then the transcript. "
        "Be faithful; don't add new information."
    )

    tr = client.models.generate_content(
        model=MODEL_TEXT,
        contents=[{"role": "user", "parts": [
            {"text": transcribe_prompt},
            {"inline_data": {"mime_type": mime_type, "data": audio_bytes}},
        ]}],
    )
    transcript_raw = (getattr(tr, "text", None) or "").strip()

    detected_language = None
    transcript = transcript_raw
    if transcript_raw.startswith("[LANG=") and "]" in transcript_raw:
        try:
            tag = transcript_raw.split("]", 1)[0]
            detected_language = tag.replace("[LANG=", "").strip() or None
            transcript = transcript_raw.split("]", 1)[1].strip()
        except Exception:
            transcript = transcript_raw

    # 2) Generate reply using history + transcript
    user_prompt = (
        "You are continuing an ongoing conversation. Do NOT restart.\n"
        "If you previously asked a question, use the user's latest answer to move forward.\n\n"
        "Conversation so far (if any):\n" + (context + "\n" if context else "(none)\n") +
        "User just said (transcript):\n" + transcript + "\n\n" +
        "Now reply as AfyaAI with the NEXT best response.\n"
        "Rules:\n"
        "- Reply in the SAME language/dialect the user spoke.\n"
        "- If the user speaks Nigerian Pidgin, reply in Nigerian Pidgin explicitly.\n"
        "- Ask at most ONE follow-up question unless emergency.\n"
        "- Keep it short and practical.\n"
        + (f"- Language hint from UI: {lang_hint}\n" if lang_hint else "") +
        "Return STRICT JSON only: {\"language\": string, \"reply\": string}."
    )

    resp = client.models.generate_content(
        model=MODEL_TEXT,
        contents=[{"role": "user", "parts": [
            {"text": SYSTEM_PROMPT},
            {"text": user_prompt},
        ]}],
    )

    raw = (getattr(resp, "text", None) or "").strip()
    reply_text = raw

    try:
        obj = json.loads(raw)
        detected_language = str(obj.get("language") or detected_language or "").strip() or detected_language
        reply_text = str(obj.get("reply") or "").strip() or raw
    except Exception:
        pass

    # 2) TTS
    tts = generate_tts_audio_b64(client, MODEL_TTS, reply_text)

    return {
        "detected_language": detected_language,
        "transcript": transcript,
        "reply_text": reply_text,
        "reply_audio_b64": tts.get("audio_b64"),
        "reply_mime_type": tts.get("mime_type"),
    }


ENABLE_LIVE = os.environ.get("ENABLE_LIVE", "false").lower() == "true"


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    # Live mode is currently unstable (preview endpoints). Keep disabled by default.
    if not ENABLE_LIVE:
        await websocket.accept()
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Realtime voice is temporarily disabled. Please refresh and use Stable Voice Mode.",
        }))
        await websocket.close()
        return

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
