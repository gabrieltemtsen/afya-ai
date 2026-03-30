"""
AfyaAI — Voice-first AI health triage for Africa
Backend: FastAPI + Gemini 3.1 Flash Live (real-time bidirectional audio + vision)
"""
import asyncio
import base64
import json
import os
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("afya")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Use the latest live model — fall back gracefully
MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-2.0-flash-live-001")

SYSTEM_PROMPT = """You are AfyaAI (Afya means "health" in Swahili) — a compassionate, voice-first AI health assistant built for communities across Africa where access to doctors is limited.

YOUR ROLE:
- Listen carefully as patients describe their symptoms
- Ask 1–2 simple follow-up questions at a time to understand severity
- Give a clear triage assessment:
  * 🟢 MILD — rest and home care, these tips will help...
  * 🟡 MODERATE — visit a clinic or health centre within 24 hours
  * 🔴 SEVERE — go to a hospital or emergency room IMMEDIATELY

YOUR COMMUNICATION STYLE:
- Speak in the SAME language the patient uses — English, Nigerian Pidgin, Yoruba, Hausa, Igbo, Swahili, French, or any other
- Be warm, calm, and reassuring — never alarm unnecessarily
- Use simple words — avoid medical jargon unless necessary
- Keep answers short and clear — patients may have limited connectivity

HARD RULES:
1. Always state you are an AI assistant, not a doctor — for serious conditions always recommend a real doctor
2. EMERGENCY signs → tell patient GO TO HOSPITAL NOW (do NOT wait): chest pain, difficulty breathing, unconscious or unresponsive, heavy uncontrolled bleeding, signs of stroke (face drooping, arm weakness, speech difficulty), poisoning, severe burns
3. Never give specific drug dosages — say "consult a pharmacist or doctor for the right dose"
4. If asked about pregnancy complications — recommend hospital immediately
5. Never diagnose definitively — say "this sounds like it could be..." not "you have..."

When a patient sends an image (photo of wound, rash, medicine, etc.) — analyse it carefully and incorporate what you see into your assessment.

Start by warmly greeting the patient in English and invite them to describe what's wrong."""

app = FastAPI(title="AfyaAI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket):
    await websocket.accept()
    log.info("New voice session started")

    if not GEMINI_API_KEY:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "GEMINI_API_KEY not configured on server"
        }))
        await websocket.close()
        return

    client = genai.Client(api_key=GEMINI_API_KEY)

    live_config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part(text=SYSTEM_PROMPT)],
            role="user"
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        ),
    )

    try:
        async with client.aio.live.connect(model=MODEL, config=live_config) as session:
            log.info("Gemini Live session established")

            # Notify client we're ready
            await websocket.send_text(json.dumps({"type": "ready"}))

            async def recv_from_browser():
                """Read messages from the browser and forward to Gemini."""
                try:
                    while True:
                        raw = await websocket.receive_text()
                        msg = json.loads(raw)
                        kind = msg.get("type")

                        if kind == "audio":
                            audio_bytes = base64.b64decode(msg["data"])
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(
                                            data=audio_bytes,
                                            mime_type="audio/pcm;rate=16000"
                                        )
                                    ]
                                )
                            )

                        elif kind == "image":
                            # Patient holds up phone to show wound / rash / medicine
                            image_bytes = base64.b64decode(msg["data"])
                            mime = msg.get("mime_type", "image/jpeg")
                            caption = msg.get("caption", "The patient has sent an image. Please analyse it as part of your health assessment.")
                            await session.send(
                                input=types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        types.Blob(data=image_bytes, mime_type=mime)
                                    ]
                                )
                            )
                            # Send text context for the image
                            await session.send(
                                input=caption,
                                end_of_turn=True
                            )

                        elif kind == "end_of_turn":
                            await session.send(input="", end_of_turn=True)

                        elif kind == "ping":
                            await websocket.send_text(json.dumps({"type": "pong"}))

                except WebSocketDisconnect:
                    log.info("Browser disconnected")
                except Exception as e:
                    log.error(f"recv_from_browser error: {e}")

            async def recv_from_gemini():
                """Stream responses from Gemini back to the browser."""
                try:
                    async for response in session.receive():
                        server_content = getattr(response, "server_content", None)
                        if server_content is None:
                            continue

                        model_turn = getattr(server_content, "model_turn", None)
                        if model_turn:
                            for part in (model_turn.parts or []):
                                # Audio chunk
                                if hasattr(part, "inline_data") and part.inline_data:
                                    audio_b64 = base64.b64encode(part.inline_data.data).decode()
                                    await websocket.send_text(json.dumps({
                                        "type": "audio",
                                        "data": audio_b64,
                                        "mime_type": part.inline_data.mime_type or "audio/pcm;rate=24000"
                                    }))
                                # Transcript
                                if hasattr(part, "text") and part.text:
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "role": "assistant",
                                        "text": part.text
                                    }))

                        # Turn complete signal
                        if getattr(server_content, "turn_complete", False):
                            await websocket.send_text(json.dumps({"type": "turn_complete"}))

                except Exception as e:
                    log.error(f"recv_from_gemini error: {e}")

            await asyncio.gather(recv_from_browser(), recv_from_gemini())

    except Exception as e:
        log.error(f"Session error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Session error: {str(e)}"
            }))
        except Exception:
            pass
    finally:
        log.info("Voice session ended")


# Serve frontend (must be last)
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
