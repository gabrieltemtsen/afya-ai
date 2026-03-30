"""Stable voice mode helpers.

We avoid Gemini Live for now and instead:
1) Generate a text response (triage) with a fast text model
2) Generate TTS audio for that response

This yields a reliable UX today (push-to-talk), and we can re-enable live later.
"""

from __future__ import annotations

import base64
from google import genai


def generate_triage_text(client: genai.Client, model: str, system_prompt: str, user_text: str) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=[
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_text}]},
        ],
    )
    return (getattr(resp, "text", None) or "").strip()


def generate_tts_audio_b64(client: genai.Client, model: str, text: str, voice_name: str = "Aoede") -> dict:
    # google-genai currently returns inline audio parts for TTS preview models.
    resp = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": [{"text": text}]}],
        config={
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": voice_name}
                }
            },
        },
    )

    # Try common shapes: resp.candidates[0].content.parts[...].inline_data.data
    audio_bytes = None
    mime_type = "audio/pcm;rate=24000"

    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            inline = getattr(p, "inline_data", None)
            if inline and getattr(inline, "data", None):
                audio_bytes = inline.data
                mt = getattr(inline, "mime_type", None)
                if mt:
                    mime_type = mt
                break

    if audio_bytes is None:
        # Fallback: some SDKs expose resp.data
        audio_bytes = getattr(resp, "data", None)

    if audio_bytes is None:
        return {"audio_b64": None, "mime_type": None}

    return {
        "audio_b64": base64.b64encode(audio_bytes).decode(),
        "mime_type": mime_type,
    }
