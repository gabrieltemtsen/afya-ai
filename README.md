# AfyaAI 🏥

> **Voice-first AI health triage for Africa** — built on Gemini 3.1 Flash Live

AfyaAI lets anyone speak to an AI health assistant in their own language — English, Nigerian Pidgin, Yorùbá, Hausa, Igbo, Swahili, French and more. No reading, no typing. Just talk.

## What it does

- 🎙️ **Real-time voice conversation** — hold the mic and speak, AI responds with voice
- 🌍 **Multi-language** — auto-detects and responds in the patient's language
- 📷 **Vision triage** — send a photo of a wound, rash, or medicine bottle → AI analyses it
- 🚦 **Severity assessment** — Mild (home care) / Moderate (visit clinic) / Severe (hospital NOW)
- ⚡ **Low latency** — powered by Gemini 3.1 Flash Live real-time audio model

## Stack

- **Backend:** FastAPI + Gemini Live WebSocket proxy
- **Frontend:** Vanilla JS, Web Audio API, mobile-first
- **Deploy:** Railway (Docker)

## Quick Start

```bash
# Clone
git clone https://github.com/gabrieltemtsen/afya-ai
cd afya-ai

# Set up env
cp .env.example .env
# → Add your GEMINI_API_KEY

# Install & run
pip install -r backend/requirements.txt
python backend/main.py

# Open http://localhost:8000
```

## Deploy to Railway

1. Push to GitHub
2. New Railway project → Deploy from GitHub
3. Set env var: `GEMINI_API_KEY=your_key`
4. Done — Railway detects the Dockerfile automatically

## Disclaimer

AfyaAI is an AI assistant and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider.

---

Built with ❤️ for Africa | Powered by Google Gemini 3.1 Flash Live
