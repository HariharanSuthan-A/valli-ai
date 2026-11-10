# Valli (வள்ளி) – Multilingual AI Voice Agent for Indian Government Services

[![Python 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![WebSockets](https://img.shields.io/badge/WebSockets-Real--Time-orange.svg)](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
[![Tests Passing](https://img.shields.io/badge/Tests-50%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Valli (வள்ளி)** is an empathetic, real-time, multilingual conversational voice agent built to bridge India's digital divide. Millions of rural, elderly, and non-English-speaking citizens struggle with intimidating government web portals and rigid robotic IVR phone trees when applying for essential public welfare services—such as **Aadhaar card updates, Ration cards (e-Sevai/TNPDS), Old Age Pensions, Birth Certificates, and Passports**.

Valli speaks, listens, and guides callers naturally in their mother tongue with zero robotic delay, low-latency streaming, and human-like patience.

---

## 🌟 Core Highlights & Capabilities

- 🌐 **5 Native Indian Languages**: Full end-to-end voice support for **Tamil (`ta-IN`)**, **Hindi (`hi-IN`)**, **Telugu (`te-IN`)**, **Malayalam (`ml-IN`)**, and **Kannada (`kn-IN`)**. Callers can switch languages dynamically on the fly without resetting the call.
- 🎙️ **Full-Duplex Real-Time Streaming Audio**: Low-latency PCM16 mono at 16 kHz streaming directly through WebSocket pipelines and browser `AudioWorklet` processors (no heavy client libraries or builds required).
- ⚡ **Ultra-Fast Groq LPU Inference (`openai/gpt-oss-20b`)**: Sub-400ms Time-To-First-Token (TTFT) streaming at 65+ tokens/second. Solves the prohibitive **63 GB+ unquantized VRAM barrier** of running 30B MoE models (`gnani-evon-v3.3-30B-A3B`) on commodity infrastructure.
- ⏱️ **Zero Dead-Air Localized Fillers**: While the LLM formulates grounded answers, Valli speaks natural acoustic thinking buffers (*"ஒரு நிமிஷம்..."*, *"एक मिनट रुकिए..."*, *"ఒక్క నిమిషం ఉండండి..."*, etc.) to eliminate awkward phone dead silence.
- 🎭 **Timbre v3.0 Prosody Speech Synthesis**: Uses expressive audio tags (`<excited>`, `<slow>`, `<sigh>`, `<pauses>`) to provide warm, human-like voice inflection rather than flat synthetic speech.
- ⏳ **45-Second Caller Patience Listening Window**: Callers are never rushed or interrupted while thinking or finding physical documents. The 45-second silence countdown arms strictly *after* audio finishes playing.
- 🎧 **Immediate Turn Transition**: The exact millisecond Valli finishes speaking, the call status switches instantly to **`Listening…`** and unblocks microphone streaming.
- 🙏 **Natural Gratitude & Courtesy Closing**: When a caller says *"Thank you"* or regional equivalents (*நன்றி*, *धन्यवाद*, *ధన్యవాదాలు*, *നന്ദി*, *ಧನ್ಯವಾದಗಳು*), Valli automatically responds with a warm, localized closing.
- 🛑 **True Barge-In Interruption**: Callers can interrupt Valli at any point during speech. Playback queue flushes instantly on the client and server returns to active listening.
- 📱 **Clean Visual Companion UI**: Beautiful orange-and-white helpline interface featuring Valli's portrait avatar, live synchronized captions, and a single direct official portal link button (e.g. `myaadhaar.uidai.gov.in`) with zero confusing clutter.

---

## 🌐 Regional Language Matrix

Automatic language switching dynamically coordinates **Gnani Prisma STT**, **Timbre TTS**, **Groq LLM system prompt persona**, thinking audio fillers, and portal actions:

| Language | Code | Spoken Thinking Filler | Spoken Response When Caller Says "Thank You" | Official Portal Action Prompt |
| :--- | :--- | :--- | :--- | :--- |
| **Tamil** | `ta-IN` | `"ஒரு நிமிஷம்..."` | `"வள்ளியைத் தொடர்பு கொண்டதற்கு நன்றி, மேலும் விவரங்கள் தேவைப்பட்டால் என்னைத் தொடர்பு கொள்ளவும்."` | `"அதிகாரப்பூர்வ பக்கத்தைப் பார்வையிட கீழே உள்ள பொத்தானைக் கிளிக் செய்யவும்"` |
| **Hindi** | `hi-IN` | `"एक मिनट रुकिए..."` | `"वल्ली से संपर्क करने के लिए धन्यवाद, यदि आपको कोई अन्य विवरण चाहिए तो मुझसे संपर्क करें।"` | `"आधिकारिक पृष्ठ पर जाने के लिए नीचे दिए गए बटन पर क्लिक करें"` |
| **Telugu** | `te-IN` | `"ఒక్క నిమిషం ఉండండి..."` | `"వల్లిని సంప్రదించినందుకు ధన్యవాదాలు, మీకు ఏవైనా మరిన్ని వివరాలు కావాలంటే నన్ను సంప్రదించండి."` | `"అధికారిక పేజీని సందర్శించడానికి క్రింది బటన్‌ను క్లిక్ చేయండి"` |
| **Malayalam** | `ml-IN` | `"ഒരു നിമിഷം നിൽക്കൂ..."` | `"വള്ളിയെ ബന്ധപ്പെട്ടതിന് നന്ദി, കൂടുതൽ വിവരങ്ങൾ ആവശ്യമുണ്ടെങ്കിൽ എന്നെ ബന്ധപ്പെടുക."` | `"ഔദ്യോഗിക പേജ് സന്ദർശിക്കാൻ താഴെയുള്ള ബട്ടണിൽ ക്ലിക്കുചെയ്യുക"` |
| **Kannada** | `kn-IN` | `"ಒಂದು ನಿಮಿಷ ಇರಿ..."` | `"ವಳ್ಳಿಯನ್ನು ಸಂಪರ್ಕಿಸಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು, ಹೆಚ್ಚಿನ ವಿವರಗಳು ಬೇಕಾದರೆ ನನ್ನನ್ನು ಸಂಪರ್ಕಿಸಿ."` | `"ಅಧಿಕೃತ ಪುಟಕ್ಕೆ ಭೇಟಿ ನೀಡಲು ಕೆಳಗಿನ ಬಟನ್ ಕ್ಲಿಕ್ ಮಾಡಿ"` |

---

## 🏗️ Architecture & Pipeline

```text
  [ Citizen's Voice ]
         │ (PCM 16kHz via AudioWorklet)
         ▼
  [ WebSocket /ws/call ]
         │
         ├──► [ Gnani Prisma STT (WebSocket Stream) ]
         │            │
         │            ▼ (Final Transcript Event)
         ├──► [ CallSession Orchestrator ]
         │            │
         │            ├─► Fast-Path Intent Filter (e.g., Gratitude / Thank You)
         │            │
         │            ▼
         │    [ Groq LPU Engine (`openai/gpt-oss-20b`) ]
         │      • Strict Grounded Persona (`app/persona.py`)
         │      • Curated Knowledge Base (`data/knowledge.json`)
         │      • Streaming Sentence Chunker + Audio Prosody Tags
         │            │
         │            ▼ (Token & Sentence Stream)
         │    [ Gnani Timbre TTS v3.0 (WebSocket / REST) ]
         │      • Pre-synthesized language fillers for instant feedback
         │            │
         │            ▼ (WAV Audio Chunks + Timed Text)
  [ WebSocket Downstream ]
         │
         ▼
  [ Browser Web Audio API ]
         • Sequential chunk playback queue
         • Sub-millisecond return to `Listening…` on finish
         • Real-time caption sync + Official Portal Visit Card
```

---

## 📂 Project Structure

```text
govt-call-agent/
├── app/
│   ├── config.py         # Central settings, environment variables & timeouts
│   ├── fillers.py        # Pre-cached regional audio fillers ("ஒரு நிமிஷம்...")
│   ├── kb.py             # Knowledge base loader, semantic/keyword matching
│   ├── llm.py            # Async streaming LLM client (Groq LPU / Local Evon)
│   ├── main.py           # FastAPI application, static file mounts & WebSocket routes
│   ├── persona.py        # Multilingual system prompt generator & prosody tag rules
│   ├── session.py        # FullCallSession state machine, STT/TTS task orchestration
│   ├── stt.py            # Gnani Prisma STT streaming WebSocket transport
│   ├── tags.py           # Audio prosody tag parser (`<slow>`, `<excited>`, `<pauses>`)
│   └── tts.py            # Gnani Timbre v3.0 TTS synthesizer with header compatibility
├── data/
│   └── knowledge.json    # Curated Indian government schemes & services knowledge base
├── static/
│   ├── index.html        # Clean, accessible helpline UI with live captions & portal card
│   ├── call.js           # AudioWorklet client, Web Audio queue & WebSocket manager
│   ├── style.css         # Orange-and-white theme, responsive layout & visual badges
│   ├── worklet.js        # PCM16 16kHz downsampling AudioWorklet processor
│   ├── valli.jpg         # Valli portrait avatar image
│   └── presentation.html # Standalone interactive presentation & simulator
├── tests/
│   ├── test_kb.py        # Knowledge base lookup & validity test cases
│   ├── test_session_mock.py # Session orchestration, 45s timer, language & gratitude tests
│   └── test_tags.py      # Audio prosody tags parser & validator tests
├── kaggle_run_valli.ipynb # Standalone GPU/Colab/Kaggle notebook with ngrok tunneling
├── render.yaml           # One-click Render cloud deployment blueprint
├── requirements.txt      # Python dependencies
└── .env.example          # Environment configuration template
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: Version `3.12` or higher recommended.
- **Gnani API Key**: For Prisma STT and Timbre TTS.
- **Groq API Key**: For ultra-fast LPU inference with `openai/gpt-oss-20b`.

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/your-username/govt4u.git
cd govt4u/govt-call-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
# Gnani API Credentials (STT + TTS)
GNANI_API_KEY=your_gnani_api_key_here

# Groq LPU Inference (Primary Low-Latency LLM)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_BASE_URL=https://api.groq.com/openai/v1

# Evon LLM (Optional: Local GPU endpoint if self-hosting)
EVON_BASE_URL=http://localhost:8001/v1
EVON_MODEL=gnani/gnani-evon-v3.3-30B-A3B
EVON_API_KEY=not-needed-for-local

# Voice & STT Defaults
STT_LANGUAGE=ta-IN
TTS_MODEL=timbre-v3.0
TTS_VOICE=Lakshmi
TTS_SAMPLE_RATE=16000

# Silence Guardrail Settings (Seconds)
SILENCE_WARN_S=45.0
SILENCE_END_S=30.0

# Mock Mode (Set to 1 for offline testing without API credits)
MOCK=0
```

---

## 🏃 Running the Application

### Local Development Server

Start the local server using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`** to access the live call interface.

> 🎧 **Tip**: Wear headphones during voice testing to avoid acoustic feedback from laptop speakers into the microphone.

---

### Mobile Testing via ngrok Tunnel

Because browsers require a secure context (`https://`) to grant microphone access on mobile devices, use `ngrok` for testing on physical smartphones:

```bash
ngrok http 8000
```

Copy the generated `https://xxxx.ngrok-free.app` URL to your smartphone or scan the QR code.

---

### Standalone Interactive Presentation Page

To review the interactive showcase and multilingual simulator for demos and project presentations:

- Open **`presentation.html`** directly in any web browser.
- Features:
  - Architecture breakdown and the rationale for migrating to Groq LPUs.
  - Interactive phone simulator with language toggles (`Tamil`, `Hindi`, `Telugu`, `Malayalam`, `Kannada`).
  - Audio waveform animations, localized fillers, official portal visit card, and gratitude closing simulation.

---

## 🧪 Automated Testing

The project includes a comprehensive test suite written with `pytest` and `pytest-asyncio`. Run all tests locally with mock mode enabled:

```bash
PYTHONPATH=. pytest tests/ -v
```

### Test Coverage Highlights:
- **`tests/test_kb.py`**: Verifies schema, ID uniqueness, document requirements, and multilingual keyword queries.
- **`tests/test_session_mock.py`**: Validates the end-to-end async state machine (`IDLE` -> `LISTENING` -> `THINKING` -> `SPEAKING`), client playback tracking, 45-second silence timers, barge-in interrupts, dynamic language switching, and thank-you fast-path detection across all 5 languages.
- **`tests/test_tags.py`**: Tests audio prosody tag extraction (`<slow>`, `<excited>`, `<pauses>`) and stripping from spoken transcripts.

---

## ☁️ Deployment

### One-Click Cloud Deployment (Render)

This repository includes a [`render.yaml`](file:///home/han/Documents/govt4u/govt-call-agent/render.yaml) blueprint:

1. Connect this repository to your **Render** dashboard.
2. Under **Blueprints**, create a new blueprint instance.
3. Provide your `GNANI_API_KEY` and `GROQ_API_KEY` as environment variables.
4. Render builds and provisions the service with Python 3.12 and WebSocket termination automatically.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).

