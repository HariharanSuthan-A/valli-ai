"""
main.py – FastAPI application entry point.

Routes:
    GET  /api/health     →  {"status": "ok", "mock": bool}
    GET  /               →  static/index.html
    WS   /ws/call        →  FullCallSession
    GET  /static/*       →  static files
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from app import fillers
from app import tts as tts_module
from app.config import MOCK, STATIC_DIR
from app.session import FullCallSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Govt Call Agent (MOCK=%s)…", MOCK)
    # Pre-synthesise filler clips so they are ready for the first call
    await fillers.preload()
    yield
    # Cleanup
    await tts_module.close()
    log.info("Govt Call Agent shut down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Govt Call Agent", lifespan=lifespan)


# ── API routes ────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "ok", "mock": MOCK}


# ── WebSocket call endpoint ───────────────────────────────────────────────────


@app.websocket("/ws/call")
async def ws_call(websocket: WebSocket):
    await websocket.accept()
    log.info("New call session from %s", websocket.client)
    session = FullCallSession(websocket)
    try:
        await session.run()
    except WebSocketDisconnect:
        log.info("Client disconnected.")
    except Exception as exc:
        log.error("Unhandled session error: %s", exc, exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Static files (UI) ─────────────────────────────────────────────────────────

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    log.warning("static/ directory not found; UI will not be served.")


# ── Dev / Production entrypoint ───────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)

