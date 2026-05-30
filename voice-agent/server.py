"""
HTTP server exposing /health for ECS health checks and /spawn for room provisioning.

The actual voice pipeline runs in a subprocess (bot.py) spawned per Daily room.
This server's job is to (a) prove liveness to the ALB target group and (b) accept
spawn requests from the Lambda that creates the Daily room.
"""

import asyncio
import os
import subprocess
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

app = FastAPI()

# Service auth shared with the Lambda that calls /spawn.
SERVICE_TOKEN = os.environ.get("VOICE_BOT_SERVICE_TOKEN", "")


class SpawnRequest(BaseModel):
    room_url: str
    meeting_token: str
    user_id: str
    auth_token: str
    service_token: str


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/spawn")
async def spawn(req: SpawnRequest):
    if not SERVICE_TOKEN or req.service_token != SERVICE_TOKEN:
        raise HTTPException(status_code=401, detail="bad service token")

    logger.info("spawning bot for room={} user={}", req.room_url, req.user_id)

    bot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.py")

    # Inherit our environment so the bot subprocess sees the service-level
    # config (NVIDIA_API_KEY, CARTESIA_*, CEKURA_*, NOMIE_API_BASE,
    # SYSTEM_PROMPT_PATH, etc.). Per-room args are passed on the CLI.
    env = os.environ.copy()

    # Fire-and-forget subprocess. The bot exits when the room closes.
    proc = subprocess.Popen(
        [
            sys.executable,
            bot_path,
            "--room-url", req.room_url,
            "--meeting-token", req.meeting_token,
            "--user-id", req.user_id,
            "--auth-token", req.auth_token,
        ],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    return {"ok": True, "pid": proc.pid}
