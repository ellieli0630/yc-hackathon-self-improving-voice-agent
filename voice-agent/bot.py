"""
Pipecat voice bot for the Nomie hackathon stack.

Pipeline: Daily transport ↔ Nvidia Parakeet STT ↔ Nvidia NIM LLM ↔ Cartesia TTS.
Wraps Cekura's PipecatTracer for transcripts/tool-calls and emits custom
per-turn latency events ourselves (Cekura SDK doesn't auto-capture latency).

Spawned per Daily room by server.py /spawn. Exits when the room closes.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Any

import httpx
from cekura.pipecat import PipecatTracer
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    OutputTransportMessageFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.transports.daily.transport import DailyParams, DailyTransport


SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "/app/system_prompt.txt")

# NVIDIA Nemotron Super — chosen over nano-9b for tool-call reliability under
# messy/ambiguous context (8/8 vs 6/8 in tests). TTFB is ~equal so streamed voice
# latency is unaffected. Has the same /no_think toggle. Override via NIM_MODEL.
DEFAULT_NIM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH) as f:
        return f.read()


# Tool schemas mirror the OpenAI Realtime tools in packages/functions/src/routes/voice.ts.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "launch_exercise",
            "description": (
                "Launch an interactive wellness exercise in the Nomie app. Use when the user wants "
                "a specific activity like breathing, worry jar, CBT reframing, or a creative "
                "worksheet. ALWAYS call this tool — do NOT try to guide the exercise through "
                "conversation alone."
            ),
            # Enum and worksheetId mirror the OpenAI Realtime tool in
            # packages/functions/src/routes/voice.ts and the mobile ExerciseLaunch
            # union in apps/mobile/src/hooks/useVoiceChat.ts. These are the only
            # types the mobile Unity handler knows how to open — emitting anything
            # else launches nothing.
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["breathing", "worry_jar", "five_column", "worksheet"],
                        "description": (
                            "'breathing' for guided breathing, 'worry_jar' for worry-to-kindness "
                            "conversion, 'five_column' for CBT cognitive reframing, 'worksheet' "
                            "for therapeutic drawing/doodling"
                        ),
                    },
                    "worksheetId": {
                        "type": "string",
                        "enum": [
                            "self_love_jar", "shield", "healing", "comfort",
                            "garden", "kindness", "storm", "blank",
                        ],
                        "description": (
                            "Required when type is 'worksheet'. Which worksheet template to open."
                        ),
                    },
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_user_memory",
            "description": (
                "Search the user's past conversations and stored facts. Use when you need to "
                "recall something the user mentioned before, like preferences, relationships, "
                "important events, or past discussions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]


def build_tools_schema() -> ToolsSchema:
    """Convert the OpenAI-format TOOLS dicts into a Pipecat ToolsSchema.

    The universal LLMContext expects standard-format tools (ToolsSchema /
    FunctionSchema), not raw OpenAI `{"type":"function","function":{...}}`
    dicts — passing those straight through is silently misparsed by the
    NVIDIA adapter, so we map them explicitly here.
    """
    functions = []
    for tool in TOOLS:
        fn = tool["function"]
        params = fn.get("parameters", {})
        functions.append(
            FunctionSchema(
                name=fn["name"],
                description=fn.get("description", ""),
                properties=params.get("properties", {}),
                required=params.get("required", []),
            ),
        )
    return ToolsSchema(standard_tools=functions)


async def send_daily_app_message(transport: DailyTransport, message: dict[str, Any]) -> None:
    """Broadcast a Daily app-message to the room.

    ``send_message`` lives on the OUTPUT transport, not the top-level
    DailyTransport (calling ``transport.send_message`` throws AttributeError).
    Errors are swallowed — a relay failure must never kill the call's audio.
    """
    try:
        await transport.output().send_message(
            OutputTransportMessageFrame(message=message),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("app-message relay failed: {}", exc)


class BotMessageForwarder(FrameProcessor):
    """Non-mutating tap that relays bot_state app-messages to the Daily client.

    Transcripts are NOT sent from here — the client reads them from Pipecat's
    RTVI app-messages instead (this processor sits after TTS, which consumes the
    LLM/transcription frames before they'd reach it). All frames pass through.
    """

    def __init__(self, *, transport: DailyTransport) -> None:
        super().__init__()
        self._transport = transport

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            await send_daily_app_message(self._transport, {"kind": "bot_state", "state": "processing"})
        elif isinstance(frame, UserStartedSpeakingFrame):
            await send_daily_app_message(self._transport, {"kind": "bot_state", "state": "listening"})
        elif isinstance(frame, BotStartedSpeakingFrame):
            await send_daily_app_message(self._transport, {"kind": "bot_state", "state": "speaking"})
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await send_daily_app_message(self._transport, {"kind": "bot_state", "state": "listening"})

        await self.push_frame(frame, direction)


class NomieToolHandler:
    """Bridges Pipecat function calls back to the Nomie REST API.

    `launch_exercise` is delivered to the mobile client as a Daily app-message
    (the client opens the Unity activity). `search_user_memory` calls
    Nomie's existing GET /memory/search.
    """

    def __init__(
        self,
        *,
        transport: DailyTransport,
        user_id: str,
        auth_token: str,
        api_base: str,
    ) -> None:
        self._transport = transport
        self._user_id = user_id
        self._auth_token = auth_token
        self._api_base = api_base.rstrip("/")
        self._http = httpx.AsyncClient(timeout=10.0)

    async def aclose(self) -> None:
        await self._http.aclose()

    # Pipecat 1.3.0 passes a FunctionCallParams (args via .arguments) and expects
    # the result delivered via params.result_callback(...) — NOT a return value.
    async def launch_exercise(self, params) -> None:
        args = params.arguments or {}
        exercise_type = args.get("type")
        worksheet_id = args.get("worksheetId")
        logger.info("tool launch_exercise type={} worksheetId={}", exercise_type, worksheet_id)
        message: dict[str, Any] = {"kind": "launch_exercise", "type": exercise_type}
        if worksheet_id is not None:
            message["worksheetId"] = worksheet_id
        await send_daily_app_message(self._transport, message)
        await params.result_callback({"ok": True, "launched": exercise_type})

    async def search_user_memory(self, params) -> None:
        query = (params.arguments or {}).get("query", "")
        try:
            resp = await self._http.get(
                f"{self._api_base}/memory/search",
                params={"q": query, "limit": 5},
                headers={"Authorization": f"Bearer {self._auth_token}"},
            )
            resp.raise_for_status()
            await params.result_callback(resp.json())
        except Exception as exc:  # noqa: BLE001 — tool failure shouldn't kill the turn
            logger.warning("search_user_memory failed: {}", exc)
            await params.result_callback({"results": {"facts": [], "conversations": []}})


async def run_bot(
    *,
    room_url: str,
    meeting_token: str,
    user_id: str,
    auth_token: str,
) -> None:
    nvidia_api_key = os.environ["NVIDIA_API_KEY"]
    cartesia_api_key = os.environ["CARTESIA_API_KEY"]
    cekura_api_key = os.environ["CEKURA_API_KEY"]
    cekura_agent_id = int(os.environ["CEKURA_AGENT_ID"])
    nomie_api_base = os.environ["NOMIE_API_BASE"]

    transport = DailyTransport(
        room_url,
        meeting_token,
        "Nomie",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            transcription_enabled=False,
        ),
    )

    stt = NvidiaSTTService(api_key=nvidia_api_key)

    nim_model = os.environ.get("NIM_MODEL", DEFAULT_NIM_MODEL)
    llm = NvidiaLLMService(api_key=nvidia_api_key, model=nim_model)

    tts = CartesiaTTSService(
        api_key=cartesia_api_key,
        # "Brooke - Big Sister" — warm, on-brand for an emotional companion.
        # Verified present in the account's voice list. Override via CARTESIA_VOICE_ID.
        voice_id=os.environ.get(
            "CARTESIA_VOICE_ID",
            "e07c00bc-4134-4eae-9ea4-1a55fb45746b",
        ),
    )

    system_prompt = load_system_prompt()
    # Nemotron models default to chain-of-thought ON; "/no_think" disables it,
    # which is essential for voice latency (we can't wait on reasoning tokens
    # before the first spoken word). Harmless on non-reasoning models, but we
    # only prepend it for nemotron to avoid confusing plain Llama.
    if "nemotron" in nim_model:
        system_prompt = "/no_think\n" + system_prompt
    context = LLMContext(
        messages=[{"role": "system", "content": system_prompt}],
        tools=build_tools_schema(),
    )

    # Cekura requires the LLMContextAggregatorPair pattern — if we use
    # llm.create_context_aggregator(), Cekura silently disables tracking
    # with a "LLMUserAggregator/LLMAssistantAggregator not found" warning.
    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    nomie_tools = NomieToolHandler(
        transport=transport,
        user_id=user_id,
        auth_token=auth_token,
        api_base=nomie_api_base,
    )
    llm.register_function("launch_exercise", nomie_tools.launch_exercise)
    llm.register_function("search_user_memory", nomie_tools.search_user_memory)

    # Single non-mutating tap placed just before the transport output: by this
    # point user TranscriptionFrames (passed through by user_agg), LLMTextFrames,
    # and VAD/bot-speaking frames have all flowed downstream, so one processor
    # sees everything it needs to relay transcript + bot_state app-messages.
    forwarder = BotMessageForwarder(transport=transport)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        tts,
        forwarder,
        transport.output(),
        assistant_agg,
    ])

    # One PipecatTracer instance per call — not thread-safe to share across
    # concurrent sessions. We're already one-process-per-room, so fine.
    # Multi-step (vs observe_and_create_task) so we can disable barge-in:
    # allow_interruptions=False means the bot finishes its turn and ignores
    # incoming audio while speaking — which also stops it from transcribing its
    # own speaker output (echo) as a new user turn and replying to itself.
    tracer = PipecatTracer(api_key=cekura_api_key, agent_id=cekura_agent_id)
    pipeline = tracer.observe_pipeline(
        pipeline,
        context,
        custom_metadata={"user_id": user_id, "room_url": room_url},
    )
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False))
    task = tracer.register_task_handlers(task, transport=transport)

    @transport.event_handler("on_first_participant_joined")
    async def on_join(_t, participant):
        logger.info("participant joined: {}", participant)
        # NIM nemotron rejects a system-only completion ("list object has no
        # element -1"), so seed a hidden user turn to kick off the spoken
        # greeting. Real turns already carry a user message and work fine.
        await task.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{
                    "role": "user",
                    "content": (
                        "[The user just opened the app and can see you on screen. "
                        "Greet them warmly in one short sentence as Nomie. They "
                        "haven't spoken yet, so don't reference their voice or tone.]"
                    ),
                }],
                run_llm=True,
            )
        )

    @transport.event_handler("on_participant_left")
    async def on_leave(_t, _p, _reason):
        logger.info("participant left, ending pipeline")
        await task.queue_frame(EndFrame())

    @transport.event_handler("on_app_message")
    async def on_app_message(_t, message, _sender):
        # Inbound client→bot app-messages mirror dailyVoice.ts senders:
        #   {kind:"user_text", text}  — typed chat message (hybrid/text-only input)
        #   {kind:"interrupt"}        — barge-in / cancel the in-flight response
        if not isinstance(message, dict):
            return
        kind = message.get("kind")
        if kind == "user_text":
            text = (message.get("text") or "").strip()
            if text:
                await task.queue_frame(
                    LLMMessagesAppendFrame(
                        messages=[{"role": "user", "content": text}],
                        run_llm=True,
                    ),
                )
        elif kind == "interrupt":
            await task.queue_frame(InterruptionFrame())

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        await nomie_tools.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room-url", required=True)
    parser.add_argument("--meeting-token", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--auth-token", required=True)
    args = parser.parse_args()

    asyncio.run(
        run_bot(
            room_url=args.room_url,
            meeting_token=args.meeting_token,
            user_id=args.user_id,
            auth_token=args.auth_token,
        ),
    )


if __name__ == "__main__":
    main()
