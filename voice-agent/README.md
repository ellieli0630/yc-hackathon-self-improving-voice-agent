# Voice Agent — the thing GEPA optimizes

This folder is the **voice companion** ("Nomie") whose system prompt the
[GEPA loop](../gepa/) evolves. It's included here as **context**: GEPA's whole
job is to make *this* agent talk better, so you need to see what it is.

It is the real production-style agent, with one deliberate omission — see
[*What was removed*](#what-was-removed-unity--react-native) below.

## How we built the voice agent

It's a **real-time voice pipeline**: your voice makes a round trip through a
handful of stages, fast enough that it feels like talking to a person. Each stage
does exactly one job:

```
You speak
   │   (your phone streams mic audio over WebRTC)
   ▼
Daily                the "phone line" — carries live audio both ways
   ▼
NVIDIA Parakeet      the ears — turns your speech into text
   ▼
NVIDIA Nemotron      the brain — reads the text and decides what to say
   │                 (may also call a tool: launch_exercise / search_user_memory)
   ▼
Cartesia             the voice — turns the reply text back into speech
   ▼
Daily ─▶ you hear the reply
        ⟫ the whole pipeline is wrapped by Cekura's PipecatTracer ⟪
```

In plain English, the pieces are:

- **Daily** — the phone line. It carries live audio between the user's phone and
  our server over WebRTC.
- **NVIDIA Parakeet** — the *ears*. Speech-to-text: it turns what you say into
  words the brain can read.
- **NVIDIA Nemotron** (`llama-3.3-nemotron-super-49b`, served via NVIDIA NIM) —
  the *brain*. It reads the conversation and decides what to say next. We picked a
  fast model on purpose: in a real conversation, waiting on the reply feels broken.
- **Cartesia** — the *voice*. Text-to-speech that turns the brain's reply back
  into natural-sounding audio ("Brooke – Big Sister", warm and on-brand).
- **Pipecat** — the *wiring*. It's the orchestration framework that connects all
  of the above into one streaming pipeline so audio flows through without us
  hand-managing buffers and turn-taking.
- **Cekura** — the *observer*. It wraps the whole pipeline and records every
  transcript and tool call, so we can later see what actually happened across
  thousands of conversations (and feed that into the GEPA loop).

| Component | Role | Tech |
|---|---|---|
| Transport | Carries live audio to/from the user | **Daily** (WebRTC) |
| STT (ears) | Speech → text | **NVIDIA Parakeet** |
| LLM (brain) | Decides what to say | **NVIDIA NIM** `llama-3.3-nemotron-super-49b-v1.5` |
| TTS (voice) | Text → speech | **Cartesia** ("Brooke – Big Sister") |
| Orchestration | Wires the pipeline together | **Pipecat** |
| Observability | Records transcripts + tool calls | **Cekura** `PipecatTracer` |

- **`bot.py`** — the Pipecat pipeline. Loads `system_prompt.txt` (the file GEPA
  rewrites), prepends `/no_think` for Nemotron (suppresses chain-of-thought so
  voice latency stays low), registers two tools, and wraps everything in Cekura.
- **`server.py`** — a tiny FastAPI server: `/health` for the load balancer and
  `/spawn` to fork one `bot.py` subprocess per call.
- **`system_prompt.txt`** — Nomie's full persona. **This is the optimization
  target** — `gepa/apply_winner.py` overwrites this file with the winning prompt.
- **`Dockerfile`** — containerizes the bot for AWS ECS Fargate.

## Implementation: the NVIDIA stack, in detail

All of this lives in [`bot.py`](bot.py), wired together with Pipecat.

### The brain — NVIDIA NIM (Nemotron)

```python
llm = NvidiaLLMService(api_key=nvidia_api_key, model=nim_model)
```

- **Model:** `nvidia/llama-3.3-nemotron-super-49b-v1.5`, served through **NVIDIA NIM**
  (an OpenAI-compatible inference endpoint, so Pipecat drives it like any chat model).
- **Why Nemotron-super-49b** (not the smaller nano-9b): it was far more reliable at
  **tool calling** under messy, ambiguous context (8/8 vs 6/8 in our tests), while
  time-to-first-token was about equal — so streamed voice latency was unaffected.
- **`/no_think` is force-prepended** to the system prompt for any `nemotron` model.
  Nemotron defaults to chain-of-thought *on*, which is fatal for voice — you can't
  wait on reasoning tokens before the first spoken word. `/no_think` disables it.
  It's prepended at load time in `bot.py` (and, in the GEPA loop, injected at the LM
  layer *below* the instruction so prompt mutations can never accidentally drop it).

### The ears — NVIDIA Parakeet (STT)

```python
stt = NvidiaSTTService(api_key=nvidia_api_key)
```

NVIDIA's **Parakeet** streaming speech-to-text turns the user's audio into text
frames the LLM consumes. Same NVIDIA API key as the LLM.

### Tool calling (the non-obvious part)

The agent exposes two functions. Raw OpenAI-format tool dicts
(`{"type":"function","function":{…}}`) are **silently misparsed by the NVIDIA
adapter**, so they're explicitly converted to Pipecat's standard schema:

```python
def build_tools_schema() -> ToolsSchema:
    functions = [FunctionSchema(name=…, description=…, properties=…, required=…)
                 for tool in TOOLS]
    return ToolsSchema(standard_tools=functions)
```

Handlers follow the Pipecat 1.3 contract — results are returned via
`params.result_callback(...)`, not a `return` value:

```python
async def launch_exercise(self, params):
    args = params.arguments or {}
    await send_daily_app_message(self._transport, {"kind": "launch_exercise", **args})
    await params.result_callback({"ok": True, "launched": args.get("type")})
```

### The pipeline (order matters)

```python
pipeline = Pipeline([
    transport.input(),   # Daily audio in
    stt,                 # Parakeet  → text
    user_agg,            # user-turn context aggregation
    llm,                 # Nemotron  → reply (+ tool calls)
    tts,                 # Cartesia  → audio
    forwarder,           # taps frames to relay bot_state app-messages
    transport.output(),  # Daily audio out
    assistant_agg,       # assistant-turn context aggregation
])
```

- **Context** uses `LLMContext` + **`LLMContextAggregatorPair`** — required by Cekura;
  using `llm.create_context_aggregator()` instead silently disables Cekura tracking.
- **VAD** is Silero (`SileroVADAnalyzer`) for turn detection.

### Observability — Cekura

```python
tracer = PipecatTracer(api_key=cekura_api_key, agent_id=cekura_agent_id)
pipeline = tracer.observe_pipeline(pipeline, context,
             custom_metadata={"user_id": user_id, "room_url": room_url})
task = tracer.register_task_handlers(task, transport=transport)
```

One tracer **per call** (not thread-safe to share) — fine here because it's
one bot process per room. It records every transcript and tool call.

### Two behavior decisions worth calling out

- **Barge-in is disabled** (`PipelineParams(allow_interruptions=False)`). On a phone
  speaker the bot's own TTS leaks into the mic; Parakeet transcribes it as a new user
  turn and the bot replies to itself. Disabling interruptions (plus muting during bot
  speech) kills that echo loop.
- **A hidden greeting seed** is queued on `on_first_participant_joined`. Nemotron
  rejects a system-only completion, so we inject an invisible user turn ("the user
  just opened the app — greet them") to trigger the first spoken hello.

## The two tools

The agent can call two functions mid-conversation:

1. **`launch_exercise`** — opens an interactive wellness activity (breathing,
   worry jar, CBT five-column, drawing worksheet).
2. **`search_user_memory`** — looks up the user's past facts/conversations via a
   REST call, so the agent "remembers" things like a friend would.

## What was removed: Unity & React Native

In the full Nomie product, the agent lives inside a **React Native (Expo) mobile
app**, and the wellness exercises are **Unity** scenes (3D breathing animations,
an interactive worry jar, drawing canvases, etc.).

**Those are not in this repo** — they're a large mobile/game codebase irrelevant
to the self-improvement loop. Here's what they did, so the agent's behavior makes
sense:

- **React Native app** — the phone client. It joined the Daily room, streamed
  mic audio, rendered the live transcript, and **listened for the agent's
  `launch_exercise` tool calls** (delivered as Daily "app-messages").
- **Unity** — when the app received a `launch_exercise` message, it opened the
  matching Unity activity on screen (the breathing visualizer, worry jar, etc.).

So in production, `launch_exercise` → Daily app-message → React Native →
**Unity scene opens on the user's phone**. In this repo, the tool call still
fires and is logged/traced; there's just no mobile UI to render the scene. The
agent's *decision* to launch an exercise is exactly what GEPA evaluates (via the
`tool_delegation` metric) — and that decision is fully present here.

## Hosting (in the full product)

Deployed to **AWS ECS Fargate** (1 vCPU / 2 GB, 2 tasks) behind a public load
balancer, health-checked on `GET /health:8080`, with one bot subprocess spawned
per Daily room. Secrets (NVIDIA, Cartesia, Cekura keys) are injected as env vars.

## Running locally (sketch)

```bash
pip install -r requirements.txt
export NVIDIA_API_KEY=...  CARTESIA_API_KEY=...  CEKURA_API_KEY=...  CEKURA_AGENT_ID=...
export NOMIE_API_BASE=...   # for search_user_memory; optional for a smoke test
uvicorn server:app --host 0.0.0.0 --port 8080
# then POST /spawn with a Daily room_url + meeting_token to start a call
```
