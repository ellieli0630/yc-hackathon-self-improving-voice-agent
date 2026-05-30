# Self-Improving Voice Agent

A mental-wellness **voice companion** ("Nomie") built on an open, self-hostable
stack — and the loop that lets it improve its own prompt from real conversations.

Built for the [YC Voice Agents Hackathon](https://github.com/pipecat-ai/yc-voice-agents-hackathon)
(Pipecat orchestration · NVIDIA open models · Cekura evaluation).

## [`voice-agent/`](voice-agent/) — the real-time voice agent

```
You speak ─▶ Daily (WebRTC) ─▶ NVIDIA Parakeet (STT) ─▶ NVIDIA NIM Nemotron (LLM)
          ─▶ Cartesia (TTS) ─▶ Daily ─▶ you hear the reply
                                    ⟫ wrapped by Cekura's PipecatTracer ⟪
```

| Layer | Tech |
|---|---|
| Orchestration | **Pipecat** |
| Transport | **Daily** (WebRTC) |
| Speech-to-text | **NVIDIA Parakeet** |
| Reasoning | **NVIDIA NIM** — `llama-3.3-nemotron-super-49b-v1.5` |
| Text-to-speech | **Cartesia** |
| Observability | **Cekura** `PipecatTracer` |

See [voice-agent/README.md](voice-agent/README.md) for the full implementation —
NVIDIA model choices, `/no_think` latency handling, tool calling, the pipeline,
and hosting on AWS ECS Fargate.
