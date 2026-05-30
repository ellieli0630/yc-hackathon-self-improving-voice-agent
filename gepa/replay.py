"""Replay a golden-set conversation against the NIM LLM with a candidate prompt.

NVIDIA NIM exposes an OpenAI-compatible Chat Completions API at
https://integrate.api.nvidia.com/v1, so we drive it with the `openai` SDK
pointed at that base_url. The network call is wrapped behind an injectable
`ReplayClient` callable so callers (and the loop) can mock it in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

try:
    from ._clients import make_chat_client
except ImportError:  # allow flat-module import (e.g. `from replay import ...` in tests)
    from _clients import make_chat_client

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Must match the live agent's model (infra/voice-bot.ts NIM_MODEL) so GEPA
# optimizes the prompt for the model we actually ship.
DEFAULT_NIM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
# Replay + judge are network-bound; NIM/OpenAI accept concurrent requests, so
# callers fan transcript scoring out across a thread pool of this size.
MAX_WORKERS = 8


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of replaying one transcript against one candidate prompt."""

    text: str
    completion_tokens: int


class ReplayClient(Protocol):
    """Injectable seam: maps (model, messages) -> (assistant_text, tokens)."""

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
    ) -> ReplayResult: ...


def build_messages(
    system_prompt: str,
    turns: list[dict[str, str]],
    cut_point: int,
) -> list[dict[str, str]]:
    """Assemble the chat-completions message list for a replay.

    `turns` is the full golden conversation; `cut_point` is the index of the
    last user turn the agent must respond to. We feed everything up to and
    including `cut_point`, then ask the model for the next assistant turn.

    PURE: no network, no env. Unit-testable.
    """
    history = turns[: cut_point + 1]
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history:
        role = turn["role"]
        # NIM/OpenAI chat roles are user|assistant|system; map anything else to user.
        if role not in ("user", "assistant", "system"):
            role = "user"
        messages.append({"role": role, "content": turn["text"]})
    return messages


def make_nim_client(
    *,
    api_key: str | None = None,
    base_url: str = NIM_BASE_URL,
    temperature: float = 0.6,
    max_tokens: int = 256,
    timeout: float = 60.0,
) -> ReplayClient:
    """Construct a real NIM-backed ReplayClient.

    `timeout` is a hard per-call ceiling (with one retry). NIM/nemotron under load
    occasionally stalls mid-generation; without this a single hung call blocks a
    rollout for the SDK's ~10-min default, dragging a whole GEPA minibatch. On
    timeout the call raises and the caller scores it a failure rather than hanging."""
    client = make_chat_client(env_key="NVIDIA_API_KEY", api_key=api_key, base_url=base_url).with_options(
        timeout=timeout, max_retries=1
    )

    def _call(*, model: str, messages: list[dict[str, str]]) -> ReplayResult:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        usage = getattr(resp, "usage", None)
        tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return ReplayResult(text=text, completion_tokens=int(tokens or 0))

    return _call


def replay(
    *,
    system_prompt: str,
    turns: list[dict[str, str]],
    cut_point: int,
    client: ReplayClient,
    model: str = DEFAULT_NIM_MODEL,
) -> ReplayResult:
    """Replay a single conversation up to `cut_point` and return the model's
    next assistant turn plus its completion-token count (our latency proxy)."""
    # Match production behavior: nemotron needs /no_think to suppress
    # chain-of-thought (see bot.py + infra/voice-bot.ts), so GEPA optimizes the
    # prompt against the responses the shipped agent actually produces.
    if "nemotron" in model:
        system_prompt = "/no_think\n" + system_prompt
    messages = build_messages(system_prompt, turns, cut_point)
    return client(model=model, messages=messages)
