"""Shared OpenAI-compatible client construction for the GEPA factories.

NIM (replay), the judge, and the mutator all build an ``openai.OpenAI`` client
the same way — lazy import, resolve the key from an explicit arg or an env var
(raising if absent), optionally pointed at a custom ``base_url``. The per-model
call shapes differ, so only the client construction is shared here; each factory
keeps its own ``_call`` closure.
"""

from __future__ import annotations

import os
from typing import Any


def make_chat_client(
    *,
    env_key: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Any:
    """Construct an OpenAI-compatible client, resolving the key from ``api_key``
    or the ``env_key`` environment variable. Import is lazy so modules that only
    use the pure helpers don't require the ``openai`` SDK to be installed."""
    from openai import OpenAI

    key = api_key or os.environ.get(env_key)
    if not key:
        raise RuntimeError(f"{env_key} not set and no api_key provided")
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)
