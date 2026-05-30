"""Build the GEPA golden set from real Nomie conversations.

Reads emotional-content conversation turns from Upstash Vector (prod, READ-ONLY),
reconstructs sessions by sorting on (userId, timestamp), selects the cut_point as the
index of the last user turn, PII-scrubs each turn via an injectable redactor (gpt-4o-mini
by default), and emits one JSON document per conversation:

    {"conversation_id": str, "turns": [{"role": str, "text": str}], "cut_point": int}

to a local directory and, optionally, to S3.

NEVER writes or deletes from Upstash. The Upstash/OpenAI/S3 clients are all injectable so
the pure reconstruction + cut_point logic can be unit-tested with NO network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Sequence

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Role = str  # "user" | "assistant"


@dataclass(frozen=True)
class Turn:
    """A single conversation turn as stored in Upstash metadata."""

    user_id: str
    timestamp: str  # ISO-8601 string; lexicographic sort == chronological
    role: Role
    text: str


@dataclass(frozen=True)
class GoldenConversation:
    conversation_id: str
    turns: list[dict]  # [{"role": str, "text": str}, ...]
    cut_point: int

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "turns": self.turns,
            "cut_point": self.cut_point,
        }


# Redactor signature: takes raw turn text, returns PII-scrubbed text.
Redactor = Callable[[str], str]


class VectorClient(Protocol):
    """Minimal read-only surface of the Upstash Vector index we depend on."""

    def query(self, *args, **kwargs):  # pragma: no cover - protocol only
        ...


# ---------------------------------------------------------------------------
# PURE logic (no network) — unit-tested
# ---------------------------------------------------------------------------


def reconstruct_session(rows: Iterable[Turn]) -> list[Turn]:
    """Sort a single user's turns into chronological order.

    Rows are expected to belong to one userId. They arrive in arbitrary order
    (vector search returns by similarity, not time). We sort by timestamp, with
    role as a stable tiebreaker so a user turn precedes an assistant turn that
    shares the exact same timestamp (the user speaks, then the bot replies).
    """
    # "assistant" > "user" lexicographically, so to put user first on a tie we
    # invert: user -> 0, assistant -> 1.
    def sort_key(t: Turn) -> tuple[str, int]:
        return (t.timestamp, 0 if t.role == "user" else 1)

    return sorted(rows, key=sort_key)


def group_by_user(rows: Iterable[Turn]) -> dict[str, list[Turn]]:
    """Group turns by userId. One user == one conversation in this golden set."""
    grouped: dict[str, list[Turn]] = {}
    for row in rows:
        grouped.setdefault(row.user_id, []).append(row)
    return grouped


def select_cut_point(turns: Sequence[Turn]) -> Optional[int]:
    """Index of the last user turn — the turn the agent must respond to.

    - Conversation ending on a user turn: cut_point is that final index.
    - Conversation with a trailing assistant turn: cut_point is the last user
      turn before it (we evaluate the agent's reply to that user turn).
    - No user turn at all: returns None (caller should skip the conversation).
    """
    last_user = None
    for i, turn in enumerate(turns):
        if turn.role == "user":
            last_user = i
    return last_user


def conversation_id_for(user_id: str, turns: Sequence[Turn]) -> str:
    """Deterministic, non-PII conversation id derived from userId + turn span."""
    first_ts = turns[0].timestamp if turns else ""
    last_ts = turns[-1].timestamp if turns else ""
    digest = hashlib.sha256(f"{user_id}:{first_ts}:{last_ts}".encode()).hexdigest()
    return f"conv_{digest[:16]}"


def build_golden_conversation(
    user_id: str,
    rows: Iterable[Turn],
    redactor: Redactor,
) -> Optional[GoldenConversation]:
    """Reconstruct, scrub, and assemble one golden conversation.

    The redactor is invoked exactly once per turn. Returns None if the
    conversation has no user turn (nothing for the agent to respond to).
    """
    ordered = reconstruct_session(rows)
    if not ordered:
        return None

    cut_point = select_cut_point(ordered)
    if cut_point is None:
        return None

    turns = [{"role": t.role, "text": redactor(t.text)} for t in ordered]
    return GoldenConversation(
        conversation_id=conversation_id_for(user_id, ordered),
        turns=turns,
        cut_point=cut_point,
    )


def build_golden_set(
    rows: Iterable[Turn],
    redactor: Redactor,
) -> list[GoldenConversation]:
    """Pure pipeline: group rows by user, reconstruct + scrub each conversation."""
    conversations: list[GoldenConversation] = []
    for user_id, user_rows in group_by_user(rows).items():
        conv = build_golden_conversation(user_id, user_rows, redactor)
        if conv is not None:
            conversations.append(conv)
    # Stable, deterministic output order.
    conversations.sort(key=lambda c: c.conversation_id)
    return conversations


# ---------------------------------------------------------------------------
# Redactor (gpt-4o-mini) — network, injected at the edges
# ---------------------------------------------------------------------------

REDACTION_SYSTEM_PROMPT = (
    "You are a strict PII scrubber. Replace any personally identifying information "
    "in the user's message with neutral placeholders: names -> [NAME], phone numbers "
    "-> [PHONE], emails -> [EMAIL], street addresses -> [ADDRESS], and any other "
    "direct identifier with a generic placeholder. Preserve the emotional content, "
    "tone, and meaning exactly. Do NOT add commentary. Return ONLY the scrubbed text."
)


def make_openai_redactor(openai_client, model: str = "gpt-4o-mini") -> Redactor:
    """Build a Redactor backed by an OpenAI chat client (injected for testing)."""

    def redact(text: str) -> str:
        if not text or not text.strip():
            return text
        resp = openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": REDACTION_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content or text

    return redact


def noop_redactor(text: str) -> str:
    """Pass-through redactor for dry runs / offline use."""
    return text


# ---------------------------------------------------------------------------
# Upstash Vector ingestion (READ-ONLY) — network, injected at the edges
# ---------------------------------------------------------------------------

EMOTIONAL_QUERY = (
    "I feel anxious, stressed, overwhelmed, sad, lonely, angry, scared, "
    "depressed, worried, hurt, grief, panic, hopeless"
)


def _row_from_metadata(metadata: dict) -> Optional[Turn]:
    """Convert one Upstash result's metadata into a Turn, or None if malformed."""
    if not metadata:
        return None
    # Skip extracted facts — they are not conversation turns.
    if metadata.get("type") == "fact":
        return None
    user_id = metadata.get("userId")
    timestamp = metadata.get("timestamp")
    role = metadata.get("role")
    text = metadata.get("text")
    if not (user_id and timestamp and role and text):
        return None
    return Turn(user_id=str(user_id), timestamp=str(timestamp), role=str(role), text=str(text))


def fetch_emotional_turns(
    vector_client: VectorClient,
    query: str = EMOTIONAL_QUERY,
    top_k: int = 1000,
) -> list[Turn]:
    """READ-ONLY query against Upstash Vector for emotional-content turns.

    Returns the conversation turns (not facts) backing the most semantically
    similar matches. Never mutates the index.
    """
    results = vector_client.query(
        data=query,
        top_k=top_k,
        include_metadata=True,
    )

    turns: list[Turn] = []
    for r in results:
        metadata = getattr(r, "metadata", None)
        if metadata is None and isinstance(r, dict):
            metadata = r.get("metadata")
        turn = _row_from_metadata(metadata or {})
        if turn is not None:
            turns.append(turn)
    return turns


# ---------------------------------------------------------------------------
# Output — local dir + optional S3 (network, injected)
# ---------------------------------------------------------------------------


def write_local(conversations: Sequence[GoldenConversation], out_dir: str | Path) -> list[Path]:
    """Write one JSON file per conversation under out_dir. Returns written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for conv in conversations:
        path = out / f"{conv.conversation_id}.json"
        path.write_text(json.dumps(conv.to_dict(), indent=2, ensure_ascii=False))
        written.append(path)
    return written


def upload_s3(
    conversations: Sequence[GoldenConversation],
    s3_client,
    bucket: str,
    prefix: str = "golden-set",
) -> list[str]:
    """Optionally upload each conversation JSON to S3. Returns object keys."""
    keys: list[str] = []
    for conv in conversations:
        key = f"{prefix.rstrip('/')}/{conv.conversation_id}.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(conv.to_dict(), ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )
        keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Client construction (lazy; only when actually running against prod)
# ---------------------------------------------------------------------------


def _build_vector_client() -> VectorClient:
    from upstash_vector import Index

    url = os.environ["UPSTASH_VECTOR_REST_URL"]
    token = os.environ["UPSTASH_VECTOR_REST_TOKEN"]
    return Index(url=url, token=token)


def _build_openai_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def _build_s3_client():
    import boto3

    return boto3.client("s3")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(
    out_dir: str,
    top_k: int = 1000,
    s3_bucket: Optional[str] = None,
    s3_prefix: str = "golden-set",
    no_redact: bool = False,
    min_turns: int = 1,
    *,
    vector_client: Optional[VectorClient] = None,
    openai_client=None,
    s3_client=None,
) -> list[GoldenConversation]:
    """End-to-end build. All clients injectable; built lazily if not provided."""
    vector_client = vector_client or _build_vector_client()

    if no_redact:
        redactor: Redactor = noop_redactor
    else:
        openai_client = openai_client or _build_openai_client()
        redactor = make_openai_redactor(openai_client)

    rows = fetch_emotional_turns(vector_client, top_k=top_k)
    conversations = build_golden_set(rows, redactor)

    # Drop trivially-short conversations (e.g. "Yeah"/"Bye" test chatter) — they
    # have no substantive emotional context for GEPA to optimize against.
    if min_turns > 1:
        kept = [c for c in conversations if len(c.turns) >= min_turns]
        print(f"Filtered {len(conversations)} -> {len(kept)} conversations (min_turns={min_turns})")
        conversations = kept

    written = write_local(conversations, out_dir)
    print(f"Wrote {len(written)} conversations to {out_dir}")

    if s3_bucket:
        s3_client = s3_client or _build_s3_client()
        keys = upload_s3(conversations, s3_client, s3_bucket, s3_prefix)
        print(f"Uploaded {len(keys)} objects to s3://{s3_bucket}/{s3_prefix}")

    return conversations


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the GEPA golden set (Upstash READ-ONLY).")
    p.add_argument("--out-dir", default="services/gepa/golden_set", help="Local output directory")
    p.add_argument("--top-k", type=int, default=1000, help="Max vectors to pull from Upstash")
    p.add_argument("--s3-bucket", default=None, help="Optional S3 bucket to also upload to")
    p.add_argument("--s3-prefix", default="golden-set", help="S3 key prefix")
    p.add_argument("--no-redact", action="store_true", help="Skip PII redaction (dry run)")
    p.add_argument("--min-turns", type=int, default=1, help="Drop conversations shorter than this")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    run(
        out_dir=args.out_dir,
        top_k=args.top_k,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        no_redact=args.no_redact,
        min_turns=args.min_turns,
    )


if __name__ == "__main__":
    main()
