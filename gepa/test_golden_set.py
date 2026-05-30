"""Unit tests for the PURE golden-set logic. NO network.

Run: .venv-test/bin/pytest services/gepa/test_golden_set.py
"""

import importlib.util
import sys
from pathlib import Path

# Load build_golden_set.py by path so this test does not import the `gepa`
# package __init__ (which pulls in sibling modules owned by other components and
# may not exist yet during parallel development). Register in sys.modules so
# dataclass machinery can resolve the module by name.
_spec = importlib.util.spec_from_file_location(
    "build_golden_set", Path(__file__).with_name("build_golden_set.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["build_golden_set"] = _mod
_spec.loader.exec_module(_mod)

Turn = _mod.Turn
build_golden_conversation = _mod.build_golden_conversation
build_golden_set = _mod.build_golden_set
fetch_emotional_turns = _mod.fetch_emotional_turns
group_by_user = _mod.group_by_user
reconstruct_session = _mod.reconstruct_session
select_cut_point = _mod.select_cut_point


# ---------------------------------------------------------------------------
# Session reconstruction
# ---------------------------------------------------------------------------


def test_reconstruct_session_sorts_unsorted_rows_by_timestamp():
    rows = [
        Turn("u1", "2026-01-01T00:00:03Z", "assistant", "second reply"),
        Turn("u1", "2026-01-01T00:00:00Z", "user", "first message"),
        Turn("u1", "2026-01-01T00:00:02Z", "user", "second message"),
        Turn("u1", "2026-01-01T00:00:01Z", "assistant", "first reply"),
    ]
    ordered = reconstruct_session(rows)
    assert [t.text for t in ordered] == [
        "first message",
        "first reply",
        "second message",
        "second reply",
    ]


def test_reconstruct_session_user_precedes_assistant_on_tie():
    # Same timestamp: the user turn must come before the assistant reply.
    rows = [
        Turn("u1", "2026-01-01T00:00:00Z", "assistant", "reply"),
        Turn("u1", "2026-01-01T00:00:00Z", "user", "message"),
    ]
    ordered = reconstruct_session(rows)
    assert [t.role for t in ordered] == ["user", "assistant"]


def test_reconstruct_session_empty():
    assert reconstruct_session([]) == []


def test_group_by_user_partitions_rows():
    rows = [
        Turn("u1", "t0", "user", "a"),
        Turn("u2", "t0", "user", "b"),
        Turn("u1", "t1", "assistant", "c"),
    ]
    grouped = group_by_user(rows)
    assert set(grouped.keys()) == {"u1", "u2"}
    assert len(grouped["u1"]) == 2
    assert len(grouped["u2"]) == 1


# ---------------------------------------------------------------------------
# cut_point selection
# ---------------------------------------------------------------------------


def test_cut_point_is_last_user_turn_when_ending_on_user():
    turns = [
        Turn("u1", "t0", "user", "hi"),
        Turn("u1", "t1", "assistant", "hello"),
        Turn("u1", "t2", "user", "i feel anxious"),
    ]
    assert select_cut_point(turns) == 2


def test_cut_point_skips_trailing_assistant_turn():
    turns = [
        Turn("u1", "t0", "user", "hi"),
        Turn("u1", "t1", "assistant", "hello"),
        Turn("u1", "t2", "user", "i feel anxious"),
        Turn("u1", "t3", "assistant", "tell me more"),
    ]
    # The last *user* turn is index 2, not the trailing assistant at index 3.
    assert select_cut_point(turns) == 2


def test_cut_point_multiple_trailing_assistant_turns():
    turns = [
        Turn("u1", "t0", "user", "hi"),
        Turn("u1", "t1", "assistant", "a"),
        Turn("u1", "t2", "assistant", "b"),
    ]
    assert select_cut_point(turns) == 0


def test_cut_point_none_when_no_user_turn():
    turns = [
        Turn("u1", "t0", "assistant", "a"),
        Turn("u1", "t1", "assistant", "b"),
    ]
    assert select_cut_point(turns) is None


# ---------------------------------------------------------------------------
# Redactor invocation
# ---------------------------------------------------------------------------


def test_redactor_invoked_once_per_turn():
    calls = []

    def spy_redactor(text: str) -> str:
        calls.append(text)
        return f"REDACTED({text})"

    rows = [
        Turn("u1", "t0", "user", "my name is Bob"),
        Turn("u1", "t1", "assistant", "hi Bob"),
        Turn("u1", "t2", "user", "call me at 555-1234"),
    ]
    conv = build_golden_conversation("u1", rows, spy_redactor)

    assert conv is not None
    # Exactly one redactor call per turn.
    assert calls == ["my name is Bob", "hi Bob", "call me at 555-1234"]
    assert [t["text"] for t in conv.turns] == [
        "REDACTED(my name is Bob)",
        "REDACTED(hi Bob)",
        "REDACTED(call me at 555-1234)",
    ]
    assert conv.cut_point == 2


def test_build_golden_conversation_returns_none_without_user_turn():
    rows = [Turn("u1", "t0", "assistant", "a")]
    assert build_golden_conversation("u1", rows, lambda t: t) is None


# ---------------------------------------------------------------------------
# Full pure pipeline
# ---------------------------------------------------------------------------


def test_build_golden_set_groups_and_orders_deterministically():
    rows = [
        # user u2, out of order
        Turn("u2", "2026-01-02T00:00:01Z", "assistant", "u2 reply"),
        Turn("u2", "2026-01-02T00:00:00Z", "user", "u2 msg"),
        # user u1, out of order
        Turn("u1", "2026-01-01T00:00:01Z", "user", "u1 second"),
        Turn("u1", "2026-01-01T00:00:00Z", "user", "u1 first"),
    ]
    convs = build_golden_set(rows, lambda t: t)
    assert len(convs) == 2
    # Deterministic order by conversation_id.
    ids = [c.conversation_id for c in convs]
    assert ids == sorted(ids)

    by_user_text = {tuple(t["text"] for t in c.turns) for c in convs}
    assert ("u1 first", "u1 second") in by_user_text
    assert ("u2 msg", "u2 reply") in by_user_text


def test_build_golden_set_drops_conversations_without_user_turn():
    rows = [
        Turn("u1", "t0", "assistant", "only assistant"),
        Turn("u2", "t0", "user", "real user turn"),
    ]
    convs = build_golden_set(rows, lambda t: t)
    assert len(convs) == 1
    assert convs[0].turns[0]["text"] == "real user turn"


# ---------------------------------------------------------------------------
# fetch_emotional_turns mapping (no network — fake client)
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, metadata):
        self.metadata = metadata


class _FakeVectorClient:
    def __init__(self, results):
        self._results = results
        self.last_kwargs = None

    def query(self, **kwargs):
        self.last_kwargs = kwargs
        return self._results


def test_fetch_emotional_turns_maps_metadata_and_is_read_only():
    client = _FakeVectorClient(
        [
            _FakeResult({"userId": "u1", "timestamp": "t0", "role": "user", "text": "hi"}),
            _FakeResult({"type": "fact", "userId": "u1", "fact": "likes tea"}),  # skipped
            _FakeResult({"userId": "u1", "role": "user"}),  # malformed -> skipped
            _FakeResult(None),  # skipped
        ]
    )
    turns = fetch_emotional_turns(client, top_k=10)
    assert len(turns) == 1
    assert turns[0] == Turn("u1", "t0", "user", "hi")
    # Confirm we only issued a read query (include_metadata, top_k) — no upsert/delete.
    assert client.last_kwargs["include_metadata"] is True
    assert client.last_kwargs["top_k"] == 10


def test_fetch_emotional_turns_supports_dict_results():
    client = _FakeVectorClient(
        [{"metadata": {"userId": "u9", "timestamp": "t5", "role": "assistant", "text": "ok"}}]
    )
    turns = fetch_emotional_turns(client)
    assert turns == [Turn("u9", "t5", "assistant", "ok")]
