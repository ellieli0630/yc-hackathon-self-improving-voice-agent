"""Score a transcript against Cekura's custom LLM-judge metrics.

Cekura is the *authoritative, independent* judge in the GEPA loop. Where the
gpt-4o-mini judge (judge.py) is the fast inner signal that ranks every
candidate, Cekura scores only the per-iteration winner — a different model and
an independently-authored rubric — so we can tell whether GEPA's gains are real
or just gaming the inner judge.

Mechanics (verified against the live API, not assumed):
  1. POST the transcript to /observability/v1/observe/ with transcript_type
     "pipecat" — the one generic type whose schema is {role, content,
     start_time, end_time}, which is what we already have. Returns a call-log id.
  2. Cekura evaluates asynchronously; poll /observability/v1/call-logs/{id}/
     until evaluation.metrics contains our custom metric ids.

Boolean metrics come back as a 0/1 (or true/false) score; we coerce to float so
a set of them averages into a pass-rate.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Sequence

CEKURA_HOST = "https://api.cekura.ai"
_OBSERVE = "/observability/v1/observe/"
_CALLLOG = "/observability/v1/call-logs/{id}/"

# Cekura's eval pipeline only fires when a recording is present; this static
# silent-but-real placeholder .wav opens the gate so it scores our transcript
# (Cekura ignores the audio's content — verified). One file, reused for every
# ingest, so it costs nothing per call.
DEFAULT_VOICE_GATE_URI = "s3://nomie-hack-assetsbucket-esvaffnd/gepa/real.wav"


def presign_voice_url(uri: str = DEFAULT_VOICE_GATE_URI, *, expires_in: int = 3600) -> str:
    """Presign the placeholder recording fresh (a long run can outlive a URL)."""
    out = subprocess.run(
        ["aws", "s3", "presign", uri, "--expires-in", str(expires_in)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()

# The 5 custom content metrics Cekura generated from the Nomie persona. Safety
# is the non-negotiable guardrail (the only "affects call success" metric).
CUSTOM_METRIC_IDS: dict[int, str] = {
    146465: "question_machine",
    146466: "therapist_tell",
    146467: "forbidden_preamble",
    146468: "tool_delegation",
    146469: "safety",
}


@dataclass(frozen=True)
class CekuraResult:
    call_log_id: int
    status: str  # success | failure | evaluating
    scores: dict[str, float]  # metric name -> 0..1 (normalized; see _coerce_score)
    complete: bool  # all requested metrics present


def _coerce_score(raw) -> float:
    """Normalize a Cekura metric score to 0..1 so a "pass" is consistent
    regardless of whether the metric reports a bool, a 0/5 rating, or a float.

      - bool true -> 1.0, false -> 0.0
      - a rating > 1 (Cekura's 0/5 scale) -> value / 5  (so 5 -> 1.0, 0 -> 0.0)
      - a float already in 0..1 -> passed through unchanged
      - strings ("true"/"pass"/"yes" -> 1.0, "false"/"fail"/"no" -> 0.0, else
        parsed numerically and run through the same normalization)

    Keeping everything on one 0..1 scale removes the latent inconsistency where a
    boolean-true metric (1.0) would fall below a 2.5 "pass" threshold tuned for
    the 0/5 scale and be miscounted as a fail."""
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return _normalize_numeric(float(raw))
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ("true", "pass", "yes"):
            return 1.0
        if low in ("false", "fail", "no"):
            return 0.0
        try:
            return _normalize_numeric(float(low))
        except ValueError:
            return 0.0
    return 0.0


def _normalize_numeric(val: float) -> float:
    """Map a numeric metric score onto 0..1. Values above 1 are treated as a 0/5
    rating (divide by 5); values already in 0..1 pass through."""
    return val / 5.0 if val > 1.0 else val


def _request(method: str, path: str, api_key: str, payload: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(
        CEKURA_HOST + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method,
    )
    req.add_header("X-CEKURA-API-KEY", api_key)
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _turns_to_transcript_json(turns: Sequence[dict]) -> list[dict]:
    """Convert golden/replay turns ([{role, text}] or [{role, content}]) into
    Cekura's pipecat transcript_json. Timestamps are synthetic but monotonic —
    the content metrics don't depend on real timing."""
    out: list[dict] = []
    t = 0.0
    for turn in turns:
        content = turn.get("text") if turn.get("text") is not None else turn.get("content", "")
        out.append({"role": turn["role"], "content": content, "start_time": round(t, 2), "end_time": round(t + 2.0, 2)})
        t += 2.5
    return out


def ingest_transcript(
    turns: Sequence[dict],
    *,
    api_key: str,
    agent_id: int,
    voice_recording_url: str,
    call_id: str | None = None,
) -> int:
    """Ingest a transcript for observability scoring; returns the call-log id.

    `voice_recording_url` is REQUIRED even though we score text: Cekura's eval
    pipeline only fires when a recording is present. A single static silent .wav
    satisfies the gate — Cekura then judges our `transcript_json`, ignoring the
    audio's content (verified: a failing transcript paired with unrelated audio
    still scored preamble/therapist-tell as fail)."""
    payload = {
        "agent": int(agent_id),
        "transcript_type": "pipecat",
        "call_id": call_id or str(uuid.uuid4()),
        "call_ended_reason": "completed",
        "voice_recording_url": voice_recording_url,
        "transcript_json": _turns_to_transcript_json(turns),
    }
    status, body = _request("POST", _OBSERVE, api_key, payload)
    if status not in (200, 201):
        raise RuntimeError(f"Cekura ingest failed ({status}): {body[:300]}")
    return int(json.loads(body)["id"])


def fetch_scores(
    call_log_id: int,
    *,
    api_key: str,
    metric_ids: dict[int, str] = CUSTOM_METRIC_IDS,
    poll_interval: float = 10.0,
    max_polls: int = 30,
    sleep=time.sleep,
) -> CekuraResult:
    """Poll the call-log until all requested metrics are evaluated (or timeout)."""
    scores: dict[str, float] = {}
    status = "evaluating"
    for _ in range(max_polls):
        sleep(poll_interval)
        http, body = _request("GET", _CALLLOG.format(id=call_log_id), api_key)
        if http != 200:
            continue
        j = json.loads(body)
        status = j.get("status", status)
        metrics = (j.get("evaluation") or {}).get("metrics") or []
        scores = {
            metric_ids[m["id"]]: _coerce_score(m.get("score"))
            for m in metrics
            if m.get("id") in metric_ids and m.get("score") is not None
        }
        if len(scores) >= len(metric_ids):
            return CekuraResult(call_log_id, status, scores, complete=True)
    return CekuraResult(call_log_id, status, scores, complete=len(scores) >= len(metric_ids))


def score_transcript(
    turns: Sequence[dict], *, api_key: str, agent_id: int, voice_recording_url: str, **poll_kwargs
) -> CekuraResult:
    """Ingest + poll in one call."""
    cl_id = ingest_transcript(
        turns, api_key=api_key, agent_id=agent_id, voice_recording_url=voice_recording_url
    )
    return fetch_scores(cl_id, api_key=api_key, **poll_kwargs)


def score_batch(
    transcripts: Sequence[Sequence[dict]],
    *,
    api_key: str,
    agent_id: int,
    voice_recording_url: str,
    metric_ids: dict[int, str] = CUSTOM_METRIC_IDS,
    poll_interval: float = 12.0,
    max_polls: int = 40,
    sleep=time.sleep,
) -> list[CekuraResult]:
    """Ingest all transcripts up front, then poll the whole set together.

    Cekura evaluates asynchronously (~1-3 min each) and in parallel, so ingesting
    the batch and polling them as a group costs roughly one eval-latency for the
    whole iteration rather than N sequential waits."""
    ids = [
        ingest_transcript(t, api_key=api_key, agent_id=agent_id, voice_recording_url=voice_recording_url)
        for t in transcripts
    ]
    results: dict[int, CekuraResult] = {}
    for _ in range(max_polls):
        sleep(poll_interval)
        for cid in ids:
            if cid in results:
                continue
            http, body = _request("GET", _CALLLOG.format(id=cid), api_key)
            if http != 200:
                continue
            j = json.loads(body)
            metrics = (j.get("evaluation") or {}).get("metrics") or []
            scores = {
                metric_ids[m["id"]]: _coerce_score(m.get("score"))
                for m in metrics
                if m.get("id") in metric_ids and m.get("score") is not None
            }
            if len(scores) >= len(metric_ids):
                results[cid] = CekuraResult(cid, j.get("status", ""), scores, complete=True)
        if len(results) >= len(ids):
            break
    return [results.get(cid, CekuraResult(cid, "timeout", {}, complete=False)) for cid in ids]


def aggregate_pass_rates(
    results: Sequence[CekuraResult], metric_ids: dict[int, str] = CUSTOM_METRIC_IDS
) -> dict[str, float | int | None]:
    """Reduce a batch of results to a per-metric pass-rate (0..1). Scores are
    normalized to 0..1 by _coerce_score (0/5 rating -> 0..1, bool true -> 1.0), so
    pass-rate = fraction at or above the 0.5 midpoint (a 5/5 rating or a True)."""
    done = [r for r in results if r.complete]
    out: dict[str, float | int | None] = {"n_scored": len(done), "n_total": len(results)}
    for name in metric_ids.values():
        vals = [r.scores[name] for r in done if name in r.scores]
        out[name] = (sum(1 for v in vals if v >= 0.5) / len(vals)) if vals else None
    rates = [out[name] for name in metric_ids.values() if isinstance(out[name], float)]
    out["overall"] = sum(rates) / len(rates) if rates else None
    return out
