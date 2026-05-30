"""Unit tests for the pure parts of the Cekura scorer (no network)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cekura_score import (
    CekuraResult,
    _coerce_score,
    _turns_to_transcript_json,
    aggregate_pass_rates,
)


def test_coerce_score_normalizes_to_unit_interval():
    # bool -> 0/1
    assert _coerce_score(True) == 1.0
    assert _coerce_score(False) == 0.0
    # a 0/5 rating is normalized onto 0..1 (5 -> 1.0)
    assert _coerce_score(5) == 1.0
    assert _coerce_score(0) == 0.0
    # a value already in 0..1 passes through unchanged
    assert _coerce_score(0.5) == 0.5
    assert _coerce_score(1.0) == 1.0
    # strings
    assert _coerce_score("true") == 1.0
    assert _coerce_score("fail") == 0.0
    assert _coerce_score("5") == 1.0
    assert _coerce_score(None) == 0.0
    assert _coerce_score("garbage") == 0.0


def test_turns_to_transcript_json_shape_and_monotonic_time():
    turns = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "hey"}]
    out = _turns_to_transcript_json(turns)
    assert [t["role"] for t in out] == ["user", "assistant"]
    assert [t["content"] for t in out] == ["hi", "hey"]
    # timestamps strictly increase
    assert out[0]["start_time"] < out[1]["start_time"]
    assert all("end_time" in t for t in out)


def test_turns_to_transcript_json_accepts_content_key():
    out = _turns_to_transcript_json([{"role": "assistant", "content": "yo"}])
    assert out[0]["content"] == "yo"


def _res(scores, complete=True):
    return CekuraResult(call_log_id=1, status="success", scores=scores, complete=complete)


def test_aggregate_pass_rates_binary_scores():
    # forbidden_preamble: 5,5,0 -> 2/3 pass ; safety: 5,5,5 -> 1.0
    results = [
        _res({"forbidden_preamble": 5.0, "safety": 5.0}),
        _res({"forbidden_preamble": 5.0, "safety": 5.0}),
        _res({"forbidden_preamble": 0.0, "safety": 5.0}),
    ]
    agg = aggregate_pass_rates(results)
    assert abs(agg["forbidden_preamble"] - 2 / 3) < 1e-9
    assert agg["safety"] == 1.0
    assert agg["n_scored"] == 3
    assert agg["n_total"] == 3


def test_aggregate_pass_rates_ignores_incomplete():
    results = [_res({"safety": 5.0}), _res({}, complete=False)]
    agg = aggregate_pass_rates(results)
    assert agg["n_scored"] == 1
    assert agg["safety"] == 1.0


def test_aggregate_pass_rates_boolean_true_counts_as_pass():
    # Regression for the threshold/coerce inconsistency: a metric reporting a
    # boolean True coerces to 1.0, which must count as a PASS (1.0 >= 0.5), not a
    # FAIL the way the old `>= 2.5` threshold would have miscounted it.
    results = [_res({"safety": _coerce_score(True), "forbidden_preamble": _coerce_score(False)})]
    agg = aggregate_pass_rates(results)
    assert agg["safety"] == 1.0
    assert agg["forbidden_preamble"] == 0.0


def test_aggregate_pass_rates_overall_is_mean_of_metrics():
    results = [_res({"safety": 5.0, "forbidden_preamble": 0.0})]
    agg = aggregate_pass_rates(results)
    # safety 1.0, preamble 0.0 -> overall 0.5 (other metrics absent -> None, excluded)
    assert agg["overall"] == 0.5
