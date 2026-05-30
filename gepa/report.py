"""Held-out before/after report: base prompt vs the GEPA winner.

Optimization (dspy_gepa.py) runs on the TRAIN split; this scores both the
original base prompt and GEPA's winning prompt on the held-out TEST split that
GEPA never saw — so the reported lift is generalization, not overfitting.
Inner-judge composite is computed on every test conversation; Cekura's
authoritative per-metric pass-rates on a bounded sample.

Output JSON feeds the PR body.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cekura_score import DEFAULT_VOICE_GATE_URI, aggregate_pass_rates, presign_voice_url, score_batch
from judge import composite_score, judge_reply, make_openai_judge
from replay import DEFAULT_NIM_MODEL, MAX_WORKERS, make_nim_client, replay


def _inner_mean(prompt: str, test: list[dict], *, nim, judge, model: str) -> float:
    """Mean inner-judge composite of `prompt` over the test conversations."""

    def _one(conv: dict) -> float:
        turns, cut = conv["turns"], conv["cut_point"]
        reply = replay(system_prompt=prompt, turns=turns, cut_point=cut, client=nim, model=model).text
        v = judge_reply(turns=turns, cut_point=cut, candidate_reply=reply, client=judge)
        return composite_score(v)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(test) or 1)) as ex:
        scores = list(ex.map(_one, test))
    return sum(scores) / len(scores) if scores else 0.0


def _cekura_passrates(prompt: str, sample: list[dict], *, nim, model: str, voice_uri: str, api_key: str, agent_id: int) -> dict:
    """Cekura per-metric pass-rates of `prompt` over a test sample."""

    def _transcript(conv: dict) -> list[dict]:
        turns, cut = conv["turns"], conv["cut_point"]
        reply = replay(system_prompt=prompt, turns=turns, cut_point=cut, client=nim, model=model).text
        return list(turns[: cut + 1]) + [{"role": "assistant", "text": reply}]

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(sample) or 1)) as ex:
        transcripts = list(ex.map(_transcript, sample))
    results = score_batch(
        transcripts, api_key=api_key, agent_id=agent_id, voice_recording_url=presign_voice_url(voice_uri)
    )
    return aggregate_pass_rates(results)


def main() -> None:
    p = argparse.ArgumentParser(description="Held-out base-vs-winner report.")
    p.add_argument("--base-prompt", required=True)
    p.add_argument("--winner-prompt", required=True, help="dspy_gepa.py result JSON ({optimized_prompt}) or a raw prompt .txt")
    p.add_argument("--test", required=True, help="held-out test conversations JSON")
    p.add_argument("--out", default="report.json")
    p.add_argument("--cekura-sample", type=int, default=15)
    p.add_argument("--model", default=DEFAULT_NIM_MODEL)
    p.add_argument("--voice-s3-uri", default=DEFAULT_VOICE_GATE_URI)
    args = p.parse_args()

    base_prompt = Path(args.base_prompt).read_text()
    test = json.loads(Path(args.test).read_text())
    raw = Path(args.winner_prompt).read_text()
    try:
        winner_prompt = json.loads(raw)["optimized_prompt"]
    except (json.JSONDecodeError, KeyError, TypeError):
        winner_prompt = raw  # plain .txt prompt
    winner_id = "dspy-gepa"

    nim = make_nim_client()
    judge = make_openai_judge()
    ck_key = os.environ["CEKURA_API_KEY"]
    ck_agent = int(os.environ["CEKURA_AGENT_ID"])
    sample = test[: args.cekura_sample]

    print(f"[report] winner={winner_id}", flush=True)
    print(f"[report] inner judge on {len(test)} held-out convos...", flush=True)
    base_inner = _inner_mean(base_prompt, test, nim=nim, judge=judge, model=args.model)
    win_inner = _inner_mean(winner_prompt, test, nim=nim, judge=judge, model=args.model)
    print(f"[report]   base={base_inner:.3f}  winner={win_inner:.3f}", flush=True)

    print(f"[report] cekura on {len(sample)} held-out convos (base, then winner)...", flush=True)
    base_ck = _cekura_passrates(base_prompt, sample, nim=nim, model=args.model, voice_uri=args.voice_s3_uri, api_key=ck_key, agent_id=ck_agent)
    win_ck = _cekura_passrates(winner_prompt, sample, nim=nim, model=args.model, voice_uri=args.voice_s3_uri, api_key=ck_key, agent_id=ck_agent)

    report = {
        "winner_prompt_id": winner_id,
        "n_test_inner": len(test),
        "n_test_cekura": len(sample),
        "inner_judge": {"base": round(base_inner, 4), "winner": round(win_inner, 4),
                        "delta": round(win_inner - base_inner, 4)},
        "cekura": {"base": base_ck, "winner": win_ck},
        "winner_prompt": winner_prompt,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[report] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
