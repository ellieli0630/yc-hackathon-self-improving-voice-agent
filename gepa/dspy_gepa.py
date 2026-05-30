"""Whole-prompt optimization of the Nomie voice prompt with dspy.GEPA.

The industry-standard path: model Nomie as a one-predictor dspy.Module whose
instruction IS the system prompt, and let dspy.GEPA (the published GEPA engine —
per-instance Pareto + reflective mutation) evolve that whole instruction against
an LLM-judge feedback metric. Cekura stays the independent held-out judge
(report.py) — GEPA never sees it, so the held-out Cekura number is an honest
arbiter, not the optimization target.

Run from services/gepa so `import gepa` (pulled in by dspy.GEPA) resolves to the
installed library, not this directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import dspy

from judge import composite_score, format_feedback, judge_reply, make_openai_judge, render_conversation
from replay import DEFAULT_NIM_MODEL, MAX_WORKERS, NIM_BASE_URL

# GEPA's optimization quality is driven by a STRONG reflection LM: it's the model
# that reads the textual feedback and proposes instruction mutations. The reflector
# fires far less often than task rollouts (once per minibatch, not per example), so
# its cost is amortized — the intended pattern is a cheap task LM (NIM, below) paired
# with a strong reflector. Use gpt-4o (not -mini); same OPENAI_API_KEY, clearly stronger.
REFLECTION_MODEL = "openai/gpt-4o"


class NimLM(dspy.LM):
    """NVIDIA NIM via dspy.LM, force-injecting `/no_think` into the system message.

    nemotron emits chain-of-thought unless told not to, which breaks structured
    output parsing. Injecting at the LM layer (below the instruction) means GEPA's
    mutations of the instruction can never drop it."""

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages:
            msgs = [dict(m) for m in messages]
            if msgs[0].get("role") == "system":
                if not msgs[0]["content"].lstrip().startswith("/no_think"):
                    msgs[0]["content"] = "/no_think\n" + msgs[0]["content"]
            else:
                msgs.insert(0, {"role": "system", "content": "/no_think"})
            messages = msgs
        return super().__call__(prompt=prompt, messages=messages, **kwargs)


# KNOWN TRAIN/DEPLOY HARNESS MISMATCH: GEPA optimizes the instruction INSIDE DSPy's
# ChatAdapter scaffolding — a single flattened `conversation` InputField plus the
# `[[ ## reply ## ]]` field markers ChatAdapter wraps around I/O. Production, however,
# ships this instruction as a raw system prompt with real multi-turn chat messages
# (no flattening, no markers). So the winner GEPA selects is optimal for a harness we
# don't actually deploy. We deliberately do NOT build a custom adapter here; instead
# report.py mitigates by scoring base-vs-winner through the raw `replay()` path (the
# real deployment shape), so the winner is validated against how it actually ships.
class NomieReply(dspy.Signature):
    """Placeholder — replaced at runtime with the full Nomie prompt as instruction."""

    conversation: str = dspy.InputField(desc="The conversation so far, most recent last.")
    reply: str = dspy.OutputField(desc="Nomie's next spoken reply. One natural, in-character turn.")


class Nomie(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(NomieReply)

    def forward(self, conversation: str):
        return self.respond(conversation=conversation)


def to_examples(golden: list[dict]) -> list[dspy.Example]:
    return [
        dspy.Example(
            conversation=render_conversation(conv["turns"], conv["cut_point"]),
            turns=conv["turns"],
            cut_point=conv["cut_point"],
        ).with_inputs("conversation")
        for conv in golden
    ]


def make_feedback_metric(judge):
    """dspy.GEPA metric: judge the generated reply -> score (0..1) + textual feedback."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        reply = getattr(pred, "reply", "") or ""
        v = judge_reply(turns=gold.turns, cut_point=gold.cut_point, candidate_reply=reply, client=judge)
        score = composite_score(v)
        return dspy.Prediction(score=score, feedback=format_feedback(v, score))

    return metric


def main() -> None:
    p = argparse.ArgumentParser(description="dspy.GEPA whole-prompt optimization of the Nomie prompt.")
    p.add_argument("--base-prompt", required=True, help="Seed system prompt (instruction)")
    p.add_argument("--trainset", required=True)
    p.add_argument("--valset", required=True)
    p.add_argument("--out", default="dspy_gepa_result.json", help="Where to write the optimized prompt")
    p.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    p.add_argument("--model", default=DEFAULT_NIM_MODEL)
    p.add_argument("--smoke", action="store_true", help="Run one example through the module + metric and exit")
    args = p.parse_args()

    base_prompt = Path(args.base_prompt).read_text()
    trainset = to_examples(json.loads(Path(args.trainset).read_text()))
    valset = to_examples(json.loads(Path(args.valset).read_text()))

    task_lm = NimLM(model=f"openai/{args.model}", api_base=NIM_BASE_URL,
                    api_key=os.environ["NVIDIA_API_KEY"], temperature=0.6, max_tokens=768)
    reflection_lm = dspy.LM(model=REFLECTION_MODEL, api_key=os.environ["OPENAI_API_KEY"],
                            temperature=1.0, max_tokens=8000)
    dspy.configure(lm=task_lm)

    student = Nomie()
    # Seed the predictor's instruction with the full Nomie prompt (the thing GEPA evolves).
    student.respond.signature = student.respond.signature.with_instructions(base_prompt)

    metric = make_feedback_metric(make_openai_judge())

    if args.smoke:
        ex = trainset[0]
        pred = student(conversation=ex.conversation)
        print("REPLY:", repr(getattr(pred, "reply", None))[:300])
        print("METRIC:", metric(ex, pred))
        return

    gepa = dspy.GEPA(
        metric=metric,
        auto=args.auto,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        use_merge=True,
        track_stats=True,
        num_threads=MAX_WORKERS,
        seed=0,
    )
    optimized = gepa.compile(student, trainset=trainset, valset=valset)

    optimized_prompt = optimized.respond.signature.instructions
    Path(args.out).write_text(
        json.dumps({"optimized_prompt": optimized_prompt, "base_prompt": base_prompt}, indent=2, ensure_ascii=False)
    )
    print(f"[dspy-gepa] wrote optimized prompt ({len(optimized_prompt)} chars) to {args.out}")


if __name__ == "__main__":
    main()
