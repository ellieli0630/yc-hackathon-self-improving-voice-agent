"""Whole-prompt GEPA optimization through the DEPLOYMENT harness.

Round 5. The dspy.GEPA path (dspy_gepa.py) optimizes the prompt inside DSPy's
ChatAdapter scaffolding — a single flattened `conversation` string field wrapped
in `[[ ## reply ## ]]` markers. Production ships the prompt as a raw system
message followed by real multi-turn chat turns. The code review flagged this as
the #1 issue: GEPA was selecting its winner in a harness we don't deploy, which
is the likely reason round 4's winner looked feasible in-harness but went neutral
on the real `replay()` path.

This module closes that gap. We call the official `gepa.optimize()` with a custom
`GEPAAdapter` whose `evaluate()` runs each candidate through the SAME `replay()`
function deployment uses (system prompt + real role-tagged turns -> NIM). So the
optimization objective and the deployment shape are identical by construction —
GEPA's selected winner is optimal for the harness we actually ship.

Run from services/gepa so `import gepa` resolves to the installed library, not
this directory.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import gepa

from functools import partial

from judge import composite_score, format_feedback, judge_reply, make_openai_judge, render_conversation
from replay import DEFAULT_NIM_MODEL, MAX_WORKERS, make_nim_client, replay

# The single named component GEPA evolves: the whole system prompt.
COMPONENT = "system_prompt"

# Strong reflector, cheap task LM — the intended GEPA pattern (the reflector fires
# once per minibatch, not per rollout, so its cost is amortized).
REFLECTION_MODEL = "gpt-4o"


@dataclass
class Trace:
    """Per-example trajectory consumed by make_reflective_dataset."""

    conversation: str
    reply: str
    score: float
    feedback: str


class NomieReplayAdapter:
    """GEPAAdapter that scores a candidate prompt through the deployment harness.

    DataInst = a golden conversation dict {turns, cut_point, ...}.
    RolloutOutput = the model's reply string.
    Trajectory = Trace.
    """

    # GEPA's reflective-mutation proposer dispatches on `self.adapter.propose_new_texts
    # is not None`, reading it as an ATTRIBUTE. The GEPAAdapter Protocol declares a
    # `= None` default, but Protocols don't inject defaults into duck-typed classes —
    # without this line the access raises AttributeError, which GEPA swallows as "did
    # not propose a new candidate", silently no-opping every mutation. Declaring it None
    # routes proposal to GEPA's built-in default proposer (which uses reflection_lm).
    propose_new_texts = None

    def __init__(self, *, model: str, max_tokens: int = 768):
        self.model = model
        # One NIM client + one judge, reused across the batch. The NIM client is a
        # thin closure over an OpenAI-compatible client (thread-safe to call).
        self._nim = make_nim_client(max_tokens=max_tokens)
        self._judge = make_openai_judge()

    def _rollout(self, ex: dict, prompt: str) -> Trace:
        """One example through replay() (the deployment harness) + judge. Never
        raises — a failed rollout becomes a hard-zero-ish Trace so the engine can
        keep going (per the GEPAAdapter error-handling contract)."""
        turns, cut = ex["turns"], ex["cut_point"]
        convo = render_conversation(turns, cut)
        try:
            reply = replay(system_prompt=prompt, turns=turns, cut_point=cut,
                           client=self._nim, model=self.model).text
        except Exception as e:  # systemic call failure for THIS example only
            return Trace(convo, "", 0.0, f"rollout failed: {e}")
        v = judge_reply(turns=turns, cut_point=cut, candidate_reply=reply, client=self._judge)
        score = composite_score(v)
        return Trace(convo, reply, score, format_feedback(v, score))

    def evaluate(self, batch, candidate, capture_traces=False):
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            traces = list(pool.map(partial(self._rollout, prompt=candidate[COMPONENT]), batch))
        outputs = [t.reply for t in traces]
        scores = [t.score for t in traces]
        return gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=traces if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [
            {
                "Inputs": t.conversation,
                "Generated Outputs": t.reply,
                "Feedback": t.feedback,
            }
            for t in (eval_batch.trajectories or [])
        ]
        return {comp: records for comp in components_to_update}


def _make_reflection_lm(model: str):
    """Plain (prompt:str)->str callable wrapping our OpenAI client — the
    LanguageModel protocol gepa.optimize accepts (it only special-cases str)."""
    from _clients import make_chat_client

    # Reflection legitimately generates more (a full prompt rewrite), so a looser
    # ceiling than the task/judge calls — but still bounded so a hung reflector
    # can't stall the whole run.
    client = make_chat_client(env_key="OPENAI_API_KEY").with_options(timeout=120.0, max_retries=1)

    def _lm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=1.0
        )
        return resp.choices[0].message.content or ""

    return _lm


def main() -> None:
    p = argparse.ArgumentParser(description="Whole-prompt GEPA through the deployment replay() harness.")
    p.add_argument("--base-prompt", required=True)
    p.add_argument("--trainset", required=True)
    p.add_argument("--valset", required=True)
    p.add_argument("--out", default="gepa_native_result.json")
    p.add_argument("--model", default=DEFAULT_NIM_MODEL)
    p.add_argument("--max-metric-calls", type=int, default=400,
                   help="Rollout budget (one call = one replay + one judge). ~light parity with round 4.")
    p.add_argument("--smoke", action="store_true", help="Evaluate the base on one train example and exit")
    args = p.parse_args()

    base_prompt = Path(args.base_prompt).read_text()
    trainset = json.loads(Path(args.trainset).read_text())
    valset = json.loads(Path(args.valset).read_text())
    adapter = NomieReplayAdapter(model=args.model)

    if args.smoke:
        eb = adapter.evaluate(trainset[:1], {COMPONENT: base_prompt}, capture_traces=True)
        print("SCORE:", eb.scores, "| REPLY:", repr(eb.outputs[0])[:200])
        print("FEEDBACK:", eb.trajectories[0].feedback)
        return

    result = gepa.optimize(
        seed_candidate={COMPONENT: base_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=_make_reflection_lm(REFLECTION_MODEL),
        candidate_selection_strategy="pareto",
        frontier_type="instance",
        use_merge=True,
        max_metric_calls=args.max_metric_calls,
        seed=0,
        display_progress_bar=True,
    )
    optimized = result.best_candidate[COMPONENT]
    Path(args.out).write_text(
        json.dumps({"optimized_prompt": optimized, "base_prompt": base_prompt}, indent=2, ensure_ascii=False)
    )
    print(f"[gepa-native] wrote optimized prompt ({len(optimized)} chars) to {args.out}; "
          f"best aggregate score {result.val_aggregate_scores[result.best_idx]:.3f} "
          f"over {result.total_metric_calls} metric calls")


if __name__ == "__main__":
    main()
