# GEPA Report — Round 4: Whole-Prompt (safe but degraded)

> *experiment(prompt): whole-prompt GEPA — round 4 (safe but degraded; DO NOT MERGE)*
> **Scope:** entire system prompt, nothing locked · **Diff:** +26 / −555 (prompt collapses 18,732 → 2,836 chars) ·
> **Outcome:** 🛑 **DO NOT MERGE** — documented negative result; the counterpart to [#report-1](report-1-section-locked-round1.md).

---

## What this is

The **whole-prompt** counterpart to #report-1. Where #report-1 optimized a single section (`THINGS YOU NEVER DO`), this run let **dspy.GEPA evolve the entire system prompt** — same engine, same NIM task LM, same two-judge design (gpt-4o-mini inner judge + Cekura held-out), but nothing locked. Applied to both prompt homes so the diff is reviewable:

- `services/voice-bot/system_prompt.txt` — the hack Pipecat/NIM agent
- `packages/functions/src/routes/voice.ts` → `NOMIE_SYSTEM_PROMPT` — the prod OpenAI-Realtime agent

**🛑 DO NOT MERGE.** This is a documented negative result, captured the same way #report-1 captured the proxy-overfit lesson. The diff itself is the finding: the prompt collapses from **18,732 → 2,836 chars**.

## Is this version actually better?

Optimized on **38 train** (30 real + 4 crisis + 4 tool-request), reported on **29 held-out test** real conversations + a **3-input crisis probe**:

| metric | base | this PR | Δ |
|---|---|---|---|
| **Crisis 988 surfaced** (3 inputs) | 2/3 | **3/3** | **+1 ✅ safer** |
| **Inner judge — feasibility** (29) | −3.016 | +0.425 | base *infeasible*, winner *feasible* |
| **Cekura overall** (authoritative, 15) | 0.67 | 0.64 | **−0.03 ❌** |
| · question-machine | 0.47 | 0.27 | **−0.20** ❌ |
| · forbidden-preamble | 0.40 | 0.47 | +0.07 ✅ |
| · therapist-tell | 0.93 | 0.93 | 0 |
| · tool-delegation | 0.53 | 0.53 | 0 (see below) |
| · safety | 1.00 | 1.00 | 0 (no crisis in the 15-sample) |
| prompt size | 18,732 | 2,836 chars | **−85%** |

**Verdict: no.** It is *safer* on crisis (the systemic safety fix worked — more on that below), but on the independent judge it's **neutral-to-worse**, it **regressed question-machine hard (−0.20)**, and — not visible in the table — it **deleted every tool-delegation / `launch_exercise` instruction** and gutted the persona down to a generic shell. The big inner-judge "+3.4" is almost entirely the safety-feasibility flip, not a quality gain.

> The tool-delegation number reads flat (0.53) only because the 15-conversation Cekura sample barely exercises tool requests — the prompt structurally lost the capability even though this eval didn't catch it. That's the same "an eval only measures what it exercises" lesson, one layer down.

## What we actually learned

Two findings, both worth keeping:

1. **The systemic safety fix works.** Rounds 2–3, whole-prompt GEPA *deleted* the 988 crisis protocol (nothing scored it, so it got optimized away — a diluted 0.0 wasn't enough to stop it). Round 4 added a **per-instance fixed-penalty constraint** (−100, à la *Certified Safe RLHF* 2510.03520 / *RePO* 2410.19933) plus a calibrated LLM crisis detector. Result: GEPA **could not** win without surfacing 988, and the winner beats base on crisis (3/3 vs 2/3). Safety as a hard gate, not a tradeable score — proven.

2. **But a safe optimizer still trades away everything you *don't* gate.** With safety gated and tools only *softly* penalized (×0.6), GEPA bought a marginal brevity gain by **dropping tools entirely** and stripping the persona — for *no* net style win. This is the whole-prompt failure mode: it preserves exactly what you hard-protect and discards the rest.

So the arc across rounds is the real deliverable:

> **Section-locking (#report-1) was right.** The only configuration that produced a prompt that's safe **and** improved **and** still full-featured was the locked-section run. Whole-prompt GEPA, even with safety correctly constrained, converges on a safe-but-stripped prompt. The fix isn't "trust the optimizer more" — it's "hard-gate every non-negotiable," which at the limit *is* section-locking by another name.

The regression is reviewable next to #report-1, and so the demo can show — not just claim — what an unconstrained whole-prompt optimizer does to a carefully-authored prompt.
