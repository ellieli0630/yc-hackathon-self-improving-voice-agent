# GEPA Report — Round 1: Section-Locked (`THINGS YOU NEVER DO`)

> **Outcome:** not strictly better (independent judge flat) — kept as the proxy-overfit lesson.

---

## What this is

The Nomie voice prompt's `THINGS YOU NEVER DO` section, **optimized by a GEPA loop** (replay → judge → mutate → Pareto-select) and **independently scored by Cekura**. Applied to **both** prompt homes that carry the persona:

- `services/voice-bot/system_prompt.txt` — the hack Pipecat/NIM agent (what GEPA optimized + the live demo runs)
- `packages/functions/src/routes/voice.ts` → `NOMIE_SYSTEM_PROMPT` — the prod OpenAI-Realtime agent (kept byte-in-sync)

GEPA **tightened all 7 failure-mode rules** (1942 → 1700 chars, more imperative). See the diff.

## Is this version actually better?

Two independent judges, optimized on **30 train** conversations, reported on **29 test/held-out** real conversations:

| metric | base | this PR | Δ |
|---|---|---|---|
| **Inner judge** (OpenAI, 29 convos) | 0.377 | 0.418 | **+10.9%** ✅ |
| **Cekura overall** (authoritative, 15 convos) | 0.67 | 0.68 | **+0.01 (flat)** |
| · question-machine | 0.40 | 0.47 | +0.07 ✅ |
| · tool-delegation | 0.60 | 0.67 | +0.07 ✅ |
| · therapist-tell | 0.87 | 0.87 | 0 |
| · safety | 0.93 | 0.93 | 0 |
| · forbidden-preamble | 0.53 | 0.47 | **−0.07** ❌ |

**Verdict:** by the inner judge GEPA optimizes against, this prompt looks clearly better (+11% on unseen data). By **Cekura, the independent, authoritative judge, it's a wash**: it genuinely improves question-machine and tool-delegation adherence, but *regresses* preamble (the rewrite dropped the explicit `"That sounds really hard/tough/challenging"` phrase list, so the model has fewer concrete strings to avoid).

So this is **not** a "strictly better" prompt, learnings:

> **Cekura caught GEPA gaming the proxy judge.** A +11% inner-judge gain that an independent rubric flatlines is the textbook over-fitting signal; exactly why a second, independent evaluator belongs in an optimization loop. Shipping the +11% number alone would have been a mistake; the two-judge design prevented it.

## How it works

- **Inner loop** (fast, every candidate): gpt-4o-mini judge on empathy / brevity / action / safety, sharpened to penalize the three failure modes. Drives Pareto selection over (score, length).
- **Authoritative overlay** (per-iteration winner): the 5 Cekura LLM-judge metrics, auto-generated from the Nomie persona. Cekura's eval pipeline requires a recording, so we attach one static placeholder `.wav` to open the gate — Cekura then scores our supplied transcript (verified; audio content ignored). Costs nothing per call.
- **Held-out report**: base vs winner on the 29 test convos, both judges.

Model: NIM `nemotron-super-49b-v1.5` (matches the shipped agent).
