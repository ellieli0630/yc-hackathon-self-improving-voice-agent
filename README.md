# Nomie — A Self-Evolving Voice Companion

> A wellness companion that lives inside a self-care open-world game — rebuilt on an
> open NVIDIA voice stack, and wired into a loop that lets it **improve its own
> prompt from real conversations**, with guardrails that stop it from improving away
> the things that matter.

**▶️ [Watch the demo](https://github.com/ellieli0630/yc-hackathon-self-improving-voice-agent/releases/download/demo-video/yc-hackathon-demo.mp4)  ·  📊 [Presentation slides](https://docs.google.com/presentation/d/10sFP2ZcR9PZ_vgu6sBed41zXESWw58oE/edit?usp=sharing&ouid=102756663402202622206&rtpof=true&sd=true)**

## 🎥 Demo

https://github.com/ellieli0630/yc-hackathon-self-improving-voice-agent/assets/demo-video/yc-hackathon-demo.mp4

<video src="https://github.com/ellieli0630/yc-hackathon-self-improving-voice-agent/releases/download/demo-video/yc-hackathon-demo.mp4" controls width="100%"></video>

> ▶️ **[Watch the demo video](https://github.com/ellieli0630/yc-hackathon-self-improving-voice-agent/releases/download/demo-video/yc-hackathon-demo.mp4)** — if the player above doesn't load inline, this link always works.

---

Built for the [YC Voice Agents Hackathon](https://github.com/pipecat-ai/yc-voice-agents-hackathon)
· Pipecat orchestration · NVIDIA open models · Cekura evaluation.

---

## 🏆 For the judges

### How I used Cekura, Nemotron, and Pipecat

- **Pipecat — *voice*.** Orchestrates the entire real-time pipeline as one streaming
  graph: Daily transport in/out → NVIDIA Parakeet STT → the NVIDIA LLM → Cartesia TTS,
  with Silero VAD, one bot process per call. It's the backbone of the live agent.
- **Nemotron / NVIDIA NIM — *open weights*.** `llama-3.3-nemotron-super-49b-v1.5` is
  the agent's brain **and** the task model the optimizer runs against — so the prompt
  is tuned for the exact open-weights model that ships. (Parakeet is the open NVIDIA STT.)
- **Cekura — *evaluating & improving agent performance*.** Two roles: (1) `PipecatTracer`
  traces every live conversation's transcript + tool calls; (2) it's the **independent,
  authoritative judge** in the loop — I auto-generated **five custom metrics** from
  Nomie's persona and score every candidate transcript against them, held-out, on a
  model the optimizer never sees.

### Cekura: what I was testing, and how much performance moved

**Goal:** an *independent, trustworthy* way to tell whether a prompt change actually
made Nomie better — separate from the fast judge I was optimizing against, so I
couldn't fool myself. Cekura's five metrics (question-machine, therapist-tell,
forbidden-preamble, tool-delegation, **crisis-safety**) are that check.

**What moved — honestly:**
- The headline isn't a big aggregate jump — it's that **Cekura caught
  over-optimization.** On run 1, my own judge showed **+11%** on held-out data; Cekura
  showed **flat (0.67 → 0.68)** and revealed that "+11%" was overfitting, not real
  improvement. That's the most valuable result.
- On individual axes Cekura measured real movement: **question-machine +0.07,
  tool-delegation +0.07**, but **forbidden-preamble −0.07** — a genuine wash, which is
  the honest read.
- The improvement I *can* stand behind is **crisis-safety**: after I made safety a
  hard gate, the agent went from **2/3 → 3/3** on an independent crisis probe —
  measured, not asserted.

So Cekura's value was less "it made the number go up" and more "it **stopped me
shipping a fake win**, and it **verified the one real improvement (safety) was real**."

### What's new in this hackathon

- **New — built during the hackathon:**
  - The **entire NVIDIA + Pipecat voice agent** — rebuilt Nomie's voice brain on open
    NVIDIA models (Nemotron + Parakeet), Pipecat, Daily, and Cartesia, replacing the
    production OpenAI-Realtime path; plus the Cekura tracing and the AWS ECS Fargate deploy.
  - The **entire GEPA self-improvement loop** — golden-set builder, replay harness, the
    inner judge with safety/tool **hard gates**, the optimizer (native + dspy paths), the
    Cekura scoring integration, the held-out report, and apply-winner.
  - The **five experiment runs + the reports** documenting them (including the negatives).
  - The **five custom Cekura metrics** and the observability-based scoring approach.
- **Pre-existing — not new:** Nomie the product — 5,000 users, the React Native app, the
  Unity game, the production OpenAI-Realtime voice agent, the persona/system prompt, and
  the user conversations the golden set is built from.
- **Borrowed — frameworks/platforms:** GEPA (via `dspy`/`gepa`), Pipecat, NVIDIA NIM
  models, Cekura, Daily, Cartesia.

---

## 1. What is Nomie

Nomie is a **wellness companion** — you talk to it like a sharp, caring friend who's
read a lot of psychology, not a therapist. It listens, reflects, and when it helps,
opens an activity in the game (a breathing exercise, a worry jar, a CBT worksheet),
and it remembers what you've told it before.

**Today ~5,000 people talk to Nomie in production** — about 5 minutes a day on
average, with power users going for hours. That real usage is the whole premise of
this project: *if people are already having real conversations, can the agent learn
from them and get better on its own?*

For the hackathon I did two things:
1. **Rebuilt the voice agent** on an open, self-hostable NVIDIA + Pipecat stack.
2. **Built a self-evolving loop** (GEPA + Cekura) that optimizes Nomie's prompt from
   those real conversations — and, more importantly, learned where that goes wrong.

> The React Native app and Unity game scenes are **not** in this repo (they're a
> large mobile/game codebase). What's here is the part that matters for the loop: the
> [voice agent](voice-agent/) and the [optimization loop](gepa/).

---

## 2. The voice agent

```
You speak ─▶ Daily (WebRTC) ─▶ NVIDIA Parakeet (STT) ─▶ NVIDIA NIM Nemotron (LLM)
          ─▶ Cartesia (TTS) ─▶ Daily ─▶ you hear the reply
                                    ⟫ traced end-to-end by Cekura ⟪
```

| Layer | Role | Tech |
|---|---|---|
| Transport | carries live audio both ways | **Daily** (WebRTC) |
| Ears (STT) | speech → text | **NVIDIA Parakeet** |
| Brain (LLM) | decides what to say + which tool to call | **NVIDIA NIM** `llama-3.3-nemotron-super-49b-v1.5` |
| Voice (TTS) | text → speech | **Cartesia** |
| Orchestration | wires the streaming pipeline | **Pipecat** |
| Observability | records transcripts + tool calls | **Cekura** |

Full build notes in **[voice-agent/README.md](voice-agent/README.md)**.

The agent has two tools — `launch_exercise` (opens an activity in the game) and
`search_user_memory` — and runs one bot process per call on **AWS ECS Fargate**.

---

## 3. Self-evolving agent

A normal agent is **frozen** — it behaves the same until an engineer rewrites the
prompt. Nomie has thousands of real conversations a day; the question was whether I
could turn that into a loop where Nomie improves **on its own**.

**GEPA** (Genetic-Pareto) is a published *reflective prompt optimizer*. It doesn't
judge quality itself — it consumes whatever metric you hand it, evolves the prompt by
**reflective mutation** (an LLM reads what went wrong and rewrites the prompt), and
keeps a **Pareto frontier** of candidates that win on different conversations. It's
the "evolve" engine; everything interesting is in *what you let it optimize* and
*what stops it*.

---

## 4. How I make it self-evolving

```
  Golden set  ─▶  Replay  ─▶  Judge  ─▶  Reflect & rewrite  ─▶  Keep winners ─┐
 (real convos)   (re-answer) (score+WHY)  (mutate the prompt)   (Pareto)      │
      ▲                                                                       │ repeat ×hundreds
      │                                                                       │
      └──────────────────────────  winning prompt  ◀──────────────────────────┘
                                         │
                                         ▼
                         Verify on a held-out split  ─▶  human review  ─▶  ship
```

The inner loop, step by step:

1. **Golden set** — real, **PII-scrubbed** conversations pulled READ-ONLY from prod
   (`build_golden_set.py`), split into train / held-out test.
2. **Replay** — re-answer each conversation through the same NVIDIA Nemotron the live
   agent uses (`replay.py`).
3. **Judge** — a fast `gpt-4o-mini` **inner judge** scores each reply *and writes down
   why it's weak* (`judge.py`).
4. **Reflect & rewrite** — GEPA reads that feedback and rewrites the prompt.
5. **Keep winners** — Pareto-select, repeat for hundreds of candidates.

### Two judges — and this is the important part

The inner judge is fast but it's *the thing GEPA optimizes against* — so GEPA can
learn to game it. So there's a second, independent judge:

- **Cekura** — an outside rubric on a **different model** that **GEPA never sees
  during training**. I use it to score every transcript against **five custom
  metrics** built from Nomie's persona (therapist-speak, preambles, question-stacking,
  tool use, and crisis-safety). It scores only the winner, on held-out data.

> **GEPA proposes. Cekura decides.** If GEPA just flattered its own judge, the
> independent one catches it.

*(How Cekura scores transcripts without running a simulation — and the rest of the
runbook — is in **[gepa/README.md](gepa/README.md)**.)*

---

## 5. The five runs — and what they taught me about guardrails

I ran the loop five times. The results were **not** what I expected — and the
negative results are the real deliverable. (Full write-ups in
**[gepa-reports/](gepa-reports/)**.)

| Run | Setup | What happened |
|---|---|---|
| **1** — [section-locked](gepa-reports/gepa-run-1.md) | evolve one section, lock the rest | **+11% on the inner judge, flat on Cekura.** A win only my own judge could see — overfitting, caught by the independent judge. |
| **2–3** — whole-prompt | nothing locked | 🛑 GEPA **silently deleted the 988 crisis-safety protocol.** Nothing scored safety, so the optimizer treated it as dead weight. |
| **4** — [+ safety gate](gepa-reports/gepa-run-2.md) | hard-gate 988 (`−100`) + crisis cases in eval | GEPA **couldn't win** without surfacing 988; the result was *safer* (3/3 vs 2/3) — but it stripped the tools and gutted the persona (−85% size). |
| **5** — + tool gate | hard-gate tools, score on the real deploy harness | Every non-negotiable now gated; scored on the shape I actually ship. |

**The takeaway:**

> **An optimizer gives you exactly what you measure — and quietly throws away
> everything you didn't.** The hard part was never making Nomie improve itself; it
> was teaching it what it's **not allowed to lose.**

What that means in practice — the guardrails:

- **Two independent judges.** Optimize against a fast judge; *verify* with a separate
  one the optimizer never sees. Disagreement is the signal.
- **Safety is a hard gate, not a score.** A crisis without 988 scores `−100` — the
  optimizer literally cannot trade it away. (Proven: 988 survived, 3/3 vs 2/3.)
- **An eval only measures what it exercises.** Tools "looked fine" only because the
  sample didn't test tool requests — the capability was actually gone. Put crisis and
  tool cases *in* the eval.
- **Bounded beats unconstrained.** The only run that was safe **and** improved **and**
  still full-featured was the **section-locked** one. "Hard-gate every non-negotiable,"
  taken to its limit, *is* section-locking.

I didn't end up with a smarter prompt. I ended up with **the system that catches a
"better" prompt that's lying.**

---

## 6. Building it into production

At 5,000 users, the loop is **offline and human-gated** — never online auto-editing.
Two halves:

```
ONLINE  (every turn)          a prompt-independent safety guardrail
                              → surfaces 988 no matter the prompt version; an
                                optimizer can never delete it

OFFLINE (weekly)              Cekura-traced convos + outcome logging
                              → stratified golden set from real voice traces
                              → section-locked, hard-gated GEPA
                              → held-out eval (2 judges + crisis & tool probes)
                              → human-review PR
                              → canary rollout 5% → 25% → 100%, measured on real
                                outcomes (next-day return), instant rollback
```

- **Online:** a separate crisis classifier on every live turn — safety lives *outside*
  the prompt, so nothing in the loop can remove it.
- **Offline:** weekly is plenty. Build the dataset from **real voice traces**,
  stratified across session lengths (5-minute venters *and* hours-long power users),
  run the **bounded, hard-gated** loop, verify on held-out data, open a
  **human-reviewed PR**, and roll out as a **canary** measured on whether people
  actually come back — with one-click rollback.
- **Calibrate the judge to reality:** with real usage you can finally check whether a
  higher judge score predicts retention — validate before you trust.

**GEPA proposes. My evaluation decides.** That's how you let a companion people trust
evolve itself — safely.

---

## Repo layout

```
.
├── voice-agent/     the real-time voice agent GEPA optimizes (NVIDIA · Pipecat · Cekura)
├── gepa/            the self-improvement loop (optimizer · judges · golden set) + runbook
└── gepa-reports/    captured run results, including the negative ones (the evidence)
```

## Run it

```bash
cd gepa
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# set NVIDIA_API_KEY, OPENAI_API_KEY (+ CEKURA_*, UPSTASH_* as needed) — see gepa/README.md
.venv/bin/python -m pytest -q          # pure-logic tests, no network
```

Full step-by-step (build the golden set, both optimize paths, the held-out report,
applying a winner) is in **[gepa/README.md](gepa/README.md)**.

---

## 📱 Talk to Nomie

Nomie is live on the App Store — **[download it here](https://apps.apple.com/us/app/nomie-ai-wellness-companion/id6757396354)**.

> Note: the production app runs on **OpenAI** models (not the NVIDIA stack in this
> repo), and includes the self-evolving prompt improvements from this work.
