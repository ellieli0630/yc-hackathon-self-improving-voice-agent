# Nomie Prompt Optimization (GEPA + Cekura)

Evolve Nomie's voice system prompt with **GEPA** (the published reflective prompt
optimizer), scored by a fast local LLM-judge, and independently validated by
**Cekura** on a held-out split. Everything runs locally against remote APIs
(NVIDIA NIM, OpenAI, Cekura) — no GPUs, no Docker.

> **Run every command from this directory (`gepa`).** The dir is named
> `gepa`, which shadows the installed `gepa` PyPI package; running from here makes
> `import gepa` resolve to the library, not this folder.

---

## 1. Setup

```bash
cd gepa
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

That single venv covers everything (judge, optimizer, golden-set build, tests).

## 2. Environment variables

| Var | Used by | Source (SST secret) |
|-----|---------|---------------------|
| `NVIDIA_API_KEY` | task LM (`replay.py` → NIM nemotron) | `NgcApiKey` |
| `OPENAI_API_KEY` | inner judge + GEPA reflector | `OpenAIApiKey` |
| `CEKURA_API_KEY` | held-out report only | `CekuraApiKey` |
| `CEKURA_AGENT_ID` | held-out report only | `CekuraAgentId` |
| `UPSTASH_VECTOR_REST_URL` / `_TOKEN` | golden-set build only | `UpstashVectorUrl` / `UpstashVectorToken` |

**Option A — pull straight from SST** (if you have repo access; no file needed).
Run from repo root:

```bash
# NVIDIA + OpenAI (enough to optimize + report-without-Cekura)
eval "$(yarn sst secret list --stage hack 2>/dev/null | awk -F= '
  /^NgcApiKey=/    {print "export NVIDIA_API_KEY=" substr($0,index($0,"=")+1)}
  /^OpenAIApiKey=/ {print "export OPENAI_API_KEY=" substr($0,index($0,"=")+1)}')"
```

For Cekura scoring also export `CEKURA_API_KEY` (`CekuraApiKey`) and
`CEKURA_AGENT_ID` (`CekuraAgentId`) the same way.

**Option B — a local `.env`** (no SST access; just bring your own keys). The
scripts read `os.environ` directly, so create `gepa/.env` and source it
into your shell before running:

```bash
cat > .env <<'EOF'
NVIDIA_API_KEY=nvapi-...        # build.nvidia.com  (Llama + Parakeet)
OPENAI_API_KEY=sk-...           # judge + GEPA reflector
CEKURA_API_KEY=...              # optional — held-out report only
CEKURA_AGENT_ID=...             # optional — held-out report only
UPSTASH_VECTOR_REST_URL=https://...   # optional — only to rebuild the golden set
UPSTASH_VECTOR_REST_TOKEN=...         # optional — only to rebuild the golden set
EOF

set -a; source .env; set +a   # export every var into the shell, then run as normal
```

`.env` is gitignored — never commit real keys. You only need `NVIDIA_API_KEY` +
`OPENAI_API_KEY` to optimize; the rest are optional for the steps noted above.

## 3. The pipeline

```
build_golden_set.py   →   <optimize>   →   report.py   →   apply_winner.py
   (real prod data)        (evolve)        (base vs winner)   (write both prompt homes)
```

### a. Build the golden set (optional — a set already lives in `golden_set/`)
Reads real, PII-redacted prod conversations from Upstash (READ-ONLY) into
`golden_set/*.json`:

```bash
.venv/bin/python build_golden_set.py --out-dir golden_set --top-k 200 --min-turns 4
```

Then split into train/test JSON (`{turns, cut_point}` per conversation). The runs
below assume `train.json` / `test.json` (held-out) already exist.

### b. Optimize — two paths

**Recommended: deployment-harness path** (`gepa_native.py`). Calls `gepa.optimize()`
directly with an adapter that scores every candidate through the *real* `replay()`
deployment harness (raw system prompt + multi-turn messages), so the winner is
optimal for the shape we actually ship:

```bash
.venv/bin/python -u gepa_native.py \
  --base-prompt seed.txt --trainset train.json --valset test.json \
  --max-metric-calls 150 --out /tmp/result.json
```

> Use `python -u` so GEPA's per-iteration log lines flush live. **Watch for
> `Proposed new text` in the output** — if you only ever see `did not propose`,
> the run is a no-op (see Gotchas). `--smoke` runs one example through the adapter
> + metric and exits.

**Alternative: dspy path** (`dspy_gepa.py`) — optimizes inside DSPy's ChatAdapter
wrapper (a different harness than deployment; kept for comparison):

```bash
.venv/bin/python dspy_gepa.py \
  --base-prompt seed.txt --trainset train.json --valset val.json \
  --auto light --out /tmp/result.json
```

Both write `{ "optimized_prompt": "...", "base_prompt": "..." }`.

### c. Report — honest base-vs-winner on held-out data
Inner judge on every test conversation + Cekura on a bounded sample:

```bash
.venv/bin/python report.py \
  --base-prompt seed.txt --winner-prompt /tmp/result.json \
  --test test.json --cekura-sample 15 --out /tmp/report.json
```

(Skip Cekura by leaving `CEKURA_*` unset — it'll report the inner judge only.)

### d. Apply the winner (only if it actually passed)
Writes the prompt into **both** homes — the hack bot and the prod route:

```bash
.venv/bin/python apply_winner.py --result /tmp/result.json
```

## 4. Tests

```bash
.venv/bin/python -m pytest -q        # judge gates, Cekura coercion, golden-set shape
```

## 5. How scoring works (the short version)

- **Inner judge** (`judge.py`, OpenAI `gpt-4o-mini`): scores every candidate on
  empathy, brevity, and style axes (no-preamble / no-therapist-tell /
  not-question-machine). **Safety and tool-delegation are *hard gates*** —
  a crisis without "988" or unsafe advice scores `-100`; a requested-but-unoffered
  exercise scores `-2`. The optimizer literally cannot win by trading these away.
- **Reflector** (GEPA's mutation engine, OpenAI `gpt-4o`): reads the judge's
  feedback and rewrites the prompt. Strong reflector + cheap task LM is the
  intended pattern.
- **Cekura** (`cekura_score.py`): an independent, separately-authored rubric on a
  different model. GEPA never sees it — it's the held-out arbiter that tells you
  whether a win is real or just gaming the inner judge.
- **Task LM**: NIM `nemotron-super-49b-v1.5` (matches the shipped agent).

## 6. Gotchas

- **Run from `gepa/`** (the `import gepa` shadow, above).
- **A progress bar advancing ≠ working.** GEPA can re-score the base forever while
  every mutation silently fails. Always confirm `Proposed new text` appears in the
  log within the first couple of iterations.
- **NIM is stochastic** (temp 0.6): whether the base surfaces "988" on a given
  crisis turn varies run-to-run, which can flip the `-100` gate and swing the
  aggregate by ~3 points on a small valset. Don't over-read a single small run.
- **Network stalls**: the NIM/judge clients have a 60s hard timeout (`max_retries=1`)
  so one hung call can't block a whole minibatch.
