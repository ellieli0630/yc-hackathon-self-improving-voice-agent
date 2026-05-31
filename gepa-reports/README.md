# GEPA Reports

Run-by-run results from optimizing the Nomie voice prompt with GEPA. Each report
is a captured pull request — kept as evidence, including the negative results.

| Report | Scope | Result |
|---|---|---|
| [section-locked (round 1)](gepa-run-1.md) | One section (`THINGS YOU NEVER DO`), everything else locked | +11% on the inner judge, **flat** on the independent judge — the proxy-overfit lesson |
| [whole-prompt (round 4)](gepa-run-2.md) | Entire prompt, nothing locked | Prompt collapses **−85%**, safe but **stripped** (lost tools + persona) — 🛑 do not merge |

## The takeaway across runs

An optimizer gives you *exactly* what you measure, on *exactly* the data you exercise — and deletes the rest.

- **Two independent judges** (inner gpt-4o-mini + Cekura) are what catch the gaming: every inner-judge "win" was flat or worse on the authoritative judge.
- **Safety can be made un-deletable** — a per-instance hard penalty + crisis detector forced 988 to survive (3/3 vs 2/3).
- **But hard-gating safety isn't enough** — a *safe* optimizer still strips everything else you don't gate.
- **So section-locking (#121) was right.** The only config that was safe **and** improved **and** still full-featured was the bounded, locked-section run. "Hard-gate every non-negotiable," taken to its limit, *is* section-locking.

> The hard part of a self-improving agent isn't the optimizer — it's the evaluation and the constraints.
