"""Apply the GEPA winning prompt to the agent's prompt home(s).

GEPA optimizes the entire Nomie system prompt, so we replace the whole prompt
wherever it lives:
  - voice-agent/system_prompt.txt          (this repo's Pipecat/NIM bot)
  - packages/functions/src/routes/voice.ts (prod OpenAI-Realtime NOMIE_SYSTEM_PROMPT,
    in the full Nomie monorepo)

In the full monorepo this keeps both prompt homes in sync. In this standalone
extract only system_prompt.txt exists, so the voice.ts step is skipped
automatically unless you pass --voice-ts pointing at a real file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _apply_to_ts(path: Path, prompt: str) -> bool:
    """Replace the whole NOMIE_SYSTEM_PROMPT template literal with `prompt`.

    The optimized text is freeform, so escape the three template-literal
    breakers (backslash, backtick, ${) — JS unescapes them back to the literal
    string at runtime."""
    src = path.read_text()
    m = re.search(r"(NOMIE_SYSTEM_PROMPT\s*=\s*`)(.*?)(`)", src, re.S)
    if not m:
        raise ValueError("NOMIE_SYSTEM_PROMPT template literal not found in voice.ts")
    safe = prompt.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    new = src[: m.start(2)] + safe + src[m.end(2):]
    if new == src:
        return False
    path.write_text(new)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Apply the dspy.GEPA winning prompt to both prompt homes.")
    p.add_argument("--result", required=True, help="optimizer result JSON ({optimized_prompt})")
    p.add_argument("--system-prompt", default="../voice-agent/system_prompt.txt")
    p.add_argument("--voice-ts", default=None,
                   help="Optional: prod voice.ts with a NOMIE_SYSTEM_PROMPT template literal")
    args = p.parse_args()

    prompt = json.loads(Path(args.result).read_text())["optimized_prompt"]
    Path(args.system_prompt).write_text(prompt if prompt.endswith("\n") else prompt + "\n")

    ts_changed: bool | None = None
    if args.voice_ts and Path(args.voice_ts).exists():
        ts_changed = _apply_to_ts(Path(args.voice_ts), prompt)
    print(f"[apply] system_prompt.txt written ({len(prompt)} chars); voice.ts changed: {ts_changed}")


if __name__ == "__main__":
    main()
