"""Phase 1 — Data preparation.

Builds the behaviour-first prompt, augments pairs with synthetic reasoning
traces, applies the quality filter and writes a HuggingFace ``datasets`` split.

Reasoning-trace generation is optional: it only runs if an Anthropic API key is
present. Without it, the module still builds a dataset using whatever reasoning
already exists on each pair (or a templated placeholder), so the pipeline is
runnable offline.
"""

from __future__ import annotations

import json
import os
from typing import Any

PROMPT_TEMPLATE = """### Task:
You are implementing code based on a formal definition. Read the definition carefully,
identify the intent and constraints, then write a complete implementation that follows
the project's established conventions.

### Definition:
{definition}

### Reasoning:
{reasoning}

### Implementation:
{implementation}"""

RESPONSE_TEMPLATE = "### Implementation:"


def format_sample(pair: dict) -> dict:
    """Render a single pair into the behaviour-first training text."""
    return {
        "text": PROMPT_TEMPLATE.format(
            definition=pair["def"].strip(),
            reasoning=pair.get("reasoning", "").strip()
            or "Analyse the definition's core responsibility, edge cases, and the "
            "project conventions it must follow before writing the implementation.",
            implementation=pair["impl"].strip(),
        )
    }


# ---------------------------------------------------------------------------
# Synthetic reasoning traces (plan 1.2)
# ---------------------------------------------------------------------------


def generate_reasoning(def_content: str, impl_content: str) -> str:
    """Use Claude to bridge def -> impl, when an API key is available."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = f"""Given this definition:
---
{def_content}
---
And this implementation:
---
{impl_content}
---
Write 3-5 sentences explaining the reasoning a developer would use to go from the
definition to this specific implementation. Focus on: why this structure was chosen,
how edge cases are handled, and what conventions are being followed. Be specific."""
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:  # pragma: no cover - network/credentials dependent
        print(f"[prepare_data] reasoning generation skipped: {exc}")
        return ""


def augment_with_reasoning(pairs: list[dict]) -> list[dict]:
    for pair in pairs:
        if not pair.get("reasoning"):
            pair["reasoning"] = generate_reasoning(pair["def"], pair["impl"])
    return pairs


# ---------------------------------------------------------------------------
# Quality filter (plan 1.4)
# ---------------------------------------------------------------------------


def is_quality_sample(text: str, token_counter, max_tokens: int = 2048) -> tuple[bool, str]:
    tokens = token_counter(text)
    if tokens > max_tokens:
        return False, f"too long ({tokens} > {max_tokens} tokens)"

    impl = text.split(RESPONSE_TEMPLATE)[-1].strip()
    if len(impl.splitlines()) < 3:
        return False, "implementation too short (likely a stub)"

    defn = text.split("### Definition:")[-1].split("### Reasoning:")[0].strip()
    if len(defn) < 30:
        return False, "definition too vague (< 30 chars)"

    return True, "ok"


def quality_report(pairs: list[dict], max_tokens: int = 2048) -> dict[str, Any]:
    """Run the filter and report what would be kept/dropped (no HF dependency)."""
    from .data_audit import count_tokens

    counter = lambda t: count_tokens(t, None)
    kept, dropped = [], []
    for pair in pairs:
        text = format_sample(pair)["text"]
        ok, reason = is_quality_sample(text, counter, max_tokens)
        (kept if ok else dropped).append(
            {"name": pair.get("name", "?"), "reason": reason}
        )
    return {
        "total": len(pairs),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_detail": dropped,
    }


# ---------------------------------------------------------------------------
# Build + save dataset (plan 1.3) — requires `datasets`
# ---------------------------------------------------------------------------


def build_dataset(
    pairs: list[dict],
    out_dir: str = "./dataset",
    test_size: float = 0.1,
    max_tokens: int = 2048,
    seed: int = 42,
):
    from datasets import Dataset
    from .data_audit import count_tokens

    pairs = augment_with_reasoning(pairs)
    counter = lambda t: count_tokens(t, None)

    formatted = []
    for pair in pairs:
        sample = format_sample(pair)
        ok, _ = is_quality_sample(sample["text"], counter, max_tokens)
        if ok:
            formatted.append(sample)

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    split.save_to_disk(out_dir)
    print(f"Saved {len(split['train'])} train / {len(split['test'])} eval to {out_dir}")
    return split


if __name__ == "__main__":
    with open("sample_data/pairs.json") as fh:
        data = json.load(fh)
    print(json.dumps(quality_report(data), indent=2))
