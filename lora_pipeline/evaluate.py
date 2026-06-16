"""Phase 4 — Behaviour evaluation.

Combines automated structural metrics (CodeBLEU when available, a lightweight
fallback otherwise) with the behaviour checklist from the plan. The web app
uses :func:`score_behaviour_checklist` and :func:`aggregate_eval` to render the
behaviour-eval tab; nothing here requires a GPU.
"""

from __future__ import annotations

import json
import re
from typing import Any

CHECKLIST_ITEMS = [
    "Naming conventions match project style",
    "Error handling matches how the codebase handles it",
    "Docstring / comment style matches",
    "Import order matches",
    "Edge cases handled the same way",
    "No hallucinated functions or libraries",
    "Structure (class vs function vs module) matches",
    "No TODO stubs — implementation is complete",
]

# Failure-mode diagnosis table (plan 4.4).
FAILURE_MODES = [
    {
        "symptom": "Generic code ignoring def specifics",
        "cause": "Memorised output shape, not intent",
        "fix": "Add more reasoning traces, increase r",
    },
    {
        "symptom": "Wrong naming style",
        "cause": "Not enough examples of that pattern",
        "fix": "Augment dataset with more of that type",
    },
    {
        "symptom": "Hallucinated imports",
        "cause": "Base model bleeding through",
        "fix": "More epochs or increase LoRA rank",
    },
    {
        "symptom": "Incomplete implementations",
        "cause": "Loss masking too aggressive or seq too short",
        "fix": "Check max_seq_length, check collator",
    },
    {
        "symptom": "Reproduces def verbatim",
        "cause": "Prompt confusion",
        "fix": "Ensure loss masking is working correctly",
    },
]


# ---------------------------------------------------------------------------
# Automated structural similarity
# ---------------------------------------------------------------------------


def code_similarity(reference: str, prediction: str, lang: str = "python") -> dict[str, Any]:
    """CodeBLEU if installed, else a token+line Jaccard fallback."""
    try:
        from codebleu import calc_codebleu

        res = calc_codebleu([reference], [prediction], lang=lang)
        res["method"] = "codebleu"
        return res
    except Exception:
        return _fallback_similarity(reference, prediction)


def _fallback_similarity(reference: str, prediction: str) -> dict[str, Any]:
    def toks(s: str) -> set[str]:
        return set(re.findall(r"\w+", s))

    ref_t, pred_t = toks(reference), toks(prediction)
    ngram = len(ref_t & pred_t) / (len(ref_t | pred_t) or 1)

    ref_lines = {l.strip() for l in reference.splitlines() if l.strip()}
    pred_lines = {l.strip() for l in prediction.splitlines() if l.strip()}
    syntax = len(ref_lines & pred_lines) / (len(ref_lines | pred_lines) or 1)

    score = round(0.5 * ngram + 0.5 * syntax, 4)
    return {
        "codebleu": score,
        "ngram_match": round(ngram, 4),
        "syntax_match": round(syntax, 4),
        "method": "fallback_jaccard",
    }


# ---------------------------------------------------------------------------
# Behaviour checklist (plan 4.2)
# ---------------------------------------------------------------------------


def score_behaviour_checklist(scores: dict[str, float]) -> dict[str, Any]:
    """Aggregate a single sample's checklist scores (0 / 0.5 / 1 each)."""
    items = []
    total = 0.0
    for item in CHECKLIST_ITEMS:
        s = float(scores.get(item, 0))
        total += s
        items.append({"item": item, "score": s})
    avg = round(total / len(CHECKLIST_ITEMS), 3)
    return {"items": items, "average": avg, "pass": avg > 0.75}


def aggregate_eval(samples: list[dict]) -> dict[str, Any]:
    """Aggregate many evaluated samples into the dashboard payload.

    Each sample: ``{"name", "checklist": {item: score}, "codebleu": float,
    "ood": bool}``.
    """
    if not samples:
        return {"count": 0, "avg_checklist": 0, "avg_codebleu": 0, "samples": []}

    rows = []
    sum_check = sum_bleu = 0.0
    ood_pass = ood_total = 0
    for s in samples:
        cl = score_behaviour_checklist(s.get("checklist", {}))
        bleu = float(s.get("codebleu", 0))
        sum_check += cl["average"]
        sum_bleu += bleu
        if s.get("ood"):
            ood_total += 1
            if cl["average"] > 0.75:
                ood_pass += 1
        rows.append(
            {
                "name": s.get("name", "?"),
                "checklist_avg": cl["average"],
                "codebleu": round(bleu, 3),
                "ood": bool(s.get("ood")),
                "pass": cl["pass"] and bleu >= 0.60,
            }
        )

    n = len(samples)
    avg_check = round(sum_check / n, 3)
    avg_bleu = round(sum_bleu / n, 3)
    return {
        "count": n,
        "avg_checklist": avg_check,            # target > 0.75
        "avg_codebleu": avg_bleu,              # target > 0.60
        "checklist_target_met": avg_check > 0.75,
        "codebleu_target_met": avg_bleu > 0.60,
        "ood_pass_rate": round(ood_pass / ood_total, 3) if ood_total else None,
        "ood_total": ood_total,
        "samples": rows,
        "failure_modes": FAILURE_MODES,
    }


if __name__ == "__main__":
    with open("sample_data/behaviour_eval.json") as fh:
        samples = json.load(fh)
    print(json.dumps(aggregate_eval(samples), indent=2))
