"""Phase 0 — Data audit.

Counts pairs, measures token lengths, categorises implementation patterns and
extracts the codebase's behavioural fingerprint. Everything here is plain
Python with graceful fallbacks so the audit runs even without a tokenizer
installed (it falls back to a whitespace+punctuation heuristic).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Tokenisation (optional transformers dependency)
# ---------------------------------------------------------------------------


def get_tokenizer(model: str = "Qwen/Qwen3-8B"):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer(text)["input_ids"])
        except Exception:
            pass
    # Heuristic: words + punctuation, roughly ~0.75 the size of a BPE count, so
    # scale up to approximate sub-word tokenisation.
    rough = len(re.findall(r"\w+|[^\w\s]", text))
    return int(rough * 1.3)


# ---------------------------------------------------------------------------
# Loading pairs
# ---------------------------------------------------------------------------


def load_pairs(
    linking_dir: str = "linking",
    def_dir: str = "def",
    impl_dir: str = "implement",
    fallback_json: str | None = None,
) -> list[dict]:
    """Load def/impl pairs.

    Primary path follows the plan: a ``linking/`` folder of JSON link files,
    each pointing at a ``def/`` and ``implement/`` file. If that layout is not
    present, fall back to a flat JSON list of ``{"def":..., "impl":...}``.
    """
    pairs: list[dict] = []
    if os.path.isdir(linking_dir):
        for link_file in sorted(os.listdir(linking_dir)):
            try:
                with open(os.path.join(linking_dir, link_file)) as fh:
                    link = json.load(fh)
                pairs.append(
                    {
                        "name": link_file,
                        "def": open(os.path.join(def_dir, link["def"])).read(),
                        "impl": open(os.path.join(impl_dir, link["impl"])).read(),
                    }
                )
            except Exception:
                continue
    if not pairs and fallback_json and os.path.exists(fallback_json):
        with open(fallback_json) as fh:
            data = json.load(fh)
        for i, p in enumerate(data):
            pairs.append(
                {
                    "name": p.get("name", f"pair_{i}"),
                    "def": p["def"],
                    "impl": p["impl"],
                }
            )
    return pairs


# ---------------------------------------------------------------------------
# Token-length statistics
# ---------------------------------------------------------------------------


def token_stats(pairs: list[dict], tokenizer=None) -> dict[str, Any]:
    lengths = []
    for p in pairs:
        combined = p["def"] + p["impl"]
        n = count_tokens(combined, tokenizer)
        p["tokens"] = n
        lengths.append(n)
    lengths.sort()
    if not lengths:
        return {"count": 0}
    n = len(lengths)
    p95 = lengths[min(int(n * 0.95), n - 1)]
    return {
        "count": n,
        "min": lengths[0],
        "median": lengths[n // 2],
        "p95": p95,
        "max": lengths[-1],
        # Set max_seq_length to P95, not max — outliers cause OOM (per plan 0.2).
        "recommended_max_seq_length": _round_up_pow2(p95),
        "histogram": _histogram(lengths, bins=12),
    }


def _round_up_pow2(n: int) -> int:
    for cap in (512, 1024, 1536, 2048, 3072, 4096, 6144, 8192):
        if n <= cap:
            return cap
    return 8192


def _histogram(values: list[int], bins: int = 12) -> list[dict]:
    if not values:
        return []
    lo, hi = values[0], values[-1]
    if hi == lo:
        return [{"range": f"{lo}", "count": len(values)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    return [
        {
            "range": f"{int(lo + i * width)}-{int(lo + (i + 1) * width)}",
            "count": c,
        }
        for i, c in enumerate(counts)
    ]


# ---------------------------------------------------------------------------
# Pattern categorisation (plan 0.3)
# ---------------------------------------------------------------------------


def categorize(pairs: list[dict]) -> dict[str, Any]:
    cats: Counter = Counter()
    for p in pairs:
        cats[_classify(p["impl"])] += 1
    total = sum(cats.values()) or 1
    rows = [
        {
            "category": cat,
            "count": cnt,
            "pct": round(100 * cnt / total, 1),
        }
        for cat, cnt in cats.most_common()
    ]
    # An imbalanced set trains skewed behaviour (plan 0.3).
    top_pct = rows[0]["pct"] if rows else 0
    balanced = top_pct <= 60
    return {"categories": rows, "balanced": balanced, "dominant_pct": top_pct}


def _classify(impl: str) -> str:
    has_class = bool(re.search(r"^\s*class\s+\w+", impl, re.M))
    has_func = bool(re.search(r"^\s*def\s+\w+", impl, re.M))
    raises = bool(re.search(r"\braise\b", impl))
    func_count = len(re.findall(r"^\s*def\s+\w+", impl, re.M))

    if has_class:
        return "Class with methods"
    if raises and has_func:
        return "Error-handling pattern"
    if func_count > 1:
        return "Multi-function module"
    if has_func:
        return "Simple function"
    return "Other / script"


# ---------------------------------------------------------------------------
# Behavioural fingerprint (plan 0.4)
# ---------------------------------------------------------------------------


def behavioural_fingerprint(pairs: list[dict]) -> dict[str, Any]:
    """Heuristically extract the codebase's conventions into a checklist."""
    snake = camel = 0
    raises = returns_none = asserts = logging = 0
    docstrings = inline_comments = 0
    n = len(pairs) or 1

    for p in pairs:
        impl = p["impl"]
        snake += len(re.findall(r"\bdef\s+[a-z]+_[a-z_]+", impl))
        camel += len(re.findall(r"\bdef\s+[a-z]+[A-Z]", impl))
        raises += len(re.findall(r"\braise\b", impl))
        returns_none += len(re.findall(r"return\s+None|return\s*$", impl, re.M))
        asserts += len(re.findall(r"\bassert\b", impl))
        logging += len(re.findall(r"\b(logging|logger)\.", impl))
        docstrings += impl.count('"""') // 2
        inline_comments += len(re.findall(r"(?m)#.+$", impl))

    naming = "snake_case" if snake >= camel else "camelCase"
    err_styles = {
        "raise": raises,
        "return None": returns_none,
        "assert": asserts,
        "logging": logging,
    }
    err_dominant = max(err_styles, key=err_styles.get) if any(err_styles.values()) else "n/a"

    return {
        "naming_convention": naming,
        "error_handling_style": err_dominant,
        "error_handling_breakdown": err_styles,
        "uses_docstrings": docstrings > 0,
        "uses_inline_comments": inline_comments > 0,
        "avg_docstrings_per_impl": round(docstrings / n, 2),
        # This is the evaluation checklist used later in Phase 4.
        "eval_checklist": [
            "Naming conventions match project style",
            "Error handling matches how the codebase handles it",
            "Docstring / comment style matches",
            "Import order matches",
            "Edge cases handled the same way",
            "No hallucinated functions or libraries",
            "Structure (class vs function vs module) matches",
            "No TODO stubs — implementation is complete",
        ],
    }


def run_full_audit(
    linking_dir: str = "linking",
    def_dir: str = "def",
    impl_dir: str = "implement",
    fallback_json: str | None = "sample_data/pairs.json",
    use_tokenizer: bool = False,
) -> dict[str, Any]:
    pairs = load_pairs(linking_dir, def_dir, impl_dir, fallback_json)
    tokenizer = get_tokenizer() if use_tokenizer else None
    stats = token_stats(pairs, tokenizer)

    # Minimum viable: 100 pairs; under 50 collect more (plan 0.1).
    count = len(pairs)
    if count == 0:
        readiness = "no_data"
    elif count < 50:
        readiness = "collect_more"
    elif count < 100:
        readiness = "viable"
    else:
        readiness = "good"

    return {
        "pair_count": count,
        "readiness": readiness,
        "token_stats": stats,
        "categories": categorize(pairs),
        "fingerprint": behavioural_fingerprint(pairs),
        "tokenizer_used": tokenizer is not None,
    }


if __name__ == "__main__":
    print(json.dumps(run_full_audit(), indent=2))
