"""LoRA training pipeline: def -> implement behaviour shaping.

Modules map 1:1 to the training plan phases:

* :mod:`data_audit`   — Phase 0 (count, token stats, categories, fingerprint)
* :mod:`prepare_data` — Phase 1 (behaviour-first prompt, reasoning, quality filter)
* :mod:`train`        — Phases 2-3 (4-bit Qwen3-8B + LoRA, loss-masked SFT)
* :mod:`evaluate`     — Phase 4 (CodeBLEU + behaviour checklist + OOD)
* :mod:`metrics`      — training-log loss / overfit / gradient analysis
"""

__all__ = ["data_audit", "prepare_data", "train", "evaluate", "metrics"]
__version__ = "1.0.0"
