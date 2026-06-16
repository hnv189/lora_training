"""Generate a realistic HuggingFace ``trainer_state.json`` for a 3-epoch LoRA run.

This mirrors what the actual ``train.py`` would write, so the dashboard shows a
real-looking loss analysis out of the box (the run exhibits the classic small-
dataset behaviour: best checkpoint at epoch 2, slight overfit by epoch 3).
"""

from __future__ import annotations

import json
import math
import os
import random

random.seed(7)

EPOCHS = 3
STEPS_PER_EPOCH = 80          # ~200 pairs, effective batch 8 (multiple of LOG_EVERY)
LOG_EVERY = 10
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
PEAK_LR = 2e-4
WARMUP = int(0.05 * TOTAL_STEPS)


def lr_at(step: int) -> float:
    """Cosine schedule with linear warmup (matches training_args)."""
    if step < WARMUP:
        return PEAK_LR * step / max(WARMUP, 1)
    progress = (step - WARMUP) / max(TOTAL_STEPS - WARMUP, 1)
    return PEAK_LR * 0.5 * (1 + math.cos(math.pi * progress))


def train_loss_at(step: int) -> float:
    """Smooth exponential decay from ~1.9 to ~0.35 with small noise."""
    base = 0.32 + 1.55 * math.exp(-3.0 * step / TOTAL_STEPS)
    return round(base + random.uniform(-0.04, 0.04), 4)


def eval_loss_at(epoch: int) -> float:
    """Eval loss bottoms out at epoch 2, then ticks up (mild overfit)."""
    table = {1: 0.92, 2: 0.71, 3: 0.78}
    return round(table[epoch] + random.uniform(-0.01, 0.01), 4)


def grad_norm_at(step: int) -> float:
    """Stays inside the healthy 0.1-0.5 band, larger early on."""
    base = 0.45 * math.exp(-2.0 * step / TOTAL_STEPS) + 0.12
    return round(base + random.uniform(-0.03, 0.03), 4)


def build_state() -> dict:
    log_history = []
    for step in range(LOG_EVERY, TOTAL_STEPS + 1, LOG_EVERY):
        epoch = round(step / STEPS_PER_EPOCH, 4)
        log_history.append(
            {
                "step": step,
                "epoch": epoch,
                "loss": train_loss_at(step),
                "learning_rate": round(lr_at(step), 8),
                "grad_norm": grad_norm_at(step),
            }
        )
        # End-of-epoch eval entry.
        if step % STEPS_PER_EPOCH == 0:
            ep = step // STEPS_PER_EPOCH
            log_history.append(
                {
                    "step": step,
                    "epoch": float(ep),
                    "eval_loss": eval_loss_at(ep),
                    "eval_runtime": round(random.uniform(8, 12), 3),
                }
            )

    best = min(
        (r for r in log_history if "eval_loss" in r), key=lambda r: r["eval_loss"]
    )
    return {
        "best_metric": best["eval_loss"],
        "best_model_checkpoint": f"./lora-checkpoints/checkpoint-{best['step']}",
        "epoch": float(EPOCHS),
        "global_step": TOTAL_STEPS,
        "max_steps": TOTAL_STEPS,
        "num_train_epochs": EPOCHS,
        "log_history": log_history,
    }


def main():
    out = os.path.join(os.path.dirname(__file__), "..", "sample_data", "trainer_state.json")
    out = os.path.abspath(out)
    state = build_state()
    with open(out, "w") as fh:
        json.dump(state, fh, indent=2)
    print(f"Wrote {len(state['log_history'])} log entries -> {out}")
    print(f"Best eval_loss {state['best_metric']} at {state['best_model_checkpoint']}")


if __name__ == "__main__":
    main()
